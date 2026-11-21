"""Metrics (tau_adapt, regret split, shares) and the experiment runner."""
import numpy as np
import pytest

from help_seeking.algorithms import DiscountedGreedy, DiscountedThompson, DiscountedUCB
from help_seeking.baselines import AlwaysAskRandom, DynamicOracle, InitialKnowledgeOracle, NeverAsk
from help_seeking.environment import SimulationEnvironment
from help_seeking.experiments import run_experiment
from help_seeking.metrics import adaptation_latency, summarize_run
from help_seeking.reliability import AbruptChangeProfile, DriftProfile, StationaryProfile
from help_seeking.simulation import SimulationRun
from help_seeking.utilities import dynamic_oracle

C = [0.1, 0.1, 0.1]


def make(policy_cls, theta=(0.7, 0.6, 0.5), U=0.35, T=300, seed=1, profile=None, **kw):
    prof = profile or StationaryProfile(theta)
    env = SimulationEnvironment(prof, R=1.0, U_self=U, c=C, lambda_mult=1.0, seed=seed)
    pol = policy_cls(3, c=C, seed=seed + 1000, **kw)
    return SimulationRun(env, pol, T)


# ----------------------------------------------------------------- metrics
def test_adaptation_latency_oracle_floor_is_26():
    prof = AbruptChangeProfile([0.8, 0.7, 0.9], [0.8, 0.7, 0.4], tau=100)
    env = SimulationEnvironment(prof, U_self=0.5, c=C, seed=1)
    rec = SimulationRun(env, DynamicOracle(3, c=C), 300).run()
    after = dynamic_oracle([0.8, 0.7, 0.4], 0.5, c=C)
    assert adaptation_latency(rec, 100, after) == (26, False)


def test_adaptation_latency_censored_for_initial_knowledge_oracle():
    prof = AbruptChangeProfile([0.8, 0.7, 0.9], [0.8, 0.7, 0.4], tau=100)
    env = SimulationEnvironment(prof, U_self=0.5, c=C, seed=1)
    pol = InitialKnowledgeOracle(3, c=C, theta_initial=[0.8, 0.7, 0.9], U_self=0.5)
    rec = SimulationRun(env, pol, 300).run()
    after = dynamic_oracle([0.8, 0.7, 0.4], 0.5, c=C)
    assert adaptation_latency(rec, 100, after) == (200, True)
    m = summarize_run(rec, 100, after)
    assert np.isclose(m["post_change_regret"], 200 * (0.7 - 0.3))
    assert m["pre_change_regret"] == 0.0


def test_summary_selection_shares_sum_to_one():
    m = summarize_run(make(DiscountedThompson, T=400).run())
    assert np.isclose(m["share_alone"] + m["share_h1"] + m["share_h2"] + m["share_h3"], 1.0)


# -------------------------------------------------------------- experiments
def test_experiment_runner_small_config(tmp_path):
    cfg = {
        "name": "T_small", "T": 120, "R": 1.0, "n_seeds": 3,
        "base": {"theta": [0.8, 0.7, 0.9], "c": C, "U_self": 0.5, "lambda": 1.0, "gamma": 0.98},
        "reliability": {"type": "abrupt", "theta_after": [0.8, 0.7, 0.4], "tau": 60},
        "sweep": {"name": "gamma", "points": [{"value": g, "set": {"gamma": g}} for g in (1.0, 0.95)]},
        "algorithms": ["Dynamic Oracle", "Initial-Knowledge Oracle", "D-TS"],
    }
    out = run_experiment(cfg, out_dir=str(tmp_path), verbose=False)
    assert len(out["rows"]) == 2 * 3 * 3
    oracle_rows = [r for r in out["rows"] if r["algorithm"] == "Dynamic Oracle"]
    assert all(r["cumulative_regret"] == 0 for r in oracle_rows)
    assert (tmp_path / "raw" / "T_small" / "runs.csv").exists()
    assert (tmp_path / "processed" / "T_small_summary.csv").exists()
    # same seed -> identical results when re-run (reproducibility)
    again = run_experiment(cfg, save=False, verbose=False)
    assert [r["cumulative_regret"] for r in again["rows"]] == [r["cumulative_regret"] for r in out["rows"]]
