"""Environment, profiles and the simulation loop."""
import numpy as np
import pytest

from help_seeking.algorithms import DiscountedGreedy, DiscountedThompson, DiscountedUCB
from help_seeking.baselines import AlwaysAskRandom, DynamicOracle, InitialKnowledgeOracle, NeverAsk
from help_seeking.environment import SimulationEnvironment
from help_seeking.experiments import run_experiment
from help_seeking.metrics import adaptation_latency, summarize_run
from help_seeking.reliability import (AbruptChangeProfile, DriftProfile, MultiShiftProfile,
                                      StationaryProfile, make_profile)
from help_seeking.simulation import SimulationRun
from help_seeking.utilities import dynamic_oracle

C = [0.1, 0.1, 0.1]


def make(policy_cls, theta=(0.7, 0.6, 0.5), U=0.35, T=300, seed=1, profile=None, **kw):
    prof = profile or StationaryProfile(theta)
    env = SimulationEnvironment(prof, R=1.0, U_self=U, c=C, lambda_mult=1.0, seed=seed)
    pol = policy_cls(3, c=C, seed=seed + 1000, **kw)
    return SimulationRun(env, pol, T)


# ---------------------------------------------------------------- profiles
def test_abrupt_change_happens_after_tau():
    p = AbruptChangeProfile([0.8, 0.7, 0.9], [0.8, 0.7, 0.4], tau=1000)
    assert np.allclose(p.get(1000), [0.8, 0.7, 0.9])
    assert np.allclose(p.get(1001), [0.8, 0.7, 0.4])


def test_drift_profile_reproducible_and_in_range():
    a = DriftProfile([0.7, 0.6], sigma=0.1, T=500, seed=4)
    b = DriftProfile([0.7, 0.6], sigma=0.1, T=500, seed=4)
    assert np.array_equal(a.path, b.path)
    assert np.all((a.path > 0) & (a.path < 1))
    assert np.allclose(a.get(1), [0.7, 0.6])


def test_multishift_profile_changes_at_rate_p_c():
    p = MultiShiftProfile([0.7, 0.6, 0.5], p_c=0.01, T=20000, seed=5, low=0.2, high=0.95)
    assert np.allclose(p.get(1), [0.7, 0.6, 0.5])
    assert abs(p.n_changes / 19999 - 0.01) < 0.002            # binomial, sd about 0.0007
    assert np.all((p.path[1:] >= 0.2) & (p.path <= 0.95)[1:])
    # piecewise constant: the vector only changes at the recorded shift rounds
    changed = [t for t in range(2, 20001) if not np.array_equal(p.get(t), p.get(t - 1))]
    assert changed == p.shift_rounds
    assert p.change_point is None


def test_multishift_reproducible_and_from_config():
    a = make_profile({"type": "multishift", "p_c": 0.005}, [0.7, 0.6, 0.5], 1000, seed=3)
    b = MultiShiftProfile([0.7, 0.6, 0.5], 0.005, 1000, seed=3)
    assert np.array_equal(a.path, b.path) and a.shift_rounds == b.shift_rounds
    assert MultiShiftProfile([0.5], 0.0, 100, seed=1).n_changes == 0


# ------------------------------------------------------------- environment
def test_outcomes_are_common_random_numbers():
    e1 = SimulationEnvironment(StationaryProfile([0.5, 0.5, 0.5]), c=C, seed=9)
    e2 = SimulationEnvironment(StationaryProfile([0.5, 0.5, 0.5]), c=C, seed=9)
    for _ in range(50):
        th = e1.start_round(); e2.start_round()
        assert [e1.generate_outcome(k, th) for k in range(3)] == \
               [e2.generate_outcome(k, th) for k in range(3)]


def test_outcome_frequency_matches_reliability():
    env = SimulationEnvironment(StationaryProfile([0.8, 0.3]), c=[0, 0], seed=2)
    xs = []
    for _ in range(20000):
        th = env.start_round()
        xs.append(env.generate_outcome(0, th))
    assert abs(np.mean(xs) - 0.8) < 0.01


def test_policy_and_environment_streams_are_independent():
    # The old code seeded both with the same integer, so Random's choice and the
    # outcome were driven by identical bit streams.  Now they come from separate seeds.
    sim = make(AlwaysAskRandom, theta=(0.5, 0.5, 0.5), T=4000)
    rec = sim.run()
    for k in range(3):
        assert abs(rec.x[rec.a == k].mean() - 0.5) < 0.05


# -------------------------------------------------------------- simulation
def test_dynamic_oracle_has_zero_pseudo_regret():
    rec = make(DynamicOracle, T=500).run()
    assert rec.regret.sum() == 0.0
    assert rec.opt_action.all()


def test_regret_is_nonnegative_every_round():
    for cls in (DiscountedGreedy, DiscountedUCB, DiscountedThompson, AlwaysAskRandom, NeverAsk):
        rec = make(cls, T=400).run()
        assert np.all(rec.regret >= -1e-12)


def test_never_ask_regret_hand_calculation():
    rec = make(NeverAsk, theta=(0.7, 0.6, 0.5), U=0.35, T=100).run()
    assert np.isclose(rec.regret.sum(), 100 * (0.6 - 0.35))
    assert np.allclose(rec.r, 0.35)
    assert rec.cost.sum() == 0


def test_realised_utility_and_cost_bookkeeping():
    rec = make(DiscountedThompson, T=300).run()
    asked = rec.d == 1
    assert np.allclose(rec.r[asked], rec.x[asked] - 0.1)
    assert np.isclose(rec.cost.sum(), 0.1 * asked.sum())


def test_learning_policies_never_receive_true_reliability():
    class Spy(DiscountedGreedy):
        def decide(self, U_self, theta_true=None):
            assert theta_true is None
            return super().decide(U_self, theta_true)
    make(Spy, T=50).run()


def test_same_seed_reproduces_identical_history():
    r1 = make(DiscountedThompson, T=300, seed=5).run()
    r2 = make(DiscountedThompson, T=300, seed=5).run()
    assert np.array_equal(r1.a, r2.a) and np.array_equal(r1.x, r2.x)


def test_step_details_for_inspector():
    sim = make(DiscountedUCB, T=5)
    info = sim.step(details=True)
    assert info["t"] == 1 and "bonus" in info["extra"]
    assert info["belief_before"]["theta_hat"] == [0.5, 0.5, 0.5]
    with pytest.raises(RuntimeError):
        for _ in range(10):
            sim.step()
