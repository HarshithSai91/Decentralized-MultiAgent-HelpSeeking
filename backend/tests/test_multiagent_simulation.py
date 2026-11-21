"""Tests for MultiAgentSimulationRun: reset, reproducibility, oracle consistency, and metrics."""
from __future__ import annotations

import numpy as np
import pytest

from help_seeking.multiagent_environment import MultiAgentEnvironment
from help_seeking.multiagent_simulation import (
    MultiAgentSimulationRun,
    MultiAgentRunRecord,
    compute_multiagent_metrics,
)
from help_seeking.reliability import StationaryProfile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_env(n_agents=3, seed=0, policy_type="D-TS"):
    profiles = [StationaryProfile([0.7]) for _ in range(n_agents)]
    return MultiAgentEnvironment(
        n_agents=n_agents,
        reliability_profiles=profiles,
        R=1.0,
        U_self=0.4,
        lambda_mult=1.0,
        requester_policy_type=policy_type,
        requester_policy_params={"gamma": 0.98},
        seed=seed,
    )


# ---------------------------------------------------------------------------
# 1. Run returns a MultiAgentRunRecord of the right shape
# ---------------------------------------------------------------------------

def test_run_returns_correct_shape():
    env = make_env(n_agents=3)
    sim = MultiAgentSimulationRun(env, T=50)
    record = sim.run()
    assert isinstance(record, MultiAgentRunRecord)
    assert record.T == 50
    assert record.n_agents == 3
    assert record.regrets.shape == (50,)
    assert record.mu_stars.shape == (50,)
    assert record.opt_actions.shape == (50,)
    assert record.workloads.shape == (50, 3)
    assert record.resources.shape == (50, 3)
    assert len(record.interactions) == 50


# ---------------------------------------------------------------------------
# 2. Same seed → identical results (reproducibility)
# ---------------------------------------------------------------------------

def test_reproducibility():
    env1 = make_env(n_agents=3, seed=42)
    sim1 = MultiAgentSimulationRun(env1, T=100)
    r1 = sim1.run()

    env2 = make_env(n_agents=3, seed=42)
    sim2 = MultiAgentSimulationRun(env2, T=100)
    r2 = sim2.run()

    np.testing.assert_array_equal(r1.regrets, r2.regrets)
    np.testing.assert_array_equal(r1.opt_actions, r2.opt_actions)
    np.testing.assert_array_equal(r1.workloads, r2.workloads)


# ---------------------------------------------------------------------------
# 3. Reset produces the same results as a fresh run
# ---------------------------------------------------------------------------

def test_reset_reproducibility():
    env = make_env(n_agents=3, seed=7, policy_type="Greedy")
    sim = MultiAgentSimulationRun(env, T=60)
    r1 = sim.run()

    # Reset and run again — should give same results
    sim.reset()
    r2 = sim.run()

    np.testing.assert_array_equal(r1.regrets, r2.regrets)
    np.testing.assert_array_equal(r1.opt_actions, r2.opt_actions)


# ---------------------------------------------------------------------------
# 4. Regret is non-negative at every step (or near-zero rounding only)
# ---------------------------------------------------------------------------

def test_regret_nonnegative():
    env = make_env(n_agents=4, seed=1, policy_type="D-UCB")
    sim = MultiAgentSimulationRun(env, T=100)
    record = sim.run()
    assert float(record.regrets.min()) >= -1e-9, f"Negative regret: {record.regrets.min()}"


# ---------------------------------------------------------------------------
# 5. Cumulative regret is non-decreasing
# ---------------------------------------------------------------------------

def test_cumulative_regret_nondecreasing():
    env = make_env(n_agents=3, seed=2)
    sim = MultiAgentSimulationRun(env, T=100)
    record = sim.run()
    cum_regret = np.cumsum(record.regrets)
    diffs = np.diff(cum_regret)
    assert np.all(diffs >= -1e-9), "Cumulative regret decreased at some step"


# ---------------------------------------------------------------------------
# 6. compute_multiagent_metrics returns expected keys
# ---------------------------------------------------------------------------

def test_compute_metrics_keys():
    env = make_env(n_agents=3, seed=3)
    sim = MultiAgentSimulationRun(env, T=50)
    record = sim.run()
    metrics = compute_multiagent_metrics(record, env)
    for key in ["cumulative_regret", "optimal_action_rate", "acceptance_rate",
                "ask_rate", "avg_workload", "avg_resources"]:
        assert key in metrics, f"Missing metric: {key}"


# ---------------------------------------------------------------------------
# 7. Optimal action rate is in [0, 1]
# ---------------------------------------------------------------------------

def test_optimal_action_rate_in_range():
    env = make_env(n_agents=3, seed=4)
    sim = MultiAgentSimulationRun(env, T=200)
    record = sim.run()
    metrics = compute_multiagent_metrics(record, env)
    oar = metrics["optimal_action_rate"]
    assert 0.0 <= oar <= 1.0, f"opt_action_rate out of range: {oar}"


# ---------------------------------------------------------------------------
# 8. history() returns the right number of records with required keys
# ---------------------------------------------------------------------------

def test_history_length():
    env = make_env(n_agents=3, seed=5)
    sim = MultiAgentSimulationRun(env, T=30)
    record = sim.run()
    h = record.history()
    assert len(h) == 30
    for entry in h:
        for k in ["round", "requester_id", "helper_id", "asked", "accepted",
                  "success", "requester_reward", "helper_reward", "regret"]:
            assert k in entry, f"Missing key '{k}' in history entry"


# ---------------------------------------------------------------------------
# 9. All three algorithms can run without error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("policy_type", ["Greedy", "D-UCB", "D-TS"])
def test_all_algorithms_run(policy_type):
    env = make_env(n_agents=3, seed=10, policy_type=policy_type)
    sim = MultiAgentSimulationRun(env, T=50)
    record = sim.run()
    assert len(record.interactions) == 50


# ---------------------------------------------------------------------------
# 10. Workload and resources are within discretized bounds [0, 2]
# ---------------------------------------------------------------------------

def test_workload_resources_in_bounds():
    env = make_env(n_agents=4, seed=6)
    sim = MultiAgentSimulationRun(env, T=100)
    record = sim.run()
    assert record.workloads.min() >= 0
    assert record.workloads.max() <= 2
    assert record.resources.min() >= 0
    assert record.resources.max() <= 2


# ---------------------------------------------------------------------------
# 11. Different seeds produce different results
# ---------------------------------------------------------------------------

def test_different_seeds_differ():
    env1 = make_env(n_agents=3, seed=100)
    sim1 = MultiAgentSimulationRun(env1, T=100)
    r1 = sim1.run()

    env2 = make_env(n_agents=3, seed=200)
    sim2 = MultiAgentSimulationRun(env2, T=100)
    r2 = sim2.run()

    # Very unlikely to be identical with different seeds
    assert not np.array_equal(r1.regrets, r2.regrets), "Different seeds produced identical regrets"


# ---------------------------------------------------------------------------
# 12. Record beliefs shape is correct when record_beliefs=True
# ---------------------------------------------------------------------------

def test_belief_recording_shape():
    n_agents = 4
    T = 30
    env = make_env(n_agents=n_agents, seed=8)
    sim = MultiAgentSimulationRun(env, T=T, record_beliefs=True)
    record = sim.run()
    # acceptance_beliefs and trust_beliefs shape: (T, n_agents, n_agents-1)
    assert record.acceptance_beliefs.shape == (T, n_agents, n_agents - 1)
    assert record.trust_beliefs.shape == (T, n_agents, n_agents - 1)
    # All values should be in [0, 1]
    assert record.acceptance_beliefs.min() >= 0.0
    assert record.acceptance_beliefs.max() <= 1.0


# ---------------------------------------------------------------------------
# 13. Policy stability metric and checkpoint tests (Section 2)
# ---------------------------------------------------------------------------

def test_q_snapshots_populated():
    """Verify that Q-table snapshots are recorded at regular intervals with visit counts."""
    env = make_env(n_agents=3, seed=1)
    sim = MultiAgentSimulationRun(env, T=150)
    record = sim.run()
    # At T=150, snapshots should be taken at round 50, 100, 150
    assert len(record.q_snapshots) >= 3
    assert len(sim.q_snapshots) >= 3
    assert record.q_snapshots[0]["t"] == 50
    assert record.q_snapshots[1]["t"] == 100
    assert record.q_snapshots[2]["t"] == 150
    # Each snapshot should have Q-tables and visit counts for all agents
    for snap in record.q_snapshots:
        assert len(snap["q"]) == 3
        assert "visits" in snap
        assert len(snap["visits"]) == 3
        for i in range(3):
            assert snap["q"][i].shape == (9, 2)
            assert snap["visits"][i].shape == (9,)


def test_helper_policy_stability_insufficient_visits_returns_nan():
    """When states have insufficient visits (< MIN_VISITS=5), stability must be None (unavailable), NOT 100%."""
    env = make_env(n_agents=2, seed=1)
    sim = MultiAgentSimulationRun(env, T=100)
    record = sim.run()

    # All states have fewer than MIN_VISITS (e.g. 2 visits)
    v_low = np.full(9, 2, dtype=int)
    record.q_snapshots = [
        {"t": 50, "q": {0: np.ones((9, 2)), 1: np.ones((9, 2))}, "visits": {0: v_low, 1: v_low}},
        {"t": 100, "q": {0: np.ones((9, 2)), 1: np.ones((9, 2))}, "visits": {0: v_low, 1: v_low}},
    ]
    metrics = compute_multiagent_metrics(record, env)
    assert metrics["helper_policy_stability"] is None, "Expected None when visits < MIN_VISITS"


def test_helper_policy_stability_metric_unchanged_policy():
    """When policies are identical across snapshots with sufficient visits, stability must be 1.0."""
    env = make_env(n_agents=2, seed=1)
    sim = MultiAgentSimulationRun(env, T=100)
    record = sim.run()

    # Artificially set identical snapshots with sufficient visits (e.g. 10 visits)
    identical_q = np.ones((9, 2))
    identical_q[:, 0] = 2.0  # action 0 is greedy everywhere
    v_high = np.full(9, 10, dtype=int)
    record.q_snapshots = [
        {"t": 50, "q": {0: identical_q.copy(), 1: identical_q.copy()}, "visits": {0: v_high, 1: v_high}},
        {"t": 100, "q": {0: identical_q.copy(), 1: identical_q.copy()}, "visits": {0: v_high, 1: v_high}},
    ]
    metrics = compute_multiagent_metrics(record, env)
    assert np.isclose(metrics["helper_policy_stability"], 1.0)
    assert np.isclose(metrics["helper_policy_convergence"], 1.0)


def test_helper_policy_stability_metric_flipped_policy():
    """When greedy actions flip between snapshots, stability drops proportionally."""
    env = make_env(n_agents=2, seed=1)
    sim = MultiAgentSimulationRun(env, T=100)
    record = sim.run()

    v_high = np.full(9, 10, dtype=int)
    q_snap1 = np.zeros((9, 2))
    q_snap1[:, 0] = 1.0  # all states prefer action 0

    q_snap2 = np.zeros((9, 2))
    q_snap2[:, 0] = 1.0
    # Flip action on 3 out of 9 states (states 0, 1, 2)
    q_snap2[0:3, 1] = 2.0  # prefer action 1

    record.q_snapshots = [
        {"t": 50, "q": {0: q_snap1, 1: q_snap1}, "visits": {0: v_high, 1: v_high}},
        {"t": 100, "q": {0: q_snap2, 1: q_snap1}, "visits": {0: v_high, 1: v_high}},
    ]
    metrics = compute_multiagent_metrics(record, env)
    # Agent 0 has 6/9 unchanged; Agent 1 has 9/9 unchanged -> avg = (6/9 + 9/9)/2 = 15/18 = 5/6 ≈ 0.8333
    assert np.isclose(metrics["helper_policy_stability"], (6.0/9.0 + 1.0) / 2.0)


def test_helper_policy_stability_unvisited_states():
    """Unvisited states (Q=0) are handled explicitly and don't produce division errors."""
    env = make_env(n_agents=2, seed=1)
    sim = MultiAgentSimulationRun(env, T=100)
    record = sim.run()

    # Only state 0 is visited (has non-zero Q)
    q_snap1 = np.zeros((9, 2))
    q_snap1[0, 0] = 1.0

    q_snap2 = np.zeros((9, 2))
    q_snap2[0, 0] = 1.0

    record.q_snapshots = [
        {"t": 50, "q": {0: q_snap1, 1: q_snap1}},
        {"t": 100, "q": {0: q_snap2, 1: q_snap1}},
    ]
    metrics = compute_multiagent_metrics(record, env)
    assert np.isclose(metrics["helper_policy_stability"], 1.0)

