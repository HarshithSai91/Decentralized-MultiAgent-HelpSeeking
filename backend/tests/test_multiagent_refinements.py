"""Comprehensive test suite for multi-agent system refinements.

Covers:
1. Explicit workload/resource activity transitions (execute_own_task, help_another_agent, rest, reject_request).
2. Decision-epoch (Semi-Markov) Q-learning: deferred update with geometric discounting across elapsed rounds.
3. Oracle tie handling and optimal-action sets for multiple optimal helpers and ties with acting alone.
4. Undefined metrics handling: returning None/NaN when observations are 0.
5. Social welfare calculation: W = r_requester + r_helper without double counting communication costs.
6. State-conditioned helper decision tracking (low workload/high resources, medium/medium, high workload/low resources).
7. E7 multi-agent abrupt changing reliability experiment mechanics.
"""
from __future__ import annotations

import numpy as np
import pytest

from help_seeking.autonomous_agent import AutonomousAgent, ACCEPT, REJECT
from help_seeking.helper_qlearning import HelperQLearning, encode_state
from help_seeking.multiagent_environment import MultiAgentEnvironment
from help_seeking.multiagent_simulation import MultiAgentSimulationRun, compute_multiagent_metrics
from help_seeking.reliability import StationaryProfile, AbruptChangeProfile


def test_workload_resource_activity_transitions():
    """Verify separate transitions for execute_own_task, help_another_agent, rest, reject_request."""
    agent = AutonomousAgent(agent_id=0, n_agents=3)
    assert agent.workload == 0
    assert agent.resources == 2

    # 1. Execute own task: workload increases, resources decrease
    w_old, r_old = agent.execute_own_task()
    assert (w_old, r_old) == (0, 2)
    assert agent.workload == 1
    assert agent.resources == 1

    # 2. Help another agent: workload increases, resources decrease
    agent.help_another_agent()
    assert agent.workload == 2
    assert agent.resources == 0

    # 3. Cap checks
    agent.help_another_agent()
    assert agent.workload == 2
    assert agent.resources == 0

    # 4. Rest: workload decreases, resources replenish
    agent.rest()
    assert agent.workload == 1
    assert agent.resources == 1

    # 5. Reject request: helper avoids task execution, relieves workload, preserves resources
    agent.reject_request()
    assert agent.workload == 0
    assert agent.resources == 1


def test_decision_epoch_qlearning():
    """Verify semi-Markov decision-epoch Q-learning with geometric discounting."""
    hq = HelperQLearning(alpha=0.5, gamma_q=0.9, epsilon=0.0)

    # Initial state (w=0, r=2) -> state 2
    # Action ACCEPT=0, immediate reward = 1.0, decided at round t=10
    hq.record_decision(workload=0, resources=2, action=0, reward=1.0, round_t=10)
    assert hq.pending_transition is not None
    assert hq.q_table[2, 0] == 0.0

    # Simulate next decision epoch at round t=14 (elapsed delta_t = 4)
    # Helper's state at this new decision is (w=1, r=1) -> state 4
    # Suppose Q(state 4, ACCEPT) = 0.8, Q(state 4, REJECT) = 0.2
    hq.q_table[4, 0] = 0.8
    hq.q_table[4, 1] = 0.2

    # Flush pending transition
    flushed = hq.flush_pending(next_workload=1, next_resources=1, current_round=14)
    assert flushed
    assert hq.pending_transition is None

    # Expected target:
    # gamma_eff = 0.9^4 = 0.6561
    # target = 1.0 + 0.6561 * 0.8 = 1.52488
    # Q(2, 0) = 0 + 0.5 * (1.52488 - 0) = 0.76244
    expected_target = 1.0 + (0.9 ** 4) * 0.8
    expected_q = 0.5 * expected_target
    assert np.isclose(hq.q_table[2, 0], expected_q)


def test_decision_epoch_episode_finalization():
    """Verify that remaining pending transitions are flushed at episode end."""
    hq = HelperQLearning(alpha=1.0, gamma_q=0.9, epsilon=0.0)
    hq.record_decision(workload=1, resources=1, action=1, reward=0.5, round_t=100)
    assert hq.pending_transition is not None

    # Finalize at round 105 at state (w=0, r=2)
    hq.finalize_episode(final_workload=0, final_resources=2, final_round=105)
    assert hq.pending_transition is None
    # With max Q(state 2) = 0: target = 0.5 + 0 = 0.5
    assert np.isclose(hq.q_table[encode_state(1, 1), 1], 0.5)


def test_oracle_multiple_optimal_helpers_tie():
    """Verify that when two helpers have identical optimal utility, both are marked optimal."""
    # Requester 0 has 2 helpers with identical theta=0.8, comm_cost=0.1, always_accept -> identical mu
    profiles = [StationaryProfile([0.5]), StationaryProfile([0.8]), StationaryProfile([0.8])]
    costs = np.array([
        [0.0, 0.1, 0.1],
        [0.1, 0.0, 0.1],
        [0.1, 0.1, 0.0],
    ])
    env = MultiAgentEnvironment(
        n_agents=3,
        reliability_profiles=profiles,
        R=1.0,
        U_self=0.3,
        comm_costs=costs,
        helper_policy_type="always_accept",
        seed=1,
    )
    oracle = env.compute_oracle(0)
    # Both helper 0 (agent 1) and helper 1 (agent 2) have mu = 0.3 + 1.0*(0.8 - 0.3) - 0.1 = 0.7
    assert np.isclose(oracle["mu_star"], 0.7)
    opt_acts = oracle["optimal_actions"]
    # Both helpers should be in optimal_actions
    assert len(opt_acts) == 2
    helpers_in_opt = {opt["a"] for opt in opt_acts if opt["d"] == 1}
    assert helpers_in_opt == {0, 1}


def test_oracle_tie_between_acting_alone_and_asking():
    """Verify that when asking yields the same utility as acting alone, both are marked optimal."""
    # U_self = 0.5. Helper has theta=0.6, comm_cost=0.1 -> mu = 0.5 + 1.0*(0.6 - 0.5) - 0.1 = 0.5 == U_self
    profiles = [StationaryProfile([0.5]), StationaryProfile([0.6])]
    costs = np.array([
        [0.0, 0.1],
        [0.1, 0.0],
    ])
    env = MultiAgentEnvironment(
        n_agents=2,
        reliability_profiles=profiles,
        R=1.0,
        U_self=0.5,
        comm_costs=costs,
        helper_policy_type="always_accept",
        seed=2,
    )
    oracle = env.compute_oracle(0)
    assert np.isclose(oracle["mu_star"], 0.5)
    opt_acts = oracle["optimal_actions"]
    assert any(opt["d"] == 0 for opt in opt_acts)
    assert any(opt["d"] == 1 and opt["a"] == 0 for opt in opt_acts)


def test_undefined_metrics_handling():
    """When an algorithm never asks, acceptance rate must be None, not 0.0."""
    profiles = [StationaryProfile([0.5]), StationaryProfile([0.8])]
    env = MultiAgentEnvironment(
        n_agents=2,
        reliability_profiles=profiles,
        R=1.0,
        U_self=0.9,  # High U_self forces acting alone
        helper_policy_type="always_accept",
        requester_policy_type="Greedy",
        requester_policy_params={"U_self": 0.9},
        seed=3,
    )
    sim = MultiAgentSimulationRun(env, T=20)
    rec = sim.run()
    metrics = compute_multiagent_metrics(rec, env)

    # No asks were performed
    assert metrics["ask_rate"] == 0.0
    assert metrics["acceptance_rate"] is None
    assert metrics["rejection_rate"] is None
    assert metrics["success_rate"] is None


def test_social_welfare_and_utilities():
    """Verify that social welfare equals requester utility plus helper reward without double subtraction."""
    profiles = [StationaryProfile([0.5]), StationaryProfile([1.0])]  # helper always succeeds
    costs = np.array([[0.0, 0.1], [0.1, 0.0]])
    env = MultiAgentEnvironment(
        n_agents=2,
        reliability_profiles=profiles,
        R=1.0,
        U_self=0.2,
        comm_costs=costs,
        helper_rewards={"B_i": 0.4, "resource_cost": 0.1, "opportunity_cost": 0.05},
        helper_policy_type="always_accept",
        requester_policy_type="Greedy",
        requester_policy_params={"U_self": 0.2},
        seed=4,
    )
    sim = MultiAgentSimulationRun(env, T=10)
    rec = sim.run()
    metrics = compute_multiagent_metrics(rec, env)

    assert "social_welfare" in metrics
    assert "requester_utility" in metrics
    assert "helper_utility" in metrics
    assert np.isclose(metrics["social_welfare"], metrics["requester_utility"] + metrics["helper_utility"])


def test_state_conditioned_helper_metrics():
    """Verify that compute_multiagent_metrics computes state-conditioned helper rates for E5."""
    profiles = [StationaryProfile([0.5]), StationaryProfile([0.8])]
    env = MultiAgentEnvironment(n_agents=2, reliability_profiles=profiles, seed=5)
    sim = MultiAgentSimulationRun(env, T=50)
    rec = sim.run()
    metrics = compute_multiagent_metrics(rec, env)

    for state_key in ["low_w_high_r", "med_w_med_r", "high_w_low_r"]:
        assert f"{state_key}_visits" in metrics
        assert f"{state_key}_accept_rate" in metrics
        assert f"{state_key}_reject_rate" in metrics


def test_e7_abrupt_reliability_change_integration():
    """Verify that MultiAgentEnvironment correctly runs with abrupt reliability change profiles."""
    profiles = [
        AbruptChangeProfile([0.9], [0.4], tau=10),
        AbruptChangeProfile([0.7], [0.8], tau=10),
    ]
    env = MultiAgentEnvironment(n_agents=2, reliability_profiles=profiles, seed=6)
    sim = MultiAgentSimulationRun(env, T=20)
    rec = sim.run()

    # Pre-change reliability of agent 0 is 0.9, post-change is 0.4
    assert np.isclose(rec.execution_reliabilities[5, 0], 0.9)
    assert np.isclose(rec.execution_reliabilities[15, 0], 0.4)
    # Agent 1 changes from 0.7 to 0.8
    assert np.isclose(rec.execution_reliabilities[5, 1], 0.7)
    assert np.isclose(rec.execution_reliabilities[15, 1], 0.8)


def test_global_belief_decay_all_agents():
    """Verify that all agents' acceptance and trust beliefs decay every round."""
    gamma = 0.9
    profiles = [StationaryProfile([0.8]), StationaryProfile([0.8]), StationaryProfile([0.8])]
    env = MultiAgentEnvironment(
        n_agents=3,
        reliability_profiles=profiles,
        requester_policy_params={"gamma": gamma},
        seed=42,
    )

    # Inject known evidence into acceptance and trust beliefs for all agents
    for agent in env.agents:
        agent.acceptance.A.fill(10.0)
        agent.acceptance.R.fill(5.0)
        agent.trust.S.fill(8.0)
        agent.trust.F.fill(2.0)

    # Execute 1 round
    rec = env.step()
    req_id = rec.requester_id

    # Verify inactive agents decayed by exactly gamma
    for i, agent in enumerate(env.agents):
        if i != req_id:
            # Acceptance evidence decayed by gamma
            assert np.allclose(agent.acceptance.A, 10.0 * gamma)
            assert np.allclose(agent.acceptance.R, 5.0 * gamma)
            # Trust evidence decayed by gamma
            assert np.allclose(agent.trust.S, 8.0 * gamma)
            assert np.allclose(agent.trust.F, 2.0 * gamma)

    # For the requester:
    req_agent = env.agents[req_id]
    if rec.asked:
        h_idx = req_agent._agent_id_to_helper_idx(rec.helper_id)
        # Teammates other than the queried helper decayed by gamma
        for idx in range(env.n_agents - 1):
            if idx != h_idx:
                assert np.isclose(req_agent.acceptance.A[idx], 10.0 * gamma)
                assert np.isclose(req_agent.acceptance.R[idx], 5.0 * gamma)
                assert np.isclose(req_agent.trust.S[idx], 8.0 * gamma)
                assert np.isclose(req_agent.trust.F[idx], 2.0 * gamma)
        # Queried teammate received exactly +1 without double-discount
        if rec.accepted:
            assert np.isclose(req_agent.acceptance.A[h_idx], 10.0 * gamma + 1.0)
            if rec.success == 1:
                assert np.isclose(req_agent.trust.S[h_idx], 8.0 * gamma + 1.0)
            else:
                assert np.isclose(req_agent.trust.F[h_idx], 2.0 * gamma + 1.0)
        else:
            assert np.isclose(req_agent.acceptance.R[h_idx], 5.0 * gamma + 1.0)
            # Trust was not queried, so it simply decayed
            assert np.isclose(req_agent.trust.S[h_idx], 8.0 * gamma)
            assert np.isclose(req_agent.trust.F[h_idx], 2.0 * gamma)
    else:
        assert np.allclose(req_agent.acceptance.A, 10.0 * gamma)
        assert np.allclose(req_agent.acceptance.R, 5.0 * gamma)
        assert np.allclose(req_agent.trust.S, 8.0 * gamma)
        assert np.allclose(req_agent.trust.F, 2.0 * gamma)


def test_final_helper_snapshot_post_finalization():
    """Verify that final Q-table snapshot matches post-finalization state."""
    profiles = [StationaryProfile([0.8]), StationaryProfile([0.8])]
    env = MultiAgentEnvironment(n_agents=2, reliability_profiles=profiles, seed=123)
    sim = MultiAgentSimulationRun(env, T=10)
    rec = sim.run()

    final_snap = rec.q_snapshots[-1]
    assert final_snap["t"] == 10

    for i, agent in enumerate(env.agents):
        np.testing.assert_array_equal(final_snap["q"][i], agent.helper_qlearning.q_table)
        np.testing.assert_array_equal(final_snap["visits"][i], agent.helper_qlearning.state_visits)


def test_temporal_environment_global_belief_decay():
    """Verify global belief decay in TemporalMultiAgentEnvironment across consecutive rounds."""
    from help_seeking.temporal_environment import TemporalMultiAgentEnvironment

    gamma = 0.95
    profiles = [StationaryProfile([0.8]), StationaryProfile([0.8]), StationaryProfile([0.8])]
    env = TemporalMultiAgentEnvironment(
        n_agents=3,
        reliability_profiles=profiles,
        timing_policy_type="never_ask",
        requester_policy_params={"gamma": gamma},
        seed=99,
    )

    for agent in env.agents:
        agent.acceptance.A.fill(10.0)
        agent.acceptance.R.fill(5.0)
        agent.trust.S.fill(8.0)
        agent.trust.F.fill(2.0)

    # Run 1 task with never_ask (all steps are solo execution)
    task_rec = env.run_task(deadline=3)
    steps = task_rec.steps_taken
    assert steps > 0

    # Over `steps` rounds, every agent's evidence should decay by gamma^steps
    expected_decay = gamma ** steps
    for agent in env.agents:
        np.testing.assert_allclose(agent.acceptance.A, 10.0 * expected_decay, rtol=1e-5)
        np.testing.assert_allclose(agent.acceptance.R, 5.0 * expected_decay, rtol=1e-5)
        np.testing.assert_allclose(agent.trust.S, 8.0 * expected_decay, rtol=1e-5)
        np.testing.assert_allclose(agent.trust.F, 2.0 * expected_decay, rtol=1e-5)
