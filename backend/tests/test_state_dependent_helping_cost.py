"""Regression tests for the state-dependent helping cost.

A helper's reward for an accepted request is

    r_j = B_j * x - cost(w, rho),   cost(w, rho) = c_res * (1 + (2 - rho)) + c_opp * (1 + w)

evaluated in the state (w, rho) the helper was in when it accepted.  Helping is cheapest
when idle with full resources and most expensive when busy and depleted, so the helper's
private state genuinely changes which action is best.
"""
import numpy as np
import pytest

from help_seeking.autonomous_agent import AutonomousAgent, ACCEPT
from help_seeking.multiagent_environment import MultiAgentEnvironment
from help_seeking.temporal_environment import TemporalMultiAgentEnvironment
from help_seeking.reliability import StationaryProfile


def _agent(**kw):
    return AutonomousAgent(agent_id=0, n_agents=3, requester_policy_type="D-TS",
                           requester_policy_params={"gamma": 0.98}, seed=0, **kw)


def test_helping_cost_grows_with_workload_and_depletion():
    a = _agent(B_i=0.5, resource_cost=0.1, opportunity_cost=0.05)
    assert np.isclose(a.helping_cost(0, 2), 0.15)   # idle, full resources
    assert np.isclose(a.helping_cost(1, 1), 0.30)   # medium
    assert np.isclose(a.helping_cost(2, 0), 0.45)   # busy, depleted
    assert np.isclose(a.helping_cost(2, 2), 0.25)   # workload term alone
    assert np.isclose(a.helping_cost(0, 0), 0.35)   # resource term alone
    # strictly increasing along the diagonal
    costs = [a.helping_cost(w, 2 - w) for w in range(3)]
    assert costs[0] < costs[1] < costs[2]


def test_helper_reward_defaults_to_current_state():
    a = _agent(B_i=0.5, resource_cost=0.1, opportunity_cost=0.05)
    a.workload, a.resources = 2, 0
    assert np.isclose(a.calculate_helper_reward(1), 0.5 - 0.45)
    assert np.isclose(a.calculate_helper_reward(0), -0.45)
    # explicit state overrides the current one
    assert np.isclose(a.calculate_helper_reward(1, 0, 2), 0.5 - 0.15)


def test_state_changes_sign_of_expected_payoff():
    """With B=0.5 and theta=0.5 accepting pays when idle but not when busy."""
    a = _agent(B_i=0.5, resource_cost=0.1, opportunity_cost=0.05)
    theta = 0.5
    assert a.B_i * theta - a.helping_cost(0, 2) > 0
    assert a.B_i * theta - a.helping_cost(2, 0) < 0


def _capture_states(agents):
    """Wrap decide_help so each call records the helper's state at decision time."""
    seen = []
    for ag in agents:
        orig = ag.decide_help

        def wrapped(requester_id, _ag=ag, _orig=orig):
            seen.append((_ag.agent_id, _ag.workload, _ag.resources))
            return _orig(requester_id)
        ag.decide_help = wrapped
    return seen


def test_multiagent_env_rewards_helper_in_pre_acceptance_state():
    profiles = [StationaryProfile([0.7]) for _ in range(3)]
    env = MultiAgentEnvironment(n_agents=3, reliability_profiles=profiles, R=1.0, U_self=0.0,
                                helper_policy_type="always_accept",
                                helper_rewards={"B_i": 0.5, "resource_cost": 0.1,
                                                "opportunity_cost": 0.05},
                                requester_policy_type="D-TS",
                                requester_policy_params={"gamma": 0.98}, seed=4)
    seen = _capture_states(env.agents)
    checked = 0
    for _ in range(60):
        for ag in env.agents:          # put every potential helper in the busy state
            ag.workload, ag.resources = 2, 0
        n_before = len(seen)
        rec = env.step()
        if rec.asked and rec.accepted:
            hid, w, rho = seen[n_before]
            assert hid == rec.helper_id and (w, rho) == (2, 0)
            assert (rec.helper_workload_before, rec.helper_resources_before) == (2, 0)
            assert np.isclose(rec.helper_reward, 0.5 * rec.success - 0.45)
            checked += 1
    assert checked > 0


def test_temporal_env_rewards_helper_in_pre_acceptance_state():
    profiles = [StationaryProfile([0.8]) for _ in range(3)]
    env = TemporalMultiAgentEnvironment(n_agents=3, reliability_profiles=profiles, U_self=0.05,
                                        timing_policy_type="immediate",
                                        helper_policy_type="always_accept",
                                        helper_rewards={"B_i": 0.5, "resource_cost": 0.1,
                                                        "opportunity_cost": 0.05},
                                        default_deadline=5, seed=11)
    seen = _capture_states(env.agents)
    checked = 0
    for _ in range(40):
        for ag in env.agents:
            ag.workload, ag.resources = 2, 0
        n_before = len(seen)
        rec = env.run_task()
        if rec.help_requested and rec.help_accepted:
            hid, w, rho = seen[n_before]
            helper = env.agents[hid]
            expected = helper.B_i * rec.helper_success - helper.helping_cost(w, rho)
            assert np.isclose(rec.helper_reward, expected)
            checked += 1
    assert checked > 0


def test_informed_threshold_helper_follows_immediate_payoff_sign():
    a = _agent(B_i=0.5, resource_cost=0.1, opportunity_cost=0.05,
               helper_policy_type="informed_threshold", execution_reliability=0.5)
    a.workload, a.resources = 0, 2          # 0.25 - 0.15 > 0 -> accept
    assert a.decide_help(1) == ACCEPT and a.get_hypothetical_p_accept(0) == 1.0
    a.workload, a.resources = 2, 0          # 0.25 - 0.45 < 0 -> reject
    assert a.decide_help(1) != ACCEPT and a.get_hypothetical_p_accept(0) == 0.0


def test_oracle_uses_informed_threshold_acceptance():
    profiles = [StationaryProfile([0.8]) for _ in range(3)]
    env = MultiAgentEnvironment(n_agents=3, reliability_profiles=profiles, R=1.0, U_self=0.3,
                                helper_policy_type="informed_threshold",
                                requester_policy_type="D-TS",
                                requester_policy_params={"gamma": 0.98}, seed=1)
    for ag in env.agents:
        ag.execution_reliability = 0.8
    env.agents[1].workload, env.agents[1].resources = 0, 2   # 0.40 - 0.15 > 0
    env.agents[2].workload, env.agents[2].resources = 2, 0   # 0.40 - 0.45 < 0
    oracle = env.compute_oracle(0)
    assert np.allclose(oracle["p_true"], [1.0, 0.0])
