"""Tests for the AutonomousAgent class."""
import numpy as np
import pytest

from help_seeking.autonomous_agent import AutonomousAgent, ACCEPT, REJECT
from help_seeking.helper_qlearning import encode_state
from help_seeking.reliability import StationaryProfile


def make_agent(agent_id=0, n_agents=3, seed=0, **kwargs):
    """Create an agent with default stationary reliability."""
    profiles = [StationaryProfile([0.7]) for _ in range(n_agents)]
    # Remove execution_reliability from kwargs if present to avoid conflict
    exec_rel = kwargs.pop("execution_reliability", 0.7)
    return AutonomousAgent(
        agent_id=agent_id,
        n_agents=n_agents,
        requester_policy_type="D-TS",
        requester_policy_params={"gamma": 0.98},
        seed=seed,
        execution_reliability=exec_rel,
        **kwargs
    )


def test_agent_initialization():
    """Test agent initializes with correct defaults."""
    agent = make_agent()
    assert agent.agent_id == 0
    assert agent.n_agents == 3
    assert agent.workload == 0
    assert agent.resources == 2
    assert agent.R == 1.0
    assert agent.U_self == 0.5
    assert agent.B_i == 0.5


def test_requester_beliefs_initialization():
    """Test acceptance and trust beliefs are initialized."""
    agent = make_agent()
    # 2 potential helpers (n_agents - 1)
    assert agent.acceptance.A.shape == (2,)
    assert agent.acceptance.R.shape == (2,)
    assert agent.trust.S.shape == (2,)
    assert agent.trust.F.shape == (2,)
    # Prior means should be 0.5
    assert np.allclose(agent.get_acceptance_beliefs(), 0.5)
    assert np.allclose(agent.get_trust_beliefs(), 0.5)


def test_helper_qlearning_initialization():
    """Test helper Q-learning is initialized."""
    agent = make_agent()
    assert agent.helper_qlearning.q_table.shape == (9, 2)
    assert np.all(agent.helper_qlearning.q_table == 0.0)


def test_should_ask_with_high_utility():
    """Agent should ask when expected utility > U_self."""
    agent = make_agent(seed=0)
    # Set high acceptance and trust for helper 0
    agent.acceptance.A[0] = 100  # p_hat ~ 1
    agent.trust.S[0] = 100       # theta_hat ~ 1
    # Expected utility = U_self + 1*(1*1 - U_self) - 0.1 = 1 - 0.1 = 0.9 > 0.5
    should_ask, helper_idx, _ = agent.should_ask(U_self=0.5)
    assert should_ask is True
    assert helper_idx == 0


def test_should_not_ask_with_low_utility():
    """Agent should not ask when expected utility <= U_self."""
    agent = make_agent(seed=0)
    # Low acceptance and trust
    agent.acceptance.A[0] = 0
    agent.trust.S[0] = 0
    # Expected utility = 0.5 + 0.5*(1*0.5 - 0.5) - 0.1 = 0.5 - 0.1 = 0.4 < 0.5
    should_ask, helper_idx, _ = agent.should_ask(U_self=0.5)
    assert should_ask is False
    assert helper_idx == -1


def test_update_acceptance_belief():
    """Test acceptance belief updates correctly."""
    agent = make_agent(seed=0)
    # Helper agent_id=1 maps to helper_idx=0 (since agent_id=0)
    agent.update_acceptance_belief(helper_agent_id=1, accepted=True)
    assert agent.acceptance.A[0] == 1.0
    assert agent.acceptance.R[0] == 0.0

    agent.update_acceptance_belief(helper_agent_id=1, accepted=False)
    # With gamma=0.98: A = 0.98*1 + 0 = 0.98, R = 0.98*0 + 1 = 1
    assert np.isclose(agent.acceptance.A[0], 0.98)
    assert np.isclose(agent.acceptance.R[0], 1.0)


def test_update_trust_belief():
    """Test trust belief updates correctly."""
    agent = make_agent(seed=0)
    agent.update_trust_belief(helper_agent_id=1, success=1)
    assert agent.trust.S[0] == 1.0
    assert agent.trust.F[0] == 0.0

    agent.update_trust_belief(helper_agent_id=1, success=0)
    # With gamma=0.98: S = 0.98*1 + 0 = 0.98, F = 0.98*0 + 1 = 1
    assert np.isclose(agent.trust.S[0], 0.98)
    assert np.isclose(agent.trust.F[0], 1.0)


def test_decay_beliefs():
    """Test beliefs decay when acting alone."""
    agent = make_agent(seed=0)
    agent.acceptance.A[0] = 10
    agent.trust.S[0] = 10
    agent.decay_beliefs()
    # With gamma=0.98
    assert np.isclose(agent.acceptance.A[0], 9.8)
    assert np.isclose(agent.trust.S[0], 9.8)


def test_decide_help():
    """Test helper decision returns valid action."""
    agent = make_agent(seed=0)
    action = agent.decide_help(requester_id=1)
    assert action in (ACCEPT, REJECT)


def test_execute_task():
    """Test task execution returns binary outcome."""
    agent = make_agent(seed=0, execution_reliability=0.8)
    outcomes = [agent.execute_task() for _ in range(5000)]
    success_rate = np.mean(outcomes)
    assert abs(success_rate - 0.8) < 0.02


def test_calculate_helper_reward():
    """Test helper reward calculation in the idle state (w=0, rho=2), where cost = c_res + c_opp."""
    agent = make_agent(seed=0, B_i=1.0, resource_cost=0.2, opportunity_cost=0.1)
    # Success
    r_success = agent.calculate_helper_reward(1)
    assert np.isclose(r_success, 1.0 - 0.2 - 0.1)  # 0.7
    # Failure
    r_fail = agent.calculate_helper_reward(0)
    assert np.isclose(r_fail, 0.0 - 0.2 - 0.1)  # -0.3


def test_update_helper_policy():
    """Test Q-table updates after helping."""
    agent = make_agent(seed=0)
    agent.helper_qlearning.epsilon = 0.0  # No exploration
    # Initial state: workload=0, resources=2
    old_q = agent.helper_qlearning.q_table[encode_state(0, 2), ACCEPT]
    # Help with success
    agent.workload = 0
    agent.resources = 2
    reward = agent.calculate_helper_reward(1)
    agent.update_helper_policy(reward, next_workload=1, next_resources=1, action=ACCEPT)
    new_q = agent.helper_qlearning.q_table[encode_state(0, 2), ACCEPT]
    # Should have increased
    assert new_q > old_q


def test_update_workload_resources_when_helping():
    """Workload increases, resources decrease when helping."""
    agent = make_agent()
    agent.workload = 0
    agent.resources = 2
    old_w, old_r = agent.update_workload_resources(helped=True)
    assert old_w == 0 and old_r == 2
    assert agent.workload == 1
    assert agent.resources == 1


def test_update_workload_resources_when_not_helping():
    """Workload decreases, resources increase when not helping."""
    agent = make_agent()
    agent.workload = 2
    agent.resources = 0
    old_w, old_r = agent.update_workload_resources(helped=False)
    assert old_w == 2 and old_r == 0
    assert agent.workload == 1
    assert agent.resources == 1


def test_workload_resource_bounds():
    """Workload and resources stay in [0, 2]."""
    agent = make_agent()
    # At max, helping shouldn't exceed bounds
    agent.workload = 2
    agent.resources = 0
    agent.update_workload_resources(helped=True)
    assert agent.workload == 2
    assert agent.resources == 0

    # At min, not helping shouldn't go below bounds
    agent.workload = 0
    agent.resources = 2
    agent.update_workload_resources(helped=False)
    assert agent.workload == 0
    assert agent.resources == 2


def test_reset():
    """Reset should clear all beliefs and state."""
    agent = make_agent(seed=0)
    # Modify state
    agent.acceptance.A[0] = 10
    agent.trust.S[0] = 10
    agent.helper_qlearning.q_table[0, 0] = 5.0
    agent.workload = 2
    agent.resources = 0
    # Reset
    agent.reset(seed=0)
    # Check everything reset
    assert np.all(agent.acceptance.A == 0)
    assert np.all(agent.trust.S == 0)
    assert np.all(agent.helper_qlearning.q_table == 0)
    assert agent.workload == 0
    assert agent.resources == 2


def test_snapshot():
    """Snapshot should contain all relevant info."""
    agent = make_agent(seed=0)
    snap = agent.snapshot()
    assert "agent_id" in snap
    assert "requester" in snap
    assert "helper" in snap
    assert "rewards" in snap
    assert snap["requester"]["acceptance"]["A"] == [0.0, 0.0]
    assert snap["helper"]["workload"] == 0
    assert snap["helper"]["resources"] == 2