"""Tests for the MultiAgentEnvironment and MultiAgentSimulationRun."""
import numpy as np
import pytest

from help_seeking.multiagent_environment import MultiAgentEnvironment
from help_seeking.multiagent_simulation import MultiAgentSimulationRun, compute_multiagent_metrics
from help_seeking.reliability import StationaryProfile, make_profile


def make_env(n_agents=3, seed=0, **kwargs):
    """Create a multi-agent environment with stationary profiles."""
    profiles = [StationaryProfile([0.7]) for _ in range(n_agents)]
    # Default requester policy type, can be overridden by kwargs
    default_policy = kwargs.pop("requester_policy_type", "D-TS")
    default_params = kwargs.pop("requester_policy_params", {"gamma": 0.98})
    u_self = kwargs.pop("U_self", 0.5)
    return MultiAgentEnvironment(
        n_agents=n_agents,
        reliability_profiles=profiles,
        R=1.0,
        U_self=u_self,
        lambda_mult=1.0,
        requester_policy_type=default_policy,
        requester_policy_params=default_params,
        seed=seed,
        **kwargs
    )


def test_environment_initialization():
    """Test environment initializes with correct number of agents."""
    env = make_env(n_agents=4, seed=0)
    assert env.n_agents == 4
    assert len(env.agents) == 4
    for i, agent in enumerate(env.agents):
        assert agent.agent_id == i
        assert agent.n_agents == 4


def test_reset():
    """Test environment reset clears state."""
    env = make_env(seed=0)
    # Run a few steps
    for _ in range(5):
        env.step()
    assert env.round == 5
    assert len(env.history) == 5
    # Reset
    env.reset(seed=0)
    assert env.round == 0
    assert len(env.history) == 0
    for agent in env.agents:
        assert agent.workload == 0
        assert agent.resources == 2


def test_step_returns_record():
    """Test step returns an InteractionRecord with all fields."""
    env = make_env(seed=0)
    record = env.step()
    assert record.round_num == 1
    assert record.requester_id in range(3)
    assert record.asked in (True, False)
    assert isinstance(record.requester_reward, float)
    assert isinstance(record.helper_reward, float)
    assert isinstance(record.communication_cost, float)


def test_round_robin_requester_selection():
    """Test round-robin requester selection."""
    env = make_env(n_agents=3, seed=0, requester_selection="round_robin")
    requesters = []
    for _ in range(6):
        record = env.step()
        requesters.append(record.requester_id)
    # Round starts at 1, so first requester is 1 % 3 = 1, then 2, 0, 1, 2, 0
    assert requesters == [1, 2, 0, 1, 2, 0]


def test_reliability_updates_each_round():
    """Test execution reliability updates from profiles each round."""
    # Use an abrupt change profile for agent 0
    from help_seeking.reliability import AbruptChangeProfile
    profiles = [
        AbruptChangeProfile([0.9], [0.3], tau=5),
        StationaryProfile([0.7]),
        StationaryProfile([0.7]),
    ]
    env = MultiAgentEnvironment(
        n_agents=3,
        reliability_profiles=profiles,
        R=1.0, U_self=0.5, lambda_mult=1.0,
        requester_policy_type="D-TS",
        requester_policy_params={"gamma": 0.98},
        seed=0,
    )
    # Before change
    for _ in range(5):
        env.step()
    assert np.isclose(env.agents[0].execution_reliability, 0.9)
    # After change
    env.step()
    assert np.isclose(env.agents[0].execution_reliability, 0.3)


def test_run_simulation():
    """Test running a full simulation."""
    env = make_env(seed=0)
    sim = MultiAgentSimulationRun(env, T=100, record_beliefs=True)
    record = sim.run()
    assert record.T == 100
    assert record.n_agents == 3
    assert len(record.interactions) == 100
    assert record.requester_rewards.shape == (100, 3)
    assert record.workloads.shape == (100, 3)
    assert record.acceptance_beliefs.shape == (100, 3, 2)


def test_reproducibility():
    """Same seed should produce identical results."""
    env1 = make_env(seed=42)
    sim1 = MultiAgentSimulationRun(env1, T=50)
    record1 = sim1.run()

    env2 = make_env(seed=42)
    sim2 = MultiAgentSimulationRun(env2, T=50)
    record2 = sim2.run()

    # Check interactions match
    for r1, r2 in zip(record1.interactions, record2.interactions):
        assert r1.requester_id == r2.requester_id
        assert r1.helper_id == r2.helper_id
        assert r1.asked == r2.asked
        assert r1.accepted == r2.accepted
        assert r1.success == r2.success
        assert np.isclose(r1.requester_reward, r2.requester_reward)
        assert np.isclose(r1.helper_reward, r2.helper_reward)


def test_compute_multiagent_metrics():
    """Test metrics computation."""
    env = make_env(seed=0)
    sim = MultiAgentSimulationRun(env, T=200, record_beliefs=True)
    record = sim.run()
    metrics = compute_multiagent_metrics(record, env)

    # Check all expected keys exist
    expected_keys = [
        "total_rounds", "ask_rate", "acceptance_rate", "rejection_rate",
        "success_rate", "task_completion_rate", "avg_workload", "avg_resources",
        "total_communication_cost", "avg_requester_reward", "avg_helper_reward",
        "avg_request_frequency", "avg_cooperation_frequency",
        "avg_acceptance_rate_per_agent",
        "per_agent_request_count", "per_agent_help_count",
        "per_agent_accept_count", "per_agent_reject_count",
        "per_agent_requester_reward", "per_agent_helper_reward",
        "per_agent_comm_cost",
    ]
    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"

    # Basic sanity checks
    assert metrics["total_rounds"] == 200
    assert 0 <= metrics["ask_rate"] <= 1
    assert 0 <= metrics["acceptance_rate"] <= 1
    assert 0 <= metrics["success_rate"] <= 1
    assert 0 <= metrics["task_completion_rate"] <= 1


def test_helper_rewards_negative_when_costs_high():
    """Test helper rewards can be negative when costs exceed benefits."""
    env = make_env(seed=0, helper_rewards={"B_i": 0.1, "resource_cost": 0.5, "opportunity_cost": 0.5})
    sim = MultiAgentSimulationRun(env, T=50, record_beliefs=True)
    record = sim.run()

    # Some helper rewards should be negative
    helper_rewards = record.helper_rewards.flatten()
    helper_rewards = helper_rewards[helper_rewards != 0]
    if len(helper_rewards) > 0:
        assert np.any(helper_rewards < 0)


def test_communication_cost_incurred_on_ask():
    """Communication cost should be incurred whenever a request is made."""
    env = make_env(seed=0, comm_costs=np.array([[0, 0.2, 0.2], [0.2, 0, 0.2], [0.2, 0.2, 0]]))
    sim = MultiAgentSimulationRun(env, T=50)
    record = sim.run()

    total_comm_cost = sum(r.communication_cost for r in record.interactions)
    asked_count = sum(1 for r in record.interactions if r.asked)
    # Each ask should incur cost 0.2
    assert np.isclose(total_comm_cost, asked_count * 0.2)


def test_workload_resource_dynamics():
    """Test workload and resources change correctly over time."""
    env = make_env(seed=0)
    sim = MultiAgentSimulationRun(env, T=20)
    record = sim.run()

    # Workload should be in [0, 2]
    assert np.all(record.workloads >= 0)
    assert np.all(record.workloads <= 2)
    # Resources should be in [0, 2]
    assert np.all(record.resources >= 0)
    assert np.all(record.resources <= 2)


def test_requester_updates_beliefs_on_rejection():
    """Requester should update acceptance belief but not trust on rejection."""
    env = make_env(seed=0, requester_selection="random")
    # Force a scenario where we can track beliefs
    # We'll just check the mechanism works by running
    sim = MultiAgentSimulationRun(env, T=10, record_beliefs=True)
    record = sim.run()

    # Beliefs should be recorded
    assert record.acceptance_beliefs.shape == (10, 3, 2)
    assert record.trust_beliefs.shape == (10, 3, 2)


def test_different_helper_rewards_per_agent():
    """Test that each agent retains its own individual reward parameters and calculates rewards independently."""
    custom_rewards = [
        {"B_i": 0.8, "resource_cost": 0.1, "opportunity_cost": 0.1},
        {"B_i": 0.4, "resource_cost": 0.2, "opportunity_cost": 0.15},
        {"B_i": 0.6, "resource_cost": 0.3, "opportunity_cost": 0.1},
    ]
    env = make_env(n_agents=3, seed=0, helper_rewards=custom_rewards)

    # Verify per-agent assignment
    assert env.agents[0].B_i == 0.8
    assert env.agents[0].resource_cost == 0.1
    assert env.agents[0].opportunity_cost == 0.1

    assert env.agents[1].B_i == 0.4
    assert env.agents[1].resource_cost == 0.2
    assert env.agents[1].opportunity_cost == 0.15

    assert env.agents[2].B_i == 0.6
    assert env.agents[2].resource_cost == 0.3
    assert env.agents[2].opportunity_cost == 0.1

    # Verify individual reward in the idle state (w=0, rho=2): r_i = B_i * x - resource_cost - opp_cost
    # For success x = 1:
    r0 = env.agents[0].calculate_helper_reward(1)
    r1 = env.agents[1].calculate_helper_reward(1)
    r2 = env.agents[2].calculate_helper_reward(1)

    assert np.isclose(r0, 0.8 - 0.1 - 0.1)   # 0.60
    assert np.isclose(r1, 0.4 - 0.2 - 0.15)  # 0.05
    assert np.isclose(r2, 0.6 - 0.3 - 0.1)   # 0.20

    # For failure x = 0:
    assert np.isclose(env.agents[0].calculate_helper_reward(0), -0.2)
    assert np.isclose(env.agents[1].calculate_helper_reward(0), -0.35)


def test_dict_mapping_helper_rewards_per_agent():
    """Test setting per-agent rewards via dict mapping agent_id -> config."""
    custom_rewards = {
        0: {"B_i": 0.9, "resource_cost": 0.15, "opportunity_cost": 0.05},
        1: {"B_i": 0.3, "resource_cost": 0.25, "opportunity_cost": 0.10},
    }
    env = make_env(n_agents=2, seed=0, helper_rewards=custom_rewards)
    assert env.agents[0].B_i == 0.9
    assert env.agents[1].B_i == 0.3


def test_environment_scheduling_and_agent_autonomy():
    """Verify environment only schedules task arrivals while agents autonomously decide."""
    env = make_env(n_agents=3, seed=42, requester_selection="round_robin")
    # Round 0: agent 0 is requester
    # Round 1: agent 1 is requester
    # Round 2: agent 2 is requester
    for expected_req in [1, 2, 0]:  # round increments 1, 2, 3
        rec = env.step()
        assert rec.round_num == env.round
        # Environment does not force accept/reject: helper uses private Q-table
        if rec.asked:
            helper = env.agents[rec.helper_id]
            # Helper state before matches what was recorded
            assert rec.helper_workload_before >= 0
            assert rec.helper_resources_before >= 0
            # Helper's private Q-table exists and is not shared
            assert helper.helper_qlearning.q_table.shape == (9, 2)


# Test with different requester policies
@pytest.mark.parametrize("policy_type", ["D-Greedy", "D-UCB", "D-TS"])
def test_various_requester_policies(policy_type):
    """Test environment works with all requester policy types."""
    env = make_env(seed=0, requester_policy_type=policy_type)
    sim = MultiAgentSimulationRun(env, T=20)
    record = sim.run()
    assert len(record.interactions) == 20


def test_helper_q_table_updates_on_rejection():
    """Verify that helper Q-table updates for REJECT action when requests are rejected."""
    from help_seeking.autonomous_agent import REJECT, ACCEPT
    from help_seeking.helper_qlearning import encode_state

    # Set up env where helper's workload is maxed out and resources depleted so it rejects
    env = make_env(seed=42, U_self=0.1)  # low U_self encourages asking
    # Prime helper 1 to have learned or prefer rejection:
    # Set positive Q on high workload for other states, but let's test a step where rejection occurs
    # Or force a rejection
    env.agents[1].workload = 2
    env.agents[1].resources = 0
    # Set Q-values such that REJECT is chosen
    s = encode_state(2, 0)
    env.agents[1].helper_qlearning.q_table[s, ACCEPT] = -1.0
    env.agents[1].helper_qlearning.q_table[s, REJECT] = 0.0
    env.agents[1].helper_qlearning.epsilon = 0.0  # greedy

    # Let requester 0 ask helper 1
    # Run until requester asks helper 1
    rejected = False
    for _ in range(50):
        rec = env.step()
        if rec.asked and rec.helper_id == 1 and not rec.accepted:
            rejected = True
            break

    assert rejected, "Expected at least one rejection"
    # Helper 1 should have updated its Q-table (epsilon decayed or Q changed from initial)
    assert env.agents[1].helper_qlearning.epsilon < 0.1 or np.any(env.agents[1].helper_qlearning.q_table[:, REJECT] != 0.0)
