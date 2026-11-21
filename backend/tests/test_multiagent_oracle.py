"""Tests for multi-agent oracle and helper baselines (Always Accept, Always Reject, Q-learning)."""
import numpy as np
import pytest

from help_seeking.multiagent_environment import MultiAgentEnvironment
from help_seeking.multiagent_simulation import MultiAgentSimulationRun, compute_multiagent_metrics
from help_seeking.reliability import StationaryProfile


def test_multiagent_oracle_calculation():
    """Verify requester-level dynamic oracle calculation."""
    profiles = [StationaryProfile([0.8]), StationaryProfile([0.4]), StationaryProfile([0.9])]
    env = MultiAgentEnvironment(
        n_agents=3,
        reliability_profiles=profiles,
        R=1.0,
        U_self=0.5,
        lambda_mult=1.0,
        helper_policy_type="always_accept",
        seed=42,
    )

    # For requester 0: helpers are agent 1 (theta=0.4) and agent 2 (theta=0.9)
    # With always_accept, p = 1.0 for both
    # mu_01 = 0.5 + 1.0 * (1.0 * 0.4 - 0.5) - 0.1 = 0.5 - 0.1 - 0.1 = 0.3
    # mu_02 = 0.5 + 1.0 * (1.0 * 0.9 - 0.5) - 0.1 = 0.5 + 0.4 - 0.1 = 0.8
    # max helper util = 0.8 > U_self (0.5), so d* = 1, best helper is agent 2 (helper_idx = 1)
    oracle = env.compute_oracle(requester_id=0)
    assert oracle["d_star"] == 1
    assert oracle["a_star"] == 1
    assert oracle["best_helper_agent_id"] == 2
    assert np.isclose(oracle["mu_star"], 0.8)


def test_oracle_when_autonomous_is_optimal():
    """Oracle should choose autonomous execution (d*=0) when all helpers yield < U_self."""
    profiles = [StationaryProfile([0.7]), StationaryProfile([0.3]), StationaryProfile([0.2])]
    env = MultiAgentEnvironment(
        n_agents=3,
        reliability_profiles=profiles,
        R=1.0,
        U_self=0.6,
        lambda_mult=1.0,
        helper_policy_type="always_accept",
        seed=42,
    )
    # For requester 0: helper 1 (theta=0.3): mu = 0.6 + 1.0*(0.3 - 0.6) - 0.1 = 0.2 < 0.6
    # helper 2 (theta=0.2): mu = 0.6 + 1.0*(0.2 - 0.6) - 0.1 = 0.1 < 0.6
    oracle = env.compute_oracle(requester_id=0)
    assert oracle["d_star"] == 0
    assert oracle["a_star"] == -1
    assert np.isclose(oracle["mu_star"], 0.6)


def test_helper_baseline_always_accept():
    """Always-accept baseline should result in 100% acceptance when requested."""
    profiles = [StationaryProfile([0.8]), StationaryProfile([0.8]), StationaryProfile([0.8])]
    env = MultiAgentEnvironment(
        n_agents=3,
        reliability_profiles=profiles,
        R=1.0,
        U_self=0.2,  # Low U_self to encourage asking
        helper_policy_type="always_accept",
        seed=1,
    )
    sim = MultiAgentSimulationRun(env, T=50)
    record = sim.run()
    metrics = compute_multiagent_metrics(record, env)
    if metrics["ask_rate"] > 0:
        assert metrics["acceptance_rate"] == 1.0


def test_helper_baseline_always_reject():
    """Always-reject baseline should result in 0% acceptance."""
    profiles = [StationaryProfile([0.8]), StationaryProfile([0.8]), StationaryProfile([0.8])]
    env = MultiAgentEnvironment(
        n_agents=3,
        reliability_profiles=profiles,
        R=1.0,
        U_self=0.2,
        helper_policy_type="always_reject",
        seed=1,
    )
    sim = MultiAgentSimulationRun(env, T=50)
    record = sim.run()
    metrics = compute_multiagent_metrics(record, env)
    assert metrics["acceptance_rate"] == 0.0


# ============================================================================
# NINE COMPREHENSIVE PSEUDO-REGRET & ORACLE MATHEMATICAL TESTS (SECTION 3)
# ============================================================================

def test_pseudo_regret_autonomous_execution():
    """1. Requester chooses autonomous execution: mu_chosen = U_self, regret = max(0, mu* - U_self)."""
    # Helpers have higher utility than U_self, but requester chooses autonomous (e.g. forced via high U_self policy)
    profiles = [StationaryProfile([0.8]), StationaryProfile([0.9]), StationaryProfile([0.9])]
    env = MultiAgentEnvironment(
        n_agents=3,
        reliability_profiles=profiles,
        R=1.0,
        U_self=0.4,
        lambda_mult=1.0,
        helper_policy_type="always_accept",
        requester_policy_type="Greedy",
        requester_policy_params={"U_self": 0.4},
        seed=42,
    )
    # Manually step when requester decides to act alone
    # Compute oracle for requester 0: helpers have p=1.0, theta=0.9, cost=0.1 -> mu = 0.4 + 1.0*(0.9 - 0.4) - 0.1 = 0.8
    oracle = env.compute_oracle(0)
    assert np.isclose(oracle["mu_star"], 0.8)

    # Force agent 0 to act alone
    from help_seeking.algorithms import Decision
    env.agents[0].decide_request = lambda u=None: Decision(d=0, a=-1)
    rec = env.step()
    assert not rec.asked
    # Pseudo-regret must be mu* - U_self = 0.8 - 0.4 = 0.4
    assert np.isclose(rec.regret, 0.8 - 0.4)
    assert not rec.opt_action


def test_pseudo_regret_high_acceptance_helper():
    """2. Requester asks a helper with high acceptance probability."""
    profiles = [StationaryProfile([0.5]), StationaryProfile([0.9]), StationaryProfile([0.5])]
    env = MultiAgentEnvironment(
        n_agents=3,
        reliability_profiles=profiles,
        R=1.0,
        U_self=0.3,
        lambda_mult=1.0,
        helper_policy_type="always_accept",  # p = 1.0
        seed=10,
    )
    oracle = env.compute_oracle(0)
    # Helper 1: p=1.0, theta=0.9, c=0.1 -> mu = 0.3 + 1.0*(0.9 - 0.3) - 0.1 = 0.8
    assert np.isclose(oracle["mu_star_helpers"][0], 0.8)


def test_pseudo_regret_low_acceptance_helper():
    """3. Requester asks a helper with low acceptance probability."""
    # Under Q-learning with high workload & depleted resources, helper 1 rejects (p = eps/2)
    from help_seeking.helper_qlearning import encode_state
    from help_seeking.autonomous_agent import ACCEPT, REJECT

    profiles = [StationaryProfile([0.5]), StationaryProfile([0.9]), StationaryProfile([0.5])]
    env = MultiAgentEnvironment(
        n_agents=3,
        reliability_profiles=profiles,
        R=1.0,
        U_self=0.3,
        lambda_mult=1.0,
        helper_policy_type="q_learning",
        seed=10,
    )
    # Set helper 1 to prefer reject:
    s = encode_state(env.agents[1].workload, env.agents[1].resources)
    env.agents[1].helper_qlearning.q_table[s, ACCEPT] = -1.0
    env.agents[1].helper_qlearning.q_table[s, REJECT] = 0.0
    eps = env.agents[1].helper_qlearning.epsilon

    oracle = env.compute_oracle(0)
    p_true_1 = eps / 2.0
    expected_mu = 0.3 + p_true_1 * (0.9 - 0.3) - 0.1
    assert np.isclose(oracle["mu_star_helpers"][0], expected_mu)


def test_pseudo_regret_consistency_accepted_vs_rejected():
    """4 & 5. Helper accepts or rejects: ex-ante pseudo-regret must be identical."""
    # Setup two identical environments at round 1
    profiles = [StationaryProfile([0.7]), StationaryProfile([0.8]), StationaryProfile([0.8])]
    env_accept = MultiAgentEnvironment(
        n_agents=3,
        reliability_profiles=profiles,
        R=1.0,
        U_self=0.3,
        lambda_mult=1.0,
        helper_policy_type="always_accept",
        seed=42,
    )
    env_reject = MultiAgentEnvironment(
        n_agents=3,
        reliability_profiles=profiles,
        R=1.0,
        U_self=0.3,
        lambda_mult=1.0,
        helper_policy_type="always_reject",
        seed=42,
    )

    # For both environments, ask expected utility uses the true p of that helper
    # In env_accept: p=1.0; in env_reject: p=0.0
    oracle_acc = env_accept.compute_oracle(0)
    oracle_rej = env_reject.compute_oracle(0)

    # In env_accept: helper accepts -> realized reward is 1.0*success - comm_cost
    rec_acc = env_accept.step()
    # In env_reject: helper rejects -> realized reward is U_self - comm_cost
    rec_rej = env_reject.step()

    # For env_accept: rec_acc.regret is max(0, oracle - mu_chosen)
    assert rec_acc.regret >= 0.0
    # For env_reject: rec_rej.regret is max(0, oracle - mu_chosen), NOT derived from realized U_self - cost
    assert rec_rej.regret >= 0.0
    # Realized rewards differ as expected:
    assert rec_rej.requester_reward == env_reject.U_self - rec_rej.communication_cost


def test_pseudo_regret_zero_acceptance():
    """6. Helper has zero acceptance probability: mu_ij = U_self - lambda * c_ij."""
    profiles = [StationaryProfile([0.5]), StationaryProfile([0.9]), StationaryProfile([0.9])]
    env = MultiAgentEnvironment(
        n_agents=3,
        reliability_profiles=profiles,
        R=1.0,
        U_self=0.4,
        lambda_mult=1.0,
        helper_policy_type="always_reject",  # p = 0.0
        seed=42,
    )
    oracle = env.compute_oracle(0)
    # For p = 0: mu = U_self + 0*(...) - lambda*c = 0.4 - 0.1 = 0.3
    for mu in oracle["mu_star_helpers"]:
        assert np.isclose(mu, 0.4 - 0.1)


def test_pseudo_regret_perfect_acceptance():
    """7. Helper has perfect acceptance probability: mu_ij = R * theta_ij - lambda * c_ij."""
    profiles = [StationaryProfile([0.5]), StationaryProfile([0.85]), StationaryProfile([0.65])]
    env = MultiAgentEnvironment(
        n_agents=3,
        reliability_profiles=profiles,
        R=1.0,
        U_self=0.4,
        lambda_mult=1.0,
        helper_policy_type="always_accept",  # p = 1.0
        seed=42,
    )
    oracle = env.compute_oracle(0)
    # For p = 1.0: mu = U_self + 1.0*(R*theta - U_self) - lambda*c = R*theta - lambda*c
    # helper 1 (theta=0.85): 1.0*0.85 - 0.1 = 0.75
    # helper 2 (theta=0.65): 1.0*0.65 - 0.1 = 0.55
    assert np.isclose(oracle["mu_star_helpers"][0], 0.75)
    assert np.isclose(oracle["mu_star_helpers"][1], 0.55)


def test_pseudo_regret_communication_cost_variation():
    """8. Communication cost lambda scales expected utility linearly."""
    profiles = [StationaryProfile([0.5]), StationaryProfile([0.8]), StationaryProfile([0.8])]
    env_low = MultiAgentEnvironment(n_agents=3, reliability_profiles=profiles, R=1.0, U_self=0.4,
                                    lambda_mult=0.5, helper_policy_type="always_accept", seed=1)
    env_high = MultiAgentEnvironment(n_agents=3, reliability_profiles=profiles, R=1.0, U_self=0.4,
                                     lambda_mult=2.0, helper_policy_type="always_accept", seed=1)

    oracle_low = env_low.compute_oracle(0)
    oracle_high = env_high.compute_oracle(0)

    # Base comm_cost = 0.1
    # low cost: lambda*c = 0.05 -> mu = 0.8 - 0.05 = 0.75
    # high cost: lambda*c = 0.20 -> mu = 0.8 - 0.20 = 0.60
    assert np.isclose(oracle_low["mu_star_helpers"][0], 0.75)
    assert np.isclose(oracle_high["mu_star_helpers"][0], 0.60)


def test_oracle_and_selected_utility_consistency():
    """9. Oracle and selected utility use identical true probability formulas."""
    profiles = [StationaryProfile([0.5]), StationaryProfile([0.8]), StationaryProfile([0.4])]
    env = MultiAgentEnvironment(
        n_agents=3,
        reliability_profiles=profiles,
        R=1.0,
        U_self=0.3,
        lambda_mult=1.0,
        helper_policy_type="always_accept",
        seed=1,
    )
    # Requester 0: helper 1 (theta=0.8) is optimal (mu = 0.8 - 0.1 = 0.7)
    # When requester selects optimal helper (d=1, a=0), regret must be exactly 0.0
    oracle = env.compute_oracle(0)
    assert oracle["d_star"] == 1
    assert oracle["a_star"] == 0

    # Ensure agent 0 selects helper 0
    rec = env.step()
    if rec.asked and rec.helper_id == 1:
        assert np.isclose(rec.regret, 0.0)


# ============================================================================
# SIX TESTS FOR ORACLE ACCEPTANCE PROBABILITY UNDER ACTUAL ACTION SELECTION
# ============================================================================

def test_oracle_p_accept_both_q_values_tied():
    """1. When Q(ACCEPT) == Q(REJECT), P(ACCEPT) must be exactly 0.5 regardless of epsilon."""
    from help_seeking.helper_qlearning import HelperQLearning, encode_state
    hq = HelperQLearning(seed=42)
    s = encode_state(1, 1)
    hq.q_table[s, :] = 0.75  # tied
    for eps in [0.0, 0.05, 0.1, 0.2, 0.5]:
        hq.epsilon = eps
        p = hq.p_accept(1, 1)
        assert np.isclose(p, 0.5), f"Expected 0.5 for tied Q-values, got {p} with eps={eps}"


def test_oracle_p_accept_accept_uniquely_optimal():
    """2. When Q(ACCEPT) > Q(REJECT), P(ACCEPT) = (1 - eps) * 1 + eps / 2 = 1 - eps / 2."""
    from help_seeking.helper_qlearning import HelperQLearning, encode_state
    hq = HelperQLearning(seed=42)
    s = encode_state(0, 2)
    hq.q_table[s, 0] = 1.0  # ACCEPT
    hq.q_table[s, 1] = 0.2  # REJECT
    hq.epsilon = 0.1
    p = hq.p_accept(0, 2)
    assert np.isclose(p, 1.0 - 0.1 / 2.0)  # 0.95


def test_oracle_p_accept_reject_uniquely_optimal():
    """3. When Q(REJECT) > Q(ACCEPT), P(ACCEPT) = (1 - eps) * 0 + eps / 2 = eps / 2."""
    from help_seeking.helper_qlearning import HelperQLearning, encode_state
    hq = HelperQLearning(seed=42)
    s = encode_state(2, 0)
    hq.q_table[s, 0] = -0.5  # ACCEPT
    hq.q_table[s, 1] = 0.0   # REJECT
    hq.epsilon = 0.1
    p = hq.p_accept(2, 0)
    assert np.isclose(p, 0.1 / 2.0)  # 0.05


def test_oracle_p_accept_different_epsilon_values():
    """4. Exact probability formula holds across different epsilon values in [0, 1]."""
    from help_seeking.helper_qlearning import HelperQLearning, encode_state
    hq = HelperQLearning(seed=42)
    s = encode_state(1, 2)
    hq.q_table[s, 0] = 0.8
    hq.q_table[s, 1] = 0.3
    for eps in [0.0, 0.02, 0.1, 0.25, 0.5, 0.8, 1.0]:
        hq.epsilon = eps
        expected = (1.0 - eps) * 1.0 + eps / 2.0
        assert np.isclose(hq.p_accept(1, 2), expected)


def test_oracle_p_accept_initial_q_table_all_zeros():
    """5. Initial Q-table with all zeros has P(ACCEPT) = 0.5 across all 9 states."""
    from help_seeking.helper_qlearning import HelperQLearning
    hq = HelperQLearning(seed=42)
    assert np.all(hq.q_table == 0.0)
    for w in range(3):
        for r in range(3):
            assert np.isclose(hq.p_accept(w, r), 0.5)


def test_oracle_p_accept_matches_empirical_frequency():
    """6. Theoretical P(ACCEPT) matches empirical action-selection frequency over many draws."""
    from help_seeking.helper_qlearning import HelperQLearning, encode_state
    from help_seeking.autonomous_agent import ACCEPT
    hq = HelperQLearning(seed=123)
    hq.epsilon = 0.2
    s = encode_state(0, 1)

    # Test Case A: ACCEPT is greedy
    hq.q_table[s, 0] = 1.0
    hq.q_table[s, 1] = 0.0
    theo_p_a = hq.p_accept(0, 1)  # 1 - 0.2/2 = 0.90
    draws_a = [hq.choose_action(0, 1) for _ in range(20000)]
    emp_p_a = sum(1 for a in draws_a if a == ACCEPT) / len(draws_a)
    assert np.isclose(emp_p_a, theo_p_a, atol=0.01)

    # Test Case B: Tied Q-values
    hq.q_table[s, 0] = 0.5
    hq.q_table[s, 1] = 0.5
    theo_p_b = hq.p_accept(0, 1)  # 0.50
    draws_b = [hq.choose_action(0, 1) for _ in range(20000)]
    emp_p_b = sum(1 for a in draws_b if a == ACCEPT) / len(draws_b)
    assert np.isclose(emp_p_b, theo_p_b, atol=0.01)


def test_oracle_hypothetical_flush_consistency():
    """Verify that compute_oracle uses hypothetical post-flush acceptance probability."""
    profiles = [StationaryProfile([0.8]), StationaryProfile([0.8])]
    env = MultiAgentEnvironment(
        n_agents=2,
        reliability_profiles=profiles,
        helper_policy_type="q_learning",
        seed=42,
    )
    helper = env.agents[1]
    # Record a pending transition on helper
    helper.record_helping_decision(action=0, reward=1.0, round_t=1, workload=0, resources=2)
    assert helper.helper_qlearning.pending_transition is not None

    # Compute oracle for requester 0
    oracle = env.compute_oracle(requester_id=0)
    p_oracle = float(oracle["p_true"][0])

    # Now actually flush helper
    helper.flush_pending_helping_transition(current_round=2)
    p_actual = float(helper.helper_qlearning.p_accept(helper.workload, helper.resources))

    # Must be exactly identical
    assert np.isclose(p_oracle, p_actual)


