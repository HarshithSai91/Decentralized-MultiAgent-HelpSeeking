"""Unit tests for MultiAgentPolicy classes (D-Greedy, D-UCB, D-TS with acceptance beliefs)."""
import numpy as np
import pytest

from help_seeking.multiagent_policies import (
    MultiAgentDiscountedGreedy,
    MultiAgentDiscountedUCB,
    MultiAgentDiscountedThompson,
    make_multiagent_policy,
)


def test_multiagent_greedy_scoring():
    """Greedy policy should score helpers based on posterior means."""
    K = 3
    pol = MultiAgentDiscountedGreedy(K, R=1.0, lambda_mult=1.0, c=np.array([0.1, 0.1, 0.1]),
                                     gamma=1.0, seed=0, U_self=0.5)
    # Initially uniform: p_hat = 0.5, theta_hat = 0.5
    # mu_hat = 0.5 + 0.5 * (1.0 * 0.5 - 0.5) - 0.1 = 0.4
    scores, extra = pol.helper_scores()
    assert np.allclose(scores, 0.4)

    # Helper 0 accepted and succeeded
    pol.update(d=1, a=0, x=1, accepted=True)
    # A[0] = 1, R[0] = 0 -> p_hat = 2/3
    # S[0] = 1, F[0] = 0 -> theta_hat = 2/3
    # mu_hat[0] = 0.5 + (2/3) * (2/3 - 0.5) - 0.1 = 0.5 + (2/3)*(1/6) - 0.1 = 0.5 + 1/9 - 0.1 = 0.5111
    scores, extra = pol.helper_scores()
    assert scores[0] > scores[1]
    assert np.isclose(scores[0], 0.5 + (2/3) * (2/3 - 0.5) - 0.1)


def test_multiagent_ucb_optimism():
    """UCB should add optimistic bonuses to both acceptance and reliability."""
    K = 2
    pol = MultiAgentDiscountedUCB(K, R=1.0, lambda_mult=1.0, c=np.array([0.1, 0.1]),
                                  gamma=1.0, seed=0, U_self=0.5, alpha_ucb=1.0, beta_p=0.5)
    scores, extra = pol.helper_scores()
    assert "p_upper" in extra
    assert "theta_upper" in extra
    assert np.all(extra["p_upper"] >= extra["p_hat"])
    assert np.all(extra["theta_upper"] >= extra["theta_hat"])


def test_multiagent_thompson_sampling():
    """Thompson sampling should draw valid Beta samples in [0, 1]."""
    K = 3
    pol = MultiAgentDiscountedThompson(K, R=1.0, lambda_mult=1.0, c=np.array([0.1, 0.1, 0.1]),
                                       gamma=0.98, seed=42, U_self=0.5)
    scores, extra = pol.helper_scores()
    assert "p_sample" in extra
    assert "theta_sample" in extra
    assert np.all((extra["p_sample"] >= 0.0) & (extra["p_sample"] <= 1.0))
    assert np.all((extra["theta_sample"] >= 0.0) & (extra["theta_sample"] <= 1.0))


def test_rejection_updates_acceptance_only():
    """Rejection must update acceptance count and leave trust count unchanged."""
    K = 2
    pol = MultiAgentDiscountedThompson(K, R=1.0, lambda_mult=1.0, c=np.array([0.1, 0.1]),
                                       gamma=1.0, seed=0, U_self=0.5)
    S_before = pol.trust.S.copy()
    F_before = pol.trust.F.copy()

    # Rejection by helper 1
    pol.update(d=1, a=1, x=None, accepted=False)

    assert pol.acceptance.R[1] == 1.0
    assert pol.acceptance.A[1] == 0.0
    # Trust must be strictly unchanged
    assert np.all(pol.trust.S == S_before)
    assert np.all(pol.trust.F == F_before)


def test_factory_function():
    """make_multiagent_policy should instantiate all supported policies."""
    c = np.array([0.1, 0.1])
    p1 = make_multiagent_policy("D-Greedy", K=2, R=1.0, lambda_mult=1.0, c=c, gamma=0.98, seed=0, U_self=0.5)
    assert isinstance(p1, MultiAgentDiscountedGreedy)

    p2 = make_multiagent_policy("D-UCB", K=2, R=1.0, lambda_mult=1.0, c=c, gamma=0.98, seed=0, U_self=0.5)
    assert isinstance(p2, MultiAgentDiscountedUCB)

    p3 = make_multiagent_policy("D-TS", K=2, R=1.0, lambda_mult=1.0, c=c, gamma=0.98, seed=0, U_self=0.5)
    assert isinstance(p3, MultiAgentDiscountedThompson)
