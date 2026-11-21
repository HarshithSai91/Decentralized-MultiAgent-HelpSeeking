# backend/src/help_seeking/multiagent_policies.py
"""Multi-agent requester policies that integrate acceptance and execution reliability.

These policies extend the original algorithms to use both:
- p_hat: acceptance probability (from AcceptanceState)
- theta_hat: conditional execution reliability (from TrustState)

Expected utility for helper j:
    mu_hat_j = U_self + p_hat_j * (R * theta_hat_j - U_self) - lambda * c_j

This replaces the original score_k = R * theta_hat_k - lambda * c_k
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .acceptance import AcceptanceState
from .algorithms import (IndexPolicy, Policy, DiscountedGreedy, DiscountedUCB, DiscountedThompson,
                         argmax_random, Decision, TIE_TOL)
from .trust import TrustState


class MultiAgentPolicy(Policy):
    """Base class for multi-agent requester policies.

    Uses both acceptance beliefs and trust beliefs.
    """

    learns = True

    def __init__(self, K: int, R: float = 1.0, lambda_mult: float = 1.0, c=None,
                 gamma: float = 1.0, seed=None, U_self: float = 0.5):
        super().__init__(K, R, lambda_mult, c, gamma, seed)
        self.U_self = float(U_self)

        # Acceptance beliefs (separate from trust beliefs)
        self.acceptance = AcceptanceState(K, gamma=self.gamma)

        # The trust state is already created in Policy.__init__
        # but we need to ensure it uses the same gamma
        self.trust = TrustState(K, gamma=self.gamma)

    def reset(self, seed=None) -> None:
        super().reset(seed)
        self.acceptance.reset()

    def update_acceptance(self, helper_idx: int, accepted: bool) -> None:
        """Update acceptance belief."""
        if accepted:
            self.acceptance.update_accept(helper_idx)
        else:
            self.acceptance.update_reject(helper_idx)

    def update_trust(self, helper_idx: int, success: int) -> None:
        """Update trust (execution reliability) belief."""
        self.trust.update(helper_idx, success)

    def record_acceptance_observation(self, helper_idx: int, accepted: bool) -> None:
        """Record acceptance observation without discounting."""
        self.acceptance.add_observation(helper_idx, accepted)

    def record_trust_observation(self, helper_idx: int, success: int) -> None:
        """Record execution reliability observation without discounting."""
        self.trust.add_observation(helper_idx, success)

    def update(self, d: int, a: int | None = None, x: int | None = None, accepted: bool = True) -> None:
        """Update beliefs based on round outcome."""
        if d == 0:
            self.decay_beliefs()
        elif d == 1 and a is not None:
            self.update_acceptance(a, accepted)
            if accepted and x is not None:
                self.update_trust(a, x)

    def decay_beliefs(self) -> None:
        """Decay both acceptance and trust beliefs."""
        self.acceptance.decay()
        self.trust.decay()

    def acceptance_means(self) -> np.ndarray:
        """Posterior mean acceptance probabilities."""
        return self.acceptance.posterior_mean()

    def trust_means(self) -> np.ndarray:
        """Posterior mean execution reliabilities."""
        return self.trust.posterior_mean()

    def compute_expected_utilities(self, p_hat: np.ndarray, theta_hat: np.ndarray) -> np.ndarray:
        """Compute expected utilities: mu = U_self + p * (R * theta - U_self) - lambda * c"""
        return self.U_self + p_hat * (self.R * theta_hat - self.U_self) - self.costs

    def compute_temporal_helper_utilities(
        self,
        p_eval: np.ndarray,
        theta_eval: np.ndarray,
        v_solo_post_help: float,
    ) -> np.ndarray:
        """Compute temporal expected value of asking helper j:
        V_ask,j = -lambda * c_j + p * theta * R + (1 - p * theta) * v_solo_post_help
        """
        success_prob = p_eval * theta_eval
        return -self.costs + success_prob * self.R + (1.0 - success_prob) * v_solo_post_help

    def decide(self, U_self: float, theta_true=None) -> Decision:
        """Two-stage decision with acceptance-aware utilities."""
        self.U_self = U_self
        scores, extra = self.helper_scores()
        k = argmax_random(scores, self.rng)
        best = float(scores[k])
        d = 1 if best > U_self else 0
        return Decision(d=d, a=k if d == 1 else None, candidate=k,
                        candidate_score=best, scores=scores, extra=extra)

    def temporal_helper_scores(self, v_solo_post_help: float):
        """Default temporal helper scores using posterior means."""
        p_hat = self.acceptance_means()
        theta_hat = self.trust_means()
        scores = self.compute_temporal_helper_utilities(p_hat, theta_hat, v_solo_post_help)
        return scores, {"p_hat": p_hat, "theta_hat": theta_hat}

    def decide_temporal(
        self,
        v_solo_current: float,
        v_solo_post_help: float,
    ) -> Decision:
        """Two-stage temporal decision matching the sequential task lifecycle.

        Compares max_j V_ask,j against v_solo_current.
        """
        scores, extra = self.temporal_helper_scores(v_solo_post_help)
        k = argmax_random(scores, self.rng)
        best = float(scores[k])
        d = 1 if best > v_solo_current else 0
        return Decision(d=d, a=k if d == 1 else None, candidate=k,
                        candidate_score=best, scores=scores, extra=extra)


class MultiAgentDiscountedGreedy(MultiAgentPolicy):
    """Discounted Greedy with acceptance probability."""

    name = "MultiAgent-Greedy"

    def helper_scores(self):
        p_hat = self.acceptance_means()
        theta_hat = self.trust_means()
        scores = self.compute_expected_utilities(p_hat, theta_hat)
        return scores, {"p_hat": p_hat, "theta_hat": theta_hat}


class MultiAgentDiscountedUCB(MultiAgentPolicy):
    """Discounted UCB with acceptance probability.

    Heuristic extension: apply confidence bounds to both p and theta.
    p_upper = min(1, p_hat + beta_p)
    theta_upper = min(1, theta_hat + beta_theta)
    mu_UCB = U_self + p_upper * max(R * theta_upper - U_self, 0) - lambda * c

    This is a heuristic non-stationary UCB extension without theoretical regret guarantees.
    """

    name = "MultiAgent-D-UCB"

    def __init__(self, K, R=1.0, lambda_mult=1.0, c=None, gamma=1.0, seed=None,
                 U_self: float = 0.5, alpha_ucb: float = 1.0, epsilon: float = 1e-6,
                 beta_p: float = 0.1, beta_theta: float = 0.1):
        super().__init__(K, R, lambda_mult, c, gamma, seed, U_self=U_self)
        if alpha_ucb <= 0:
            raise ValueError("alpha_ucb must be > 0")
        self.alpha_ucb = float(alpha_ucb)
        self.epsilon = float(epsilon)
        self.beta_p = float(beta_p)
        self.beta_theta = float(beta_theta)

    def acceptance_bonus(self) -> np.ndarray:
        """Confidence bonus for acceptance probability."""
        N_k = self.acceptance.effective_sample_size()
        N = float(N_k.sum())
        return self.beta_p * np.sqrt(2.0 * np.log(max(N, 2.0)) / (N_k + self.epsilon))

    def trust_bonus(self) -> np.ndarray:
        """Confidence bonus for execution reliability (original UCB bonus)."""
        N_k = self.trust.effective_sample_size()
        N = float(N_k.sum())
        return self.R * self.alpha_ucb * np.sqrt(2.0 * np.log(max(N, 2.0)) / (N_k + self.epsilon))

    def helper_scores(self):
        p_hat = self.acceptance_means()
        theta_hat = self.trust_means()

        p_bonus = self.acceptance_bonus()
        theta_bonus = self.trust_bonus()

        p_upper = np.minimum(1.0, p_hat + p_bonus)
        theta_upper = np.minimum(1.0, theta_hat + theta_bonus)

        # mu_UCB = U_self + p_upper * max(R * theta_upper - U_self, 0) - lambda * c
        net_gain = np.maximum(self.R * theta_upper - self.U_self, 0)
        scores = self.U_self + p_upper * net_gain - self.costs

        return scores, {
            "p_hat": p_hat, "theta_hat": theta_hat,
            "p_bonus": p_bonus, "theta_bonus": theta_bonus,
            "p_upper": p_upper, "theta_upper": theta_upper,
        }

    def temporal_helper_scores(self, v_solo_post_help: float):
        p_hat = self.acceptance_means()
        theta_hat = self.trust_means()

        p_bonus = self.acceptance_bonus()
        theta_bonus = self.trust_bonus()

        p_upper = np.minimum(1.0, p_hat + p_bonus)
        theta_upper = np.minimum(1.0, theta_hat + theta_bonus)

        scores = self.compute_temporal_helper_utilities(p_upper, theta_upper, v_solo_post_help)
        return scores, {
            "p_hat": p_hat, "theta_hat": theta_hat,
            "p_bonus": p_bonus, "theta_bonus": theta_bonus,
            "p_upper": p_upper, "theta_upper": theta_upper,
        }


class MultiAgentDiscountedThompson(MultiAgentPolicy):
    """Discounted Thompson Sampling with acceptance probability.

    Samples:
    p_sample ~ Beta(1 + A, 1 + R)
    theta_sample ~ Beta(1 + S, 1 + F)

    mu_TS = U_self + p_sample * (R * theta_sample - U_self) - lambda * c
    """

    name = "MultiAgent-D-TS"

    def helper_scores(self):
        # Sample acceptance probability
        p_alpha, p_beta = self.acceptance.posterior_params()
        p_sample = self.rng.beta(p_alpha, p_beta)

        # Sample execution reliability
        theta_alpha, theta_beta = self.trust.posterior_params()
        theta_sample = self.rng.beta(theta_alpha, theta_beta)

        scores = self.compute_expected_utilities(p_sample, theta_sample)

        return scores, {
            "p_sample": p_sample, "theta_sample": theta_sample,
            "p_hat": self.acceptance_means(), "theta_hat": self.trust_means(),
        }

    def temporal_helper_scores(self, v_solo_post_help: float):
        p_alpha, p_beta = self.acceptance.posterior_params()
        p_sample = self.rng.beta(p_alpha, p_beta)

        theta_alpha, theta_beta = self.trust.posterior_params()
        theta_sample = self.rng.beta(theta_alpha, theta_beta)

        scores = self.compute_temporal_helper_utilities(p_sample, theta_sample, v_solo_post_help)
        return scores, {
            "p_sample": p_sample, "theta_sample": theta_sample,
            "p_hat": self.acceptance_means(), "theta_hat": self.trust_means(),
        }


# Factory function for creating multi-agent policies
def make_multiagent_policy(name: str, K: int, R: float, lambda_mult: float, c,
                           gamma: float, seed: Optional[int], U_self: float,
                           alpha_ucb: float = 1.0, beta_p: float = 0.1, beta_theta: float = 0.1,
                           overrides: dict | None = None) -> MultiAgentPolicy:
    """Create a multi-agent requester policy by name."""
    overrides = overrides or {}
    gamma = float(overrides.get("gamma", gamma))
    alpha_ucb = float(overrides.get("alpha_ucb", alpha_ucb))
    beta_p = float(overrides.get("beta_p", beta_p))
    beta_theta = float(overrides.get("beta_theta", beta_theta))

    name_lower = name.lower().replace("multiagent-", "").replace("-", "").replace(" ", "")
    # Handle various naming conventions
    if name_lower in ("greedy", "dgreedy", "discountedgreedy"):
        return MultiAgentDiscountedGreedy(K, R, lambda_mult, c, gamma, seed, U_self)
    elif name_lower in ("ducb", "ducb", "discounteducb"):
        return MultiAgentDiscountedUCB(K, R, lambda_mult, c, gamma, seed, U_self,
                                       alpha_ucb=alpha_ucb, beta_p=beta_p, beta_theta=beta_theta)
    elif name_lower in ("dts", "dts", "discountedthompson", "thompson"):
        return MultiAgentDiscountedThompson(K, R, lambda_mult, c, gamma, seed, U_self)
    else:
        raise ValueError(f"Unknown multi-agent policy: {name}")