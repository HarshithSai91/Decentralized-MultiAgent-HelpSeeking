# backend/src/help_seeking/trust.py
"""Discounted Beta-Bernoulli trust belief model (proposal Sec. 2.4)."""
from __future__ import annotations

import numpy as np


class TrustState:
    """Discounted pseudo-counts S_k, F_k for K teammates.

    End-of-round update (Eqs. 3-4), applied every round:
        S_k(t+1) = gamma * S_k(t) + 1[d_t = 1 and a_t = k] * x_t
        F_k(t+1) = gamma * F_k(t) + 1[d_t = 1 and a_t = k] * (1 - x_t)
    Posterior mean under a uniform Beta(1, 1) prior (Eq. 5):
        theta_hat_k = (S_k + 1) / (S_k + F_k + 2)
    """

    def __init__(self, K: int, gamma: float = 1.0):
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"gamma must lie in (0, 1], got {gamma}")
        self.K = int(K)
        self.gamma = float(gamma)
        self.S = np.zeros(self.K)
        self.F = np.zeros(self.K)

    def reset(self) -> None:
        self.S.fill(0.0)
        self.F.fill(0.0)

    # --- beliefs ------------------------------------------------------------
    def posterior_mean(self) -> np.ndarray:
        """Eq. (5)."""
        return (self.S + 1.0) / (self.S + self.F + 2.0)

    def posterior_params(self):
        """Beta(alpha_k, beta_k) = Beta(1 + S_k, 1 + F_k)."""
        return 1.0 + self.S, 1.0 + self.F

    def effective_sample_size(self) -> np.ndarray:
        """N_k = S_k + F_k (the discounted number of observations)."""
        return self.S + self.F

    # --- updates ------------------------------------------------------------
    def discount(self) -> None:
        """Decay all evidence by gamma (the first term of Eqs. 3-4)."""
        if self.gamma != 1.0:
            self.S *= self.gamma
            self.F *= self.gamma

    def update(self, k: int, x: int) -> None:
        """Round in which teammate k was queried and returned outcome x."""
        self.discount()
        self.S[k] += float(x)
        self.F[k] += 1.0 - float(x)

    def add_observation(self, k: int, x: int) -> None:
        """Record outcome x for teammate k without discounting (when discount was already applied)."""
        self.S[k] += float(x)
        self.F[k] += 1.0 - float(x)

    def decay(self) -> None:
        """Round in which nobody was queried (agent acted alone)."""
        self.discount()

    def snapshot(self) -> dict:
        return {"S": self.S.tolist(), "F": self.F.tolist(),
                "theta_hat": self.posterior_mean().tolist()}
