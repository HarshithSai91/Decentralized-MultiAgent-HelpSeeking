# backend/src/help_seeking/acceptance.py
"""Discounted acceptance probability belief model.

For each ordered agent pair (i, j), maintain discounted pseudo-counts of
accepted and rejected requests to estimate p_ij = P(agent j accepts a request
from agent i).

This is separate from the execution reliability theta_ij, which is estimated
using the discounted Beta-Bernoulli trust model in trust.py.
"""
from __future__ import annotations

import numpy as np


class AcceptanceState:
    """Discounted pseudo-counts A_ij, R_ij for acceptance probability estimation.

    For each potential helper j (from requester i's perspective):
        A_ij(t+1) = gamma * A_ij(t) + I[request to j accepted]
        R_ij(t+1) = gamma * R_ij(t) + I[request to j rejected]

    Posterior mean under uniform Beta(1, 1) prior:
        p_hat_ij = (1 + A_ij) / (2 + A_ij + R_ij)

    Both acceptance and rejection observations are discounted every round,
    including rounds where no request was made (decay only).
    """

    def __init__(self, n_helpers: int, gamma: float = 1.0):
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"gamma must lie in (0, 1], got {gamma}")
        self.n_helpers = int(n_helpers)
        self.gamma = float(gamma)
        self.A = np.zeros(self.n_helpers)  # accepted counts
        self.R = np.zeros(self.n_helpers)  # rejected counts

    def reset(self) -> None:
        self.A.fill(0.0)
        self.R.fill(0.0)

    # --- beliefs ------------------------------------------------------------
    def posterior_mean(self) -> np.ndarray:
        """Posterior mean acceptance probability: (1 + A) / (2 + A + R)."""
        return (1.0 + self.A) / (2.0 + self.A + self.R)

    def posterior_params(self):
        """Beta(alpha, beta) = Beta(1 + A, 1 + R)."""
        return 1.0 + self.A, 1.0 + self.R

    def effective_sample_size(self) -> np.ndarray:
        """N = A + R (discounted number of observations)."""
        return self.A + self.R

    # --- updates ------------------------------------------------------------
    def discount(self) -> None:
        """Decay all evidence by gamma."""
        if self.gamma != 1.0:
            self.A *= self.gamma
            self.R *= self.gamma

    def update_accept(self, helper_idx: int) -> None:
        """Record that a request to helper_idx was accepted."""
        self.discount()
        self.A[helper_idx] += 1.0

    def update_reject(self, helper_idx: int) -> None:
        """Record that a request to helper_idx was rejected."""
        self.discount()
        self.R[helper_idx] += 1.0

    def add_observation(self, helper_idx: int, accepted: bool) -> None:
        """Record observation for helper_idx without discounting (when discount was already applied)."""
        if accepted:
            self.A[helper_idx] += 1.0
        else:
            self.R[helper_idx] += 1.0

    def decay(self) -> None:
        """Round in which no request was made (acted alone)."""
        self.discount()

    def snapshot(self) -> dict:
        return {"A": self.A.tolist(), "R": self.R.tolist(),
                "p_hat": self.posterior_mean().tolist()}