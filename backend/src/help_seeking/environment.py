# backend/src/help_seeking/environment.py
"""Simulation environment (proposal Sec. 2.2): ground truth and outcome generation.

The environment is the only component that holds the true reliabilities.
Learning agents never receive them; they only observe the binary outcome of
the helper they actually queried.

Common random numbers: for round t and teammate k a single uniform number
u[t, k] is drawn from the environment's own random stream, and the outcome of
querying k at round t is x_t = 1[u[t, k] < theta_k(t)].  Every algorithm run
with the same environment seed therefore sees exactly the same "would-have"
outcomes, which removes most outcome noise from paired comparisons.
"""
from __future__ import annotations

import numpy as np

from .reliability import ReliabilityProfile
from .utilities import dynamic_oracle, OracleDecision


class SimulationEnvironment:

    _BLOCK = 1024

    def __init__(self, reliability_profile: ReliabilityProfile, R: float = 1.0,
                 U_self: float = 0.5, c=None, lambda_mult: float = 1.0, seed=None):
        self.profile = reliability_profile
        self.K = reliability_profile.K
        self.R = float(R)
        self.U_self = float(U_self)
        self.c = np.zeros(self.K) if c is None else np.asarray(c, dtype=float)
        if self.c.shape != (self.K,):
            raise ValueError(f"cost vector has length {self.c.size}, expected K={self.K}")
        self.lambda_mult = float(lambda_mult)
        self._seed = seed
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self, seed=None) -> None:
        if seed is not None:
            self._seed = seed
        self.rng = np.random.default_rng(self._seed)
        self._u = np.empty((0, self.K))
        self.t = 0

    def _uniforms(self, t: int) -> np.ndarray:
        # Drawn lazily in blocks; numpy Generators produce the same sequence
        # whether drawn in one block or several, so this is reproducible.
        while self._u.shape[0] < t:
            self._u = np.vstack([self._u, self.rng.random((self._BLOCK, self.K))])
        return self._u[t - 1]

    # ------------------------------------------------------------------ round
    def start_round(self) -> np.ndarray:
        """Advance to the next round and return the (hidden) true reliabilities."""
        self.t += 1
        return self.profile.get(self.t)

    def theta(self, t: int | None = None) -> np.ndarray:
        return self.profile.get(self.t if t is None else t)

    def costs(self) -> np.ndarray:
        """Effective communication cost lambda * c_k of each helper."""
        return self.lambda_mult * self.c

    def oracle(self, theta=None) -> OracleDecision:
        theta = self.theta() if theta is None else theta
        return dynamic_oracle(theta, self.U_self, R=self.R, lambda_mult=self.lambda_mult, c=self.c)

    def generate_outcome(self, k: int, theta=None) -> int:
        """x_t ~ Bernoulli(theta_k(t)), via the common random number u[t, k]."""
        theta = self.theta() if theta is None else theta
        return int(self._uniforms(self.t)[k] < theta[k])

    def realized_utility(self, d: int, a, x) -> float:
        """r_t = R x_t - lambda c_{a_t} if d_t = 1, else the autonomous utility.

        The autonomous outcome is modelled by its expectation U_self (the
        proposal only fixes E[x_self] = U_self).  Regret uses expected
        utilities, so this choice does not affect any regret number.
        """
        if d == 0:
            return self.U_self
        return self.R * float(x) - self.lambda_mult * float(self.c[a])

    def expected_utility(self, d: int, a, theta=None) -> float:
        """mu of the chosen action: U_self (Eq. 6) or mu_a(t) (Eq. 7)."""
        if d == 0:
            return self.U_self
        theta = self.theta() if theta is None else theta
        return self.R * float(theta[a]) - self.lambda_mult * float(self.c[a])

    # backwards compatibility with the first version of the code
    def compute_utility(self, d, a, x, theta_true=None) -> float:
        return self.realized_utility(d, a, x)
