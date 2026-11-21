# backend/src/help_seeking/reliability.py
"""True teammate reliability profiles theta_k(t) (proposal Sec. 2.3).

Every profile is a deterministic function of the round index t (1-indexed)
once it has been constructed.  All algorithms evaluated under the same seed
therefore face exactly the same reliability path (common random numbers).
Only the environment and the oracle baselines may read these values.
"""
from __future__ import annotations

import numpy as np


def _prob_vector(theta) -> np.ndarray:
    arr = np.asarray(theta, dtype=float).copy()
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("theta must be a non-empty 1-D vector")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"reliabilities must lie in [0, 1], got {arr}")
    arr.setflags(write=False)
    return arr


class ReliabilityProfile:
    """Base class: ``get(t)`` returns the true reliability vector at round t."""

    kind = "base"

    @property
    def K(self) -> int:
        return int(self.get(1).shape[0])

    def get(self, t: int) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def change_point(self):
        """Round tau after which reliabilities shift (None if no single shift)."""
        return None

    def describe(self) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


class StationaryProfile(ReliabilityProfile):
    """theta_k(t) = theta_k(0) for all t."""

    kind = "stationary"

    def __init__(self, theta):
        self.theta = _prob_vector(theta)

    def get(self, t: int) -> np.ndarray:
        return self.theta

    def describe(self) -> dict:
        return {"type": "stationary", "theta": self.theta.tolist()}


class AbruptChangeProfile(ReliabilityProfile):
    """Eq. (1): theta^(1) for t <= tau and theta^(2) for t > tau."""

    kind = "abrupt"

    def __init__(self, theta_before, theta_after, tau: int):
        self.theta_before = _prob_vector(theta_before)
        self.theta_after = _prob_vector(theta_after)
        if self.theta_before.shape != self.theta_after.shape:
            raise ValueError("theta_before and theta_after must have the same length")
        self.tau = int(tau)

    def get(self, t: int) -> np.ndarray:
        return self.theta_before if t <= self.tau else self.theta_after

    @property
    def change_point(self):
        return self.tau

    def describe(self) -> dict:
        return {"type": "abrupt", "theta_before": self.theta_before.tolist(),
                "theta_after": self.theta_after.tolist(), "tau": self.tau}


class DriftProfile(ReliabilityProfile):
    """Optional extension, Eq. (2): Gaussian random walk on the logit scale.

    logit(theta_k(t+1)) = logit(theta_k(t)) + sigma * eps_{k,t},  eps ~ N(0, 1).
    The whole path is pre-computed from ``seed`` so it is identical for every
    algorithm run with that seed.
    """

    kind = "drift"

    def __init__(self, theta0, sigma: float, T: int, seed=None, eps: float = 1e-4):
        theta0 = np.clip(_prob_vector(theta0), eps, 1.0 - eps)
        rng = np.random.default_rng(seed)
        K = theta0.shape[0]
        logits = np.empty((T, K))
        logits[0] = np.log(theta0 / (1.0 - theta0))
        if T > 1:
            steps = sigma * rng.standard_normal((T - 1, K))
            logits[1:] = logits[0] + np.cumsum(steps, axis=0)
        self.path = 1.0 / (1.0 + np.exp(-logits))
        self.path.setflags(write=False)
        self.theta0 = theta0
        self.sigma = float(sigma)
        self.T = int(T)

    def get(self, t: int) -> np.ndarray:
        idx = min(max(int(t), 1), self.T) - 1
        return self.path[idx]

    def describe(self) -> dict:
        return {"type": "drift", "theta0": self.theta0.tolist(), "sigma": self.sigma, "T": self.T}


class MultiShiftProfile(ReliabilityProfile):
    """Generalised multi-shift setting (proposal Sec. 2.3, item 2).

    Reliabilities are piecewise constant.  Before every round t >= 2 a change
    occurs independently with probability ``p_c``; when it does, every
    theta_k is redrawn uniformly from [low, high].  The proposal does not fix
    the post-change distribution; uniform redraws are our (documented) choice.
    The whole path is pre-computed from ``seed`` so that every algorithm run
    with that seed faces the same change points and values.
    """

    kind = "multishift"

    def __init__(self, theta0, p_c: float, T: int, seed=None, low: float = 0.2, high: float = 0.95):
        if not (0.0 <= p_c <= 1.0):
            raise ValueError("p_c must lie in [0, 1]")
        if not (0.0 <= low < high <= 1.0):
            raise ValueError("need 0 <= low < high <= 1")
        theta0 = _prob_vector(theta0)
        rng = np.random.default_rng(seed)
        K = theta0.shape[0]
        T = int(T)
        flips = rng.random(T) < p_c
        flips[0] = False
        draws = rng.uniform(low, high, size=(T, K))
        path = np.empty((T, K))
        path[0] = theta0
        for i in range(1, T):
            path[i] = draws[i] if flips[i] else path[i - 1]
        self.path = path
        self.path.setflags(write=False)
        # 1-indexed rounds at which a new reliability vector takes effect
        self.shift_rounds = [int(i) + 1 for i in np.flatnonzero(flips)]
        self.theta0, self.p_c, self.T = theta0, float(p_c), T
        self.low, self.high = float(low), float(high)

    @property
    def n_changes(self) -> int:
        return len(self.shift_rounds)

    def get(self, t: int) -> np.ndarray:
        idx = min(max(int(t), 1), self.T) - 1
        return self.path[idx]

    def describe(self) -> dict:
        return {"type": "multishift", "theta0": self.theta0.tolist(), "p_c": self.p_c,
                "low": self.low, "high": self.high, "T": self.T, "shift_rounds": self.shift_rounds}


# --- convenience constructors (kept for backwards compatibility) -------------

def stationary(theta) -> StationaryProfile:
    return StationaryProfile(theta)


def abrupt_change(theta_initial, theta_after, tau) -> AbruptChangeProfile:
    return AbruptChangeProfile(theta_initial, theta_after, tau)


def continuous_drift(theta_initial, sigma, T, seed=None) -> DriftProfile:
    return DriftProfile(theta_initial, sigma, T, seed=seed)


def multi_shift(theta_initial, p_c, T, seed=None, low=0.2, high=0.95) -> MultiShiftProfile:
    return MultiShiftProfile(theta_initial, p_c, T, seed=seed, low=low, high=high)


def make_profile(spec: dict, theta, T: int, seed=None) -> ReliabilityProfile:
    """Build a profile from a config ``reliability`` block.

    spec examples::
        {"type": "stationary"}
        {"type": "abrupt", "theta_after": [0.8, 0.7, 0.4], "tau": 1000}
        {"type": "drift", "sigma": 0.05}
        {"type": "multishift", "p_c": 0.002, "low": 0.2, "high": 0.95}
    """
    kind = (spec or {}).get("type", "stationary")
    if kind == "stationary":
        return StationaryProfile(theta)
    if kind == "abrupt":
        return AbruptChangeProfile(theta, spec["theta_after"], spec["tau"])
    if kind == "drift":
        return DriftProfile(theta, spec["sigma"], T, seed=seed)
    if kind == "multishift":
        return MultiShiftProfile(theta, spec["p_c"], T, seed=seed,
                                 low=spec.get("low", 0.2), high=spec.get("high", 0.95))
    raise ValueError(f"Unknown reliability profile type: {kind!r}")
