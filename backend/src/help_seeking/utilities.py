# backend/src/help_seeking/utilities.py
"""Utility model and the dynamic oracle (proposal Sec. 3, Eqs. 6-13)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Two expected utilities closer than TOL are treated as equal.  Needed because
# some configurations in Table 2 sit exactly on a decision boundary
# (e.g. E2 with U_self = 0.65 where R*theta_1 - lambda*c_1 = 0.65).
TOL = 1e-9


def expected_utility_helper(theta, R, lambda_mult, c):
    """mu_k(t) = R * theta_k(t) - lambda * c_k  (Eq. 7)."""
    return R * np.asarray(theta, dtype=float) - lambda_mult * np.asarray(c, dtype=float)


def estimated_utility_helper(theta_hat, R, lambda_mult, c):
    """U_hat_k(t) = R * theta_hat_k(t) - lambda * c_k  (Eq. 8)."""
    return R * np.asarray(theta_hat, dtype=float) - lambda_mult * np.asarray(c, dtype=float)


@dataclass(frozen=True)
class OracleDecision:
    """Full-information optimum for one round.

    d, a        : the oracle's choice (Eqs. 11-12); a is the best helper
                  (argmax mu_k) and is None when the oracle acts alone.
    mu_star     : max(U_self, max_k mu_k)  (Eq. 13).
    mu          : expected net utility of each helper, mu_k(t).
    best_helper : boolean mask of helpers attaining max_k mu_k (ties allowed).
    ask_ok      : asking the best helper is (weakly) optimal.
    alone_ok    : acting alone is (weakly) optimal.
    """
    d: int
    a: int | None
    mu_star: float
    mu: np.ndarray
    best_helper: np.ndarray
    ask_ok: bool
    alone_ok: bool

    def is_optimal_action(self, d: int, a: int | None) -> bool:
        if d == 0:
            return self.alone_ok
        return self.ask_ok and bool(self.best_helper[a])

    def is_correct_ask_decision(self, d: int) -> bool:
        return self.ask_ok if d == 1 else self.alone_ok


def dynamic_oracle(theta, U_self, R=1.0, lambda_mult=1.0, c=None) -> OracleDecision:
    """Dynamic oracle (Eqs. 11-13) with explicit tie handling."""
    theta = np.asarray(theta, dtype=float)
    c = np.zeros_like(theta) if c is None else np.asarray(c, dtype=float)
    mu = expected_utility_helper(theta, R, lambda_mult, c)
    best = float(mu.max())
    best_mask = mu >= best - TOL
    ask_ok = best >= U_self - TOL
    alone_ok = U_self >= best - TOL
    d = 1 if best > U_self + TOL else 0          # Eq. 11 (strict inequality)
    a = int(np.flatnonzero(best_mask)[0]) if d == 1 else None
    return OracleDecision(d=d, a=a, mu_star=max(float(U_self), best), mu=mu,
                          best_helper=best_mask, ask_ok=bool(ask_ok), alone_ok=bool(alone_ok))


def dynamic_oracle_decision(theta_true, U_self, R=1.0, lambda_mult=1.0, c=None):
    """Backwards-compatible tuple form: (d*, a*, max_k mu_k)."""
    o = dynamic_oracle(theta_true, U_self, R=R, lambda_mult=lambda_mult, c=c)
    return o.d, o.a, float(o.mu.max())
