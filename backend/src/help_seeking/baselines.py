# backend/src/help_seeking/baselines.py
"""Benchmark baselines (proposal Sec. 4.2)."""
from __future__ import annotations

import numpy as np

from .algorithms import Decision, Policy, argmax_random
from .utilities import dynamic_oracle


class NeverAsk(Policy):
    """1. Never-Ask / Always-Act-Alone: d_t = 0 in every round."""

    name = "Never-Ask"

    def decide(self, U_self, theta_true=None) -> Decision:
        return Decision(d=0, a=None)


class AlwaysAskRandom(Policy):
    """2. Always-Ask + Random Selection: a_t ~ Uniform(H)."""

    name = "Always-Ask + Random"

    def decide(self, U_self, theta_true=None) -> Decision:
        return Decision(d=1, a=int(self.rng.integers(self.K)))


class AlwaysAskFixedCheapest(Policy):
    """3. Always-Ask + Fixed-Cheapest: a_t = argmin_k c_k for all t.

    When several helpers share the minimum cost (E1 uses c = (0.1, 0.1, 0.1))
    the fixed helper is drawn uniformly among them once, at the start of each
    run.  Always taking index 0 would silently hand this baseline the best
    helper in E1, because helper 1 is the most reliable there.
    """

    name = "Always-Ask + Fixed-Cheapest"

    def __init__(self, K, R=1.0, lambda_mult=1.0, c=None, gamma=1.0, seed=None):
        super().__init__(K, R, lambda_mult, c, gamma, seed)
        self._pick()

    def _pick(self):
        cheapest = np.flatnonzero(self.c <= self.c.min() + 1e-12)
        self.fixed = int(self.rng.choice(cheapest))

    def reset(self, seed=None):
        super().reset(seed)
        self._pick()

    def decide(self, U_self, theta_true=None) -> Decision:
        return Decision(d=1, a=self.fixed)


class AlwaysAskEmpiricalGreedy(Policy):
    """4. Always-Ask + Empirical Greedy: always asks, picks argmax R theta_hat - lambda c
    with undiscounted counts (gamma = 1) and never considers acting alone."""

    name = "Always-Ask + Empirical Greedy"
    learns = True

    def __init__(self, K, R=1.0, lambda_mult=1.0, c=None, gamma=1.0, seed=None):
        super().__init__(K, R, lambda_mult, c, gamma=1.0, seed=seed)

    def decide(self, U_self, theta_true=None) -> Decision:
        theta_hat = self.trust.posterior_mean()
        scores = self.R * theta_hat - self.costs
        k = argmax_random(scores, self.rng)
        return Decision(d=1, a=k, candidate=k, candidate_score=float(scores[k]),
                        scores=scores, extra={"theta_hat": theta_hat})


class DynamicOracle(Policy):
    """5. Dynamic Oracle: knows theta(t), c and U_self every round (Eqs. 11-12)."""

    name = "Dynamic Oracle"
    uses_truth = True

    def decide(self, U_self, theta_true=None) -> Decision:
        if theta_true is None:
            raise ValueError("DynamicOracle needs the true reliabilities")
        o = dynamic_oracle(theta_true, U_self, R=self.R, lambda_mult=self.lambda_mult, c=self.c)
        return Decision(d=o.d, a=o.a, candidate=int(np.argmax(o.mu)),
                        candidate_score=float(o.mu.max()), scores=o.mu)


class InitialKnowledgeOracle(Policy):
    """6. Initial-Knowledge Oracle: knows theta(0) only and repeats the joint
    action that was optimal at t = 1 for the whole horizon."""

    name = "Initial-Knowledge Oracle"

    def __init__(self, K, R=1.0, lambda_mult=1.0, c=None, gamma=1.0, seed=None,
                 theta_initial=None, U_self=None):
        super().__init__(K, R, lambda_mult, c, gamma, seed)
        if theta_initial is None or U_self is None:
            raise ValueError("InitialKnowledgeOracle needs theta_initial and U_self")
        o = dynamic_oracle(theta_initial, U_self, R=self.R, lambda_mult=self.lambda_mult, c=self.c)
        self.fixed_d, self.fixed_a = o.d, o.a

    def decide(self, U_self, theta_true=None) -> Decision:
        return Decision(d=self.fixed_d, a=self.fixed_a)
