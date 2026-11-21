# backend/src/help_seeking/algorithms.py
"""Adaptive help-seeking policies (proposal Sec. 4.1, Algorithm 1).

All three learning methods share the same two-stage rule and differ only in
the helper score they compute from the discounted Beta-Bernoulli belief:

    Stage 2 candidate : k* = argmax_k score_k        (ties broken uniformly)
    Stage 1 ask?      : d_t = 1  iff  score_{k*} > U_self
    if d_t = 1        : a_t = k*

    Discounted Greedy : score_k = R theta_hat_k - lambda c_k
    Discounted UCB    : score_k = R theta_hat_k + R alpha sqrt(2 ln max(N, 2) / (N_k + eps)) - lambda c_k
    Discounted TS     : score_k = R theta_tilde_k - lambda c_k,  theta_tilde_k ~ Beta(1 + S_k, 1 + F_k)

Policies never see the true reliabilities: ``decide`` receives only U_self
(learning policies ignore the ``theta_true`` argument, which is passed only to
policies that declare ``uses_truth = True``, i.e. the dynamic oracle).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .trust import TrustState

TIE_TOL = 1e-12


def argmax_random(scores: np.ndarray, rng: np.random.Generator) -> int:
    """argmax with ties broken uniformly at random (Algorithm 1, line 6)."""
    best = scores.max()
    tied = np.flatnonzero(scores >= best - TIE_TOL)
    if tied.size == 1:
        return int(tied[0])
    return int(rng.choice(tied))


@dataclass
class Decision:
    d: int                              # 1 = ask for help, 0 = act alone
    a: int | None                       # queried helper (None if d = 0)
    candidate: int | None = None        # k*, the best candidate helper
    candidate_score: float | None = None
    scores: np.ndarray | None = None    # score of every helper
    extra: dict = field(default_factory=dict)


class Policy:
    """Common interface for every decision rule (learning methods and baselines)."""

    name = "policy"
    uses_truth = False     # True only for the dynamic oracle
    learns = False         # True if decisions depend on the trust belief

    def __init__(self, K: int, R: float = 1.0, lambda_mult: float = 1.0, c=None,
                 gamma: float = 1.0, seed=None):
        self.K = int(K)
        self.R = float(R)
        self.lambda_mult = float(lambda_mult)
        self.c = np.zeros(self.K) if c is None else np.asarray(c, dtype=float)
        self.gamma = float(gamma)
        self.trust = TrustState(self.K, gamma=self.gamma)
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None) -> None:
        if seed is not None:
            self._seed = seed
        self.trust.reset()
        self.rng = np.random.default_rng(self._seed)

    @property
    def costs(self) -> np.ndarray:
        return self.lambda_mult * self.c

    def belief(self) -> np.ndarray:
        """Current trust estimate theta_hat_k (Eq. 5)."""
        return self.trust.posterior_mean()

    def decide(self, U_self: float, theta_true=None) -> Decision:  # pragma: no cover
        raise NotImplementedError

    def update(self, d: int, a, x) -> None:
        """End-of-round belief update (Eqs. 3-4 / Algorithm 1 lines 22-23, 27)."""
        if d == 1:
            self.trust.update(a, x)
        else:
            self.trust.decay()

    # Backwards-compatible names used by the first version of the code.
    def select_action(self, U_self, theta_true=None):
        dec = self.decide(U_self, theta_true)
        return dec.d, dec.a

    def observe_and_update(self, a_t, x_t):
        self.update(1, a_t, x_t)


class IndexPolicy(Policy):
    """Two-stage sequential decision shared by D-Greedy, D-UCB and D-TS."""

    learns = True

    def helper_scores(self):  # pragma: no cover - interface
        """Return (scores, extra_info_for_inspection)."""
        raise NotImplementedError

    def decide(self, U_self: float, theta_true=None) -> Decision:
        scores, extra = self.helper_scores()
        k = argmax_random(scores, self.rng)
        best = float(scores[k])
        d = 1 if best > U_self else 0
        return Decision(d=d, a=k if d == 1 else None, candidate=k,
                        candidate_score=best, scores=scores, extra=extra)


class DiscountedGreedy(IndexPolicy):
    """Method A: exploit the posterior mean only."""

    name = "Greedy"

    def helper_scores(self):
        theta_hat = self.trust.posterior_mean()
        return self.R * theta_hat - self.costs, {"theta_hat": theta_hat}


class DiscountedUCB(IndexPolicy):
    """Method B: optimistic index with a discounted confidence bonus (Eq. 16)."""

    name = "D-UCB"

    def __init__(self, K, R=1.0, lambda_mult=1.0, c=None, gamma=1.0, seed=None,
                 alpha_ucb: float = 1.0, epsilon: float = 1e-6):
        super().__init__(K, R, lambda_mult, c, gamma, seed)
        if alpha_ucb <= 0:
            raise ValueError("alpha_ucb must be > 0")
        self.alpha_ucb = float(alpha_ucb)
        self.epsilon = float(epsilon)

    def bonus(self) -> np.ndarray:
        N_k = self.trust.effective_sample_size()
        N = float(N_k.sum())
        return self.R * self.alpha_ucb * np.sqrt(2.0 * np.log(max(N, 2.0)) / (N_k + self.epsilon))

    def helper_scores(self):
        theta_hat = self.trust.posterior_mean()
        bonus = self.bonus()
        return self.R * theta_hat + bonus - self.costs, {"theta_hat": theta_hat, "bonus": bonus}


class DiscountedThompson(IndexPolicy):
    """Method C: posterior sampling from the discounted Beta belief (Eq. 17)."""

    name = "D-TS"

    def helper_scores(self):
        alpha, beta = self.trust.posterior_params()
        sample = self.rng.beta(alpha, beta)
        return self.R * sample - self.costs, {"theta_sample": sample,
                                              "theta_hat": self.trust.posterior_mean()}
