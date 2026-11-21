# backend/src/help_seeking/simulation.py
"""One simulation run: environment + policy + logging (Algorithm 1 loop)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .algorithms import Policy
from .environment import SimulationEnvironment


@dataclass
class RunRecord:
    """Per-round log of one run, stored as arrays (index i = round t = i + 1).

    a = -1 and x = -1 mark rounds where the agent acted alone.
    mu_chosen is the *expected* utility of the action taken and regret is
    mu_star - mu_chosen, i.e. the pseudo-regret of Eq. (15) without
    Bernoulli noise.  theta_hat is the belief the agent held when deciding.
    """
    T: int
    K: int
    d: np.ndarray
    a: np.ndarray
    x: np.ndarray
    r: np.ndarray
    cost: np.ndarray
    mu_chosen: np.ndarray
    mu_star: np.ndarray
    regret: np.ndarray
    d_star: np.ndarray
    opt_action: np.ndarray
    ask_correct: np.ndarray
    best_helper_chosen: np.ndarray
    theta_true: np.ndarray
    theta_hat: np.ndarray | None

    @property
    def action(self) -> np.ndarray:
        """0 = act alone, k + 1 = asked helper k."""
        return np.where(self.d == 1, self.a + 1, 0)

    def history(self, start: int = 0, end: int | None = None) -> list[dict]:
        end = self.T if end is None else min(end, self.T)
        out = []
        for i in range(start, end):
            out.append({
                "t": i + 1, "d": int(self.d[i]),
                "a": None if self.d[i] == 0 else int(self.a[i]),
                "x": None if self.d[i] == 0 else int(self.x[i]),
                "r": float(self.r[i]), "cost": float(self.cost[i]),
                "mu_chosen": float(self.mu_chosen[i]), "mu_star": float(self.mu_star[i]),
                "regret": float(self.regret[i]), "d_star": int(self.d_star[i]),
                "optimal": bool(self.opt_action[i]),
                "theta_true": self.theta_true[i].tolist(),
                "theta_hat": None if self.theta_hat is None else self.theta_hat[i].tolist(),
            })
        return out


class SimulationRun:
    """Runs ``policy`` in ``environment`` for T rounds.

    Use ``run()`` for experiments or ``step(details=True)`` to advance one
    round at a time (dashboard / decision inspector).
    """

    def __init__(self, environment: SimulationEnvironment, policy: Policy, T: int,
                 record_beliefs: bool = True):
        if policy.K != environment.K:
            raise ValueError("policy and environment disagree on the number of teammates")
        self.env = environment
        self.policy = policy
        self.T = int(T)
        self.record_beliefs = record_beliefs
        self._oracle_cache: dict = {}
        self._alloc()

    def _alloc(self):
        T, K = self.T, self.env.K
        self._d = np.zeros(T, dtype=np.int8)
        self._a = np.full(T, -1, dtype=np.int16)
        self._x = np.full(T, -1, dtype=np.int8)
        self._r = np.zeros(T)
        self._cost = np.zeros(T)
        self._mu = np.zeros(T)
        self._mu_star = np.zeros(T)
        self._d_star = np.zeros(T, dtype=np.int8)
        self._opt = np.zeros(T, dtype=bool)
        self._ask_ok = np.zeros(T, dtype=bool)
        self._best_h = np.zeros(T, dtype=bool)
        self._theta = np.zeros((T, K))
        self._theta_hat = np.zeros((T, K)) if self.record_beliefs else None
        self.t = 0

    def reset(self, env_seed=None, policy_seed=None):
        self.env.reset(env_seed)
        self.policy.reset(policy_seed)
        self._oracle_cache.clear()
        self._alloc()

    @property
    def done(self) -> bool:
        return self.t >= self.T

    def _oracle(self, theta):
        key = theta.tobytes()
        o = self._oracle_cache.get(key)
        if o is None:
            o = self.env.oracle(theta)
            if len(self._oracle_cache) < 4096:
                self._oracle_cache[key] = o
        return o

    def step(self, details: bool = False):
        """Play one round.  Returns a dict describing it when ``details``."""
        if self.done:
            raise RuntimeError("simulation horizon reached")
        env, pol = self.env, self.policy
        theta = env.start_round()
        i = env.t - 1
        belief = pol.belief() if (self.record_beliefs or details) else None
        if details:
            S_before, F_before = pol.trust.S.copy(), pol.trust.F.copy()

        dec = pol.decide(env.U_self, theta if pol.uses_truth else None)
        oracle = self._oracle(theta)

        if dec.d == 1:
            x = env.generate_outcome(dec.a, theta)
            cost = env.lambda_mult * float(env.c[dec.a])
        else:
            x, cost = -1, 0.0
        r = env.realized_utility(dec.d, dec.a, x)
        mu = env.expected_utility(dec.d, dec.a, theta)
        pol.update(dec.d, dec.a, x)

        self._d[i] = dec.d
        if dec.d == 1:
            self._a[i] = dec.a
            self._x[i] = x
            self._best_h[i] = bool(oracle.best_helper[dec.a])
        self._r[i] = r
        self._cost[i] = cost
        self._mu[i] = mu
        self._mu_star[i] = oracle.mu_star
        self._d_star[i] = oracle.d
        self._opt[i] = oracle.is_optimal_action(dec.d, dec.a)
        self._ask_ok[i] = oracle.is_correct_ask_decision(dec.d)
        self._theta[i] = theta
        if self._theta_hat is not None:
            self._theta_hat[i] = belief
        self.t = env.t

        if not details:
            return None
        info = {
            "t": env.t, "U_self": env.U_self,
            "theta_true": theta.tolist(),
            "belief_before": {"theta_hat": belief.tolist(), "S": S_before.tolist(), "F": F_before.tolist()},
            "scores": None if dec.scores is None else np.asarray(dec.scores).tolist(),
            "extra": {k: np.asarray(v).tolist() for k, v in dec.extra.items()},
            "candidate": dec.candidate, "candidate_score": dec.candidate_score,
            "d": dec.d, "a": dec.a, "x": None if dec.d == 0 else x,
            "r": r, "cost": cost, "mu_chosen": mu, "mu_star": oracle.mu_star,
            "regret": oracle.mu_star - mu,
            "oracle": {"d": oracle.d, "a": oracle.a, "mu": oracle.mu.tolist()},
            "optimal": bool(self._opt[i]),
            "belief_after": pol.trust.snapshot(),
        }
        return info

    def run(self, T: int | None = None) -> RunRecord:
        if T is not None and T != self.T:
            self.T = int(T)
            self._alloc()
        while not self.done:
            self.step()
        return self.record()

    def record(self) -> RunRecord:
        n = self.t
        return RunRecord(
            T=n, K=self.env.K, d=self._d[:n], a=self._a[:n], x=self._x[:n], r=self._r[:n],
            cost=self._cost[:n], mu_chosen=self._mu[:n], mu_star=self._mu_star[:n],
            regret=self._mu_star[:n] - self._mu[:n], d_star=self._d_star[:n],
            opt_action=self._opt[:n], ask_correct=self._ask_ok[:n],
            best_helper_chosen=self._best_h[:n], theta_true=self._theta[:n],
            theta_hat=None if self._theta_hat is None else self._theta_hat[:n])
