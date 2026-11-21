# backend/src/help_seeking/requester_timing.py
"""Learned and heuristic timing policies for the WHEN TO ASK decision.

Each autonomous agent maintains its own private RequesterTimingPolicy.
Action space:
    CONTINUE_SELF (0): Continue executing the task independently.
    EVALUATE_HELP (1): Trigger the whether/whom requester decision process.

State space:
    Compact 27-state discretization of task lifecycle:
        - progress_level (3 levels: Low < 0.34, Med 0.34-0.66, High >= 0.67)
        - time_left_level (3 levels: Critical <= 1, Moderate 2-3, Ample >= 4)
        - attempts_level (3 levels: 0 attempts, 1 attempt, >= 2 attempts)
    state = progress_level * 9 + time_left_level * 3 + attempts_level in [0, 26]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .task_lifecycle import TemporalTask

CONTINUE_SELF = 0
EVALUATE_HELP = 1


def encode_timing_state(task: TemporalTask) -> int:
    """Discretize task state into an integer index in [0, 26]."""
    p = task.progress_fraction
    if p < 0.34:
        p_lvl = 0
    elif p < 0.67:
        p_lvl = 1
    else:
        p_lvl = 2

    rem = task.remaining_time
    if rem <= 1:
        t_lvl = 0
    elif rem <= 3:
        t_lvl = 1
    else:
        t_lvl = 2

    att = task.attempts
    if att == 0:
        a_lvl = 0
    elif att == 1:
        a_lvl = 1
    else:
        a_lvl = 2

    return p_lvl * 9 + t_lvl * 3 + a_lvl


def decode_timing_state(state: int) -> tuple[int, int, int]:
    """Decode integer index into (progress_level, time_left_level, attempts_level)."""
    p_lvl = state // 9
    rem = state % 9
    t_lvl = rem // 3
    a_lvl = rem % 3
    return p_lvl, t_lvl, a_lvl


class BaseTimingPolicy:
    """Abstract base class for requester timing policies."""

    def select_action(self, task: TemporalTask) -> int:
        raise NotImplementedError

    def update(self, state: int, action: int, reward: float, next_state: Optional[int], done: bool) -> None:
        pass

    def reset(self, seed: Optional[int] = None) -> None:
        pass


@dataclass
class RequesterTimingPolicy(BaseTimingPolicy):
    """Independent tabular Q-learning for an agent's WHEN TO ASK decision.

    Learns when to stop executing solo and trigger help evaluation before deadlines expire.
    """
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 0.15
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.998
    seed: Optional[int] = field(default=None, repr=False)

    q_table: np.ndarray = field(default_factory=lambda: np.zeros((27, 2), dtype=float))
    state_visits: np.ndarray = field(default_factory=lambda: np.zeros(27, dtype=int))
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(), repr=False)

    def __post_init__(self):
        if self.seed is not None:
            self._rng = np.random.default_rng(self.seed)
        else:
            self._rng = np.random.default_rng()
        if self.state_visits is None or len(self.state_visits) != 27:
            self.state_visits = np.zeros(27, dtype=int)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset Q-table, state visits, and RNG."""
        self.q_table.fill(0.0)
        self.state_visits.fill(0)
        self.epsilon = 0.15
        if seed is not None:
            self.seed = seed
            self._rng = np.random.default_rng(seed)

    def select_action(self, task: TemporalTask) -> int:
        """Select timing action (CONTINUE_SELF or EVALUATE_HELP).

        If help was already requested on this task, the agent cannot ask again;
        it is forced to CONTINUE_SELF.
        """
        if task.help_requested:
            return CONTINUE_SELF

        state = encode_timing_state(task)
        self.state_visits[state] += 1

        # Epsilon-greedy exploration
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(2))

        # Greedy with random tie-breaking
        q_vals = self.q_table[state]
        best_val = q_vals.max()
        tied = np.flatnonzero(q_vals >= best_val - 1e-12)
        return int(self._rng.choice(tied))

    def update(self, state: int, action: int, reward: float, next_state: Optional[int], done: bool) -> None:
        """Standard Q-learning update for the timing policy."""
        q_sa = self.q_table[state, action]
        if done or next_state is None:
            target = float(reward)
        else:
            max_next_q = float(self.q_table[next_state].max())
            target = float(reward) + self.gamma * max_next_q

        self.q_table[state, action] = q_sa + self.alpha * (target - q_sa)

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def snapshot(self) -> dict:
        return {
            "q_table": self.q_table.tolist(),
            "state_visits": self.state_visits.tolist(),
            "epsilon": self.epsilon,
            "policy": self.q_table.argmax(axis=1).tolist(),
        }


# --- Model-based timing: optimal stopping with the requester's learned beliefs ---

class PlannedTimingPolicy(BaseTimingPolicy):
    """Model-based WHEN-to-ask policy: approximate optimal stopping with frozen learned beliefs.

    At every step the requester plans with its *current learned beliefs* about its
    teammates (posterior means p_hat_j, theta_hat_j) and the known solo-progress model.
    Two simplifications make this an approximation of the optimal policy for the full
    stochastic system: (i) the beliefs are held fixed during the backward induction,
    although in the environment they decay every global round and change with each
    observation; (ii) it plans with posterior means, while the actual whether/whom
    decision is taken by the requester policy (posterior samples under MA-D-TS).
    With q_j = p_hat_j * theta_hat_j and the one-request-per-task rule, the value of
    asking now at progress p with tau steps left is

        Ask(p, tau) = max_j [ -lambda c_j + q_j R + (1 - q_j) F(p, tau) ],
        F(p, tau)   = -c_step + gamma_T V_solo(p, tau - 1)   (tau > 1),   -c_fail (tau = 1),

    and the value of taking one more solo step while keeping the option to ask later is

        Wait(p, tau) = E_dp[ R 1{p + dp >= 1} + 1{p + dp < 1} ( -c_step + gamma_T V*(p + dp, tau - 1) ) ]
                       (the bracket is -c_fail when tau = 1),
        V*(p, tau)   = max(Ask(p, tau), Wait(p, tau)).

    The policy returns EVALUATE_HELP iff Ask >= Wait at the current state.  Unlike the
    whether-rule (ask iff V_ask > V_solo, i.e. "ask now" versus "never ask"), Wait includes
    the option value of asking later.  V* is computed by backward induction on a progress
    grid; V_solo is the exact Irwin-Hall value from the environment.

    The policy must be bound to its agent and environment (``bind``), which the temporal
    environment does automatically.
    """

    def __init__(self, grid_points: int = 401, quadrature_points: int = 41):
        self.grid = np.linspace(0.0, 1.0, int(grid_points))
        self.n_quad = int(quadrature_points)
        self.agent = None
        self.env = None
        self._vsolo: Optional[dict] = None

    def bind(self, agent, env) -> None:
        self.agent, self.env = agent, env
        self._vsolo = None

    def reset(self, seed: Optional[int] = None) -> None:  # no learned state of its own
        pass

    def _solo_value(self, tau: int) -> np.ndarray:
        """Exact solo value V_solo(., tau) on the progress grid, computed lazily and cached per tau,
        so tasks with any deadline (not only the environment default) are supported."""
        if self._vsolo is None:
            self._vsolo = {}
        tau = int(tau)
        if tau not in self._vsolo:
            env = self.env
            kappa = env.solo_capability
            self._vsolo[tau] = np.array([env.compute_v_solo(float(p), tau, kappa) for p in self.grid])
        return self._vsolo[tau]

    def ask_and_wait(self, q: np.ndarray, comm_cost: np.ndarray, tau: int):
        """Return grids (Ask(., tau), Wait(., tau)) for helper success probabilities q and costs."""
        env = self.env
        R, c_step, c_fail, g = env.R, env.step_cost, env.c_fail, env.gamma_timing
        kappa = env.solo_capability
        dp = kappa * (0.20 + 0.25 * (np.arange(self.n_quad) + 0.5) / self.n_quad)  # U(0.20, 0.45) * kappa
        G = self.grid
        q = np.asarray(q, dtype=float)
        comm_cost = np.asarray(comm_cost, dtype=float)
        v_star_prev = None
        ask = wait = None
        for t in range(1, int(tau) + 1):
            nxt = np.minimum(G[:, None] + dp[None, :], 1.0)
            done = nxt >= 1.0 - 1e-12
            cont = np.full_like(nxt, -c_fail) if t == 1 else (-c_step + g * np.interp(nxt, G, v_star_prev))
            wait = np.where(done, R, cont).mean(axis=1)
            fallback = np.full_like(G, -c_fail) if t == 1 else (-c_step + g * self._solo_value(t - 1))
            ask = (-comm_cost[None, :] + q[None, :] * R + (1.0 - q[None, :]) * fallback[:, None]).max(axis=1)
            v_star_prev = np.maximum(ask, wait)
        return ask, wait

    def select_action(self, task: TemporalTask) -> int:
        if task.help_requested or task.remaining_time <= 0:
            return CONTINUE_SELF
        if self.agent is None or self.env is None:
            raise RuntimeError("PlannedTimingPolicy must be bound to an agent and environment")
        pol = self.agent.requester_policy
        q = pol.acceptance_means() * pol.trust_means()
        ask, wait = self.ask_and_wait(q, pol.costs, int(task.remaining_time))
        p = float(task.progress)
        a = float(np.interp(p, self.grid, ask))
        w = float(np.interp(p, self.grid, wait))
        return EVALUATE_HELP if a >= w else CONTINUE_SELF


# --- Heuristic Timing Baselines for Fair Comparison (E8) ------------------

class ImmediateHelpTimingPolicy(BaseTimingPolicy):
    """Baseline: Triggers help evaluation immediately upon task arrival and at every subsequent eligible decision epoch until help is requested."""
    def select_action(self, task: TemporalTask) -> int:
        return CONTINUE_SELF if task.help_requested else EVALUATE_HELP


class FixedDelayTimingPolicy(BaseTimingPolicy):
    """Baseline: Executes independently for K attempts before evaluating help."""
    def __init__(self, delay: int = 2):
        self.delay = delay

    def select_action(self, task: TemporalTask) -> int:
        if task.help_requested:
            return CONTINUE_SELF
        return EVALUATE_HELP if task.attempts >= self.delay else CONTINUE_SELF


class UrgencyTimingPolicy(BaseTimingPolicy):
    """Baseline: Evaluates help only when deadline slack becomes urgent (slack <= threshold)."""
    def __init__(self, urgency_threshold: int = 2):
        self.urgency_threshold = urgency_threshold

    def select_action(self, task: TemporalTask) -> int:
        if task.help_requested:
            return CONTINUE_SELF
        return EVALUATE_HELP if task.remaining_time <= self.urgency_threshold else CONTINUE_SELF


class NeverAskTimingPolicy(BaseTimingPolicy):
    """Baseline: Completes tasks entirely independently; never requests help."""
    def select_action(self, task: TemporalTask) -> int:
        return CONTINUE_SELF


def make_timing_policy(name: str, seed: Optional[int] = None, **kwargs) -> BaseTimingPolicy:
    """Factory for timing policies."""
    canonical = name.lower().replace("-", "_").replace(" ", "_")
    if canonical in ("learned", "learned_timing", "q_learning", "q_learning_timing"):
        return RequesterTimingPolicy(seed=seed, **kwargs)
    elif canonical in ("planned", "planned_timing", "model_based", "model_based_timing"):
        return PlannedTimingPolicy(**kwargs)
    elif canonical in ("immediate", "immediate_help", "immediate_timing"):
        return ImmediateHelpTimingPolicy()
    elif canonical in ("fixed_delay", "fixed_delay_timing", "delay"):
        delay = kwargs.get("delay", 2)
        return FixedDelayTimingPolicy(delay=delay)
    elif canonical in ("urgency", "urgency_based", "urgency_timing"):
        threshold = kwargs.get("urgency_threshold", 2)
        return UrgencyTimingPolicy(urgency_threshold=threshold)
    elif canonical in ("never", "never_ask", "never_ask_timing", "solo_only"):
        return NeverAskTimingPolicy()
    raise ValueError(f"Unknown timing policy: {name}")
