# backend/src/help_seeking/helper_qlearning.py
"""Independent tabular Q-learning for helper agents' accept/reject decisions.

Each agent maintains its own Q-table for the helping decision:
- State: (workload_level, resource_level) discretized into 3 levels each -> 9 states
- Action: ACCEPT (0) or REJECT (1)
- Q-table: shape (9, 2)

Update: Q(s,a) = Q(s,a) + alpha * [r + gamma_Q * max_a' Q(s',a') - Q(s,a)]

Parameters:
- Learning rate alpha = 0.1
- Discount factor gamma_Q = 0.9
- Epsilon = 0.1 (decays to 0.02 minimum)
- Epsilon-greedy with random tie-breaking
- Independent RNG per agent
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# State encoding: workload (0,1,2) x resources (0,1,2) -> 9 states
def encode_state(workload: int, resources: int) -> int:
    """Encode (workload, resources) into a single state index 0-8."""
    if not (0 <= workload <= 2 and 0 <= resources <= 2):
        raise ValueError(f"workload and resources must be in {{0,1,2}}, got ({workload}, {resources})")
    return workload * 3 + resources


def decode_state(state: int) -> tuple[int, int]:
    """Decode state index into (workload, resources)."""
    return state // 3, state % 3


@dataclass
class PendingHelpingTransition:
    """Stores pending transition between decision epochs in semi-Markov RL."""
    state: int
    action: int
    reward: float
    decision_round: int


@dataclass
class HelperQLearning:
    """Independent Q-learning for a single agent's helping policy."""

    # Q-table: 9 states x 2 actions (ACCEPT=0, REJECT=1)
    q_table: np.ndarray = field(default_factory=lambda: np.zeros((9, 2), dtype=float))

    # Q-learning parameters
    alpha: float = 0.1          # learning rate
    gamma_q: float = 0.9        # RL discount factor (not to be confused with belief discount gamma)
    epsilon: float = 0.1        # exploration rate
    epsilon_min: float = 0.02   # minimum exploration rate
    epsilon_decay: float = 0.9995  # decay per update (very slow)

    # Explicit state visitation counts (one counter per state index 0-8)
    state_visits: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=int))

    # Decision-epoch pending transition (Semi-Markov decision epoch update)
    pending_transition: PendingHelpingTransition | None = field(default=None, repr=False)

    # Agent's own RNG for independent exploration
    seed: int | None = field(default=None, repr=False)
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(), repr=False)

    def __post_init__(self):
        if self.seed is not None:
            self._rng = np.random.default_rng(self.seed)
        else:
            self._rng = np.random.default_rng()
        if self.state_visits is None or len(self.state_visits) != 9:
            self.state_visits = np.zeros(9, dtype=int)
        self.pending_transition = None

    def reset(self, seed: int | None = None) -> None:
        """Reset Q-table, visitation counts, pending transitions, and RNG."""
        self.q_table.fill(0.0)
        self.state_visits.fill(0)
        self.pending_transition = None
        if seed is not None:
            self.seed = seed
            self._rng = np.random.default_rng(seed)
        self.epsilon = 0.1

    # --- action selection & exact policy probabilities ----------------------
    def choose_action(self, workload: int, resources: int) -> int:
        """Epsilon-greedy action selection with random tie-breaking."""
        state = encode_state(workload, resources)
        self.state_visits[state] += 1
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(2))  # ACCEPT=0 or REJECT=1
        # Greedy with random tie-breaking
        q_vals = self.q_table[state]
        best = q_vals.max()
        tied = np.flatnonzero(q_vals >= best - 1e-12)
        return int(self._rng.choice(tied))

    def p_accept(self, workload: int, resources: int) -> float:
        """Exact probability of selecting ACCEPT (0) under actual action-selection policy.

        P(ACCEPT) = (1 - epsilon) * greedy_probability(ACCEPT) + (epsilon / 2)
        where greedy_probability(ACCEPT) is:
          - 1.0 if ACCEPT is the unique greedy action (Q(ACCEPT) > Q(REJECT))
          - 0.0 if REJECT is the unique greedy action (Q(REJECT) > Q(ACCEPT))
          - 0.5 if both actions are tied (Q(ACCEPT) == Q(REJECT))
        """
        state = encode_state(workload, resources)
        q_vals = self.q_table[state]
        diff = q_vals[0] - q_vals[1]  # ACCEPT = 0, REJECT = 1
        if diff > 1e-12:
            greedy_p = 1.0
        elif diff < -1e-12:
            greedy_p = 0.0
        else:
            greedy_p = 0.5
        return float((1.0 - self.epsilon) * greedy_p + (self.epsilon / 2.0))

    def hypothetical_p_accept(self, next_workload: int, next_resources: int, current_round: int) -> float:
        """Compute p_accept after hypothetical flush_pending, without mutating q_table or epsilon."""
        if self.pending_transition is None:
            return self.p_accept(next_workload, next_resources)

        pt = self.pending_transition
        delta_t = max(1, int(current_round) - pt.decision_round)
        gamma_eff = float(self.gamma_q ** delta_t)
        next_state = encode_state(next_workload, next_resources)

        q_sa = self.q_table[pt.state, pt.action]
        max_next_q = self.q_table[next_state].max()
        target = pt.reward + gamma_eff * max_next_q
        new_q_sa = q_sa + self.alpha * (target - q_sa)

        q_next = self.q_table[next_state].copy()
        if pt.state == next_state:
            q_next[pt.action] = new_q_sa

        hypo_eps = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        diff = q_next[0] - q_next[1]
        if diff > 1e-12:
            greedy_p = 1.0
        elif diff < -1e-12:
            greedy_p = 0.0
        else:
            greedy_p = 0.5
        return float((1.0 - hypo_eps) * greedy_p + (hypo_eps / 2.0))

    def greedy_action(self, workload: int, resources: int) -> int:
        """Pure greedy action (for evaluation)."""
        state = encode_state(workload, resources)
        q_vals = self.q_table[state]
        best = q_vals.max()
        tied = np.flatnonzero(q_vals >= best - 1e-12)
        return int(self._rng.choice(tied))

    # --- Decision-epoch (Semi-Markov) RL transitions ----------------------
    def record_decision(self, workload: int, resources: int, action: int,
                        reward: float, round_t: int) -> None:
        """Store pending transition for helper decision epoch.

        The Q-learning target will be evaluated when the helper is next called upon,
        using its true state at that future decision epoch as s'.
        """
        state = encode_state(workload, resources)
        self.pending_transition = PendingHelpingTransition(
            state=state,
            action=action,
            reward=float(reward),
            decision_round=int(round_t),
        )

    def flush_pending(self, next_workload: int, next_resources: int, current_round: int) -> bool:
        """Complete pending Q-learning update using the true next decision state.

        Applies geometric discounting over elapsed rounds:
            gamma_eff = gamma_q ** (current_round - decision_round)

        Returns:
            True if an update was performed, False if no transition was pending.
        """
        if self.pending_transition is None:
            return False

        pt = self.pending_transition
        delta_t = max(1, int(current_round) - pt.decision_round)
        gamma_eff = float(self.gamma_q ** delta_t)

        next_state = encode_state(next_workload, next_resources)
        q_sa = self.q_table[pt.state, pt.action]
        max_next_q = self.q_table[next_state].max()
        target = pt.reward + gamma_eff * max_next_q
        self.q_table[pt.state, pt.action] = q_sa + self.alpha * (target - q_sa)

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.pending_transition = None
        return True

    def finalize_episode(self, final_workload: int, final_resources: int, final_round: int) -> bool:
        """Finalize any pending decision transition at episode termination without bootstrapping.

        At episode termination, there is no subsequent decision epoch, so:
            Q(s, a) <- Q(s, a) + alpha * [r - Q(s, a)]
        """
        if self.pending_transition is None:
            return False

        pt = self.pending_transition
        q_sa = self.q_table[pt.state, pt.action]
        target = pt.reward  # No bootstrap from next state at episode termination
        self.q_table[pt.state, pt.action] = q_sa + self.alpha * (target - q_sa)

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.pending_transition = None
        return True

    # --- Q-learning update --------------------------------------------------
    def update(self, workload: int, resources: int, action: int,
               reward: float, next_workload: int, next_resources: int) -> None:
        """Q-learning update: Q(s,a) += alpha * [r + gamma_Q * max Q(s',a') - Q(s,a)]."""
        state = encode_state(workload, resources)
        next_state = encode_state(next_workload, next_resources)

        q_sa = self.q_table[state, action]
        max_next_q = self.q_table[next_state].max()
        target = reward + self.gamma_q * max_next_q
        self.q_table[state, action] = q_sa + self.alpha * (target - q_sa)

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # --- inspection ---------------------------------------------------------
    def get_q_values(self, workload: int, resources: int) -> np.ndarray:
        """Get Q-values for a state."""
        return self.q_table[encode_state(workload, resources)].copy()

    def get_policy(self) -> np.ndarray:
        """Get the greedy policy for all 9 states (0=ACCEPT, 1=REJECT)."""
        return self.q_table.argmax(axis=1)

    def snapshot(self) -> dict:
        return {
            "q_table": self.q_table.tolist(),
            "state_visits": self.state_visits.tolist(),
            "alpha": self.alpha,
            "gamma_q": self.gamma_q,
            "epsilon": self.epsilon,
            "policy": self.get_policy().tolist(),
        }