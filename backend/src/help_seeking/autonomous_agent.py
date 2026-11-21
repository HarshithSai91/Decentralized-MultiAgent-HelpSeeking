# backend/src/help_seeking/autonomous_agent.py
"""Autonomous agent that can act as both requester and helper.

Each agent maintains:
- Requester side: acceptance beliefs, trust beliefs, requester policy (D-Greedy/D-UCB/D-TS)
- Helper side: Q-table for accept/reject decisions, workload, resources
- Individual rewards for both roles
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .acceptance import AcceptanceState
from .algorithms import Decision, argmax_random
from .helper_qlearning import HelperQLearning, encode_state
from .multiagent_policies import (
    MultiAgentPolicy,
    MultiAgentDiscountedGreedy,
    MultiAgentDiscountedUCB,
    MultiAgentDiscountedThompson,
    make_multiagent_policy,
)
from .requester_timing import BaseTimingPolicy, RequesterTimingPolicy, make_timing_policy
from .task_lifecycle import TemporalTask
from .trust import TrustState

# Helper action constants
ACCEPT = 0
REJECT = 1


@dataclass
class AutonomousAgent:
    """An agent that can act as both requester and helper.

    Requester capabilities:
    - Decide whether to execute independently or request help
    - Select which helper to contact
    - Learn acceptance probabilities of other agents
    - Learn conditional execution reliability of other agents
    - Retain D-Greedy, D-UCB, and D-TS algorithms

    Helper capabilities:
    - Receive requests from other agents
    - Observe own workload and resources
    - Independently decide ACCEPT/REJECT using Q-learning (or baseline policies)
    - Learn helping policy from individual rewards
    """

    agent_id: int
    n_agents: int  # total number of agents in the system (including self)
    requester_policy_type: str = "D-TS"  # "D-Greedy", "D-UCB", "D-TS"
    requester_policy_params: dict = field(default_factory=dict)
    helper_policy_type: str = "q_learning"  # "q_learning", "always_accept", "always_reject", "informed_threshold"
    timing_policy_type: str = "learned"     # "learned", "immediate", "fixed_delay", "urgency", "never_ask"
    timing_policy_params: dict = field(default_factory=dict)

    # --- Requester-side beliefs ---
    acceptance: AcceptanceState = field(init=False)
    trust: TrustState = field(init=False)
    requester_policy: MultiAgentPolicy = field(init=False)
    timing_policy: BaseTimingPolicy = field(init=False)

    # --- Helper-side state ---
    helper_qlearning: HelperQLearning = field(init=False)
    workload: int = 0  # 0=Low, 1=Medium, 2=High
    resources: int = 2  # 0=Low, 1=Medium, 2=High

    # --- Individual rewards ---
    # Requester rewards
    R: float = 1.0                    # task success reward
    U_self: float = 0.5               # autonomous execution utility
    lambda_mult: float = 1.0          # communication cost multiplier
    comm_costs: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.1, 0.1]))

    # Helper rewards
    B_i: float = 0.5                  # cooperation reward
    resource_cost: float = 0.1        # cost of helping
    opportunity_cost: float = 0.05    # value of own task progress sacrificed

    # --- Execution reliability (ground truth, for environment) ---
    execution_reliability: float = 0.7  # true theta for this agent when helping
    _reliability_rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(), repr=False)

    # --- Random generators ---
    seed: int | None = field(default=None, repr=False)
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(), repr=False)

    def __post_init__(self):
        # Initialize RNG
        if self.seed is not None:
            self._rng = np.random.default_rng(self.seed)
            self._reliability_rng = np.random.default_rng(self.seed + 10000)
        else:
            self._rng = np.random.default_rng()
            self._reliability_rng = np.random.default_rng()

        # Communication costs for each potential helper (indexed by helper_idx 0..n_agents-2)
        if self.comm_costs.size != self.n_agents - 1:
            self.comm_costs = np.full(self.n_agents - 1, 0.1)

        # Create multi-agent requester policy
        self._init_requester_policy()

        # Helper-side: Q-learning for accept/reject
        self.helper_qlearning = HelperQLearning(
            alpha=0.1,
            gamma_q=0.9,
            epsilon=0.1,
            epsilon_min=0.02,
            epsilon_decay=0.9995,
            seed=self.seed + 20000 if self.seed is not None else None,
        )

        # Timing policy: WHEN TO ASK
        timing_seed = self.seed + 50000 if self.seed is not None else None
        self.timing_policy = make_timing_policy(
            name=self.timing_policy_type,
            seed=timing_seed,
            **self.timing_policy_params,
        )

    def _init_requester_policy(self) -> None:
        """Initialize or reinitialize the requester policy and link beliefs."""
        k_helpers = self.n_agents - 1
        policy_seed = self.seed + 30000 if self.seed is not None else None
        gamma = float(self.requester_policy_params.get("gamma", 1.0))

        self.requester_policy = make_multiagent_policy(
            name=self.requester_policy_type,
            K=k_helpers,
            R=self.R,
            lambda_mult=self.lambda_mult,
            c=self.comm_costs,
            gamma=gamma,
            seed=policy_seed,
            U_self=self.U_self,
            overrides=self.requester_policy_params,
        )
        self.acceptance = self.requester_policy.acceptance
        self.trust = self.requester_policy.trust

    # --- Properties required by Part 10 ---
    @property
    def helper_q_table(self) -> np.ndarray:
        return self.helper_qlearning.q_table

    @property
    def acceptance_beliefs(self) -> AcceptanceState:
        return self.acceptance

    @property
    def trust_beliefs(self) -> TrustState:
        return self.trust

    @property
    def timing_q_table(self) -> Optional[np.ndarray]:
        if hasattr(self.timing_policy, "q_table"):
            return self.timing_policy.q_table
        return None

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    # ========================================================================
    # REQUESTER-SIDE METHODS
    # ========================================================================

    def decide_timing(self, task: TemporalTask) -> int:
        """Decide WHEN to trigger help evaluation vs CONTINUE_SELF (Decision 3)."""
        return self.timing_policy.select_action(task)

    def update_timing_policy(self, state: int, action: int, reward: float,
                             next_state: Optional[int], done: bool) -> None:
        """Update Q-table of the learned timing policy."""
        self.timing_policy.update(state, action, reward, next_state, done)

    def _helper_idx_to_agent_id(self, helper_idx: int) -> int:
        """Map helper index (0..n_agents-2) to actual agent ID."""
        if helper_idx >= self.agent_id:
            return helper_idx + 1
        return helper_idx

    def _agent_id_to_helper_idx(self, agent_id: int) -> int:
        """Map actual agent ID to helper index (0..n_agents-2)."""
        if agent_id > self.agent_id:
            return agent_id - 1
        return agent_id

    def decide_request(self, U_self: Optional[float] = None) -> Decision:
        """Decide whether to ask for help and whom to ask.

        Uses the active requester policy (D-Greedy, D-UCB, or D-TS).
        """
        if U_self is not None:
            self.U_self = U_self
            self.requester_policy.U_self = U_self

        return self.requester_policy.decide(self.U_self)

    def decide_temporal_request(self, v_solo_current: float, v_solo_post_help: float) -> Decision:
        """Decide whether to ask for help and whom to ask under temporal task constraints.

        Compares max_j V_ask,j against v_solo_current, where:
            V_ask,j = -lambda * c_j + p * theta * R + (1 - p * theta) * v_solo_post_help
        """
        if hasattr(self.requester_policy, "decide_temporal"):
            return self.requester_policy.decide_temporal(v_solo_current, v_solo_post_help)
        return self.decide_request(v_solo_current)

    def compute_expected_utilities(self) -> np.ndarray:
        """Compute estimated expected utility for each potential helper.

        mu_hat_j = U_self + p_hat_j * (R * theta_hat_j - U_self) - lambda * c_j

        Returns:
            Array of length n_agents-1 with estimated utilities.
        """
        p_hat = self.acceptance.posterior_mean()
        theta_hat = self.trust.posterior_mean()
        costs = self.lambda_mult * self.comm_costs
        return self.U_self + p_hat * (self.R * theta_hat - self.U_self) - costs

    def select_helper(self, U_self: Optional[float] = None) -> tuple[int, float, np.ndarray]:
        """Select the best helper based on estimated expected utility.

        Returns:
            (helper_idx, best_utility, all_utilities)
        """
        if U_self is not None:
            self.U_self = U_self
            self.requester_policy.U_self = U_self

        utilities = self.compute_expected_utilities()
        best_idx = int(np.argmax(utilities))
        best_util = float(utilities[best_idx])
        return best_idx, best_util, utilities

    def should_ask(self, U_self: Optional[float] = None) -> tuple[bool, int, np.ndarray]:
        """Decide whether to ask and whom based on estimated utility.

        Returns:
            (should_ask, helper_idx, all_utilities)
        """
        if U_self is not None:
            self.U_self = U_self
            self.requester_policy.U_self = U_self

        utilities = self.compute_expected_utilities()
        best_idx = int(np.argmax(utilities))
        best_util = float(utilities[best_idx])
        should_ask = best_util > self.U_self
        return should_ask, best_idx if should_ask else -1, utilities

    def update_acceptance_belief(self, helper_agent_id: int, accepted: bool) -> None:
        """Update acceptance belief after a request.

        If rejected, also decay trust beliefs for elapsed round without adding
        success/failure counts.
        """
        helper_idx = self._agent_id_to_helper_idx(helper_agent_id)
        self.requester_policy.update_acceptance(helper_idx, accepted)
        if not accepted:
            self.requester_policy.trust.decay()

    def update_trust_belief(self, helper_agent_id: int, success: int) -> None:
        """Update trust (execution reliability) belief after an accepted request."""
        helper_idx = self._agent_id_to_helper_idx(helper_agent_id)
        self.requester_policy.update_trust(helper_idx, success)

    def record_acceptance_observation(self, helper_agent_id: int, accepted: bool) -> None:
        """Record acceptance observation without discounting (when discount was already applied globally)."""
        helper_idx = self._agent_id_to_helper_idx(helper_agent_id)
        if hasattr(self.requester_policy, "record_acceptance_observation"):
            self.requester_policy.record_acceptance_observation(helper_idx, accepted)
        else:
            self.requester_policy.acceptance.add_observation(helper_idx, accepted)

    def record_trust_observation(self, helper_agent_id: int, success: int) -> None:
        """Record trust observation without discounting (when discount was already applied globally)."""
        helper_idx = self._agent_id_to_helper_idx(helper_agent_id)
        if hasattr(self.requester_policy, "record_trust_observation"):
            self.requester_policy.record_trust_observation(helper_idx, success)
        else:
            self.requester_policy.trust.add_observation(helper_idx, success)

    def update_beliefs(self, helper_agent_id: int, accepted: bool, success: Optional[int] = None) -> None:
        """Unified belief update (Part 10 interface)."""
        self.update_acceptance_belief(helper_agent_id, accepted)
        if accepted and success is not None:
            self.update_trust_belief(helper_agent_id, success)

    def decay_beliefs(self) -> None:
        """Decay all beliefs (when acting alone)."""
        self.requester_policy.decay_beliefs()

    def get_acceptance_beliefs(self) -> np.ndarray:
        """Get acceptance probability estimates for all helpers."""
        return self.acceptance.posterior_mean()

    def get_trust_beliefs(self) -> np.ndarray:
        """Get execution reliability estimates for all helpers."""
        return self.trust.posterior_mean()

    @property
    def state_visits(self) -> np.ndarray:
        """Get state visitation counts for the helper policy."""
        return self.helper_qlearning.state_visits

    # ========================================================================
    # HELPER-SIDE METHODS
    # ========================================================================

    def decide_help(self, requester_id: int) -> int:
        """Decide whether to ACCEPT or REJECT a help request.

        Supports Q-learning policy as well as baseline policies (Always Accept, Always Reject).
        """
        if self.helper_policy_type == "always_accept":
            return ACCEPT
        elif self.helper_policy_type == "always_reject":
            return REJECT
        elif self.helper_policy_type == "informed_threshold":
            return ACCEPT if self.informed_accept() else REJECT
        return self.helper_qlearning.choose_action(self.workload, self.resources)

    def informed_accept(self) -> bool:
        """Reference (non-learning) helper rule: accept iff the immediate expected payoff
        B_i * theta_i - cost(w, rho) is positive.  It uses the helper's true execution
        reliability, so it serves as an informed benchmark for the learned policy."""
        return self.B_i * self.execution_reliability - self.helping_cost() > 0.0

    def get_hypothetical_p_accept(self, current_round: int) -> float:
        """Get hypothetical acceptance probability after flushing any pending transition.

        Used by the dynamic oracle to compute exact expected utility using the consistent
        post-flush policy state without mutating actual Q-tables.
        """
        if self.helper_policy_type == "always_accept":
            return 1.0
        elif self.helper_policy_type == "always_reject":
            return 0.0
        elif self.helper_policy_type == "informed_threshold":
            return 1.0 if self.informed_accept() else 0.0
        return self.helper_qlearning.hypothetical_p_accept(self.workload, self.resources, current_round)

    def execute_task(self) -> int:
        """Execute a task as a helper. Returns success (1) or failure (0)."""
        return int(self._reliability_rng.random() < self.execution_reliability)

    def helping_cost(self, workload: Optional[int] = None, resources: Optional[int] = None) -> float:
        """State-dependent cost of helping in state (w, rho).

        cost(w, rho) = c_res * (1 + (2 - rho)) + c_opp * (1 + w)

        Helping is cheapest when the helper is idle with full resources (w=0, rho=2:
        cost = c_res + c_opp) and becomes more expensive as its own workload grows
        (opportunity cost) and its resources deplete (resource cost).
        (w, rho) default to the agent's current state.
        """
        w = self.workload if workload is None else int(workload)
        rho = self.resources if resources is None else int(resources)
        return self.resource_cost * (1 + (2 - rho)) + self.opportunity_cost * (1 + w)

    def calculate_helper_reward(self, success: int, workload: Optional[int] = None,
                                resources: Optional[int] = None) -> float:
        """Individual reward for helping, evaluated in the state where the request was accepted.

        r_i = B_i * x - cost(w, rho),  cost(w, rho) = c_res * (3 - rho) + c_opp * (1 + w)
        """
        return self.B_i * success - self.helping_cost(workload, resources)

    def calculate_individual_reward(self, success: int, workload: Optional[int] = None,
                                    resources: Optional[int] = None) -> float:
        """Alias for calculate_helper_reward (Part 10 interface)."""
        return self.calculate_helper_reward(success, workload, resources)

    def flush_pending_helping_transition(self, current_round: int) -> bool:
        """Flush pending helping transition using the agent's current state as s'."""
        return self.helper_qlearning.flush_pending(self.workload, self.resources, current_round)

    def record_helping_decision(self, action: int, reward: float, round_t: int,
                                workload: Optional[int] = None,
                                resources: Optional[int] = None) -> None:
        """Record a helping decision transition pending until the next decision epoch."""
        w = self.workload if workload is None else workload
        r = self.resources if resources is None else resources
        self.helper_qlearning.record_decision(w, r, action, reward, round_t)

    def finalize_helper_episode(self, final_round: int) -> bool:
        """Flush remaining pending transition at episode termination."""
        return self.helper_qlearning.finalize_episode(self.workload, self.resources, final_round)

    def update_helper_policy(self, reward: float, next_workload: int, next_resources: int,
                             action: int, current_workload: Optional[int] = None,
                             current_resources: Optional[int] = None) -> None:
        """Update Q-table after helping decision.

        Args:
            reward: immediate reward (positive on net benefit, 0.0 on reject).
            next_workload: workload in state s'.
            next_resources: resources in state s'.
            action: ACCEPT (0) or REJECT (1).
            current_workload: workload in state s (defaults to self.workload).
            current_resources: resources in state s (defaults to self.resources).
        """
        cur_w = self.workload if current_workload is None else current_workload
        cur_r = self.resources if current_resources is None else current_resources
        self.helper_qlearning.update(
            cur_w, cur_r, action,
            reward, next_workload, next_resources
        )

    def update_helping_policy(self, reward: float, next_workload: int, next_resources: int,
                              action: int, current_workload: Optional[int] = None,
                              current_resources: Optional[int] = None) -> None:
        """Alias for update_helper_policy (Part 10 interface)."""
        self.update_helper_policy(
            reward, next_workload, next_resources, action,
            current_workload=current_workload, current_resources=current_resources
        )

    def execute_own_task(self) -> tuple[int, int]:
        """Agent executes its own task independently (or after a request rejection).

        Workload increases (capped at 2), resources decrease (floored at 0).

        Returns:
            (old_workload, old_resources) before update
        """
        old_workload, old_resources = self.workload, self.resources
        self.workload = min(2, self.workload + 1)
        self.resources = max(0, self.resources - 1)
        return old_workload, old_resources

    def help_another_agent(self) -> tuple[int, int]:
        """Agent accepts and executes another agent's task as a helper.

        Workload increases (capped at 2), resources decrease (floored at 0).

        Returns:
            (old_workload, old_resources) before update
        """
        old_workload, old_resources = self.workload, self.resources
        self.workload = min(2, self.workload + 1)
        self.resources = max(0, self.resources - 1)
        return old_workload, old_resources

    def rest(self) -> tuple[int, int]:
        """Agent is idle/resting (neither executing its own task nor helping another agent).

        Also applies to a requester whose task was successfully accepted and executed by a helper.
        Workload decreases (floored at 0), resources replenish (capped at 2).

        Returns:
            (old_workload, old_resources) before update
        """
        old_workload, old_resources = self.workload, self.resources
        self.workload = max(0, self.workload - 1)
        self.resources = min(2, self.resources + 1)
        return old_workload, old_resources

    def reject_request(self) -> tuple[int, int]:
        """Helper declines an incoming help request.

        Declining assistance avoids task execution: current workload relieves toward baseline,
        while resources are preserved without expending energy on task execution.

        Returns:
            (old_workload, old_resources) before update
        """
        old_workload, old_resources = self.workload, self.resources
        self.workload = max(0, self.workload - 1)
        # Resources preserved
        return old_workload, old_resources

    def wait(self) -> tuple[int, int]:
        """Requester waits for peer response during a communication decision round.

        Waiting incurs no active task execution: workload relieves toward baseline
        while energy resources are preserved without recovery.

        Returns:
            (old_workload, old_resources) before update
        """
        old_workload, old_resources = self.workload, self.resources
        self.workload = max(0, self.workload - 1)
        # Resources preserved
        return old_workload, old_resources

    def update_workload_resources(self, helped: bool, success: Optional[int] = None) -> tuple[int, int]:
        """Update workload and resources (backward-compatible wrapper).

        If helped: calls help_another_agent().
        If not helped: calls rest().

        Returns:
            (old_workload, old_resources) before update
        """
        if helped:
            return self.help_another_agent()
        return self.rest()

    # ========================================================================
    # RESET AND INSPECTION
    # ========================================================================

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset all beliefs, Q-table, and state."""
        if seed is not None:
            self.seed = seed
            self._rng = np.random.default_rng(seed)
            self._reliability_rng = np.random.default_rng(seed + 10000)

        self._init_requester_policy()
        self.helper_qlearning.reset(seed + 20000 if seed is not None else None)
        self.timing_policy.reset(seed + 50000 if seed is not None else None)

        self.workload = 0
        self.resources = 2

    def snapshot(self) -> dict:
        """Get complete state snapshot for logging/inspection."""
        return {
            "agent_id": self.agent_id,
            "requester": {
                "acceptance": self.acceptance.snapshot(),
                "trust": self.trust.snapshot(),
                "policy": self.requester_policy.name,
                "U_self": self.U_self,
            },
            "helper": {
                "policy_type": self.helper_policy_type,
                "qlearning": self.helper_qlearning.snapshot(),
                "workload": self.workload,
                "resources": self.resources,
                "execution_reliability": self.execution_reliability,
            },
            "rewards": {
                "R": self.R,
                "U_self": self.U_self,
                "lambda_mult": self.lambda_mult,
                "comm_costs": self.comm_costs.tolist(),
                "B_i": self.B_i,
                "resource_cost": self.resource_cost,
                "opportunity_cost": self.opportunity_cost,
            },
        }
