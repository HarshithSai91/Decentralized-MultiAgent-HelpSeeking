# backend/src/help_seeking/multiagent_environment.py
"""Multi-agent environment managing N autonomous agents interacting.

Each round:
1. An agent receives a task (selected round-robin or randomly).
2. The agent evaluates autonomous execution utility and potential helpers.
3. Requester decides whether to ask for help and whom (via its requester policy).
4. If asking: request sent to selected helper.
5. Helper observes own workload and resources and independently decides ACCEPT/REJECT.
6. If REJECTED:
   - Requester performs autonomous execution.
   - Communication cost incurred.
   - Acceptance belief updated.
   - Execution reliability is NOT updated.
   - Helper receives zero reward.
7. If ACCEPTED:
   - Helper executes task.
   - Success/failure generated from helper's current execution reliability.
   - Requester receives R * x - lambda * c.
   - Helper receives individual reward B_i * x - cost(w, rho), where the helping cost
     grows with its workload w and resource depletion (2 - rho) at the time it accepted.
   - Helper updates its Q-table.
   - Requester updates acceptance and execution reliability beliefs.
8. Workloads and resources updated for all agents.
9. Dynamic oracle and pseudo-regret recorded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .autonomous_agent import AutonomousAgent, ACCEPT, REJECT
from .helper_qlearning import encode_state
from .reliability import ReliabilityProfile, make_profile


@dataclass
class InteractionRecord:
    """Record of a single interaction."""
    round_num: int
    requester_id: int
    helper_id: int | None
    asked: bool
    accepted: bool | None
    success: int | None
    requester_reward: float
    helper_reward: float
    communication_cost: float
    requester_workload_before: int
    requester_resources_before: int
    helper_workload_before: int
    helper_resources_before: int
    mu_star: float = 0.0
    d_star: int = 0
    regret: float = 0.0
    opt_action: bool = False


class MultiAgentEnvironment:
    """Environment managing N autonomous agents."""

    def __init__(self, n_agents: int, reliability_profiles: list[ReliabilityProfile],
                 R: float = 1.0, U_self: float = 0.5, lambda_mult: float = 1.0,
                 comm_costs: Optional[np.ndarray] = None,  # shape (n_agents, n_agents)
                 helper_rewards: Optional[dict] = None,  # per-agent B_i, resource_cost, opportunity_cost
                 requester_policy_type: str = "D-TS",
                 requester_policy_params: Optional[dict] = None,
                 helper_policy_type: str = "q_learning",
                 seed: Optional[int] = None,
                 requester_selection: str = "round_robin"):  # "round_robin" or "random"
        self.n_agents = n_agents
        self.R = float(R)
        self.U_self = float(U_self)
        self.lambda_mult = float(lambda_mult)
        self.requester_selection = requester_selection
        self.helper_policy_type = helper_policy_type

        # Validate inputs
        if n_agents < 2:
            raise ValueError(f"n_agents must be >= 2, got {n_agents}")
        if len(reliability_profiles) != n_agents:
            raise ValueError(f"reliability_profiles length ({len(reliability_profiles)}) must equal n_agents ({n_agents})")

        # Communication cost matrix: comm_costs[i, j] = cost for agent i to ask agent j
        if comm_costs is None:
            comm_costs = np.full((n_agents, n_agents), 0.1)
            np.fill_diagonal(comm_costs, 0.0)
        self.comm_costs = np.asarray(comm_costs, dtype=float)
        if self.comm_costs.shape != (n_agents, n_agents):
            raise ValueError(f"comm_costs shape ({self.comm_costs.shape}) must be ({n_agents}, {n_agents})")

        # Resolve per-agent helper reward parameters
        default_hr = {
            "B_i": 0.5,
            "resource_cost": 0.1,
            "opportunity_cost": 0.05,
        }
        self.per_agent_helper_rewards: list[dict] = []
        for i in range(n_agents):
            agent_hr = dict(default_hr)
            if helper_rewards is not None:
                if isinstance(helper_rewards, list):
                    if i < len(helper_rewards) and isinstance(helper_rewards[i], dict):
                        agent_hr.update(helper_rewards[i])
                elif isinstance(helper_rewards, dict):
                    # Check if keyed by agent ID (e.g. {0: {...}, 1: {...}})
                    if i in helper_rewards and isinstance(helper_rewards[i], dict):
                        agent_hr.update(helper_rewards[i])
                    elif str(i) in helper_rewards and isinstance(helper_rewards[str(i)], dict):
                        agent_hr.update(helper_rewards[str(i)])
                    elif any(k in helper_rewards for k in ["B_i", "resource_cost", "opportunity_cost"]):
                        # Uniform reward settings across all agents
                        agent_hr.update(helper_rewards)
            self.per_agent_helper_rewards.append(agent_hr)

        # Retain backward-compatible reference
        self.helper_rewards = self.per_agent_helper_rewards[0] if self.per_agent_helper_rewards else default_hr

        # Create agents
        self.agents: list[AutonomousAgent] = []
        for i in range(n_agents):
            agent_comm_costs = np.delete(self.comm_costs[i], i)
            hr = self.per_agent_helper_rewards[i]

            agent = AutonomousAgent(
                agent_id=i,
                n_agents=n_agents,
                requester_policy_type=requester_policy_type,
                requester_policy_params=requester_policy_params or {},
                helper_policy_type=helper_policy_type,
                R=R,
                U_self=U_self,
                lambda_mult=lambda_mult,
                comm_costs=agent_comm_costs,
                B_i=hr["B_i"],
                resource_cost=hr["resource_cost"],
                opportunity_cost=hr["opportunity_cost"],
                execution_reliability=reliability_profiles[i].get(1)[0] if hasattr(reliability_profiles[i], 'get') else 0.7,
                seed=seed + i * 1000 if seed is not None else None,
            )
            self.agents.append(agent)

        self.reliability_profiles = reliability_profiles
        self.round = 0
        self._rng = np.random.default_rng(seed)
        self.history: list[InteractionRecord] = []

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset all agents and environment state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        for i, agent in enumerate(self.agents):
            agent.reset(seed + i * 1000 if seed is not None else None)
            agent.execution_reliability = self.reliability_profiles[i].get(1)[0]
        self.round = 0
        self.history.clear()

    def _select_requester(self) -> int:
        """Select which agent acts as requester this round."""
        if self.requester_selection == "round_robin":
            return self.round % self.n_agents
        elif self.requester_selection == "random":
            return int(self._rng.integers(self.n_agents))
        else:
            raise ValueError(f"Unknown requester selection: {self.requester_selection}")

    def _get_helper_ids(self, requester_id: int) -> list[int]:
        """Get list of potential helper agent IDs for a requester."""
        return [j for j in range(self.n_agents) if j != requester_id]

    def _helper_idx_to_agent_id(self, requester_id: int, helper_idx: int) -> int:
        """Map helper index (0..n_agents-2) to actual agent ID."""
        helpers = self._get_helper_ids(requester_id)
        return helpers[helper_idx]

    def _agent_id_to_helper_idx(self, requester_id: int, helper_agent_id: int) -> int:
        """Map actual helper agent ID to helper index."""
        helpers = self._get_helper_ids(requester_id)
        return helpers.index(helper_agent_id)

    def compute_oracle(self, requester_id: int) -> dict:
        """Compute requester-level dynamic oracle for requester_id (Part 11).

        Uses true conditional execution reliability theta_j(t) and true
        acceptance probability p_j(t) under helper j's current state and policy.

        mu_ij* = U_self + p_ij * (R * theta_ij - U_self) - lambda * c_ij
        """
        helper_ids = self._get_helper_ids(requester_id)
        K = len(helper_ids)
        if K == 0:
            return {
                "d_star": 0,
                "a_star": -1,
                "best_helper_agent_id": None,
                "mu_star": self.U_self,
                "p_true": np.array([]),
                "theta_true": np.array([]),
                "mu_star_helpers": np.array([]),
            }

        p_true = np.zeros(K)
        theta_true = np.zeros(K)
        mu_star_helpers = np.zeros(K)

        for idx, h_id in enumerate(helper_ids):
            helper = self.agents[h_id]
            theta_true[idx] = helper.execution_reliability

            if helper.helper_policy_type == "always_accept":
                p_true[idx] = 1.0
            elif helper.helper_policy_type == "always_reject":
                p_true[idx] = 0.0
            elif helper.helper_policy_type == "informed_threshold":
                p_true[idx] = 1.0 if helper.informed_accept() else 0.0
            else:
                # Exact probability under actual post-flush epsilon-greedy policy with random tie-breaking:
                # Evaluates hypothetical Q-table and epsilon without mutating state
                p_true[idx] = helper.get_hypothetical_p_accept(self.round)

            comm_cost = self.lambda_mult * self.comm_costs[requester_id, h_id]
            mu_star_helpers[idx] = self.U_self + p_true[idx] * (self.R * theta_true[idx] - self.U_self) - comm_cost

        best_helper_idx = int(np.argmax(mu_star_helpers))
        max_helper_util = float(mu_star_helpers[best_helper_idx])
        mu_star = max(self.U_self, max_helper_util)

        # Build full set of optimal actions within numerical tolerance (1e-9)
        # Any action with expected utility within 1e-9 of mu_star is strictly optimal
        optimal_actions = []
        if abs(self.U_self - mu_star) < 1e-9:
            optimal_actions.append({"d": 0, "a": -1, "helper_agent_id": None})
        for idx in range(K):
            if abs(mu_star_helpers[idx] - mu_star) < 1e-9:
                optimal_actions.append({"d": 1, "a": idx, "helper_agent_id": helper_ids[idx]})

        if max_helper_util > self.U_self + 1e-9:
            d_star = 1
            a_star = best_helper_idx
        elif self.U_self > max_helper_util + 1e-9:
            d_star = 0
            a_star = -1
        else:
            # Tied between acting alone and asking
            d_star = 0
            a_star = -1

        return {
            "d_star": d_star,
            "a_star": a_star,
            "best_helper_agent_id": helper_ids[best_helper_idx] if d_star == 1 else None,
            "mu_star": mu_star,
            "optimal_actions": optimal_actions,
            "p_true": p_true,
            "theta_true": theta_true,
            "mu_star_helpers": mu_star_helpers,
        }

    def step(self) -> InteractionRecord:
        """Execute one round of interaction following Part 9 protocol."""
        self.round += 1

        # 1. Update execution reliabilities from profiles
        for i, agent in enumerate(self.agents):
            agent.execution_reliability = self.reliability_profiles[i].get(self.round)[0]

        # Global belief decay across all agents for the elapsed round
        for agent in self.agents:
            agent.decay_beliefs()

        # 2. Select requester
        requester_id = self._select_requester()
        requester = self.agents[requester_id]

        # Compute requester-level dynamic oracle
        oracle = self.compute_oracle(requester_id)

        # 3. Requester autonomously decides whether to ask and whom
        dec = requester.decide_request(self.U_self)
        should_ask = (dec.d == 1)
        helper_idx = dec.a if should_ask else -1

        # Calculate ex-ante expected utility of the chosen action and pseudo-regret.
        # An action is optimal if its expected utility is within epsilon of mu_star,
        # which correctly accounts for ties among helpers or between asking and acting alone.
        if not should_ask:
            mu_chosen = self.U_self
            opt_action = any(opt["d"] == 0 for opt in oracle["optimal_actions"])
        else:
            mu_chosen = float(oracle["mu_star_helpers"][helper_idx])
            opt_action = any(opt["d"] == 1 and opt["a"] == helper_idx for opt in oracle["optimal_actions"])

        regret = max(0.0, oracle["mu_star"] - mu_chosen)

        record = InteractionRecord(
            round_num=self.round,
            requester_id=requester_id,
            helper_id=None,
            asked=should_ask,
            accepted=None,
            success=None,
            requester_reward=0.0,
            helper_reward=0.0,
            communication_cost=0.0,
            requester_workload_before=requester.workload,
            requester_resources_before=requester.resources,
            helper_workload_before=0,
            helper_resources_before=0,
            mu_star=oracle["mu_star"],
            d_star=oracle["d_star"],
            regret=regret,
            opt_action=opt_action,
        )

        if not should_ask:
            # Requester acts alone autonomously: executes its own task (beliefs already decayed globally)
            requester.execute_own_task()
            for j, agent in enumerate(self.agents):
                if j != requester_id:
                    agent.rest()

            requester_reward = requester.U_self
            record.requester_reward = requester_reward
            record.helper_id = None
            self.history.append(record)
            return record

        # Requester asks a helper
        helper_agent_id = self._helper_idx_to_agent_id(requester_id, helper_idx)
        helper = self.agents[helper_agent_id]

        record.helper_id = helper_agent_id
        record.asked = True
        record.helper_workload_before = helper.workload
        record.helper_resources_before = helper.resources

        # Communication cost
        comm_cost = self.lambda_mult * self.comm_costs[requester_id, helper_agent_id]
        record.communication_cost = comm_cost

        # Decision-epoch RL: flush helper's pending transition using its current state at this decision epoch
        helper.flush_pending_helping_transition(current_round=self.round)

        # Helper decides ACCEPT/REJECT autonomously using its private Q-learning policy
        old_helper_w = helper.workload
        old_helper_r = helper.resources
        helper_action = helper.decide_help(requester_id)
        accepted = (helper_action == ACCEPT)

        # Update requester's acceptance belief (only acceptance belief updated, no double discount)
        requester.record_acceptance_observation(helper_agent_id, accepted)

        if not accepted:
            # REJECTED: requester performs autonomous execution fallback
            # Helper rejected: avoids task execution
            # All other agents rest
            requester.execute_own_task()
            helper.reject_request()
            for j, agent in enumerate(self.agents):
                if j != requester_id and j != helper_agent_id:
                    agent.rest()

            requester_reward = requester.U_self - comm_cost
            helper_reward = 0.0

            # Store pending decision-epoch transition for helper (pure semi-Markov Q-learning)
            helper.record_helping_decision(
                helper_action, helper_reward, round_t=self.round,
                workload=old_helper_w, resources=old_helper_r
            )

            record.accepted = False
            record.success = None
            record.requester_reward = requester_reward
            record.helper_reward = helper_reward
            self.history.append(record)
            return record

        # ACCEPTED: helper executes task, requester offloaded its task
        requester.rest()
        helper.help_another_agent()
        for j, agent in enumerate(self.agents):
            if j != requester_id and j != helper_agent_id:
                agent.rest()

        # Helper executes the task
        success = helper.execute_task()

        # Update requester's trust belief (execution reliability, no double discount)
        requester.record_trust_observation(helper_agent_id, success)

        # Calculate realized rewards
        requester_reward = requester.R * success - comm_cost
        helper_reward = helper.calculate_helper_reward(success, old_helper_w, old_helper_r)

        # Store pending decision-epoch transition for helper (pure semi-Markov Q-learning)
        helper.record_helping_decision(
            helper_action, helper_reward, round_t=self.round,
            workload=old_helper_w, resources=old_helper_r
        )

        record.accepted = True
        record.success = success
        record.requester_reward = requester_reward
        record.helper_reward = helper_reward
        self.history.append(record)
        return record

    def finalize(self) -> None:
        """Finalize any pending decision-epoch transitions across all agents at episode end."""
        for agent in self.agents:
            agent.finalize_helper_episode(self.round)

    def run(self, T: int) -> list[InteractionRecord]:
        """Run T rounds and finalize pending transitions."""
        for _ in range(T):
            self.step()
        self.finalize()
        return self.history

    def get_agent_snapshots(self) -> list[dict]:
        """Get snapshots of all agents."""
        return [agent.snapshot() for agent in self.agents]
