# backend/src/help_seeking/multiagent_simulation.py
"""Multi-agent simulation runner with logging, metrics, and convergence tracking."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .multiagent_environment import MultiAgentEnvironment, InteractionRecord
from .reliability import ReliabilityProfile


@dataclass
class MultiAgentRunRecord:
    """Complete record of a multi-agent simulation run."""
    T: int
    n_agents: int
    interactions: list[InteractionRecord] = field(default_factory=list)

    # Per-agent time series
    requester_rewards: np.ndarray = field(default_factory=lambda: np.array([]))  # shape (T, n_agents)
    helper_rewards: np.ndarray = field(default_factory=lambda: np.array([]))     # shape (T, n_agents)
    workloads: np.ndarray = field(default_factory=lambda: np.array([]))          # shape (T, n_agents)
    resources: np.ndarray = field(default_factory=lambda: np.array([]))          # shape (T, n_agents)
    execution_reliabilities: np.ndarray = field(default_factory=lambda: np.array([]))  # shape (T, n_agents)

    # System-wide metrics
    regrets: np.ndarray = field(default_factory=lambda: np.array([]))            # shape (T,)
    mu_stars: np.ndarray = field(default_factory=lambda: np.array([]))           # shape (T,)
    opt_actions: np.ndarray = field(default_factory=lambda: np.array([]))        # shape (T,)

    # Beliefs (if recorded)
    acceptance_beliefs: np.ndarray = field(default_factory=lambda: np.array([]))  # shape (T, n_agents, n_agents-1)
    trust_beliefs: np.ndarray = field(default_factory=lambda: np.array([]))       # shape (T, n_agents, n_agents-1)

    # Q-table snapshots recorded at regular checkpoints for policy stability tracking
    q_snapshots: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.requester_rewards.size == 0:
            self.requester_rewards = np.zeros((self.T, self.n_agents))
        if self.helper_rewards.size == 0:
            self.helper_rewards = np.zeros((self.T, self.n_agents))
        if self.workloads.size == 0:
            self.workloads = np.zeros((self.T, self.n_agents), dtype=int)
        if self.resources.size == 0:
            self.resources = np.zeros((self.T, self.n_agents), dtype=int)
        if self.execution_reliabilities.size == 0:
            self.execution_reliabilities = np.zeros((self.T, self.n_agents))
        if self.regrets.size == 0:
            self.regrets = np.zeros(self.T)
        if self.mu_stars.size == 0:
            self.mu_stars = np.zeros(self.T)
        if self.opt_actions.size == 0:
            self.opt_actions = np.zeros(self.T, dtype=bool)

    def add_interaction(self, t: int, record: InteractionRecord, agents: list):
        """Add interaction data for round t."""
        self.interactions.append(record)
        self.regrets[t] = record.regret
        self.mu_stars[t] = record.mu_star
        self.opt_actions[t] = record.opt_action

        # Record per-agent state
        for i, agent in enumerate(agents):
            self.workloads[t, i] = agent.workload
            self.resources[t, i] = agent.resources
            self.execution_reliabilities[t, i] = agent.execution_reliability

            if record.requester_id == i:
                self.requester_rewards[t, i] = record.requester_reward
            if record.helper_id == i:
                self.helper_rewards[t, i] = record.helper_reward

            if hasattr(agent, 'acceptance') and hasattr(agent, 'trust'):
                p_hat = agent.get_acceptance_beliefs()
                theta_hat = agent.get_trust_beliefs()
                if self.acceptance_beliefs.size > 0:
                    self.acceptance_beliefs[t, i] = p_hat
                    self.trust_beliefs[t, i] = theta_hat

    def history(self, start: int = 0, end: int | None = None) -> list[dict]:
        """Export serialized interaction history."""
        end = len(self.interactions) if end is None else min(end, len(self.interactions))
        out = []
        for i in range(start, end):
            r = self.interactions[i]
            out.append({
                "round": r.round_num,
                "requester_id": r.requester_id,
                "helper_id": r.helper_id,
                "asked": r.asked,
                "accepted": r.accepted,
                "success": r.success,
                "requester_reward": float(r.requester_reward),
                "helper_reward": float(r.helper_reward),
                "comm_cost": float(r.communication_cost),
                "regret": float(r.regret),
                "mu_star": float(r.mu_star),
                "optimal": bool(r.opt_action),
            })
        return out


class MultiAgentSimulationRun:
    """Runs multi-agent simulation with logging."""

    def __init__(self, environment: MultiAgentEnvironment, T: int,
                 record_beliefs: bool = True):
        self.env = environment
        self.T = int(T)
        self.record_beliefs = record_beliefs

        n_agents = environment.n_agents
        self.record = MultiAgentRunRecord(T=T, n_agents=n_agents)

        if record_beliefs:
            self.record.acceptance_beliefs = np.zeros((T, n_agents, n_agents - 1))
            self.record.trust_beliefs = np.zeros((T, n_agents, n_agents - 1))

        self.t = 0
        self.q_snapshots: list[dict] = []

    def reset(self, env_seed=None):
        self.env.reset(env_seed)
        n_agents = self.env.n_agents
        self.record = MultiAgentRunRecord(T=self.T, n_agents=n_agents)
        if self.record_beliefs:
            self.record.acceptance_beliefs = np.zeros((self.T, n_agents, n_agents - 1))
            self.record.trust_beliefs = np.zeros((self.T, n_agents, n_agents - 1))
        self.t = 0
        self.q_snapshots.clear()
        self.record.q_snapshots.clear()

    @property
    def done(self) -> bool:
        return self.t >= self.T

    def step(self) -> InteractionRecord:
        """Execute one round and record data."""
        if self.done:
            raise RuntimeError("Simulation horizon reached")

        record = self.env.step()
        agents = self.env.agents

        self.record.add_interaction(self.t, record, agents)
        self.t = self.env.round

        # Periodic Q-table snapshot for policy stability tracking
        if self.t % 50 == 0 or self.t == self.T:
            snap_q = {i: agent.helper_qlearning.q_table.copy() for i, agent in enumerate(agents)}
            snap_v = {i: agent.helper_qlearning.state_visits.copy() for i, agent in enumerate(agents)}
            self.q_snapshots.append({"t": self.t, "q": snap_q, "visits": snap_v})
            self.record.q_snapshots = list(self.q_snapshots)

        return record

    def run(self, T: int | None = None) -> MultiAgentRunRecord:
        if T is not None and T != self.T:
            self.T = int(T)
            n_agents = self.env.n_agents
            self.record = MultiAgentRunRecord(T=self.T, n_agents=n_agents)
            if self.record_beliefs:
                self.record.acceptance_beliefs = np.zeros((self.T, n_agents, n_agents - 1))
                self.record.trust_beliefs = np.zeros((self.T, n_agents, n_agents - 1))
            self.t = 0
            self.q_snapshots.clear()
            self.record.q_snapshots.clear()

        while not self.done:
            self.step()
        self.env.finalize()

        # Refresh the final snapshot with post-finalization Q-values and visit counts
        agents = self.env.agents
        final_snap = {
            "t": self.T,
            "q": {i: agent.helper_qlearning.q_table.copy() for i, agent in enumerate(agents)},
            "visits": {i: agent.helper_qlearning.state_visits.copy() for i, agent in enumerate(agents)},
        }
        if self.q_snapshots and self.q_snapshots[-1]["t"] == self.T:
            self.q_snapshots[-1] = final_snap
        else:
            self.q_snapshots.append(final_snap)

        self.record.q_snapshots = list(self.q_snapshots)
        return self.record


# ========================================================================
# METRICS COMPUTATION (PART 13)
# ========================================================================

def compute_multiagent_metrics(record: MultiAgentRunRecord, env: MultiAgentEnvironment) -> dict:
    """Compute aggregate metrics for a multi-agent run matching Part 13."""
    T = record.T
    n_agents = record.n_agents
    interactions = record.interactions

    # Global metrics
    total_interactions = len(interactions)
    asked_count = sum(1 for r in interactions if r.asked)
    accepted_count = sum(1 for r in interactions if r.accepted)
    rejected_count = sum(1 for r in interactions if r.asked and not r.accepted)
    success_count = sum(1 for r in interactions if r.success == 1)

    ask_rate = float(asked_count / T) if T > 0 else 0.0
    acceptance_rate = float(accepted_count / asked_count) if asked_count > 0 else None
    rejection_rate = float(rejected_count / asked_count) if asked_count > 0 else None
    success_rate = float(success_count / accepted_count) if accepted_count > 0 else None

    # Per-agent metrics
    agent_request_count = np.zeros(n_agents)         # Requests initiated by agent i as requester
    agent_requests_received = np.zeros(n_agents)     # Requests received by agent i as candidate helper
    agent_accept_count = np.zeros(n_agents)          # Requests accepted by agent i as helper (actual helping)
    agent_reject_count = np.zeros(n_agents)          # Requests rejected by agent i
    agent_requester_reward = np.zeros(n_agents)
    agent_helper_reward = np.zeros(n_agents)
    agent_comm_cost = np.zeros(n_agents)

    # State-conditioned helper decisions (E5 state breakdown):
    # - low_w_high_r: (w=0, r=2)
    # - med_w_med_r: (w=1, r=1)
    # - high_w_low_r: (w=2, r=0)
    state_cond = {
        "low_w_high_r": {"w": 0, "r": 2, "visits": 0, "accepts": 0, "rejects": 0},
        "med_w_med_r": {"w": 1, "r": 1, "visits": 0, "accepts": 0, "rejects": 0},
        "high_w_low_r": {"w": 2, "r": 0, "visits": 0, "accepts": 0, "rejects": 0},
    }

    for r in interactions:
        agent_request_count[r.requester_id] += 1
        agent_requester_reward[r.requester_id] += r.requester_reward
        agent_comm_cost[r.requester_id] += r.communication_cost

        if r.asked and r.helper_id is not None:
            agent_requests_received[r.helper_id] += 1
            agent_helper_reward[r.helper_id] += r.helper_reward
            if r.accepted:
                agent_accept_count[r.helper_id] += 1
            else:
                agent_reject_count[r.helper_id] += 1

            hw = r.helper_workload_before
            hr = r.helper_resources_before
            for sc in state_cond.values():
                if hw == sc["w"] and hr == sc["r"]:
                    sc["visits"] += 1
                    if r.accepted:
                        sc["accepts"] += 1
                    else:
                        sc["rejects"] += 1

    per_agent_acc_rate: list[float | None] = []
    for i in range(n_agents):
        if agent_requests_received[i] > 0:
            per_agent_acc_rate.append(float(agent_accept_count[i] / agent_requests_received[i]))
        else:
            per_agent_acc_rate.append(None)

    valid_acc_rates = [r for r in per_agent_acc_rate if r is not None]
    avg_acc_rate_per_agent = float(np.mean(valid_acc_rates)) if valid_acc_rates else None

    # State-conditioned breakdown metrics
    state_metrics = {}
    for key, sc in state_cond.items():
        v = sc["visits"]
        acc_rate = float(sc["accepts"] / v) if v > 0 else None
        rej_rate = float(sc["rejects"] / v) if v > 0 else None
        state_metrics[f"{key}_visits"] = v
        state_metrics[f"{key}_accept_rate"] = acc_rate
        state_metrics[f"{key}_reject_rate"] = rej_rate

    # Task completion proxy under non-temporal model assumptions:
    # Under the non-temporal model (E5-E7), self-execution deterministically yields baseline utility U_self
    # (guaranteed self-completion), while helper execution succeeds probabilistically with rate theta_j.
    completed = 0
    for r in interactions:
        if not r.asked:
            completed += 1  # acted alone
        elif r.accepted and r.success == 1:
            completed += 1  # helper succeeded
        elif not r.accepted:
            completed += 1  # acted alone after rejection
    task_completion_rate = completed / T if T > 0 else 0.0

    # Average workload and resources
    avg_workload = float(record.workloads.mean()) if T > 0 else 0.0
    avg_resources = float(record.resources.mean()) if T > 0 else 0.0

    # Regret & Oracle
    cum_regret = float(record.regrets.sum())
    avg_regret = float(record.regrets.mean()) if T > 0 else 0.0
    opt_action_rate = float(record.opt_actions.mean()) if T > 0 else 0.0

    # Helper policy stability (empirical greedy-policy consistency metric):
    # Evaluated strictly over non-tied states with sufficient cumulative observations at both checkpoints (>= 5):
    MIN_VISITS = 5
    snapshots = record.q_snapshots
    stabilities = []
    eval_counts = []

    if len(snapshots) >= 2:
        # Evaluate over the last up to 4 consecutive checkpoint intervals
        window_snaps = snapshots[-5:] if len(snapshots) >= 5 else snapshots

        for snap_idx in range(len(window_snaps) - 1):
            snap_prev = window_snaps[snap_idx]
            snap_curr = window_snaps[snap_idx + 1]

            q_prev = snap_prev["q"]
            q_curr = snap_curr["q"]
            v_prev = snap_prev.get("visits", {})
            v_curr = snap_curr.get("visits", {})

            for i in range(n_agents):
                if i not in q_prev or i not in q_curr:
                    continue
                qp = q_prev[i]
                qc = q_curr[i]
                vp = np.asarray(v_prev[i]) if (isinstance(v_prev, dict) and i in v_prev) else None
                vc = np.asarray(v_curr[i]) if (isinstance(v_curr, dict) and i in v_curr) else None

                if vp is not None and vc is not None:
                    sufficient_mask = (vp >= MIN_VISITS) & (vc >= MIN_VISITS)
                else:
                    sufficient_mask = (np.abs(qp).sum(axis=1) > 1e-6) & (np.abs(qc).sum(axis=1) > 1e-6)

                # Exclude unresolved ties (|Q(s, ACCEPT) - Q(s, REJECT)| <= 1e-12)
                diff_p = qp[:, 0] - qp[:, 1]
                diff_c = qc[:, 0] - qc[:, 1]
                untied_mask = (np.abs(diff_p) > 1e-12) & (np.abs(diff_c) > 1e-12)

                eval_mask = sufficient_mask & untied_mask
                if np.any(eval_mask):
                    act_p = np.where(diff_p > 0, 0, 1)
                    act_c = np.where(diff_c > 0, 0, 1)
                    unchanged = int((act_p[eval_mask] == act_c[eval_mask]).sum())
                    n_eval = int(eval_mask.sum())
                    stabilities.append(unchanged / n_eval)
                    eval_counts.append(n_eval)

    q_stability = float(np.mean(stabilities)) if stabilities else None
    helper_states_evaluated = float(np.mean(eval_counts)) if eval_counts else None

    # Requester and helper utilities & social welfare
    total_requester_reward = float(agent_requester_reward.sum())
    total_helper_reward = float(agent_helper_reward.sum())
    requester_utility = total_requester_reward / T if T > 0 else 0.0
    helper_utility = total_helper_reward / T if T > 0 else 0.0
    social_welfare = requester_utility + helper_utility

    # Individual agent scalar metrics for per-agent analysis
    agent_scalars = {}
    for i in range(n_agents):
        agent_scalars[f"agent_{i}_acceptance_rate"] = per_agent_acc_rate[i]
        agent_scalars[f"agent_{i}_incoming_request_frequency"] = float(agent_requests_received[i] / T) if T > 0 else 0.0
        agent_scalars[f"agent_{i}_help_frequency"] = float(agent_accept_count[i] / T) if T > 0 else 0.0
        agent_scalars[f"agent_{i}_request_frequency"] = float(agent_request_count[i] / T) if T > 0 else 0.0
        agent_scalars[f"agent_{i}_helper_reward"] = float(agent_helper_reward[i])

    return {
        # Global
        "total_rounds": T,
        "ask_rate": ask_rate,
        "acceptance_rate": acceptance_rate,
        "rejection_rate": rejection_rate,
        "success_rate": success_rate,
        "task_completion_rate": task_completion_rate,
        "avg_workload": avg_workload,
        "avg_resources": avg_resources,
        "total_communication_cost": float(agent_comm_cost.sum()),
        "cumulative_regret": cum_regret,
        "average_regret": avg_regret,
        "optimal_action_rate": opt_action_rate,
        "requester_utility": requester_utility,
        "helper_utility": helper_utility,
        "social_welfare": social_welfare,
        "requester_net_utility": requester_utility,  # backward compatibility alias
        "helper_policy_stability": q_stability,
        "helper_policy_convergence": q_stability,
        "helper_states_evaluated": helper_states_evaluated,

        # State-conditioned helper metrics (E5)
        **state_metrics,

        # Per-agent (averaged)
        "avg_requester_reward": float(agent_requester_reward.mean()),
        "avg_helper_reward": float(agent_helper_reward.mean()),
        "avg_request_frequency": float(agent_request_count.mean() / T) if T > 0 else 0.0,
        "avg_incoming_request_frequency": float(agent_requests_received.mean() / T) if T > 0 else 0.0,
        "avg_cooperation_frequency": float(agent_accept_count.mean() / T) if T > 0 else 0.0,
        "avg_acceptance_rate_per_agent": avg_acc_rate_per_agent,

        # Per-agent scalar breakdown
        **agent_scalars,

        # Per-agent arrays
        "per_agent_acceptance_rate": per_agent_acc_rate,
        "per_agent_request_count": agent_request_count.tolist(),
        "per_agent_requests_received": agent_requests_received.tolist(),
        "per_agent_help_count": agent_accept_count.tolist(),  # actual helping count
        "per_agent_accept_count": agent_accept_count.tolist(),
        "per_agent_reject_count": agent_reject_count.tolist(),
        "per_agent_requester_reward": agent_requester_reward.tolist(),
        "per_agent_helper_reward": agent_helper_reward.tolist(),
        "per_agent_comm_cost": agent_comm_cost.tolist(),
    }


def compute_requester_metrics(record: MultiAgentRunRecord, env: MultiAgentEnvironment,
                              agent_id: int) -> dict:
    """Compute requester-side metrics for a specific agent."""
    interactions = [r for r in record.interactions if r.requester_id == agent_id]
    T = len(interactions)

    if T == 0:
        return {}

    asked = [r for r in interactions if r.asked]
    asked_count = len(asked)
    accepted = [r for r in asked if r.accepted]
    accepted_count = len(accepted)

    ask_rate = asked_count / T
    acceptance_rate = accepted_count / asked_count if asked_count > 0 else 0.0

    requester_rewards = [r.requester_reward for r in interactions]
    avg_net_utility = float(np.mean(requester_rewards))
    total_comm_cost = float(sum(r.communication_cost for r in interactions))

    return {
        "agent_id": agent_id,
        "ask_rate": ask_rate,
        "acceptance_rate": acceptance_rate,
        "avg_net_utility": avg_net_utility,
        "total_communication_cost": total_comm_cost,
        "n_interactions": T,
    }
