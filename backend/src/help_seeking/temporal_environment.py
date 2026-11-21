# backend/src/help_seeking/temporal_environment.py
"""Temporal multi-agent environment integrating WHEN, WHETHER, and WHOM to ask.

Models explicit task lifecycles with deadlines, solo execution attempts,
learned timing decisions (RequesterTimingPolicy), whether/whom decisions (MA-D-TS, MA-D-UCB, MA-Greedy),
and independent helper Q-learning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Any
import numpy as np

from .autonomous_agent import AutonomousAgent, ACCEPT, REJECT
from .task_lifecycle import TemporalTask
from .requester_timing import CONTINUE_SELF, EVALUATE_HELP, encode_timing_state
from .reliability import ReliabilityProfile


def irwin_hall_cdf(x: float, n: int) -> float:
    """Cumulative distribution function of the sum of n i.i.d. Uniform(0, 1) variables."""
    if x <= 0.0:
        return 0.0
    if x >= n:
        return 1.0
    s = 0.0
    for k in range(int(math.floor(x)) + 1):
        term = ((-1) ** k) * math.comb(n, k) * ((x - k) ** n)
        s += term
    return s / math.factorial(n)


def solo_completion_probability(rem_work: float, rem_time: int, a: float, b: float) -> float:
    """Exact probability that the sum of rem_time Uniform(a, b) variables >= rem_work."""
    if rem_work <= 0.0:
        return 1.0
    if rem_time <= 0:
        return 0.0
    if rem_time * b < rem_work:
        return 0.0
    if rem_time * a >= rem_work:
        return 1.0
    x = (rem_work - rem_time * a) / (b - a)
    return 1.0 - irwin_hall_cdf(x, rem_time)


@dataclass
class TemporalTaskRecord:
    """Record of a complete task lifecycle under temporal help-seeking."""
    task_id: int
    round_started: int
    round_finished: int
    requester_id: int
    helper_id: Optional[int]
    deadline: int
    steps_taken: int
    solo_attempts: int
    help_requested: bool
    help_accepted: Optional[bool]
    helper_success: Optional[int]
    completed: bool
    deadline_exceeded: bool
    requester_reward: float
    helper_reward: float
    communication_cost: float
    final_progress: float
    timing_action_history: list[int] = field(default_factory=list)
    actual_request_step: Optional[int] = None


class TemporalMultiAgentEnvironment:
    """Environment coordinating temporal help-seeking across autonomous agents.

    Decision flow per task:
        1. Task arrives with deadline D and required progress 1.0.
        2. At each task decision epoch:
            a. Requester evaluates WHEN to ask (RequesterTimingPolicy).
            b. If CONTINUE_SELF:
               Requester attempts solo execution. Progress updates.
            c. If EVALUATE_HELP (and help not previously requested):
               Requester evaluates WHETHER and WHOM to ask using state-dependent continuation utility.
               If ask: candidate helper evaluates WHETHER TO ACCEPT (HelperQLearning).
               Outcomes resolve and beliefs update.
            d. If task completes or deadline expires, task terminates.
    """

    def __init__(
        self,
        n_agents: int = 4,
        reliability_profiles: Optional[list[ReliabilityProfile]] = None,
        R: float = 1.0,
        U_self: float = 0.5,
        lambda_mult: float = 1.0,
        comm_costs: Optional[np.ndarray] = None,
        helper_rewards: Optional[dict | list[dict]] = None,
        requester_policy_type: str = "MultiAgent-D-TS",
        requester_policy_params: Optional[dict] = None,
        helper_policy_type: str = "q_learning",
        timing_policy_type: str = "learned",
        timing_policy_params: Optional[dict] = None,
        default_deadline: int = 5,
        step_cost: float = 0.02,
        c_fail: float = 0.5,
        solo_capability: float = 0.60,
        kappa_solo: Optional[float] = None,
        seed: Optional[int] = None,
        requester_selection: str = "round_robin",
    ):
        self.n_agents = int(n_agents)
        if self.n_agents < 2:
            raise ValueError("n_agents must be >= 2")

        self.R = float(R)
        self.U_self = float(U_self)
        self.lambda_mult = float(lambda_mult)
        self.requester_selection = requester_selection
        self.default_deadline = int(default_deadline)
        self.step_cost = float(step_cost)
        self.c_fail = float(c_fail)
        self.solo_capability = float(kappa_solo) if kappa_solo is not None else float(solo_capability)
        self.gamma_timing = float(timing_policy_params.get("gamma", 0.95) if timing_policy_params else 0.95)

        if reliability_profiles is None:
            from .reliability import StationaryProfile
            self.reliability_profiles = [StationaryProfile([0.7]) for _ in range(self.n_agents)]
        else:
            if len(reliability_profiles) != self.n_agents:
                raise ValueError(f"reliability_profiles length ({len(reliability_profiles)}) != n_agents ({self.n_agents})")
            self.reliability_profiles = reliability_profiles

        if comm_costs is None:
            comm_costs = np.full((self.n_agents, self.n_agents), 0.1)
            np.fill_diagonal(comm_costs, 0.0)
        self.comm_costs = np.asarray(comm_costs, dtype=float)

        # Per-agent helper reward configs
        default_hr = {"B_i": 0.5, "resource_cost": 0.1, "opportunity_cost": 0.05}
        self.per_agent_helper_rewards: list[dict] = []
        for i in range(self.n_agents):
            hr = dict(default_hr)
            if helper_rewards is not None:
                if isinstance(helper_rewards, list) and i < len(helper_rewards):
                    hr.update(helper_rewards[i])
                elif isinstance(helper_rewards, dict):
                    if i in helper_rewards:
                        hr.update(helper_rewards[i])
                    elif str(i) in helper_rewards:
                        hr.update(helper_rewards[str(i)])
                    elif any(k in helper_rewards for k in ["B_i", "resource_cost", "opportunity_cost"]):
                        hr.update(helper_rewards)
            self.per_agent_helper_rewards.append(hr)

        # Create autonomous agents
        self.agents: list[AutonomousAgent] = []
        for i in range(self.n_agents):
            agent_comm_costs = np.delete(self.comm_costs[i], i)
            hr = self.per_agent_helper_rewards[i]
            agent = AutonomousAgent(
                agent_id=i,
                n_agents=self.n_agents,
                requester_policy_type=requester_policy_type,
                requester_policy_params=requester_policy_params or {},
                helper_policy_type=helper_policy_type,
                timing_policy_type=timing_policy_type,
                timing_policy_params=timing_policy_params or {},
                R=self.R,
                U_self=self.U_self,
                lambda_mult=self.lambda_mult,
                comm_costs=agent_comm_costs,
                B_i=hr["B_i"],
                resource_cost=hr["resource_cost"],
                opportunity_cost=hr["opportunity_cost"],
                execution_reliability=self.reliability_profiles[i].get(1)[0],
                seed=seed + i * 1000 if seed is not None else None,
            )
            self.agents.append(agent)

        # Model-based timing policies plan with their agent's beliefs and this environment's task model
        for agent in self.agents:
            if hasattr(agent.timing_policy, "bind"):
                agent.timing_policy.bind(agent, self)

        self.global_round = 0
        self.task_counter = 0
        self._rng = np.random.default_rng(seed)
        self.history: list[TemporalTaskRecord] = []
        self.timing_snapshots: list[dict] = []

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset environment and all agents."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        for i, agent in enumerate(self.agents):
            agent.reset(seed + i * 1000 if seed is not None else None)
            agent.execution_reliability = self.reliability_profiles[i].get(1)[0]
        self.global_round = 0
        self.task_counter = 0
        self.history.clear()
        self.timing_snapshots.clear()

    def _select_requester(self) -> int:
        if self.requester_selection == "round_robin":
            return self.task_counter % self.n_agents
        return int(self._rng.integers(self.n_agents))

    def _get_helper_ids(self, requester_id: int) -> list[int]:
        return [j for j in range(self.n_agents) if j != requester_id]

    def _helper_idx_to_agent_id(self, requester_id: int, helper_idx: int) -> int:
        helpers = self._get_helper_ids(requester_id)
        return helpers[helper_idx]

    def compute_v_solo(self, progress: float, remaining_time: int, solo_capability: float) -> float:
        """Compute exact expected continuation utility of solo execution.

        Calculates exact completion probability distribution across remaining steps using
        the Irwin-Hall distribution for the sum of uniform progress increments:
            Delta p ~ Uniform(0.20 * kappa_solo, 0.45 * kappa_solo)

        Consistent with run_task(), completion on attempt k incurs (k - 1) step costs
        and earns terminal reward R, discounted by gamma_timing^(k - 1). Failure after
        remaining_time attempts incurs (remaining_time - 1) step costs and failure penalty
        -c_fail, discounted by gamma_timing^(remaining_time - 1).

        Args:
            progress: Current cumulative task progress in [0.0, 1.0].
            remaining_time: Integer slack steps remaining before deadline expiry.
            solo_capability: Physical scaling factor kappa_solo for progress increment.

        Returns:
            Exact expected continuation utility value.
        """
        remaining_work = max(0.0, 1.0 - progress)
        if remaining_work <= 0.0:
            return self.R
        if remaining_time <= 0:
            return -self.c_fail

        a = 0.20 * solo_capability
        b = 0.45 * solo_capability
        gamma = self.gamma_timing

        def geom_sum(g: float, m: int) -> float:
            if m <= 0:
                return 0.0
            if abs(g - 1.0) < 1e-9:
                return float(m)
            return (1.0 - (g ** m)) / (1.0 - g)

        expected_val = 0.0
        prev_p = 0.0
        for k in range(1, remaining_time + 1):
            pk = solo_completion_probability(remaining_work, k, a, b)
            p_step_k = max(0.0, pk - prev_p)
            r_k = (gamma ** (k - 1)) * self.R - self.step_cost * geom_sum(gamma, k - 1)
            expected_val += p_step_k * r_k
            prev_p = pk

        p_fail = max(0.0, 1.0 - prev_p)
        r_fail = -(gamma ** (remaining_time - 1)) * self.c_fail - self.step_cost * geom_sum(gamma, remaining_time - 1)
        expected_val += p_fail * r_fail
        return expected_val

    def compute_continuation_utility(self, task: TemporalTask, solo_capability: float) -> float:
        """Backwards-compatible wrapper delegating to compute_v_solo."""
        return self.compute_v_solo(task.progress, task.remaining_time, solo_capability)

    def run_task(self, deadline: Optional[int] = None) -> TemporalTaskRecord:
        """Execute one complete task lifecycle under the temporal interaction protocol."""
        self.task_counter += 1
        d_limit = self.default_deadline if deadline is None else int(deadline)

        # 1. Select task owner
        requester_id = self._select_requester()
        requester = self.agents[requester_id]

        task = TemporalTask(
            task_id=self.task_counter,
            requester_id=requester_id,
            progress=0.0,
            required_progress=1.0,
            deadline=d_limit,
            remaining_time=d_limit,
        )

        task_start_round = self.global_round + 1
        step_count = 0
        action_history = []
        total_requester_reward = 0.0
        total_helper_reward = 0.0
        total_comm_cost = 0.0
        helper_agent_id: Optional[int] = None
        helper_accepted: Optional[bool] = None
        helper_success: Optional[int] = None
        actual_request_step: Optional[int] = None

        # Solo execution capability (kappa_solo decoupled from U_self/R)
        solo_capability = self.solo_capability

        # Main timing decision loop: runs while task is active and help not yet requested
        while not (task.completed or task.failed or task.help_requested):
            self.global_round += 1
            step_count += 1

            # Update reliabilities aligned with task arrival index (Problem 7)
            for i, agent in enumerate(self.agents):
                agent.execution_reliability = self.reliability_profiles[i].get(self.task_counter)[0]

            # Global belief decay across all agents for elapsed decision round
            for agent in self.agents:
                agent.decay_beliefs()

            s_time = encode_timing_state(task)
            timing_action = requester.decide_timing(task)
            action_history.append(timing_action)

            # ---------------------------------------------------------------
            # CASE A: CONTINUE_SELF
            # ---------------------------------------------------------------
            if timing_action == CONTINUE_SELF:
                requester.execute_own_task()
                for j, agent in enumerate(self.agents):
                    if j != requester_id:
                        agent.rest()

                inc = task.attempt_solo(solo_capability, requester.rng)

                if task.completed:
                    r_step = self.R
                elif task.failed:
                    r_step = -self.c_fail
                else:
                    r_step = -self.step_cost

                total_requester_reward += r_step
                s_next = None if (task.completed or task.failed) else encode_timing_state(task)
                is_done = task.completed or task.failed
                requester.update_timing_policy(s_time, CONTINUE_SELF, r_step, s_next, is_done)
                continue

            # ---------------------------------------------------------------
            # CASE B: EVALUATE_HELP
            # ---------------------------------------------------------------
            # Temporal requester utility accounting for sequential task lifecycle (Problem 3)
            v_solo_current = self.compute_v_solo(task.progress, task.remaining_time, solo_capability)
            if task.remaining_time <= 1:
                v_fallback = -self.c_fail
            else:
                v_solo_next = self.compute_v_solo(task.progress, task.remaining_time - 1, solo_capability)
                v_fallback = -self.step_cost + self.gamma_timing * v_solo_next
            dec = requester.decide_temporal_request(v_solo_current, v_fallback)

            if dec.d == 0:
                # Requester evaluated help but chose autonomous execution (beliefs already decayed globally)
                requester.execute_own_task()
                for j, agent in enumerate(self.agents):
                    if j != requester_id:
                        agent.rest()

                inc = task.attempt_solo(solo_capability, requester.rng)
                if task.completed:
                    r_step = self.R
                elif task.failed:
                    r_step = -self.c_fail
                else:
                    r_step = -self.step_cost

                total_requester_reward += r_step
                s_next = None if (task.completed or task.failed) else encode_timing_state(task)
                is_done = task.completed or task.failed
                requester.update_timing_policy(s_time, EVALUATE_HELP, r_step, s_next, is_done)
                continue

            # Help request is transmitted to candidate helper j* (Problem 6)
            actual_request_step = step_count
            task.help_requested = True
            helper_idx = dec.a
            helper_agent_id = self._helper_idx_to_agent_id(requester_id, helper_idx)
            helper = self.agents[helper_agent_id]

            c_comm = self.lambda_mult * self.comm_costs[requester_id, helper_agent_id]
            total_comm_cost += c_comm

            # Helper decides whether to accept
            helper.flush_pending_helping_transition(current_round=self.global_round)
            old_hw, old_hr = helper.workload, helper.resources
            h_action = helper.decide_help(requester_id)
            helper_accepted = (h_action == ACCEPT)

            requester.record_acceptance_observation(helper_agent_id, helper_accepted)

            if not helper_accepted:
                # REJECTED (Problem 1):
                # Helper declines assistance (workload relieves, resources preserved)
                helper.reject_request()
                # Requester waited for peer response during communication round
                requester.wait()
                # Inactive peer agents rest and recover
                for j, agent in enumerate(self.agents):
                    if j != requester_id and j != helper_agent_id:
                        agent.rest()

                task.mark_help_rejected()  # Consumes exactly 1 deadline unit

                helper.record_helping_decision(h_action, 0.0, self.global_round, old_hw, old_hr)

                # Communication step cost
                if task.failed:
                    r_help = -self.c_fail - c_comm
                else:
                    r_help = -self.step_cost - c_comm

                total_requester_reward += r_help
                help_decision_reward = r_help
            else:
                # ACCEPTED:
                # Requester rests; helper executes task
                requester.rest()
                helper.help_another_agent()
                for j, agent in enumerate(self.agents):
                    if j != requester_id and j != helper_agent_id:
                        agent.rest()

                h_success = helper.execute_task()
                helper_success = h_success
                requester.record_trust_observation(helper_agent_id, h_success)

                task.apply_helper_execution(h_success)  # Consumes exactly 1 deadline unit

                h_reward = helper.calculate_helper_reward(h_success, old_hw, old_hr)
                total_helper_reward += h_reward

                helper.record_helping_decision(h_action, h_reward, self.global_round, old_hw, old_hr)

                if task.completed:
                    r_help = self.R - c_comm
                elif task.failed:
                    r_help = -self.c_fail - c_comm
                else:
                    r_help = -self.step_cost - c_comm

                total_requester_reward += r_help
                help_decision_reward = r_help

            # Help request has concluded; break from timing loop
            break

        # -------------------------------------------------------------------
        # POST-REQUEST SOLO PHASE (Problem 3)
        # If the task is not yet finished, execute remaining solo steps deterministically.
        # DO NOT train the timing policy on these forced post-request actions!
        # -------------------------------------------------------------------
        post_request_discounted_return = 0.0
        post_step = 0
        gamma_timing = getattr(requester.timing_policy, "gamma", 0.95) if hasattr(requester, "timing_policy") else 0.95
        while not (task.completed or task.failed):
            self.global_round += 1
            step_count += 1
            post_step += 1

            for i, agent in enumerate(self.agents):
                agent.execution_reliability = self.reliability_profiles[i].get(self.task_counter)[0]

            # Global belief decay across all agents for elapsed decision round
            for agent in self.agents:
                agent.decay_beliefs()

            requester.execute_own_task()
            for j, agent in enumerate(self.agents):
                if j != requester_id:
                    agent.rest()

            inc = task.attempt_solo(solo_capability, requester.rng)
            if task.completed:
                r_post = self.R
            elif task.failed:
                r_post = -self.c_fail
            else:
                r_post = -self.step_cost

            total_requester_reward += r_post
            post_request_discounted_return += (gamma_timing ** post_step) * r_post

        # If a help request was made, update timing policy for EVALUATE_HELP with the discounted outcome (Problem 5)
        # G = r_0 + gamma * r_1 + gamma^2 * r_2 + ...
        if task.help_requested:
            timing_return = help_decision_reward + post_request_discounted_return
            requester.update_timing_policy(s_time, EVALUATE_HELP, timing_return, None, True)

        rec = TemporalTaskRecord(
            task_id=task.task_id,
            round_started=task_start_round,
            round_finished=self.global_round,
            requester_id=requester_id,
            helper_id=helper_agent_id,
            deadline=d_limit,
            steps_taken=step_count,
            solo_attempts=task.attempts,
            help_requested=task.help_requested,
            help_accepted=helper_accepted,
            helper_success=helper_success,
            completed=task.completed,
            deadline_exceeded=task.failed,
            requester_reward=total_requester_reward,
            helper_reward=total_helper_reward,
            communication_cost=total_comm_cost,
            final_progress=task.progress,
            timing_action_history=action_history,
            actual_request_step=actual_request_step,
        )
        self.history.append(rec)
        return rec

    def run(self, n_tasks: int, deadline: Optional[int] = None) -> list[TemporalTaskRecord]:
        """Run n_tasks through the temporal interaction protocol."""
        for t_idx in range(n_tasks):
            self.run_task(deadline=deadline)
            # Periodic timing policy snapshot for policy stability tracking (Problem 5)
            if (t_idx + 1) % 50 == 0 or (t_idx + 1) == n_tasks:
                snap_q = {i: agent.timing_policy.q_table.copy() for i, agent in enumerate(self.agents) if hasattr(agent.timing_policy, 'q_table')}
                snap_v = {i: agent.timing_policy.state_visits.copy() for i, agent in enumerate(self.agents) if hasattr(agent.timing_policy, 'state_visits')}
                self.timing_snapshots.append({"task": self.task_counter, "q": snap_q, "visits": snap_v})

        # Finalize pending epoch transitions across helpers
        for agent in self.agents:
            agent.finalize_helper_episode(self.global_round)
        return self.history
