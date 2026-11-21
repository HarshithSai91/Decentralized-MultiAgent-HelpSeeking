# backend/src/help_seeking/temporal_simulation.py
"""Runner and metrics computation for temporal multi-agent simulation."""
from __future__ import annotations

from typing import Optional
import numpy as np

from .temporal_environment import TemporalMultiAgentEnvironment, TemporalTaskRecord


def compute_temporal_metrics(records: list[TemporalTaskRecord], env: TemporalMultiAgentEnvironment) -> dict:
    """Compute aggregate metrics across temporal task lifecycles."""
    N = len(records)
    if N == 0:
        return {}

    completed_count = sum(1 for r in records if r.completed)
    failed_count = sum(1 for r in records if r.deadline_exceeded)
    asked_count = sum(1 for r in records if r.help_requested)
    accepted_count = sum(1 for r in records if r.help_accepted is True)
    success_count = sum(1 for r in records if r.helper_success == 1)

    completion_rate = float(completed_count / N)
    deadline_failure_rate = float(failed_count / N)
    ask_rate = float(asked_count / N)
    acceptance_rate = float(accepted_count / asked_count) if asked_count > 0 else None
    assistance_success_rate = float(success_count / accepted_count) if accepted_count > 0 else None

    avg_steps = float(np.mean([r.steps_taken for r in records]))
    avg_solo_attempts = float(np.mean([r.solo_attempts for r in records]))
    avg_requester_reward = float(np.mean([r.requester_reward for r in records]))
    avg_helper_reward = float(np.mean([r.helper_reward for r in records]))
    avg_comm_cost = float(np.mean([r.communication_cost for r in records]))
    social_welfare = avg_requester_reward + avg_helper_reward

    # Timing metrics: when was help requested?
    # Use actual_request_step recorded when a request was transmitted (Problem 6)
    request_steps = [
        r.actual_request_step
        for r in records
        if r.actual_request_step is not None
    ]
    avg_time_to_request = float(np.mean(request_steps)) if request_steps else None

    # Per-agent metrics
    agent_tasks = np.zeros(env.n_agents)
    agent_completions = np.zeros(env.n_agents)
    agent_rewards = np.zeros(env.n_agents)
    agent_helps_given = np.zeros(env.n_agents)
    agent_helper_rewards = np.zeros(env.n_agents)

    for r in records:
        agent_tasks[r.requester_id] += 1
        if r.completed:
            agent_completions[r.requester_id] += 1
        agent_rewards[r.requester_id] += r.requester_reward
        if r.helper_id is not None:
            agent_helps_given[r.helper_id] += 1
            agent_helper_rewards[r.helper_id] += r.helper_reward

    # Timing policy stability: measured by comparing greedy actions between consecutive snapshots (Problem 5)
    timing_stability = None
    timing_states_evaluated = None
    if hasattr(env, "timing_snapshots") and len(env.timing_snapshots) >= 2:
        snaps = env.timing_snapshots[-5:] if len(env.timing_snapshots) >= 5 else env.timing_snapshots
        stabs = []
        eval_counts = []
        for s_idx in range(len(snaps) - 1):
            s_prev = snaps[s_idx]
            s_curr = snaps[s_idx + 1]
            qp_dict = s_prev["q"]
            qc_dict = s_curr["q"]
            vp_dict = s_prev.get("visits", {})
            vc_dict = s_curr.get("visits", {})
            for i in range(env.n_agents):
                if i in qp_dict and i in qc_dict:
                    qp = qp_dict[i]
                    qc = qc_dict[i]
                    vp = vp_dict.get(i)
                    vc = vc_dict.get(i)
                    if vp is not None and vc is not None:
                        # Require sufficient cumulative visits at both checkpoints (at least 5)
                        sufficient_mask = (vp >= 5) & (vc >= 5)
                    else:
                        sufficient_mask = (qp.sum(axis=1) != 0) & (qc.sum(axis=1) != 0)

                    # Exclude unresolved ties (|Q(s, CONTINUE) - Q(s, EVALUATE)| <= 1e-12)
                    diff_p = qp[:, 0] - qp[:, 1]
                    diff_c = qc[:, 0] - qc[:, 1]
                    untied_mask = (np.abs(diff_p) > 1e-12) & (np.abs(diff_c) > 1e-12)

                    eval_mask = sufficient_mask & untied_mask
                    if np.any(eval_mask):
                        act_p = np.where(diff_p > 0, 0, 1)
                        act_c = np.where(diff_c > 0, 0, 1)
                        same = int((act_p[eval_mask] == act_c[eval_mask]).sum())
                        n_eval = int(eval_mask.sum())
                        stabs.append(same / n_eval)
                        eval_counts.append(n_eval)
        if stabs:
            timing_stability = float(np.mean(stabs))
            timing_states_evaluated = float(np.mean(eval_counts))

    return {
        "total_tasks": N,
        "task_completion_rate": completion_rate,
        "deadline_failure_rate": deadline_failure_rate,
        "ask_rate": ask_rate,
        "acceptance_rate": acceptance_rate,
        "assistance_success_rate": assistance_success_rate,
        "avg_steps_per_task": avg_steps,
        "avg_solo_attempts": avg_solo_attempts,
        "avg_time_to_request": avg_time_to_request,
        "avg_requester_reward": avg_requester_reward,
        "avg_helper_reward": avg_helper_reward,
        "social_welfare": social_welfare,
        "avg_communication_cost": avg_comm_cost,
        "timing_policy_stability": timing_stability,
        "timing_states_evaluated": timing_states_evaluated,
    }
