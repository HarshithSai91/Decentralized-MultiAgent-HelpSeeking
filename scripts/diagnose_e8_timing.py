#!/usr/bin/env python3
"""Diagnostic for E8: is the learned (Q-learning) timing policy behind the heuristics only
because of exploration noise or too little training?

Each policy is run for a training phase of T_train tasks and then measured on the next T_eval
tasks (30 seeds, same environment as E8).  For the learned policy we also freeze exploration
(epsilon = 0) for the evaluation phase.  Writes results/processed/e8_timing_diagnostics.txt.

    python scripts/diagnose_e8_timing.py            # T_train = 400 and 2000, T_eval = 400
"""
import json
import os

import numpy as np

import _paths  # noqa: F401
from help_seeking.reliability import make_profile
from help_seeking.temporal_environment import TemporalMultiAgentEnvironment
from help_seeking.temporal_simulation import compute_temporal_metrics

CFG = json.load(open("configs/e8_temporal_help_seeking.json"))
P = CFG["base"]


def run(timing, params, seed, t_train, t_eval, greedy_eval):
    _, _, prof_ss = np.random.SeedSequence(seed).spawn(3)
    ps = prof_ss.spawn(4)
    spec = dict(CFG["reliability"])
    profiles = [make_profile({**spec, "theta_after": [spec["theta_after"][i]]}, [P["theta"][i]],
                             t_train + t_eval, seed=ps[i]) for i in range(4)]
    env = TemporalMultiAgentEnvironment(
        n_agents=4, reliability_profiles=profiles, R=CFG["R"], U_self=P["U_self"], lambda_mult=P["lambda"],
        comm_costs=np.array(P["c"]), helper_rewards=P["helper_rewards"], requester_policy_type=P["requester_policy"],
        requester_policy_params={"gamma": P["gamma"]}, timing_policy_type=timing, timing_policy_params=params,
        default_deadline=CFG["default_deadline"], step_cost=CFG["step_cost"], c_fail=CFG["c_fail"],
        solo_capability=CFG["solo_capability"], seed=seed)
    env.run(n_tasks=t_train)
    if greedy_eval:
        for agent in env.agents:
            if hasattr(agent.timing_policy, "epsilon"):
                agent.timing_policy.epsilon = 0.0
                agent.timing_policy.epsilon_min = 0.0
    m = compute_temporal_metrics(env.run(n_tasks=t_eval), env)
    return m["task_completion_rate"], m["avg_requester_reward"], m["ask_rate"]


def main():
    policies = [("Learned (online)", "learned", {}, False), ("Learned (greedy evaluation)", "learned", {}, True),
                ("Planned", "planned", {}, False), ("Fixed delay (k=2)", "fixed_delay", {"delay": 2}, False),
                ("Urgency", "urgency", {"urgency_threshold": 2}, False)]
    lines = ["E8 timing diagnostics: mean over 30 seeds of the evaluation phase (next 400 tasks)", ""]
    for t_train in (400, 2000):
        lines.append(f"Training phase: {t_train} tasks")
        for name, timing, params, greedy in policies:
            r = np.array([run(timing, params, s, t_train, 400, greedy) for s in range(1, 31)])
            lines.append(f"  {name:28s} completion {r[:, 0].mean():.3f}  requester reward {r[:, 1].mean():.3f}  "
                         f"ask rate {r[:, 2].mean():.3f}")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join("results", "processed", "e8_timing_diagnostics.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
