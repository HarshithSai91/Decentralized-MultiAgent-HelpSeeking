#!/usr/bin/env python3
"""Diagnostic for E5: does the gap between Q-learning helpers and the informed threshold close
with a decreasing learning rate or with longer training?

Variants (MA-D-TS requesters, E5 setting, 30 seeds):
  * Q-learning with the default constant learning rate alpha = 0.1,
  * Q-learning with a count-based rate alpha = 1 / n(s, a) (sample average),
  * the informed threshold reference,
each for T = 2000 and T = 8000 rounds.  Writes results/processed/e5_helper_diagnostics.txt.
The count-based variant is applied here only; the experiments use the default learner.
"""
import json
import os

import numpy as np

import _paths  # noqa: F401
from help_seeking import helper_qlearning as HQ
from help_seeking.multiagent_environment import MultiAgentEnvironment
from help_seeking.multiagent_simulation import MultiAgentSimulationRun, compute_multiagent_metrics
from help_seeking.reliability import make_profile

CFG = json.load(open("configs/e5_helper_learning.json"))
P = CFG["base"]
DEFAULT_FLUSH = HQ.HelperQLearning.flush_pending


def count_based_flush(self, next_workload, next_resources, current_round):
    if self.pending_transition is None:
        return False
    pt = self.pending_transition
    if not hasattr(self, "_n_sa"):
        self._n_sa = np.zeros((9, 2))
    self._n_sa[pt.state, pt.action] += 1
    alpha = 1.0 / self._n_sa[pt.state, pt.action]
    dt = max(1, int(current_round) - pt.decision_round)
    target = pt.reward + self.gamma_q ** dt * self.q_table[HQ.encode_state(next_workload, next_resources)].max()
    self.q_table[pt.state, pt.action] += alpha * (target - self.q_table[pt.state, pt.action])
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    self.pending_transition = None
    return True


def run(seed, T, helper_policy, count_based):
    HQ.HelperQLearning.flush_pending = count_based_flush if count_based else DEFAULT_FLUSH
    _, _, prof_ss = np.random.SeedSequence(seed).spawn(3)
    ps = prof_ss.spawn(4)
    profiles = [make_profile({"type": "stationary"}, [P["theta"][i]], T, seed=ps[i]) for i in range(4)]
    env = MultiAgentEnvironment(4, profiles, R=CFG["R"], U_self=P["U_self"], lambda_mult=P["lambda"],
                                comm_costs=np.array(P["c"]), helper_rewards=P["helper_rewards"],
                                requester_policy_type="MultiAgent-D-TS", requester_policy_params={"gamma": P["gamma"]},
                                helper_policy_type=helper_policy, seed=seed)
    m = compute_multiagent_metrics(MultiAgentSimulationRun(env, T=T).run(), env)
    return m["acceptance_rate"], m["helper_utility"], m["requester_utility"], m["social_welfare"]


def main():
    lines = ["E5 helper diagnostics (MA-D-TS requesters, 30 seeds): utilities per round", ""]
    for T in (2000, 8000):
        lines.append(f"T = {T} rounds")
        for name, pol, cb in [("Q-learning, alpha = 0.1 (default)", "q_learning", False),
                              ("Q-learning, alpha = 1/n(s,a)", "q_learning", True),
                              ("Informed threshold", "informed_threshold", False)]:
            r = np.array([run(s, T, pol, cb) for s in range(1, 31)])
            lines.append(f"  {name:34s} accept {r[:, 0].mean():.3f}  helper util {r[:, 1].mean():.3f}  "
                         f"requester util {r[:, 2].mean():.3f}  welfare {r[:, 3].mean():.3f}")
        lines.append("")
    HQ.HelperQLearning.flush_pending = DEFAULT_FLUSH
    text = "\n".join(lines)
    print(text)
    with open(os.path.join("results", "processed", "e5_helper_diagnostics.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
