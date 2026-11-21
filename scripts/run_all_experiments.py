#!/usr/bin/env python3
"""Run every experiment of the project with the full protocol (30 seeds, T = 2000).

Core experiments (proposal Table 2): E1-E4.
Supplementary studies: S1 (gamma in a stationary world, H2), S2 (alpha_ucb
sensitivity), S3 (optional continuous-drift extension), S4 (ask accuracy vs gamma),
S5 (generalised multi-shift setting).
"""
import argparse
import os
import time

import _paths  # noqa: F401
from help_seeking.experiments import expand_scenarios, run_experiment

CONFIGS = [
    "configs/e1_stationary.json",
    "configs/e2_when_to_ask.json",
    "configs/e3_abrupt_change.json",
    "configs/e4_communication_cost.json",
    "configs/s1_gamma_stationary.json",
    "configs/s2_alpha_ucb.json",
    "configs/s3_drift.json",
    "configs/s4_when_to_ask_gamma.json",
    "configs/s5_multishift.json",
    # Multi-agent experiments (E5 / E6 / E7 / E8)
    "configs/e5_helper_learning.json",
    "configs/e6_multiagent_cooperation.json",
    "configs/e7_changing_reliability.json",
    "configs/e8_temporal_help_seeking.json",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seeds", type=int, default=None, help="override (e.g. 3 for a smoke test)")
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None, help="substring filter on config names")
    ap.add_argument("--out", default="results",
                    help="output root (default: results/); use another folder for smoke tests")
    args = ap.parse_args()
    t0 = time.time()
    for path in CONFIGS:
        if args.only and not any(s in path for s in args.only):
            continue
        for cfg in expand_scenarios(path):
            run_experiment(cfg, out_dir=args.out, workers=args.workers, n_seeds=args.seeds, T=args.T)
    print(f"All experiments finished in {time.time() - t0:.0f}s. Next: python scripts/analyze_results.py")


if __name__ == "__main__":
    main()
