#!/usr/bin/env python3
"""Run one or more experiment configs.

    python scripts/run_experiment.py configs/e3_abrupt_change.json
    python scripts/run_experiment.py configs/e1_stationary.json --seeds 5 --T 500   # quick check
"""
import argparse
import os

import _paths  # noqa: F401
from help_seeking.experiments import expand_scenarios, run_experiment


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+", help="config JSON file(s)")
    ap.add_argument("--seeds", type=int, default=None, help="override number of seeds")
    ap.add_argument("--T", type=int, default=None, help="override horizon")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="results", help="output root (default: results/)")
    args = ap.parse_args()
    for path in args.configs:
        for cfg in expand_scenarios(path):
            run_experiment(cfg, out_dir=args.out, workers=args.workers, n_seeds=args.seeds, T=args.T)


if __name__ == "__main__":
    main()
