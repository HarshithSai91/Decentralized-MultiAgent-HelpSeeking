"""Shared helpers for the analysis and figure scripts."""
import json
import os

import numpy as np
import pandas as pd

import _paths  # noqa: F401  (sets cwd to project root, adds backend/src)
from help_seeking.stats import mean_ci, paired_difference  # noqa: F401

RESULTS = "results"


def manifest(name):
    with open(os.path.join(RESULTS, "manifests", f"{name}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def runs(name) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS, "raw", name, "runs.csv"))


def summary(name) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS, "processed", f"{name}_summary.csv"))


def curves(name):
    return np.load(os.path.join(RESULTS, "raw", name, "curves.npz"))


def slug(label: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in label).strip("_")


def curve(cv, point: int, algo: str, field: str):
    return cv[f"p{point}__{slug(algo)}__{field}"]


def available(name) -> bool:
    return os.path.exists(os.path.join(RESULTS, "manifests", f"{name}.json"))


def cell(df: pd.DataFrame, algo: str, point: int, metric: str) -> np.ndarray:
    """Per-seed values of a metric for one algorithm at one sweep point (sorted by seed)."""
    sub = df[(df.algorithm == algo) & (df.point == point)].sort_values("seed")
    return sub[metric].to_numpy(dtype=float)


def stat(df, algo, point, metric) -> dict:
    return mean_ci(cell(df, algo, point, metric))


def paired(df, algo_a, algo_b, point, metric, point_b=None) -> dict:
    """mean(a - b) over seeds with a 95% CI; pairs by seed."""
    a = df[(df.algorithm == algo_a) & (df.point == point)].set_index("seed")[metric]
    b = df[(df.algorithm == algo_b) & (df.point == (point if point_b is None else point_b))].set_index("seed")[metric]
    common = a.index.intersection(b.index)
    return paired_difference(a.loc[common].to_numpy(float), b.loc[common].to_numpy(float))


def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def fmt_ci(s: dict, nd=1, pct=False):
    """'mean ± half-width' string from a mean_ci dict."""
    if s["n"] == 0 or np.isnan(s["mean"]):
        return "--"
    k = 100.0 if pct else 1.0
    half = (s["ci_high"] - s["mean"]) * k if not np.isnan(s["ci_high"]) else 0.0
    return f"{s['mean'] * k:.{nd}f} ± {half:.{nd}f}"
