# backend/src/help_seeking/stats.py
"""Small statistics helpers (95% confidence intervals across independent seeds)."""
from __future__ import annotations

import math

import numpy as np

# Two-sided 95% Student-t critical values, used when SciPy is not installed.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
         16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
         23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
         30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t_critical(df: int, conf: float = 0.95) -> float:
    if df < 1:
        return float("nan")
    try:
        from scipy.stats import t as student_t
        return float(student_t.ppf(0.5 + conf / 2.0, df))
    except Exception:  # pragma: no cover - fallback path
        if conf != 0.95:
            raise
        keys = sorted(_T975)
        for k in keys:
            if df <= k:
                return _T975[k]
        return 1.96


def mean_ci(values, conf: float = 0.95) -> dict:
    """Mean, std, standard error and t-based CI of independent replications."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = int(x.size)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "se": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan")}
    mean = float(x.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "se": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan")}
    std = float(x.std(ddof=1))
    se = std / math.sqrt(n)
    h = t_critical(n - 1, conf) * se
    return {"n": n, "mean": mean, "std": std, "se": se, "ci_low": mean - h, "ci_high": mean + h}


def paired_difference(a, b, conf: float = 0.95) -> dict:
    """CI of mean(a - b) for seed-paired samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    res = mean_ci(diff, conf)
    res["frac_a_better_lower"] = float(np.mean(diff < 0)) if diff.size else float("nan")
    return res
