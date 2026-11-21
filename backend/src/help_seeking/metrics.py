# backend/src/help_seeking/metrics.py
"""Evaluation metrics (proposal Sec. 4.3) computed from a RunRecord."""
from __future__ import annotations

import numpy as np

from .simulation import RunRecord
from .utilities import OracleDecision


def cumulative_regret(rec: RunRecord, start: int = 0, end: int | None = None) -> float:
    """Reg(T) = sum_t (mu*(t) - mu_{chosen}(t))  (Eq. 15, pseudo-regret: no Bernoulli noise)."""
    return float(rec.regret[start:end].sum())


def regret_curve(rec: RunRecord) -> np.ndarray:
    return np.cumsum(rec.regret)


def average_net_utility(rec: RunRecord, start: int = 0, end: int | None = None) -> float:
    """(1/T) sum_t r_t (realised utility, includes outcome noise)."""
    return float(rec.r[start:end].mean())


def average_expected_utility(rec: RunRecord, start: int = 0, end: int | None = None) -> float:
    """(1/T) sum_t mu_{chosen}(t) (noise-free version of the same quantity)."""
    return float(rec.mu_chosen[start:end].mean())


def ask_rate(rec: RunRecord, start: int = 0, end: int | None = None) -> float:
    """(1/T) sum_t d_t."""
    return float(rec.d[start:end].mean())


def ask_decision_accuracy(rec: RunRecord, start: int = 0, end: int | None = None) -> float:
    """(1/T) sum_t 1[d_t = d*_t].  When asking and acting alone are exactly
    tied in expectation both decisions count as correct."""
    return float(rec.ask_correct[start:end].mean())


def optimal_teammate_selection_rate(rec: RunRecord, start: int = 0, end: int | None = None) -> float:
    """sum_t 1[d_t = 1 and a_t = a*_t] / sum_t d_t, where a*_t = argmax_k mu_k(t).
    NaN if the agent never asked in the window."""
    d = rec.d[start:end] == 1
    if not d.any():
        return float("nan")
    return float(rec.best_helper_chosen[start:end][d].mean())


def optimal_action_rate(rec: RunRecord, start: int = 0, end: int | None = None) -> float:
    """Fraction of rounds in which the joint action (d_t, a_t) was optimal."""
    return float(rec.opt_action[start:end].mean())


def total_communication_cost(rec: RunRecord, start: int = 0, end: int | None = None) -> float:
    """sum_t d_t * lambda * c_{a_t}."""
    return float(rec.cost[start:end].sum())


def mae_curve(rec: RunRecord) -> np.ndarray | None:
    """MAE_theta(t) = (1/K) sum_k |theta_hat_k(t) - theta_k(t)| for every round."""
    if rec.theta_hat is None:
        return None
    return np.abs(rec.theta_hat - rec.theta_true).mean(axis=1)


def mean_absolute_estimation_error(rec: RunRecord, which: str = "mean") -> float:
    """Time-averaged MAE_theta ("mean") or its value in the last round ("final")."""
    curve = mae_curve(rec)
    if curve is None:
        return float("nan")
    return float(curve[-1] if which == "final" else curve.mean())


def selection_share(rec: RunRecord, start: int = 0, end: int | None = None) -> np.ndarray:
    """Share of rounds spent [acting alone, asking helper 1, ..., helper K]."""
    act = rec.action[start:end]
    return np.bincount(act, minlength=rec.K + 1)[: rec.K + 1] / max(act.size, 1)


def new_optimum_matches(rec: RunRecord, oracle_after: OracleDecision) -> np.ndarray:
    """1 if the action of round t is optimal under the *post-change* reliabilities."""
    d, a = rec.d, rec.a
    ok = np.zeros(rec.T, dtype=bool)
    ok[d == 0] = oracle_after.alone_ok
    asked = d == 1
    if oracle_after.ask_ok:
        ok[asked] = oracle_after.best_helper[a[asked]]
    return ok


def adaptation_latency(rec: RunRecord, change_point: int, oracle_after: OracleDecision,
                       window: int = 50, threshold: float = 0.5):
    """tau_adapt: rounds after the change point until the agent first selects the
    newly optimal decision in more than ``threshold`` of the preceding ``window``
    rounds (proposal Sec. 4.3).

    The trailing window may straddle the change point, exactly as in the
    definition; a policy that switches perfectly at tau + 1 therefore scores
    floor(window * threshold) + 1 = 26.  Returns (latency, censored); if the
    criterion is never met, latency = T - tau and censored = True.
    """
    match = new_optimum_matches(rec, oracle_after).astype(float)
    csum = np.concatenate([[0.0], np.cumsum(match)])
    for t in range(change_point + 1, rec.T + 1):          # t is a 1-indexed round
        lo = max(0, t - window)
        frac = (csum[t] - csum[lo]) / (t - lo)
        if frac > threshold:
            return t - change_point, False
    return rec.T - change_point, True


def summarize_run(rec: RunRecord, change_point: int | None = None,
                  oracle_after: OracleDecision | None = None, window: int = 50) -> dict:
    """All scalar metrics for one run."""
    share = selection_share(rec)
    out = {
        "cumulative_regret": cumulative_regret(rec),
        "average_net_utility": average_net_utility(rec),
        "average_expected_utility": average_expected_utility(rec),
        "ask_rate": ask_rate(rec),
        "ask_accuracy": ask_decision_accuracy(rec),
        "optimal_teammate_rate": optimal_teammate_selection_rate(rec),
        "optimal_action_rate": optimal_action_rate(rec),
        "communication_cost": total_communication_cost(rec),
        "mae_theta": mean_absolute_estimation_error(rec, "mean"),
        "mae_theta_final": mean_absolute_estimation_error(rec, "final"),
        "share_alone": float(share[0]),
    }
    for k in range(rec.K):
        out[f"share_h{k + 1}"] = float(share[k + 1])
    if change_point is not None and oracle_after is not None:
        tau = int(change_point)
        lat, cens = adaptation_latency(rec, tau, oracle_after, window=window)
        out.update({
            "adaptation_latency": float(lat),
            "adaptation_censored": bool(cens),
            "pre_change_regret": cumulative_regret(rec, 0, tau),
            "post_change_regret": cumulative_regret(rec, tau),
            "post_change_utility": average_net_utility(rec, tau),
            "post_change_optimal_teammate_rate": optimal_teammate_selection_rate(rec, tau),
            "post_change_optimal_action_rate": optimal_action_rate(rec, tau),
            "post_change_ask_rate": ask_rate(rec, tau),
        })
    return out
