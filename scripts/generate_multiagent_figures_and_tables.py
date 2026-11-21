#!/usr/bin/env python3
"""Tables, figures and key numbers for the multi-agent experiments E5-E8.

Everything is computed directly from the per-seed raw results
(results/raw/<experiment>/runs.csv and curves.npz):

    report/tables/e5_table.tex            helper behaviour vs. requester/helper utility (E5)
    report/tables/e5_state_table.tex      acceptance by helper state (E5)
    report/tables/e6_table.tex            cooperation incentive sweep (E6)
    report/tables/e6_heterogeneous_table.tex  per-agent behaviour in the heterogeneous team (E6)
    report/tables/e7_table.tex            changing reliability vs. discount factor (E7)
    report/tables/e8_table.tex            timing policies under deadlines (E8)
    figures/e5_e6_helping_behaviour.{pdf,png}
    figures/e7_changing_reliability.{pdf,png}
    figures/e8_temporal_help_seeking.{pdf,png}
    results/processed/multiagent_key_numbers.json   (every E5-E8 number quoted in the report,
                                                     including seed-paired Wilcoxon tests)

Mean +/- half-width of the 95% Student-t confidence interval over 30 seeds.
Paired tests: two-sided Wilcoxon signed-rank over the 30 matching seeds (needs SciPy).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import mean_ci, runs, curves  # noqa: E402  (sets cwd, sys.path)

FIG_DIR = "figures"
TABLE_DIR = os.path.join("report", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# Categorical palette (fixed order) and neutral ink for text/axes.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.4, "legend.frameon": False, "pdf.fonttype": 42,
})

ALGOS = [("MultiAgent-D-TS", "MA-D-TS"), ("MultiAgent-D-UCB", "MA-D-UCB")]
KEY = {}


def stat(values):
    r = mean_ci(values)
    half = r["ci_high"] - r["mean"] if r["n"] > 1 and np.isfinite(r["ci_high"]) else 0.0
    return float(r["mean"]), float(half)


def fmt(values, digits=3, scale=1.0):
    m, h = stat(np.asarray(values, dtype=float) * scale)
    if not np.isfinite(m):
        return "--"
    if h < 1e-12:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f}{{\\scriptsize$\\pm${h:.{digits}f}}}"


def record(path, values, scale=1.0):
    """Store mean and CI half-width under KEY[path...] for the report."""
    m, h = stat(np.asarray(values, dtype=float) * scale)
    node = KEY
    for p in path[:-1]:
        node = node.setdefault(p, {})
    node[path[-1]] = {"mean": m, "ci95_half_width": h}


def wilcoxon(a, b):
    try:
        from scipy.stats import wilcoxon as _w
    except Exception:  # pragma: no cover
        return None
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return 1.0
    return float(_w(a, b).pvalue)


def paired(df, col, sel_a, sel_b):
    a = df[sel_a].sort_values("seed")
    b = df[sel_b].sort_values("seed")
    assert list(a["seed"]) == list(b["seed"]), "seed mismatch"
    return {"mean_a": float(a[col].mean()), "mean_b": float(b[col].mean()),
            "wilcoxon_p": wilcoxon(a[col].values, b[col].values), "n_pairs": int(len(a))}


def write(name, lines):
    with open(os.path.join(TABLE_DIR, name), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {TABLE_DIR}/{name}")


def save(fig, stem):
    fig.savefig(os.path.join(FIG_DIR, f"{stem}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, f"{stem}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_DIR}/{stem}.pdf/.png")


# --------------------------------------------------------------------------- E5
E5 = "E5_helper_learning"
HELPERS = [("Always Accept", "Always accept"), ("Always Reject", "Always reject"),
           ("Informed Threshold", "Informed threshold$^{\\dagger}$"), ("Q-Learning", "Q-learning (ours)")]
STATES = [("low_w_high_r", "$(0,2)$ idle", 0.15), ("med_w_med_r", "$(1,1)$ medium", 0.30),
          ("high_w_low_r", "$(2,0)$ busy", 0.45)]


def e5():
    d = runs(E5)
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Helper policy & Requester & Accept\,\% & Ask\,\% & Requester util. & Helper util. & Welfare \\",
             r"\midrule"]
    for i, (pt, label) in enumerate(HELPERS):
        for j, (algo, alabel) in enumerate(ALGOS):
            s = d[(d.point_label == pt) & (d.algorithm == algo)]
            acc = fmt(s.acceptance_rate, 1, 100) if s.acceptance_rate.notna().any() else "--"
            lines.append(f"{label if j == 0 else ''} & {alabel} & {acc} & {fmt(s.ask_rate, 1, 100)} & "
                         f"{fmt(s.requester_utility)} & {fmt(s.helper_utility)} & {fmt(s.social_welfare)} \\\\")
            for col in ["acceptance_rate", "ask_rate", "requester_utility", "helper_utility",
                        "social_welfare", "cumulative_regret", "helper_policy_stability",
                        "helper_states_evaluated"]:
                if s[col].notna().any():
                    record(["E5", pt, alabel, col], s[col])
        if i < len(HELPERS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("e5_table.tex", lines)

    # acceptance by helper state
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Helper state $(w,\rho)$ & Help cost & Q-learning (MA-D-TS) & Q-learning (MA-D-UCB) & Informed threshold$^{\dagger}$ \\",
             r"\midrule"]
    for key, label, cost in STATES:
        cells = []
        for pt, algo in [("Q-Learning", "MultiAgent-D-TS"), ("Q-Learning", "MultiAgent-D-UCB"),
                         ("Informed Threshold", "MultiAgent-D-TS")]:
            s = d[(d.point_label == pt) & (d.algorithm == algo)][f"{key}_accept_rate"].dropna()
            cells.append(fmt(s, 1, 100))
            record(["E5", "state_acceptance", pt, algo, key], s, 100)
        lines.append(f"{label} & {cost:.2f} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("e5_state_table.tex", lines)

    # Q-learning vs informed threshold / always accept, paired by seed
    for algo, alabel in ALGOS:
        for col in ["helper_utility", "requester_utility", "social_welfare"]:
            KEY.setdefault("E5", {}).setdefault("tests", {})[f"{alabel} Q-learning vs informed {col}"] = paired(
                d, col, (d.point_label == "Q-Learning") & (d.algorithm == algo),
                (d.point_label == "Informed Threshold") & (d.algorithm == algo))
            KEY["E5"]["tests"][f"{alabel} Q-learning vs always-accept {col}"] = paired(
                d, col, (d.point_label == "Q-Learning") & (d.algorithm == algo),
                (d.point_label == "Always Accept") & (d.algorithm == algo))
    return d


# --------------------------------------------------------------------------- E6
E6 = "E6_multiagent_cooperation"
BPOINTS = [(0.2, "$B=0.2$"), (0.5, "$B=0.5$"), (0.8, "$B=0.8$"), (-1.0, "Heterogeneous")]


def e6():
    d = runs(E6)
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{MA-D-TS} & \multicolumn{3}{c}{MA-D-UCB} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Benefit $B_j$ & Accept\,\% & Requester util. & Welfare & Accept\,\% & Requester util. & Welfare \\",
             r"\midrule"]
    for val, label in BPOINTS:
        row = [label]
        for algo, alabel in ALGOS:
            s = d[(d.point_value == val) & (d.algorithm == algo)]
            row += [fmt(s.acceptance_rate, 1, 100), fmt(s.requester_utility), fmt(s.social_welfare)]
            for col in ["acceptance_rate", "requester_utility", "helper_utility", "social_welfare"]:
                record(["E6", label, alabel, col], s[col])
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("e6_table.tex", lines)

    het = d[d.point_value == -1.0]
    thetas = [0.7, 0.6, 0.5, 0.8]
    Bs = [0.2, 0.4, 0.6, 0.8]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & & & \multicolumn{2}{c}{MA-D-TS} & \multicolumn{2}{c}{MA-D-UCB} \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Agent & $B_j$ & $\theta_j$ & Accept\,\% & Helps/round & Accept\,\% & Helps/round \\",
             r"\midrule"]
    for k in range(4):
        row = [f"$A_{k + 1}$", f"{Bs[k]:.1f}", f"{thetas[k]:.1f}"]
        for algo, alabel in ALGOS:
            s = het[het.algorithm == algo]
            row += [fmt(s[f"agent_{k}_acceptance_rate"].dropna(), 1, 100),
                    fmt(s[f"agent_{k}_help_frequency"], 3)]
            record(["E6", "heterogeneous", alabel, f"agent_{k + 1}", "acceptance_rate"],
                   s[f"agent_{k}_acceptance_rate"].dropna(), 100)
            record(["E6", "heterogeneous", alabel, f"agent_{k + 1}", "help_frequency"],
                   s[f"agent_{k}_help_frequency"])
            record(["E6", "heterogeneous", alabel, f"agent_{k + 1}", "incoming_request_frequency"],
                   s[f"agent_{k}_incoming_request_frequency"])
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("e6_heterogeneous_table.tex", lines)
    return d


def fig_helping(d5, d6):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.35))
    # (a) acceptance by helper state
    series = [("Q-Learning", "MultiAgent-D-TS", "Q-learning, MA-D-TS", C[0], None),
              ("Q-Learning", "MultiAgent-D-UCB", "Q-learning, MA-D-UCB", C[1], None),
              ("Informed Threshold", "MultiAgent-D-TS", "Informed threshold", C[2], "////")]
    x = np.arange(len(STATES))
    w = 0.26
    for k, (pt, algo, lab, col, hatch) in enumerate(series):
        ms, hs = [], []
        for key, _, _ in STATES:
            s = d5[(d5.point_label == pt) & (d5.algorithm == algo)][f"{key}_accept_rate"].dropna() * 100
            m, h = stat(s)
            ms.append(m)
            hs.append(h)
        ax1.bar(x + (k - 1) * w, ms, w * 0.92, yerr=hs, color=col, label=lab, hatch=hatch,
                edgecolor="white", linewidth=0.6, error_kw={"elinewidth": 0.7, "ecolor": INK2, "capsize": 1.5})
        for xi, m in zip(x + (k - 1) * w, ms):
            if m < 0.5:
                ax1.text(xi, 1.5, "0", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax1.set_xticks(x, [r"idle $(0,2)$" + "\ncost 0.15", r"medium $(1,1)$" + "\ncost 0.30",
                       r"busy $(2,0)$" + "\ncost 0.45"])
    ax1.set_ylabel("Requests accepted (%)")
    ax1.set_ylim(0, 138)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_title("(a) E5: acceptance by the helper's own state")
    ax1.legend(loc="upper right", fontsize=6.3, ncol=1)
    ax1.grid(axis="x", visible=False)

    # (b) heterogeneous team: acceptance per agent
    het = d6[d6.point_value == -1.0]
    x = np.arange(4)
    w = 0.38
    for k, (algo, alabel) in enumerate(ALGOS):
        ms, hs = [], []
        for a in range(4):
            m, h = stat(het[het.algorithm == algo][f"agent_{a}_acceptance_rate"].dropna() * 100)
            ms.append(m)
            hs.append(h)
        ax2.bar(x + (k - 0.5) * w, ms, w * 0.92, yerr=hs, color=C[k], label=alabel, edgecolor="white",
                linewidth=0.6, error_kw={"elinewidth": 0.7, "ecolor": INK2, "capsize": 1.5})
    ax2.set_xticks(x, [r"$A_1$" + "\n$B$=0.2, $\\theta$=0.7", r"$A_2$" + "\n$B$=0.4, $\\theta$=0.6",
                       r"$A_3$" + "\n$B$=0.6, $\\theta$=0.5", r"$A_4$" + "\n$B$=0.8, $\\theta$=0.8"])
    ax2.set_ylabel("Requests accepted (%)")
    ax2.set_ylim(0, 138)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.set_title("(b) E6: heterogeneous team, per helper")
    ax2.legend(loc="upper left", fontsize=6.5)
    ax2.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.5)
    save(fig, "e5_e6_helping_behaviour")


# --------------------------------------------------------------------------- E7
E7 = "E7_changing_reliability"
GAMMAS = [(1.0, "1.00"), (0.995, "0.995"), (0.98, "0.98"), (0.95, "0.95")]


def e7():
    d = runs(E7)
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Requester & $\gamma$ & Pre-change regret & Post-change regret & Total regret & Post opt.\ action\,\% \\",
             r"\midrule"]
    for j, (algo, alabel) in enumerate(ALGOS):
        for i, (g, glabel) in enumerate(GAMMAS):
            s = d[(d.algorithm == algo) & np.isclose(d.point_value, g)]
            lines.append(f"{alabel if i == 0 else ''} & {glabel} & {fmt(s.pre_change_regret, 1)} & "
                         f"{fmt(s.post_change_regret, 1)} & {fmt(s.cumulative_regret, 1)} & "
                         f"{fmt(s.post_change_optimal_action_rate, 1, 100)} \\\\")
            for col in ["pre_change_regret", "post_change_regret", "cumulative_regret",
                        "post_change_optimal_action_rate", "post_change_ask_rate"]:
                record(["E7", alabel, f"gamma={glabel}", col], s[col])
        if j == 0:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("e7_table.tex", lines)
    g = d[d.algorithm == "MultiAgent-Greedy"]
    record(["E7", "MA-Greedy", "ask_rate"], g.ask_rate)
    for algo, alabel in ALGOS:
        for gl, gv in [("0.995", 0.995), ("0.98", 0.98), ("0.95", 0.95)]:
            for col in ["post_change_regret", "pre_change_regret", "cumulative_regret"]:
                KEY.setdefault("E7", {}).setdefault("tests", {})[f"{alabel} gamma={gl} vs gamma=1.00 {col}"] = paired(
                    d, col, (d.algorithm == algo) & np.isclose(d.point_value, gv),
                    (d.algorithm == algo) & np.isclose(d.point_value, 1.0))
    return d


def fig_e7():
    cv = curves(E7)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    for ax, (algo, alabel) in zip(axes, ALGOS):
        slug_a = algo.replace("-", "_")
        for k, (g, glabel) in enumerate(GAMMAS):
            y = cv[f"p{k}__{slug_a}__cum_regret_mean"]
            se = cv[f"p{k}__{slug_a}__cum_regret_se"]
            t = np.arange(1, len(y) + 1)
            ax.plot(t, y, color=C[k], label=f"$\\gamma$ = {glabel}",
                    linestyle=["-", "-", "--", ":"][k])
            ax.fill_between(t, y - 1.96 * se, y + 1.96 * se, color=C[k], alpha=0.12, linewidth=0)
        ax.axvline(1000, color=INK2, linestyle=":", linewidth=0.8)
        ax.text(1015, 8, "reliability change", color=INK2, fontsize=6.5)
        ax.set_xlabel("Round $t$")
        ax.set_title(f"{'(a)' if alabel == 'MA-D-TS' else '(b)'} E7: {alabel} requesters")
    axes[0].set_ylabel("Cumulative pseudo-regret")
    axes[0].legend(loc="upper left", fontsize=6.5)
    fig.tight_layout(w_pad=1.0)
    save(fig, "e7_changing_reliability")


# --------------------------------------------------------------------------- E8
E8 = "E8_temporal_help_seeking"
TIMING = [("Learned Timing", "Learned timing (Q-learning)"), ("Planned Timing", "Planned timing (model-based)"),
          ("Immediate Evaluation", "Immediate evaluation"),
          ("Fixed Delay (k=2)", "Fixed delay ($k=2$)"), ("Urgency-Based", r"Urgency ($\tau_{\mathrm{rem}}\le 2$)"),
          ("Never Ask", "Never ask")]


def e8():
    d = runs(E8)
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Timing policy & Completion\,\% & Ask\,\% & Solo attempts & Requester reward & Welfare \\",
             r"\midrule"]
    for name, label in TIMING:
        s = d[d.algorithm == name]
        lines.append(f"{label} & {fmt(s.task_completion_rate, 1, 100)} & {fmt(s.ask_rate, 1, 100)} & "
                     f"{fmt(s.avg_solo_attempts, 2)} & {fmt(s.avg_requester_reward)} & {fmt(s.social_welfare)} \\\\")
        for col in ["task_completion_rate", "ask_rate", "acceptance_rate", "avg_solo_attempts",
                    "avg_time_to_request", "avg_requester_reward", "avg_helper_reward", "social_welfare",
                    "timing_policy_stability", "timing_states_evaluated"]:
            if s[col].notna().any():
                record(["E8", name, col], s[col])
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("e8_table.tex", lines)
    for mine, short in [("Learned Timing", "Learned"), ("Planned Timing", "Planned")]:
        sel = d.algorithm == mine
        others = ["Immediate Evaluation", "Fixed Delay (k=2)", "Urgency-Based", "Never Ask"]
        if mine == "Planned Timing":
            others = ["Learned Timing"] + others
        for other in others:
            for col in ["task_completion_rate", "avg_requester_reward", "ask_rate"]:
                KEY.setdefault("E8", {}).setdefault("tests", {})[f"{short} vs {other} {col}"] = paired(
                    d, col, sel, d.algorithm == other)
    return d


def fig_e8():
    cv = curves(E8)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.3))
    keys = ["Learned_Timing", "Planned_Timing", "Immediate_Evaluation", "Fixed_Delay__k_2", "Urgency_Based", "Never_Ask"]
    styles = ["-", "-", "--", "-.", ":", (0, (1, 3))]
    for k, ((name, label), key) in enumerate(zip(TIMING, keys)):
        comp = cv[f"p0__{key}__completion_rate"]
        win = 40
        sm = np.convolve(comp, np.ones(win) / win, mode="valid")
        label = label.replace(r"\le ", r"\leq ")
        ax1.plot(np.arange(win, len(comp) + 1), sm * 100, color=C[k], linestyle=styles[k], label=label)
        rew = cv[f"p0__{key}__reward_mean"]
        ax2.plot(np.arange(1, len(rew) + 1), np.cumsum(rew), color=C[k], linestyle=styles[k], label=label)
    for ax in (ax1, ax2):
        ax.axvline(200, color=INK2, linestyle=":", linewidth=0.8)
        ax.set_xlabel("Task index")
    ax1.set_ylabel("Completion rate (%, 40-task window)")
    ax1.set_title("(a) E8: task completion over time")
    ax2.set_ylabel("Cumulative requester reward")
    ax2.set_title("(b) E8: cumulative requester reward")
    ax2.legend(loc="upper left", fontsize=6.3)
    fig.tight_layout(w_pad=1.5)
    save(fig, "e8_temporal_help_seeking")


def _m(node, scale=1.0, digits=3):
    return f"{node['mean'] * scale:.{digits}f} ± {node['ci95_half_width'] * scale:.{digits}f}"


def write_summary():
    """Human-readable summary of E5-E8 (same numbers as the report tables)."""
    L = ["# Multi-agent results summary (E5–E8)", "",
         "Generated by `scripts/generate_multiagent_figures_and_tables.py` from `results/raw/*/runs.csv`.",
         "Mean ± half-width of the 95% t-CI over 30 seeds; p-values: two-sided Wilcoxon signed-rank, paired by seed.", ""]
    L += ["## E5 — helper behaviour (utilities per round)", "",
          "| Helper policy | Requester | Accept % | Ask % | Requester util. | Helper util. | Welfare |", "|---|---|---|---|---|---|---|"]
    for pt, _ in HELPERS:
        for _, alabel in ALGOS:
            n = KEY["E5"][pt][alabel]
            acc = _m(n["acceptance_rate"], 100, 1) if "acceptance_rate" in n else "–"
            L.append(f"| {pt} | {alabel} | {acc} | {_m(n['ask_rate'], 100, 1)} | {_m(n['requester_utility'])} | "
                     f"{_m(n['helper_utility'])} | {_m(n['social_welfare'])} |")
    L += ["", "Acceptance by helper state (%):", "", "| State | Q-learning (MA-D-TS) | Q-learning (MA-D-UCB) | Informed threshold (MA-D-TS) |",
          "|---|---|---|---|"]
    sa = KEY["E5"]["state_acceptance"]
    for key, label, cost in STATES:
        L.append(f"| {label.replace('$', '')} (cost {cost:.2f}) | {_m(sa['Q-Learning']['MultiAgent-D-TS'][key], 1, 1)} | "
                 f"{_m(sa['Q-Learning']['MultiAgent-D-UCB'][key], 1, 1)} | {_m(sa['Informed Threshold']['MultiAgent-D-TS'][key], 1, 1)} |")
    L += ["", "Paired tests:", ""] + [f"- {k}: {v['mean_a']:.3f} vs {v['mean_b']:.3f}, p = {v['wilcoxon_p']:.3g}"
                                       for k, v in KEY["E5"]["tests"].items()]
    L += ["", "## E6 — cooperation benefit B", "", "| B | Requester | Accept % | Requester util. | Helper util. | Welfare |",
          "|---|---|---|---|---|---|"]
    for _, label in BPOINTS:
        for _, alabel in ALGOS:
            n = KEY["E6"][label][alabel]
            L.append(f"| {label.replace('$', '')} | {alabel} | {_m(n['acceptance_rate'], 100, 1)} | {_m(n['requester_utility'])} | "
                     f"{_m(n['helper_utility'])} | {_m(n['social_welfare'])} |")
    L += ["", "Heterogeneous team (B = 0.2, 0.4, 0.6, 0.8 for A1–A4):", "",
          "| Agent | Requester | Accept % | Helps / round | Asked / round |", "|---|---|---|---|---|"]
    for _, alabel in ALGOS:
        for k in range(1, 5):
            n = KEY["E6"]["heterogeneous"][alabel][f"agent_{k}"]
            L.append(f"| A{k} | {alabel} | {_m(n['acceptance_rate'], 1, 1)} | {_m(n['help_frequency'])} | "
                     f"{_m(n['incoming_request_frequency'])} |")
    L += ["", "## E7 — reliability change at t = 1000", "",
          "| Requester | γ | Pre-change regret | Post-change regret | Total regret | Post opt. action % |", "|---|---|---|---|---|---|"]
    for _, alabel in ALGOS:
        for _, gl in GAMMAS:
            n = KEY["E7"][alabel][f"gamma={gl}"]
            L.append(f"| {alabel} | {gl} | {_m(n['pre_change_regret'], 1, 1)} | {_m(n['post_change_regret'], 1, 1)} | "
                     f"{_m(n['cumulative_regret'], 1, 1)} | {_m(n['post_change_optimal_action_rate'], 100, 1)} |")
    L += ["", "Paired tests (vs γ = 1.00):", ""] + [f"- {k}: {v['mean_a']:.1f} vs {v['mean_b']:.1f}, p = {v['wilcoxon_p']:.3g}"
                                                 for k, v in KEY["E7"]["tests"].items()]
    L += ["", "## E8 — timing under deadlines (D = 5, 400 tasks)", "",
          "| Timing policy | Completion % | Ask % | Accept % | Solo attempts | Requester reward | Welfare |", "|---|---|---|---|---|---|---|"]
    for name, _ in TIMING:
        n = KEY["E8"][name]
        acc = _m(n["acceptance_rate"], 100, 1) if "acceptance_rate" in n else "–"
        L.append(f"| {name} | {_m(n['task_completion_rate'], 100, 1)} | {_m(n['ask_rate'], 100, 1)} | {acc} | "
                 f"{_m(n['avg_solo_attempts'], 1, 2)} | {_m(n['avg_requester_reward'])} | {_m(n['social_welfare'])} |")
    st = KEY["E8"]["Learned Timing"]
    L += ["", f"Learned timing greedy-policy stability: {_m(st['timing_policy_stability'], 100, 1)} % "
              f"over {_m(st['timing_states_evaluated'], 1, 1)} evaluated states.", "", "Paired tests:", ""]
    L += [f"- {k}: {v['mean_a']:.3f} vs {v['mean_b']:.3f}, p = {v['wilcoxon_p']:.3g}" for k, v in KEY["E8"]["tests"].items()]
    out = os.path.join("results", "MULTIAGENT_RESULTS_SUMMARY.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    d5 = e5()
    d6 = e6()
    fig_helping(d5, d6)
    e7()
    fig_e7()
    e8()
    fig_e8()
    out = os.path.join("results", "processed", "multiagent_key_numbers.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(KEY, fh, indent=1, ensure_ascii=False)
    print(f"wrote {out}")
    write_summary()
