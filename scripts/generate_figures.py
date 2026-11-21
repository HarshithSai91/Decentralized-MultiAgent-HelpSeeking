#!/usr/bin/env python3
"""Draw every report figure from the saved results (no re-simulation).

Reads   results/raw/<exp>/{runs.csv, curves.npz}, results/manifests/<exp>.json
Writes  figures/*.pdf (vector, used by the LaTeX report) and figures/*.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import available, cell, curve, curves, manifest, runs, stat  # noqa: E402

FIG = "figures"
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- style
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
# Colour follows the entity: every method keeps the same colour/marker in every figure.
STYLE = {
    "D-UCB": ("#2a78d6", "o", "D-UCB"),
    "D-TS": ("#eb6834", "s", "D-TS"),
    "Greedy": ("#1baf7a", "^", "D-Greedy"),
    "Greedy (extra)": ("#1baf7a", "^", "D-Greedy"),
    "Always-Ask + Random": ("#eda100", "D", "Always-Ask + Random"),
    "Always-Ask + Fixed-Cheapest": ("#e87ba4", "v", "Always-Ask + Fixed-Cheapest"),
    "Never-Ask": ("#008300", "P", "Never-Ask"),
    "Always-Ask + Empirical Greedy": ("#6250d6", "X", "Always-Ask + Greedy (γ=1)"),
    "Initial-Knowledge Oracle": ("#e34948", "h", "Initial-Knowledge Oracle"),
    "Dynamic Oracle": (INK, None, "Dynamic oracle"),
    "Stationary Oracle": (INK, None, "Oracle"),
}
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # ordinal, light -> dark
ALONE_GREY = "#b4b2a9"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200,
    "legend.frameon": False, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def style(algo):
    if algo.startswith("D-UCB [alpha="):
        return ("#2a78d6", "o", "D-UCB (α = " + algo.split("=")[1].rstrip("]") + ")")
    return STYLE.get(algo, (INK2, "o", algo))


def plot_series(ax, x, y, algo, err=None, label=True, **kw):
    col, mk, lab = style(algo)
    oracle = "Oracle" in algo and "Initial" not in algo
    ls = (0, (4, 2)) if oracle else ((0, (1, 1.2)) if "[alpha" in algo else "-")
    lw = 1.2 if oracle else 1.6
    if err is not None:
        ax.errorbar(x, y, yerr=err, color=col, lw=0, elinewidth=0.9, capsize=2, zorder=2)
    ax.plot(x, y, color=col, lw=lw, ls=ls, marker=mk if not oracle else None, ms=5,
            markeredgecolor="white", markeredgewidth=0.8, label=lab if label else None, zorder=3, **kw)


def ci_half(s):
    return 0.0 if np.isnan(s["ci_high"]) else s["ci_high"] - s["mean"]


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight")
    plt.close(fig)
    print("  figures/" + name + ".{pdf,png}")


def smooth(y, w=50):
    """Trailing moving average (same window as the tau_adapt definition)."""
    c = np.cumsum(np.insert(np.asarray(y, float), 0, 0.0))
    out = np.empty(len(y))
    for i in range(len(y)):
        lo = max(0, i + 1 - w)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def shared_legend(fig, axes, ncol=6, y=1.04):
    handles, labels = [], []
    for ax in np.atleast_1d(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, y),
               handlelength=2.2, columnspacing=1.4)


# ================================================================== E1
def fig_e1():
    name = "E1_stationary"
    if not available(name):
        return
    m, df, cv = manifest(name), runs(name), curves(name)
    algos = [a["label"] for a in m["algorithms"]]
    pts = m["points"]
    p10 = [p for p in pts if abs(p["value"] - 0.10) < 1e-9][0]["index"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    ax = axes[0]
    t = np.arange(1, m["config"]["T"] + 1)
    for a in algos:
        y = curve(cv, p10, a, "cum_regret_mean")
        se = curve(cv, p10, a, "cum_regret_se")
        col, _, _ = style(a)
        if "Oracle" not in a:
            ax.fill_between(t, y - 1.96 * se, y + 1.96 * se, color=col, alpha=0.12, lw=0)
        plot_series(ax, t, y, a, markevery=400)
    ax.set_title("(a) Cumulative regret, Δ = 0.10")
    ax.set_xlabel("round t")
    ax.set_ylabel("pseudo-regret")
    deltas = [p["value"] for p in pts]
    for metric, axi, title, ylab, scale in (("cumulative_regret", axes[1], "(b) Final regret vs gap Δ", "regret at T = 2000", 1),
                                           ("optimal_teammate_rate", axes[2], "(c) Optimal-teammate rate", "% of asks to best helper", 100)):
        for a in algos:
            st = [stat(df, a, p["index"], metric) for p in pts]
            plot_series(axi, deltas, [s["mean"] * scale for s in st], a,
                        err=[ci_half(s) * scale for s in st], label=False)
        axi.set_title(title)
        axi.set_xlabel("utility gap Δ between top helpers")
        axi.set_ylabel(ylab)
        axi.set_xticks(deltas)
    axes[2].set_ylim(0, 105)
    shared_legend(fig, axes, ncol=6, y=1.1)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig1_e1_stationary")


# ================================================================== E2
def fig_e2():
    name = "E2_when_to_ask"
    if not available(name):
        return
    m, df = manifest(name), runs(name)
    algos = [a["label"] for a in m["algorithms"]]
    pts = m["points"]
    us = [p["value"] for p in pts]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    spec = (("ask_rate", "(a) Ask rate", "% of rounds asking", 100),
            ("ask_accuracy", "(b) Ask-decision accuracy", r"Acc$_{ask}$ (%)", 100),
            ("average_net_utility", "(c) Average net utility", "utility per round", 1))
    for (metric, title, ylab, sc), ax in zip(spec, axes):
        for a in algos:
            st = [stat(df, a, p["index"], metric) for p in pts]
            plot_series(ax, us, [s["mean"] * sc for s in st], a, err=[ci_half(s) * sc for s in st])
        ax.set_title(title)
        ax.set_xlabel("acting-alone utility $U^{self}$")
        ax.set_ylabel(ylab)
        ax.set_xticks(us)
        # oracle switches from asking to acting alone where U_self reaches max_k mu_k = 0.65
        ax.axvspan(0.65, 0.83, color="#f0efec", zorder=0, lw=0)
    axes[1].axhline(90, color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
    axes[1].text(0.29, 91.5, "90% success criterion", color=INK2, fontsize=6.5)
    axes[0].text(0.735, 78, "acting alone\noptimal (tie\nat 0.65)", color=INK2, fontsize=6.5, ha="center")
    shared_legend(fig, axes, ncol=6, y=1.1)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig2_e2_when_to_ask")


# ================================================================== E3
def fig_e3_adaptation():
    name = "E3_abrupt_change"
    if not available(name):
        return
    m, cv = manifest(name), curves(name)
    pts = m["points"]
    tau = m["config"]["reliability"]["tau"]
    T = m["config"]["T"]
    t = np.arange(1, T + 1)
    learners = ["Greedy", "D-UCB", "D-TS"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4), sharey=True)
    for ax, a in zip(axes, learners):
        for p, col in zip(pts, BLUE_RAMP[::-1]):       # gamma = 1 darkest
            y = smooth(curve(cv, p["index"], a, "opt_action")) * 100
            ax.plot(t, y, color=col, lw=1.4, label=f"γ = {p['value']}")
        ax.plot(t, smooth(curve(cv, 0, "Initial-Knowledge Oracle", "opt_action")) * 100,
                color=style("Initial-Knowledge Oracle")[0], lw=1.1, ls=(0, (4, 2)), label="Initial-Knowledge Oracle")
        ax.axvline(tau, color=MUTED, lw=0.8)
        ax.set_title(f"{style(a)[2]}")
        ax.set_xlabel("round t")
        ax.set_xlim(600, 1600)
        ax.set_ylim(-3, 103)
    axes[0].set_ylabel("optimal action, % (50-round avg.)")
    axes[0].text(tau + 12, 92, "τ = 1000", color=INK2, fontsize=6.5)
    shared_legend(fig, axes, ncol=6, y=1.1)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig3_e3_adaptation")


def fig_e3_summary():
    name = "E3_abrupt_change"
    if not available(name):
        return
    m, df = manifest(name), runs(name)
    pts = m["points"]
    gam = [p["value"] for p in pts]
    xs = np.arange(len(gam))
    learners = ["Greedy", "D-UCB", "D-TS"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    ax = axes[0]
    for a in learners:
        med, lo, hi = [], [], []
        for p in pts:
            lat = cell(df, a, p["index"], "adaptation_latency")
            med.append(np.median(lat))
            lo.append(np.median(lat) - np.percentile(lat, 25))
            hi.append(np.percentile(lat, 75) - np.median(lat))
        col, mk, lab = style(a)
        ax.errorbar(xs, med, yerr=[lo, hi], color=col, lw=0, elinewidth=0.9, capsize=2)
        ax.plot(xs, med, color=col, marker=mk, ms=5, lw=1.6, markeredgecolor="white", label=lab)
    ax.axhline(26, color=INK, lw=1.0, ls=(0, (4, 2)), label="Dynamic oracle (floor 26)")
    ax.set_yscale("log")
    ax.set_title(r"(a) Adaptation latency $\tau_{adapt}$")
    ax.set_ylabel("rounds (median, IQR; log)")
    for metric, axi, title in (("pre_change_regret", axes[1], "(b) Regret before change (t ≤ τ)"),
                               ("post_change_regret", axes[2], "(c) Regret after change (t > τ)")):
        for a in learners:
            st = [stat(df, a, p["index"], metric) for p in pts]
            plot_series(axi, xs, [s["mean"] for s in st], a, err=[ci_half(s) for s in st], label=False)
        ref = stat(df, "Initial-Knowledge Oracle", 0, metric)["mean"]
        if metric == "post_change_regret":
            axi.axhline(ref, color=style("Initial-Knowledge Oracle")[0], lw=1.0, ls=(0, (4, 2)),
                        label="Initial-Knowledge Oracle")
        axi.set_title(title)
        axi.set_ylabel("pseudo-regret")
    for axi in axes:
        axi.set_xticks(xs)
        axi.set_xticklabels([f"{g:g}" for g in gam])
        axi.set_xlabel("discount factor γ")
    shared_legend(fig, axes, ncol=5, y=1.1)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig4_e3_tradeoff")


def fig_e3_beliefs():
    name = "E3_abrupt_change"
    if not available(name):
        return
    m, cv = manifest(name), curves(name)
    pts = m["points"]
    T = m["config"]["T"]
    t = np.arange(1, T + 1)
    show = [p for p in pts if p["value"] in (1.0, 0.98, 0.9)]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.4), sharey=True)
    for ax, (k, title) in zip(axes, ((2, "Helper 3 (0.90 → 0.40)"), (0, "Helper 1 (0.80, constant)"))):
        true = curve(cv, 0, "Dynamic Oracle", "theta_true")[:, k]
        ax.plot(t, true, color=INK, lw=1.2, ls=(0, (4, 2)), label=r"true $\theta_k(t)$")
        for p, col in zip(show, [BLUE_RAMP[4], BLUE_RAMP[2], BLUE_RAMP[0]]):
            ax.plot(t, curve(cv, p["index"], "D-TS", "theta_hat")[:, k], color=col, lw=1.5,
                    label=f"D-TS belief, γ = {p['value']}")
        ax.set_title(title)
        ax.set_xlabel("round t")
        ax.set_ylim(0.3, 1.0)
    axes[0].set_ylabel(r"trust estimate $\hat{\theta}_k$ (mean of 30 seeds)")
    shared_legend(fig, axes, ncol=4, y=1.1)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig5_e3_beliefs")


# ================================================================== E4
def fig_e4():
    name = "E4_communication_cost"
    if not available(name):
        return
    m, df = manifest(name), runs(name)
    algos = [a["label"] for a in m["algorithms"]]
    pts = m["points"]
    lam = [p["value"] for p in pts]
    xs = np.arange(len(lam))
    fig, axes = plt.subplots(1, len(algos), figsize=(7.4, 2.3), sharey=True)
    seg = [("share_alone", ALONE_GREY, "act alone"), ("share_h1", BLUE_RAMP[0], "helper 1 (θ=.60, c=.05)"),
           ("share_h2", BLUE_RAMP[2], "helper 2 (θ=.75, c=.10)"), ("share_h3", BLUE_RAMP[4], "helper 3 (θ=.90, c=.20)")]
    for ax, a in zip(axes, algos):
        bottom = np.zeros(len(pts))
        for metric, col, lab in seg:
            vals = np.array([stat(df, a, p["index"], metric)["mean"] * 100 for p in pts])
            ax.bar(xs, vals, bottom=bottom, width=0.62, color=col, edgecolor="white", linewidth=1.0,
                   label=lab)
            bottom += vals
        short = {"Always-Ask + Fixed-Cheapest": "Fixed-Cheapest", "Dynamic Oracle": "Dynamic oracle"}
        ax.set_title(short.get(a, style(a)[2]), fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{v:g}" for v in lam])
        ax.set_xlabel("λ")
        ax.grid(False)
    axes[0].set_ylabel("share of rounds (%)")
    shared_legend(fig, axes[:1], ncol=4, y=1.1)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig6_e4_selection")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.4))
    for metric, ax, title, ylab in (("average_net_utility", axes[0], "(a) Average net utility", "utility per round"),
                                    ("communication_cost", axes[1], "(b) Total communication cost", r"$\sum_t \lambda c_{a_t}$ over 2000 rounds")):
        for a in algos:
            st = [stat(df, a, p["index"], metric) for p in pts]
            plot_series(ax, xs, [s["mean"] for s in st], a, err=[ci_half(s) for s in st])
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{v:g}" for v in lam])
        ax.set_xlabel("cost multiplier λ")
        ax.set_title(title)
        ax.set_ylabel(ylab)
    axes[0].axhline(0.45, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    axes[0].text(1.15, 0.415, r"$U^{self} = 0.45$", color=INK2, fontsize=6.5)
    shared_legend(fig, axes, ncol=5, y=1.1)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig7_e4_utility_cost")


# ======================================================== supplementary
def fig_supp():
    have_s1, have_s3 = available("S1_gamma_stationary"), available("S3_drift")
    if not (have_s1 and have_s3):
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.4))
    for ax, name, title in ((axes[0], "S1_gamma_stationary", "(a) Stationary world (E1, Δ = 0.10)"),
                            (axes[1], "S3_drift", "(b) Continuous logit drift, σ = 0.05")):
        m, df = manifest(name), runs(name)
        pts = m["points"]
        xs = np.arange(len(pts))
        for a in [x["label"] for x in m["algorithms"] if x["label"] not in ("Dynamic Oracle",)]:
            st = [stat(df, a, p["index"], "cumulative_regret") for p in pts]
            plot_series(ax, xs, [s["mean"] for s in st], a, err=[ci_half(s) for s in st])
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{p['value']:g}" for p in pts])
        ax.set_xlabel("discount factor γ")
        ax.set_ylabel("cumulative regret")
        ax.set_title(title)
    shared_legend(fig, axes, ncol=4, y=1.1)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig8_s1_s3_discounting")


def fig_alpha():
    scen = [("S2_alpha_ucb_E1", "E1: stationary, γ = 1"),
            ("S2_alpha_ucb_E2_U080", r"E2: $U^{self}$ = 0.80, γ = 0.98"),
            ("S2_alpha_ucb_E3_g098", "E3: abrupt change, γ = 0.98")]
    if not all(available(n) for n, _ in scen):
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.3))
    for ax, (name, title) in zip(axes, scen):
        m, df = manifest(name), runs(name)
        pts = m["points"]
        al = [p["value"] for p in pts]
        st = [stat(df, "D-UCB", p["index"], "cumulative_regret") for p in pts]
        plot_series(ax, al, [s["mean"] for s in st], "D-UCB", err=[ci_half(s) for s in st])
        ax.set_xscale("log")
        ax.set_xticks(al)
        ax.set_xticklabels([f"{a:g}" for a in al])
        ax.minorticks_off()
        ax.set_xlabel(r"exploration intensity $\alpha_{ucb}$ (log)")
        ax.set_title(title, fontsize=8)
        ax.axvline(1.0, color=AXIS, lw=0.8, zorder=0)
        ax.text(1.0, 0.97, " default α = 1", transform=ax.get_xaxis_transform(), fontsize=6.5,
                color=INK2, ha="right", va="top")
    axes[0].set_ylabel("D-UCB cumulative regret")
    fig.tight_layout(w_pad=1.4)
    save(fig, "fig9_s2_alpha")


def fig_s4():
    scen = [("S4_when_to_ask_gamma_U050", r"$U^{self}$ = 0.50 (asking optimal)"),
            ("S4_when_to_ask_gamma_U080", r"$U^{self}$ = 0.80 (acting alone optimal)")]
    if not all(available(n) for n, _ in scen):
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.4), sharey=True)
    for ax, (name, title) in zip(axes, scen):
        m, df = manifest(name), runs(name)
        pts = m["points"]
        xs = np.arange(len(pts))
        for a in [x["label"] for x in m["algorithms"] if x["label"] != "Dynamic Oracle"]:
            st = [stat(df, a, p["index"], "ask_accuracy") for p in pts]
            plot_series(ax, xs, [s["mean"] * 100 for s in st], a, err=[ci_half(s) * 100 for s in st])
        ax.axhline(90, color=MUTED, lw=0.8, ls=(0, (2, 2)))
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{p['value']:g}" for p in pts])
        ax.set_xlabel("discount factor γ")
        ax.set_title(title, fontsize=8.5)
    axes[0].set_ylabel(r"Acc$_{ask}$ (%)")
    shared_legend(fig, axes, ncol=3, y=1.1)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig10_s4_ask_accuracy_gamma")


def fig_s5():
    name = "S5_multishift"
    if not available(name):
        return
    m, df = manifest(name), runs(name)
    pts = m["points"]
    pcs = sorted({p["params"]["p_c"] for p in pts})
    gammas = sorted({p["params"]["gamma"] for p in pts}, reverse=True)
    idx = {(p["params"]["p_c"], p["params"]["gamma"]): p["index"] for p in pts}
    xs = np.arange(len(gammas))
    fig, axes = plt.subplots(1, len(pcs), figsize=(7.4, 2.4), sharey=True)
    for ax, pc in zip(axes, pcs):
        nch = stat(df, "Dynamic Oracle", idx[(pc, gammas[0])], "n_changes")["mean"]
        for a in ("Greedy", "D-UCB", "D-TS"):
            st = [stat(df, a, idx[(pc, g)], "cumulative_regret") for g in gammas]
            plot_series(ax, xs, [s["mean"] for s in st], a, err=[ci_half(s) for s in st])
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{g:g}" for g in gammas])
        ax.set_xlabel("discount factor γ")
        ax.set_title(f"$p_c$ = {pc:g}  (≈{nch:.0f} changes)", fontsize=8.5)
    axes[0].set_ylabel("cumulative regret")
    shared_legend(fig, axes, ncol=3, y=1.1)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig11_s5_multishift")


if __name__ == "__main__":
    for f in (fig_e1, fig_e2, fig_e3_adaptation, fig_e3_summary, fig_e3_beliefs, fig_e4, fig_supp, fig_alpha, fig_s4, fig_s5):
        f()
