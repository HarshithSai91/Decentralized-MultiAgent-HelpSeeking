# backend/src/help_seeking/experiments.py
"""Configuration-driven experiment runner for E1-E4 and the supplementary studies.

Config format (JSON)::

    {
      "name": "E1_stationary",
      "title": "E1: Stationary learning",
      "T": 2000, "R": 1.0, "n_seeds": 30,
      "base": {"theta": [...], "c": [...], "U_self": 0.35, "lambda": 1.0,
               "gamma": 1.0, "alpha_ucb": 1.0},
      "reliability": {"type": "stationary"}            # or abrupt / drift / multishift
      "sweep": {"name": "delta", "symbol": "Δ",
                "points": [{"value": 0.02, "set": {"theta": [0.62, 0.60, 0.50]}}, ...]},
      "algorithms": ["Dynamic Oracle", {"name": "D-UCB", "overrides": {"alpha_ucb": 0.5}}, ...]
    }

Seeding: run i (i = 1..n_seeds) uses ``SeedSequence(seed_offset + i)`` spawned
into independent streams for (environment outcomes, policy randomness,
reliability path).  The environment stream is identical for every algorithm
and sweep point with the same seed index -> seed-paired evaluation.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import numpy as np

from .environment import SimulationEnvironment
from .metrics import mae_curve, summarize_run
from .multiagent_environment import MultiAgentEnvironment
from .multiagent_simulation import MultiAgentSimulationRun, compute_multiagent_metrics
from .temporal_environment import TemporalMultiAgentEnvironment
from .temporal_simulation import compute_temporal_metrics
from .policies import canonical_name, make_policy
from .reliability import make_profile
from .simulation import SimulationRun
from .stats import mean_ci
from .utilities import dynamic_oracle

NO_BELIEF_POLICIES = {"Dynamic Oracle", "Initial-Knowledge Oracle", "Never-Ask"}
RELIABILITY_KEYS = ("theta_after", "tau", "sigma", "p_c", "low", "high")


# --------------------------------------------------------------------- config
def load_config(config) -> dict:
    if isinstance(config, (str, os.PathLike)):
        with open(config, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return deepcopy(config)


def sweep_points(config: dict) -> list[dict]:
    sweep = config.get("sweep")
    if not sweep:
        return [{"value": None, "label": "base", "params": dict(config["base"])}]
    pts = []
    for p in sweep["points"]:
        params = dict(config["base"])
        params.update(p.get("set", {}))
        label = p.get("label") or f"{sweep.get('symbol', sweep['name'])}={p['value']}"
        pts.append({"value": p["value"], "label": label, "params": params})
    return pts


def algorithm_specs(config: dict) -> list[dict]:
    out = []
    is_temporal = config.get("temporal", False)
    for a in config["algorithms"]:
        spec = {"name": a} if isinstance(a, str) else dict(a)
        spec.setdefault("label", spec["name"])
        if is_temporal:
            spec["canonical"] = spec["name"]
        else:
            spec["canonical"] = canonical_name(spec["name"])
        spec.setdefault("overrides", {})
        out.append(spec)
    return out


def seed_list(config: dict) -> list[int]:
    if "seeds" in config:
        return [int(s) for s in config["seeds"]]
    off = int(config.get("seed_offset", 0))
    return [off + i for i in range(1, int(config.get("n_seeds", 30)) + 1)]


# ------------------------------------------------------------------ one cell
def build_run(config: dict, params: dict, algo: dict, seed: int, record_beliefs=True):
    T = int(config["T"])
    R = float(config.get("R", 1.0))
    env_ss, pol_ss, prof_ss = np.random.SeedSequence(int(seed)).spawn(3)
    spec = dict(config.get("reliability", {"type": "stationary"}))
    for key in RELIABILITY_KEYS:          # a sweep point may override reliability settings
        if key in params:
            spec[key] = params[key]
    profile = make_profile(spec, params["theta"], T, seed=prof_ss)
    env = SimulationEnvironment(profile, R=R, U_self=params["U_self"], c=params["c"],
                                lambda_mult=params["lambda"], seed=env_ss)
    pol = make_policy(algo["name"], K=env.K, R=R, lambda_mult=params["lambda"], c=params["c"],
                      gamma=params.get("gamma", 1.0), seed=pol_ss,
                      alpha_ucb=params.get("alpha_ucb", 1.0),
                      theta_initial=profile.get(1), U_self=params["U_self"],
                      overrides=algo.get("overrides"))
    return SimulationRun(env, pol, T, record_beliefs=record_beliefs), profile


def run_cell(args):
    """Run every seed of one (sweep point, algorithm) cell.  Top-level for pickling."""
    config, p_idx, point, algo, seeds = args
    params = point["params"]
    T = int(config["T"])
    R = float(config.get("R", 1.0))
    rows = []
    K = len(params["theta"])
    acc = {"cum_regret": np.zeros(T), "cum_regret_sq": np.zeros(T), "ask": np.zeros(T),
           "opt_action": np.zeros(T), "action_share": np.zeros((K + 1, T)),
           "theta_hat": np.zeros((T, K)), "mae": np.zeros(T), "theta_true": np.zeros((T, K))}
    track_belief = algo["canonical"] not in NO_BELIEF_POLICIES
    t0 = time.time()
    for seed in seeds:
        sim, profile = build_run(config, params, algo, seed)
        rec = sim.run()
        tau = profile.change_point
        if tau is not None and tau >= T:      # change point outside the horizon
            tau = None
        oracle_after = None
        if tau is not None:
            oracle_after = dynamic_oracle(profile.get(tau + 1), params["U_self"], R=R,
                                          lambda_mult=params["lambda"], c=params["c"])
        m = summarize_run(rec, change_point=tau, oracle_after=oracle_after)
        n_changes = getattr(profile, "n_changes", None)
        if n_changes is not None:
            m["n_changes"] = float(n_changes)
        if not track_belief:
            m["mae_theta"] = float("nan")
            m["mae_theta_final"] = float("nan")
        rows.append({"experiment": config["name"], "point": p_idx, "point_label": point["label"],
                     "point_value": point["value"], "algorithm": algo["label"], "seed": seed, **m})
        cr = np.cumsum(rec.regret)
        acc["cum_regret"] += cr
        acc["cum_regret_sq"] += cr * cr
        acc["ask"] += rec.d
        acc["opt_action"] += rec.opt_action
        act = rec.action
        for j in range(K + 1):
            acc["action_share"][j] += (act == j)
        acc["theta_true"] += rec.theta_true
        if track_belief:
            acc["theta_hat"] += rec.theta_hat
            acc["mae"] += mae_curve(rec)
    n = len(seeds)
    mean_cr = acc["cum_regret"] / n
    var_cr = np.maximum(acc["cum_regret_sq"] / n - mean_cr ** 2, 0.0) * n / max(n - 1, 1)
    curves = {"cum_regret_mean": mean_cr, "cum_regret_se": np.sqrt(var_cr / n),
              "ask": acc["ask"] / n, "opt_action": acc["opt_action"] / n,
              "action_share": acc["action_share"] / n, "theta_true": acc["theta_true"] / n}
    if track_belief:
        curves["theta_hat"] = acc["theta_hat"] / n
        curves["mae"] = acc["mae"] / n
    curves = {k: v.astype(np.float32) for k, v in curves.items()}
    return p_idx, algo["label"], rows, curves, time.time() - t0


def run_cell_multiagent(args):
    """Run every seed of one multi-agent (sweep point, algorithm) cell."""
    config, p_idx, point, algo, seeds = args
    params = point["params"]
    T = int(config["T"])
    R = float(config.get("R", 1.0))
    n_agents = int(config.get("n_agents", len(params["theta"])))
    rows = []
    t0 = time.time()

    acc = {
        "cum_regret": np.zeros(T),
        "cum_regret_sq": np.zeros(T),
        "ask": np.zeros(T),
        "opt_action": np.zeros(T),
        "workload": np.zeros(T),
        "resources": np.zeros(T),
    }

    helper_policy = params.get("helper_policy", "q_learning")

    for seed in seeds:
        env_ss, pol_ss, prof_ss = np.random.SeedSequence(int(seed)).spawn(3)
        prof_seeds = prof_ss.spawn(n_agents)
        spec = dict(config.get("reliability", {"type": "stationary"}))
        for key in RELIABILITY_KEYS:
            if key in params:
                spec[key] = params[key]

        thetas = params["theta"]
        profiles = []
        for i in range(n_agents):
            th = thetas[i] if isinstance(thetas[i], (list, np.ndarray)) else [thetas[i]]
            agent_spec = dict(spec)
            if "theta_after" in spec and isinstance(spec["theta_after"], (list, np.ndarray)):
                if len(spec["theta_after"]) == n_agents:
                    th_after = spec["theta_after"][i]
                    agent_spec["theta_after"] = th_after if isinstance(th_after, (list, np.ndarray)) else [th_after]
            profiles.append(make_profile(agent_spec, th, T, seed=prof_seeds[i]))

        env = MultiAgentEnvironment(
            n_agents=n_agents,
            reliability_profiles=profiles,
            R=R,
            U_self=params["U_self"],
            lambda_mult=params.get("lambda", 1.0),
            comm_costs=np.array(params["c"]),
            helper_rewards=params.get("helper_rewards"),
            requester_policy_type=algo["name"],
            requester_policy_params={"gamma": params.get("gamma", 1.0), **algo.get("overrides", {})},
            helper_policy_type=helper_policy,
            seed=seed,
            requester_selection="round_robin",
        )
        sim = MultiAgentSimulationRun(env, T=T, record_beliefs=True)
        rec = sim.run()
        m = compute_multiagent_metrics(rec, env)

        tau = spec.get("tau")
        if tau is not None and 0 < int(tau) < T:
            tau_int = int(tau)
            m["pre_change_regret"] = float(rec.regrets[:tau_int].sum())
            m["post_change_regret"] = float(rec.regrets[tau_int:].sum())
            m["post_change_optimal_action_rate"] = float(rec.opt_actions[tau_int:].mean())
            m["post_change_ask_rate"] = float(np.mean([1.0 if r.asked else 0.0 for r in rec.interactions[tau_int:]]))

        rows.append({
            "experiment": config["name"],
            "point": p_idx,
            "point_label": point["label"],
            "point_value": point["value"],
            "algorithm": algo["label"],
            "seed": seed,
            **m,
        })

        cr = np.cumsum(rec.regrets)
        acc["cum_regret"] += cr
        acc["cum_regret_sq"] += cr * cr
        asked_arr = np.array([1.0 if r.asked else 0.0 for r in rec.interactions])
        acc["ask"] += asked_arr
        acc["opt_action"] += rec.opt_actions.astype(float)
        acc["workload"] += rec.workloads.mean(axis=1)
        acc["resources"] += rec.resources.mean(axis=1)

    n = len(seeds)
    mean_cr = acc["cum_regret"] / n
    var_cr = np.maximum(acc["cum_regret_sq"] / n - mean_cr ** 2, 0.0) * n / max(n - 1, 1)
    curves = {
        "cum_regret_mean": mean_cr.astype(np.float32),
        "cum_regret_se": np.sqrt(var_cr / n).astype(np.float32),
        "ask": (acc["ask"] / n).astype(np.float32),
        "opt_action": (acc["opt_action"] / n).astype(np.float32),
        "workload": (acc["workload"] / n).astype(np.float32),
        "resources": (acc["resources"] / n).astype(np.float32),
    }
    return p_idx, algo["label"], rows, curves, time.time() - t0


def run_cell_temporal(args):
    """Run every seed of one temporal (sweep point, algorithm) cell."""
    config, p_idx, point, algo, seeds = args
    params = point["params"]
    T = int(config["T"])
    R = float(config.get("R", 1.0))
    n_agents = int(config.get("n_agents", len(params["theta"])))
    default_deadline = int(config.get("default_deadline", 5))
    step_cost = float(config.get("step_cost", 0.02))
    c_fail = float(config.get("c_fail", 0.5))
    rows = []
    t0 = time.time()

    acc = {
        "completed": np.zeros(T),
        "reward": np.zeros(T),
        "ask": np.zeros(T),
    }

    timing_policy_type = algo.get("timing_policy", "learned")
    timing_params = algo.get("timing_params", {})
    requester_policy = params.get("requester_policy", "MultiAgent-D-TS")

    for seed in seeds:
        env_ss, pol_ss, prof_ss = np.random.SeedSequence(int(seed)).spawn(3)
        prof_seeds = prof_ss.spawn(n_agents)
        spec = dict(config.get("reliability", {"type": "stationary"}))
        for key in RELIABILITY_KEYS:
            if key in params:
                spec[key] = params[key]

        thetas = params["theta"]
        profiles = []
        for i in range(n_agents):
            th = thetas[i] if isinstance(thetas[i], (list, np.ndarray)) else [thetas[i]]
            agent_spec = dict(spec)
            if "theta_after" in spec and isinstance(spec["theta_after"], (list, np.ndarray)):
                if len(spec["theta_after"]) == n_agents:
                    agent_spec["theta_after"] = [spec["theta_after"][i]]
            profiles.append(make_profile(agent_spec, th, T, seed=prof_seeds[i]))

        env = TemporalMultiAgentEnvironment(
            n_agents=n_agents,
            reliability_profiles=profiles,
            R=R,
            U_self=params["U_self"],
            lambda_mult=params.get("lambda", 1.0),
            comm_costs=np.array(params["c"]),
            helper_rewards=params.get("helper_rewards"),
            requester_policy_type=requester_policy,
            requester_policy_params={"gamma": params.get("gamma", 0.98)},
            timing_policy_type=timing_policy_type,
            timing_policy_params=timing_params,
            default_deadline=default_deadline,
            step_cost=step_cost,
            c_fail=c_fail,
            solo_capability=config.get("solo_capability", config.get("kappa_solo", params.get("solo_capability", 0.60))),
            seed=int(seed),
        )

        records = env.run(n_tasks=T)
        m = compute_temporal_metrics(records, env)

        rows.append({
            "experiment": config["name"],
            "point": p_idx,
            "point_label": point["label"],
            "point_value": point["value"],
            "algorithm": algo["label"],
            "seed": seed,
            **m,
        })

        for t_idx, r in enumerate(records):
            if r.completed:
                acc["completed"][t_idx] += 1.0
            acc["reward"][t_idx] += r.requester_reward
            if r.help_requested:
                acc["ask"][t_idx] += 1.0

    n = len(seeds)
    curves = {
        "completion_rate": (acc["completed"] / n).astype(np.float32),
        "reward_mean": (acc["reward"] / n).astype(np.float32),
        "ask_rate": (acc["ask"] / n).astype(np.float32),
    }
    return p_idx, algo["label"], rows, curves, time.time() - t0


# ------------------------------------------------------------- aggregation
SUMMARY_METRICS = [
    "cumulative_regret", "average_net_utility", "average_expected_utility", "ask_rate",
    "ask_accuracy", "optimal_teammate_rate", "optimal_action_rate", "communication_cost",
    "mae_theta", "mae_theta_final", "share_alone", "share_h1", "share_h2", "share_h3",
    "adaptation_latency", "pre_change_regret", "post_change_regret", "post_change_utility",
    "post_change_optimal_teammate_rate", "post_change_optimal_action_rate", "post_change_ask_rate",
    "n_changes",
    "acceptance_rate", "rejection_rate", "success_rate", "task_completion_rate",
    "deadline_failure_rate", "assistance_success_rate", "avg_steps_per_task",
    "avg_solo_attempts", "avg_time_to_request", "avg_communication_cost", "timing_policy_stability",
    "timing_states_evaluated",
    "avg_workload", "avg_resources", "total_communication_cost", "avg_requester_reward",
    "avg_helper_reward", "avg_request_frequency", "avg_cooperation_frequency",
    "avg_acceptance_rate_per_agent", "requester_net_utility", "requester_utility",
    "helper_utility", "social_welfare", "helper_policy_convergence", "helper_policy_stability",
    "helper_states_evaluated",
    "low_w_high_r_visits", "low_w_high_r_accept_rate", "low_w_high_r_reject_rate",
    "med_w_med_r_visits", "med_w_med_r_accept_rate", "med_w_med_r_reject_rate",
    "high_w_low_r_visits", "high_w_low_r_accept_rate", "high_w_low_r_reject_rate",
    "agent_0_acceptance_rate", "agent_1_acceptance_rate", "agent_2_acceptance_rate", "agent_3_acceptance_rate",
    "agent_0_help_frequency", "agent_1_help_frequency", "agent_2_help_frequency", "agent_3_help_frequency",
    "agent_0_incoming_request_frequency", "agent_1_incoming_request_frequency", "agent_2_incoming_request_frequency", "agent_3_incoming_request_frequency",
    "agent_0_helper_reward", "agent_1_helper_reward", "agent_2_helper_reward", "agent_3_helper_reward",
]


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["point"], r["point_label"], r["point_value"], r["algorithm"]), []).append(r)
    out = []
    for (p, lab, val, algo), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][3])):
        for metric in SUMMARY_METRICS:
            if metric not in rs[0]:
                continue
            vals = [float(r[metric]) if (r.get(metric) is not None and not (isinstance(r[metric], float) and np.isnan(r[metric]))) else float("nan") for r in rs]
            s = mean_ci(vals)
            row = {"experiment": rs[0]["experiment"], "point": p, "point_label": lab,
                   "point_value": val, "algorithm": algo, "metric": metric, **s}
            if metric == "adaptation_latency":
                row["median"] = float(np.median(vals))
                row["n_censored"] = int(sum(bool(r["adaptation_censored"]) for r in rs))
            out.append(row)
    return out


def _write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")


# ------------------------------------------------------------------- driver
def run_experiment(config, out_dir: str = "results", workers: int = 1, n_seeds: int | None = None,
                   T: int | None = None, save: bool = True, verbose: bool = True) -> dict:
    """Run every (sweep point x algorithm x seed) of a config and save the results."""
    config = load_config(config)
    if T is not None:
        config["T"] = int(T)
    if n_seeds is not None:
        config["n_seeds"] = int(n_seeds)
        config.pop("seeds", None)
    seeds = seed_list(config)
    points = sweep_points(config)
    algos = algorithm_specs(config)
    tasks = [(config, i, pt, algo, seeds) for i, pt in enumerate(points) for algo in algos]
    is_temporal = config.get("temporal", False)
    is_multiagent = "n_agents" in config or config.get("multiagent", False)
    if is_temporal:
        cell_fn = run_cell_temporal
    elif is_multiagent:
        cell_fn = run_cell_multiagent
    else:
        cell_fn = run_cell

    t0 = time.time()
    rows, curves = [], {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(cell_fn, tasks))
    else:
        results = [cell_fn(t) for t in tasks]
    for p_idx, label, r, c, dt in results:
        rows.extend(r)
        for k, v in c.items():
            curves[f"p{p_idx}__{_slug(label)}__{k}"] = v
        if verbose:
            print(f"  [{config['name']}] {points[p_idx]['label']:>14s} | {label:<32s} {dt:6.1f}s", flush=True)
    rows.sort(key=lambda r: (r["point"], r["algorithm"], r["seed"]))
    summary = aggregate(rows)
    elapsed = time.time() - t0

    if save:
        raw_dir = os.path.join(out_dir, "raw", config["name"])
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "processed"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "manifests"), exist_ok=True)
        _write_csv(os.path.join(raw_dir, "runs.csv"), rows)
        np.savez_compressed(os.path.join(raw_dir, "curves.npz"), **curves)
        _write_csv(os.path.join(out_dir, "processed", f"{config['name']}_summary.csv"), summary)
        manifest = {
            "experiment": config["name"], "title": config.get("title"),
            "config": config, "seeds": seeds,
            "points": [{"index": i, "label": p["label"], "value": p["value"], "params": p["params"]}
                       for i, p in enumerate(points)],
            "algorithms": [{"label": a["label"], "canonical": a["canonical"],
                            "overrides": a["overrides"], "slug": _slug(a["label"])} for a in algos],
            "n_runs": len(rows), "runtime_seconds": round(elapsed, 2),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python": platform.python_version(), "numpy": np.__version__,
            "files": {"runs": f"raw/{config['name']}/runs.csv",
                      "curves": f"raw/{config['name']}/curves.npz",
                      "summary": f"processed/{config['name']}_summary.csv"},
        }
        with open(os.path.join(out_dir, "manifests", f"{config['name']}.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
    if verbose:
        print(f"[{config['name']}] {len(rows)} runs in {elapsed:.1f}s", flush=True)
    return {"rows": rows, "summary": summary, "curves": curves, "points": points, "algorithms": algos}


def expand_scenarios(config) -> list[dict]:
    """A config may define ``scenarios``: named overrides of base / reliability.
    Returns one concrete config per scenario (or the config itself)."""
    config = load_config(config)
    scen = config.pop("scenarios", None)
    if not scen:
        return [config]
    out = []
    for name, ov in scen.items():
        c = deepcopy(config)
        c["name"] = f"{config['name']}_{name}"
        c["scenario"] = name
        if "base" in ov:
            base = dict(c["base"])
            base.update(ov["base"])
            c["base"] = base
        if "reliability" in ov:
            c["reliability"] = ov["reliability"]
        out.append(c)
    return out
