# backend/src/help_seeking/api.py
"""FastAPI backend for the interactive dashboard.

Every number the dashboard shows comes from the same simulation engine used
for the experiments (``SimulationRun``) or from the saved experiment results;
nothing is simulated in the browser.

Run:  uvicorn help_seeking.api:app --app-dir backend/src --reload --port 8000
"""
from __future__ import annotations

import csv
import json
import os
import uuid
from collections import OrderedDict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .environment import SimulationEnvironment
from .metrics import summarize_run
from .multiagent_environment import MultiAgentEnvironment
from .multiagent_simulation import MultiAgentRunRecord, MultiAgentSimulationRun, compute_multiagent_metrics
from .policies import CANONICAL, LEARNING_METHODS, canonical_name, make_policy
from .reliability import StationaryProfile, make_profile
from .simulation import SimulationRun
from .utilities import dynamic_oracle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESULTS = os.path.join(ROOT, "results")

app = FastAPI(title="Help-Seeking Simulation API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

PRESETS = {
    "E1": {"label": "E1 stationary (Δ = 0.10)", "theta": [0.70, 0.60, 0.50], "c": [0.1, 0.1, 0.1],
           "U_self": 0.35, "lambda": 1.0, "gamma": 1.0, "reliability": {"type": "stationary"}},
    "E2": {"label": "E2 when to ask (U_self = 0.80)", "theta": [0.75, 0.65, 0.50], "c": [0.1, 0.1, 0.1],
           "U_self": 0.80, "lambda": 1.0, "gamma": 0.98, "reliability": {"type": "stationary"}},
    "E3": {"label": "E3 abrupt change at t = 1000", "theta": [0.80, 0.70, 0.90], "c": [0.1, 0.1, 0.1],
           "U_self": 0.50, "lambda": 1.0, "gamma": 0.98,
           "reliability": {"type": "abrupt", "theta_after": [0.80, 0.70, 0.40], "tau": 1000}},
    "E4": {"label": "E4 cost variation (λ = 2.5)", "theta": [0.60, 0.75, 0.90], "c": [0.05, 0.10, 0.20],
           "U_self": 0.45, "lambda": 2.5, "gamma": 0.98, "reliability": {"type": "stationary"}},
    "S5": {"label": "S5 random changes (p_c = 0.0025)", "theta": [0.70, 0.60, 0.50], "c": [0.1, 0.1, 0.1],
           "U_self": 0.35, "lambda": 1.0, "gamma": 0.99,
           "reliability": {"type": "multishift", "p_c": 0.0025, "low": 0.2, "high": 0.95}},
}


class SimConfig(BaseModel):
    algorithm: str = "D-TS"
    preset: str | None = "E3"
    theta: list[float] | None = None
    c: list[float] | None = None
    U_self: float | None = None
    lambda_mult: float | None = Field(default=None, alias="lambda")
    gamma: float | None = None
    alpha_ucb: float = 1.0
    reliability: dict | None = None
    T: int = Field(default=2000, ge=1, le=20000)
    seed: int = 1

    model_config = {"populate_by_name": True}

    def resolved(self) -> dict:
        base = dict(PRESETS.get(self.preset or "", PRESETS["E1"]))
        for key, val in (("theta", self.theta), ("c", self.c), ("U_self", self.U_self),
                         ("lambda", self.lambda_mult), ("gamma", self.gamma), ("reliability", self.reliability)):
            if val is not None:
                base[key] = val
        base.pop("label", None)
        base["alpha_ucb"] = self.alpha_ucb
        return base


def build(cfg: SimConfig, algorithm: str | None = None) -> SimulationRun:
    p = cfg.resolved()
    env_ss, pol_ss, prof_ss = np.random.SeedSequence(int(cfg.seed)).spawn(3)
    profile = make_profile(p["reliability"], p["theta"], cfg.T, seed=prof_ss)
    env = SimulationEnvironment(profile, R=1.0, U_self=p["U_self"], c=p["c"],
                                lambda_mult=p["lambda"], seed=env_ss)
    pol = make_policy(algorithm or cfg.algorithm, K=env.K, R=1.0, lambda_mult=p["lambda"], c=p["c"],
                      gamma=p["gamma"], seed=pol_ss, alpha_ucb=p["alpha_ucb"],
                      theta_initial=profile.get(1), U_self=p["U_self"])
    return SimulationRun(env, pol, cfg.T)


SESSIONS: "OrderedDict[str, dict]" = OrderedDict()
MAX_SESSIONS = 50


def _session(sid: str) -> dict:
    if sid not in SESSIONS:
        raise HTTPException(404, "unknown session")
    return SESSIONS[sid]


def _state(sess: dict) -> dict:
    sim: SimulationRun = sess["sim"]
    rec = sim.record()
    n = rec.T
    env, pol = sim.env, sim.policy
    return {
        "session_id": sess["id"], "algorithm": pol.name, "t": n, "T": sim.T, "done": sim.done,
        "U_self": env.U_self, "costs": env.costs().tolist(),
        "theta_true_now": env.theta(max(n, 1)).tolist(),
        "theta_hat": pol.belief().tolist(), "S": pol.trust.S.tolist(), "F": pol.trust.F.tolist(),
        "totals": {
            "regret": float(rec.regret.sum()) if n else 0.0,
            "utility": float(rec.r.sum()) if n else 0.0,
            "comm_cost": float(rec.cost.sum()) if n else 0.0,
            "asks": int(rec.d.sum()) if n else 0,
            "optimal_actions": int(rec.opt_action.sum()) if n else 0,
        },
        "change_point": env.profile.change_point,
        "config": sess["config"],
    }


@app.get("/api/status")
def status():
    return {"status": "ok", "algorithms": list(CANONICAL), "presets": list(PRESETS)}


@app.get("/api/presets")
def presets():
    return {"presets": PRESETS, "algorithms": list(CANONICAL), "learning": list(LEARNING_METHODS)}


@app.post("/api/sim/start")
def sim_start(cfg: SimConfig):
    try:
        canonical_name(cfg.algorithm)
        sim = build(cfg)
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc))
    sid = uuid.uuid4().hex[:12]
    SESSIONS[sid] = {"id": sid, "sim": sim, "config": cfg.model_dump(by_alias=True), "last": None}
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)
    return _state(SESSIONS[sid])


class StepRequest(BaseModel):
    n: int = Field(default=1, ge=1, le=20000)


@app.post("/api/sim/{sid}/step")
def sim_step(sid: str, req: StepRequest):
    sess = _session(sid)
    sim: SimulationRun = sess["sim"]
    if sim.done:
        raise HTTPException(409, "horizon reached; reset or start a new session")
    start = sim.t
    last = None
    for i in range(req.n):
        if sim.done:
            break
        last = sim.step(details=(i == req.n - 1) or sim.t + 1 == sim.T)
    sess["last"] = last
    rec = sim.record()
    return {"state": _state(sess), "last_round": last, "rounds": rec.history(start, sim.t)}


@app.post("/api/sim/{sid}/reset")
def sim_reset(sid: str):
    sess = _session(sid)
    cfg = SimConfig(**sess["config"])
    sess["sim"] = build(cfg)
    sess["last"] = None
    return _state(sess)


@app.get("/api/sim/{sid}/state")
def sim_state(sid: str):
    sess = _session(sid)
    return {**_state(sess), "last_round": sess["last"]}


@app.get("/api/sim/{sid}/history")
def sim_history(sid: str, start: int = 0, end: int | None = None):
    sim: SimulationRun = _session(sid)["sim"]
    return {"rounds": sim.record().history(start, end)}


class CompareRequest(SimConfig):
    algorithms: list[str] = ["Dynamic Oracle", "Greedy", "D-UCB", "D-TS"]
    points: int = Field(default=200, ge=10, le=2000)


@app.post("/api/compare")
def compare(req: CompareRequest):
    """Run several algorithms on the same configuration and seed (paired by seed)."""
    out = []
    p = req.resolved()
    for name in req.algorithms:
        try:
            sim = build(req, name)
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, str(exc))
        rec = sim.run()
        tau = sim.env.profile.change_point
        after = None
        if tau is not None and tau < rec.T:
            after = dynamic_oracle(sim.env.profile.get(tau + 1), p["U_self"], lambda_mult=p["lambda"], c=p["c"])
        else:
            tau = None
        m = summarize_run(rec, change_point=tau, oracle_after=after)
        idx = np.unique(np.linspace(0, rec.T - 1, req.points).astype(int))
        cr = np.cumsum(rec.regret)
        ask = np.convolve(rec.d.astype(float), np.ones(50) / 50, mode="full")[: rec.T]
        out.append({"algorithm": name, "metrics": {k: (None if isinstance(v, float) and np.isnan(v) else v)
                                                   for k, v in m.items()},
                    "t": (idx + 1).tolist(), "cum_regret": cr[idx].round(3).tolist(),
                    "ask_rate_50": ask[idx].round(3).tolist()})
    return {"config": req.model_dump(by_alias=True), "results": out}


# ------------------------------------------------------- saved experiments
@app.get("/api/experiments")
def experiments():
    mdir = os.path.join(RESULTS, "manifests")
    items = []
    if os.path.isdir(mdir):
        for f in sorted(os.listdir(mdir)):
            if f.endswith(".json"):
                with open(os.path.join(mdir, f), encoding="utf-8") as fh:
                    m = json.load(fh)
                items.append({"name": m["experiment"], "title": m.get("title"), "n_runs": m["n_runs"],
                              "points": [p["label"] for p in m["points"]],
                              "algorithms": [a["label"] for a in m["algorithms"]],
                              "completed_at": m.get("completed_at")})
    return {"experiments": items}


@app.get("/api/experiments/{name}")
def experiment(name: str):
    path = os.path.join(RESULTS, "processed", f"{name}_summary.csv")
    mpath = os.path.join(RESULTS, "manifests", f"{name}.json")
    if not (os.path.exists(path) and os.path.exists(mpath)):
        raise HTTPException(404, "experiment not found; run scripts/run_all_experiments.py first")
    with open(mpath, encoding="utf-8") as fh:
        manifest = json.load(fh)
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in ("mean", "std", "se", "ci_low", "ci_high", "point_value", "median"):
                if k in r and r[k] not in ("", None):
                    try:
                        v = float(r[k])
                        r[k] = None if np.isnan(v) else v
                    except ValueError:
                        pass
            r["point"] = int(r["point"])
            rows.append(r)
    return {"manifest": manifest, "summary": rows}


# ------------------------------------------------------- multi-agent simulation API
MA_SESSIONS: "OrderedDict[str, dict]" = OrderedDict()


class MultiAgentSimConfig(BaseModel):
    n_agents: int = 4
    requester_policy: str = "D-TS"
    helper_policy: str = "q_learning"
    thetas: list[float] | None = None
    U_self: float = 0.5
    lambda_mult: float = Field(default=1.0, alias="lambda")
    B_i: float = 0.5
    resource_cost: float = 0.1
    opportunity_cost: float = 0.05
    gamma: float = 0.98
    T: int = Field(default=2000, ge=1, le=20000)
    seed: int = 1

    model_config = {"populate_by_name": True}


def build_multiagent(cfg: MultiAgentSimConfig) -> MultiAgentSimulationRun:
    n = cfg.n_agents
    thetas = cfg.thetas or [0.7, 0.6, 0.5, 0.8][:n]
    if len(thetas) < n:
        thetas = (thetas * ((n // len(thetas)) + 1))[:n]

    profiles = [StationaryProfile([th]) for th in thetas]
    comm_costs = np.full((n, n), 0.1)
    np.fill_diagonal(comm_costs, 0.0)

    env = MultiAgentEnvironment(
        n_agents=n,
        reliability_profiles=profiles,
        R=1.0,
        U_self=cfg.U_self,
        lambda_mult=cfg.lambda_mult,
        comm_costs=comm_costs,
        helper_rewards={
            "B_i": cfg.B_i,
            "resource_cost": cfg.resource_cost,
            "opportunity_cost": cfg.opportunity_cost,
        },
        requester_policy_type=cfg.requester_policy,
        requester_policy_params={"gamma": cfg.gamma},
        helper_policy_type=cfg.helper_policy,
        seed=cfg.seed,
    )
    return MultiAgentSimulationRun(env, T=cfg.T, record_beliefs=True)


def _ma_session(sid: str) -> dict:
    if sid not in MA_SESSIONS:
        raise HTTPException(404, "unknown multi-agent session")
    return MA_SESSIONS[sid]


def _record_so_far(sim: MultiAgentSimulationRun) -> MultiAgentRunRecord:
    """View of the run record restricted to the rounds played so far.

    The record pre-allocates arrays for all T rounds, so computing live metrics on the
    full record would divide by T instead of by the number of rounds actually played.
    """
    rec = sim.record
    t = int(sim.t)
    if t >= rec.T:
        return rec
    if t <= 0:
        return MultiAgentRunRecord(T=0, n_agents=rec.n_agents)
    return MultiAgentRunRecord(
        T=t, n_agents=rec.n_agents, interactions=list(rec.interactions[:t]),
        requester_rewards=rec.requester_rewards[:t], helper_rewards=rec.helper_rewards[:t],
        workloads=rec.workloads[:t], resources=rec.resources[:t],
        execution_reliabilities=rec.execution_reliabilities[:t],
        regrets=rec.regrets[:t], mu_stars=rec.mu_stars[:t], opt_actions=rec.opt_actions[:t],
        acceptance_beliefs=rec.acceptance_beliefs[:t] if rec.acceptance_beliefs.size else rec.acceptance_beliefs,
        trust_beliefs=rec.trust_beliefs[:t] if rec.trust_beliefs.size else rec.trust_beliefs,
        q_snapshots=list(rec.q_snapshots),
    )


def _ma_state(sess: dict) -> dict:
    sim: MultiAgentSimulationRun = sess["sim"]
    env = sim.env
    metrics = compute_multiagent_metrics(_record_so_far(sim), env)
    if sim.t == 0:
        # Before the first round the averages describe the agents' initial state.
        metrics["avg_workload"] = float(np.mean([a.workload for a in env.agents]))
        metrics["avg_resources"] = float(np.mean([a.resources for a in env.agents]))
    return {
        "session_id": sess["id"],
        "t": sim.t,
        "T": sim.T,
        "done": sim.done,
        "n_agents": env.n_agents,
        "requester_policy": sess["config"].get("requester_policy", "D-TS"),
        "helper_policy": sess["config"].get("helper_policy", "q_learning"),
        "agents": env.get_agent_snapshots(),
        "metrics": metrics,
    }


@app.post("/api/multiagent/sim/new")
def ma_sim_new(cfg: MultiAgentSimConfig):
    sim = build_multiagent(cfg)
    sid = uuid.uuid4().hex[:12]
    sess = {"id": sid, "sim": sim, "config": cfg.model_dump(by_alias=True), "last_interaction": None}
    MA_SESSIONS[sid] = sess
    if len(MA_SESSIONS) > MAX_SESSIONS:
        MA_SESSIONS.popitem(last=False)
    return {"session_id": sid, "state": _ma_state(sess)}


@app.post("/api/multiagent/sim/{sid}/step")
def ma_sim_step(sid: str, req: StepRequest):
    sess = _ma_session(sid)
    sim: MultiAgentSimulationRun = sess["sim"]
    if sim.done:
        raise HTTPException(409, "horizon reached; reset or start a new session")
    start = sim.t
    last = None
    for _ in range(req.n):
        if sim.done:
            break
        last = sim.step()
    sess["last_interaction"] = last
    history = sim.record.history(start, sim.t)
    return {
        "state": _ma_state(sess),
        "last_interaction": {
            "round": last.round_num,
            "requester_id": last.requester_id,
            "helper_id": last.helper_id,
            "asked": last.asked,
            "accepted": last.accepted,
            "success": last.success,
            "requester_reward": float(last.requester_reward),
            "helper_reward": float(last.helper_reward),
            "comm_cost": float(last.communication_cost),
        } if last else None,
        "history": history,
    }


@app.get("/api/multiagent/sim/{sid}/state")
def ma_sim_state(sid: str):
    sess = _ma_session(sid)
    return _ma_state(sess)


@app.post("/api/multiagent/sim/{sid}/reset")
def ma_sim_reset(sid: str):
    sess = _ma_session(sid)
    cfg = MultiAgentSimConfig(**sess["config"])
    sess["sim"] = build_multiagent(cfg)
    sess["last_interaction"] = None
    return _ma_state(sess)


# ---------------------------------------------------------------------------
# Optional: serve the built dashboard (frontend/dist) from the same server, so
#   uvicorn help_seeking.api:app --app-dir backend/src --port 8000
# is enough and the dashboard is at http://localhost:8000/.
# Mounted last, so every /api route above keeps precedence.
# ---------------------------------------------------------------------------
_DIST_DIR = os.path.join(ROOT, "frontend", "dist")
if os.path.isdir(_DIST_DIR):
    app.mount("/", StaticFiles(directory=_DIST_DIR, html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return status()
