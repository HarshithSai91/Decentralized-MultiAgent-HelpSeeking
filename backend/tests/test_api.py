"""API smoke tests: the dashboard endpoints return real simulation state."""
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from help_seeking.api import app  # noqa: E402

client = TestClient(app)


def test_start_step_reset_cycle():
    s = client.post("/api/sim/start", json={"algorithm": "D-TS", "preset": "E3", "T": 50, "seed": 3}).json()
    sid = s["session_id"]
    assert s["t"] == 0 and s["theta_hat"] == [0.5, 0.5, 0.5]
    r = client.post(f"/api/sim/{sid}/step", json={"n": 10}).json()
    assert r["state"]["t"] == 10 and len(r["rounds"]) == 10
    last = r["last_round"]
    assert last["t"] == 10 and "theta_sample" in last["extra"]
    assert r["state"]["totals"]["regret"] >= 0
    assert client.post(f"/api/sim/{sid}/reset").json()["t"] == 0


def test_same_seed_same_history_via_api():
    def run():
        sid = client.post("/api/sim/start", json={"algorithm": "D-UCB", "preset": "E1", "T": 30, "seed": 9}).json()["session_id"]
        return client.post(f"/api/sim/{sid}/step", json={"n": 30}).json()["rounds"]
    assert run() == run()


def test_compare_oracle_has_zero_regret():
    res = client.post("/api/compare", json={"preset": "E4", "T": 200, "seed": 1,
                                            "algorithms": ["Dynamic Oracle", "D-TS"]}).json()["results"]
    assert res[0]["metrics"]["cumulative_regret"] == 0
    assert res[1]["cum_regret"][-1] >= 0


def test_unknown_algorithm_rejected():
    assert client.post("/api/sim/start", json={"algorithm": "Nope"}).status_code == 400


def test_saved_experiment_endpoint_is_json_safe():
    names = [e["name"] for e in client.get("/api/experiments").json()["experiments"]]
    if not names:
        pytest.skip("no saved results yet")
    r = client.get(f"/api/experiments/{names[0]}")
    assert r.status_code == 200 and r.json()["summary"]


def test_multiagent_api_cycle():
    """Verify multi-agent simulation API endpoints."""
    res = client.post("/api/multiagent/sim/new", json={
        "n_agents": 3,
        "requester_policy": "D-TS",
        "helper_policy": "q_learning",
        "T": 100,
        "seed": 42,
    })
    assert res.status_code == 200
    data = res.json()
    sid = data["session_id"]
    state = data["state"]
    assert state["t"] == 0
    assert state["n_agents"] == 3
    assert len(state["agents"]) == 3

    # Step 10 rounds
    step_res = client.post(f"/api/multiagent/sim/{sid}/step", json={"n": 10})
    assert step_res.status_code == 200
    step_data = step_res.json()
    assert step_data["state"]["t"] == 10
    assert len(step_data["history"]) == 10
    assert "metrics" in step_data["state"]

    # Reset
    reset_res = client.post(f"/api/multiagent/sim/{sid}/reset")
    assert reset_res.status_code == 200
    assert reset_res.json()["t"] == 0



def test_multiagent_live_metrics_use_rounds_played_so_far():
    """Mid-run dashboard metrics are rates over the rounds played, not over the horizon T."""
    sid = client.post("/api/multiagent/sim/new", json={"seed": 3}).json()["state"]["session_id"]
    st = client.post(f"/api/multiagent/sim/{sid}/step", json={"n": 40}).json()["state"]
    assert st["t"] == 40
    m = st["metrics"]
    # each round has exactly one requester, so the per-agent request frequency over 40 rounds is 1/4
    assert abs(m["avg_request_frequency"] - 0.25) < 1e-9
    # resources start at 2 and live in [0, 2]; a mean computed over 2000 pre-allocated rounds would be ~0.04
    assert 0.5 < m["avg_resources"] <= 2.0


def test_multiagent_metrics_before_first_round_describe_initial_state():
    st = client.post("/api/multiagent/sim/new", json={"T": 100, "seed": 1}).json()["state"]
    assert st["t"] == 0
    m = st["metrics"]
    assert m["total_rounds"] == 0
    assert m["avg_workload"] == 0.0 and m["avg_resources"] == 2.0   # all agents start idle with full resources
    assert m["ask_rate"] == 0.0


def test_built_dashboard_is_served_when_present():
    import os
    from help_seeking.api import ROOT
    if not os.path.isdir(os.path.join(ROOT, "frontend", "dist")):
        pytest.skip("frontend/dist not built")
    r = client.get("/")
    assert r.status_code == 200 and 'id="root"' in r.text
    # API routes still take precedence over the static mount
    assert client.get("/api/status").json()["status"] == "ok"
    assert client.post("/api/multiagent/sim/new", json={"T": 10}).status_code == 200
