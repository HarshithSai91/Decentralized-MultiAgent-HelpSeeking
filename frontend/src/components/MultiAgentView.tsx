import React, { useState } from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, MultiAgentConfig, MultiAgentInteraction, MultiAgentState } from "../api";

const fmt = (x: number | null | undefined, d = 3) =>
  x === null || x === undefined || Number.isNaN(x) ? "–" : x.toFixed(d);

export function MultiAgentView() {
  const [config, setConfig] = useState<MultiAgentConfig>({
    n_agents: 4,
    requester_policy: "D-TS",
    helper_policy: "q_learning",
    U_self: 0.35,
    lambda: 1.0,
    B_i: 0.5,
    resource_cost: 0.1,
    opportunity_cost: 0.05,
    gamma: 0.98,
    T: 2000,
    seed: 42,
  });

  const [state, setState] = useState<MultiAgentState | null>(null);
  const [history, setHistory] = useState<MultiAgentInteraction[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chartData, setChartData] = useState<any[]>([]);

  const startSession = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.multiAgentNew(config);
      setState(res.state);
      setHistory([]);
      setChartData([]);
    } catch (err: any) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  const stepSession = async (n: number) => {
    if (!state) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.multiAgentStep(state.session_id, n);
      setState(res.state);
      setHistory((prev) => [...prev, ...res.history].slice(-100));

      // Append chart point
      if (res.state.metrics) {
        setChartData((prev) => [
          ...prev,
          {
            round: res.state.t,
            avg_workload: res.state.metrics.avg_workload,
            avg_resources: res.state.metrics.avg_resources,
            acceptance_rate: res.state.metrics.acceptance_rate,
            cum_regret: res.state.metrics.cumulative_regret,
          },
        ].slice(-200));
      }
    } catch (err: any) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  const resetSession = async () => {
    if (!state) return;
    setLoading(true);
    setError(null);
    try {
      const s = await api.multiAgentReset(state.session_id);
      setState(s);
      setHistory([]);
      setChartData([]);
    } catch (err: any) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  const WORKLOAD_LABELS = ["Low", "Medium", "High"];
  const RESOURCE_LABELS = ["Low", "Medium", "High"];

  return (
    <div className="multiagent-view">
      <section className="panel controls">
        <label>
          Agents (N)
          <select
            value={config.n_agents}
            onChange={(e) => setConfig({ ...config, n_agents: Number(e.target.value) })}
          >
            <option value={3}>3 Agents</option>
            <option value={4}>4 Agents</option>
            <option value={5}>5 Agents</option>
          </select>
        </label>

        <label>
          Requester Policy
          <select
            value={config.requester_policy}
            onChange={(e) => setConfig({ ...config, requester_policy: e.target.value })}
          >
            <option value="D-TS">MultiAgent-D-TS</option>
            <option value="D-UCB">MultiAgent-D-UCB</option>
            <option value="D-Greedy">MultiAgent-Greedy</option>
          </select>
        </label>

        <label>
          Helper Policy
          <select
            value={config.helper_policy}
            onChange={(e) => setConfig({ ...config, helper_policy: e.target.value })}
          >
            <option value="q_learning">Tabular Q-Learning</option>
            <option value="always_accept">Always Accept</option>
            <option value="always_reject">Always Reject</option>
            <option value="informed_threshold">Informed Threshold (reference)</option>
          </select>
        </label>

        <label>
          Cooperation B_i
          <input
            type="number"
            step="0.1"
            min="0"
            max="1.5"
            value={config.B_i}
            onChange={(e) => setConfig({ ...config, B_i: Number(e.target.value) })}
          />
        </label>

        <label>
          U_self
          <input
            type="number"
            step="0.05"
            min="0"
            max="1"
            value={config.U_self}
            onChange={(e) => setConfig({ ...config, U_self: Number(e.target.value) })}
          />
        </label>

        <label>
          Seed
          <input
            type="number"
            value={config.seed}
            onChange={(e) => setConfig({ ...config, seed: Number(e.target.value) })}
          />
        </label>

        <div style={{ display: "flex", gap: "8px", alignItems: "flex-end" }}>
          {!state ? (
            <button className="primary" onClick={startSession} disabled={loading}>
              Initialize System
            </button>
          ) : (
            <>
              <button onClick={() => stepSession(1)} disabled={loading || state.done}>
                Step 1
              </button>
              <button onClick={() => stepSession(50)} disabled={loading || state.done}>
                Step 50
              </button>
              <button onClick={() => stepSession(500)} disabled={loading || state.done}>
                Step 500
              </button>
              <button onClick={resetSession} disabled={loading}>
                Reset
              </button>
            </>
          )}
        </div>
      </section>

      {error && <div className="error">{error}</div>}

      {state && (
        <>
          {/* Summary metrics header */}
          <section className="cards">
            <div className="card">
              <span className="card-lbl">Round</span>
              <span className="card-val">
                {state.t} / {state.T}
              </span>
            </div>
            <div className="card">
              <span className="card-lbl">Acceptance Rate</span>
              <span className="card-val">
                {fmt((state.metrics.acceptance_rate || 0) * 100, 1)}%
              </span>
            </div>
            <div className="card">
              <span className="card-lbl">Ask Rate</span>
              <span className="card-val">{fmt((state.metrics.ask_rate || 0) * 100, 1)}%</span>
            </div>
            <div className="card">
              <span className="card-lbl">Avg Workload</span>
              <span className="card-val">{fmt(state.metrics.avg_workload, 2)}</span>
            </div>
            <div className="card">
              <span className="card-lbl">Avg Resources</span>
              <span className="card-val">{fmt(state.metrics.avg_resources, 2)}</span>
            </div>
            <div className="card">
              <span className="card-lbl">Task Completion</span>
              <span className="card-val">
                {fmt((state.metrics.task_completion_rate || 0) * 100, 1)}%
              </span>
            </div>
          </section>

          {/* Agents Grid */}
          <h3>Autonomous Agents State & Beliefs</h3>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
              gap: "16px",
              marginBottom: "24px",
            }}
          >
            {state.agents.map((agent: any) => {
              const qVals = agent.helper.qlearning.q_table[agent.helper.workload * 3 + agent.helper.resources];
              const pAccept = qVals ? qVals[0] : 0;
              const pReject = qVals ? qVals[1] : 0;

              return (
                <div key={agent.agent_id} className="panel" style={{ padding: "16px" }}>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      borderBottom: "1px solid #e1e0d9",
                      paddingBottom: "8px",
                      marginBottom: "12px",
                    }}
                  >
                    <strong style={{ fontSize: "1.1rem" }}>Agent A{agent.agent_id + 1}</strong>
                    <span style={{ fontSize: "0.85rem", color: "#666" }}>
                      Reliability θ: {fmt(agent.helper.execution_reliability, 2)}
                    </span>
                  </div>

                  <div style={{ marginBottom: "12px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem" }}>
                      <span>Workload: <strong>{WORKLOAD_LABELS[agent.helper.workload]} ({agent.helper.workload})</strong></span>
                      <span>Resources: <strong>{RESOURCE_LABELS[agent.helper.resources]} ({agent.helper.resources})</strong></span>
                    </div>
                  </div>

                  <div style={{ marginBottom: "12px", fontSize: "0.85rem" }}>
                    <div style={{ color: "#444", fontWeight: 600, marginBottom: "4px" }}>
                      Helper Q-Values (Current State):
                    </div>
                    <div style={{ display: "flex", gap: "12px" }}>
                      <span>ACCEPT: <strong>{fmt(pAccept, 3)}</strong></span>
                      <span>REJECT: <strong>{fmt(pReject, 3)}</strong></span>
                    </div>
                  </div>

                  <div style={{ fontSize: "0.85rem" }}>
                    <div style={{ color: "#444", fontWeight: 600, marginBottom: "4px" }}>
                      Beliefs about Teammates:
                    </div>
                    <table style={{ width: "100%", fontSize: "0.8rem", textAlign: "left" }}>
                      <thead>
                        <tr>
                          <th>Teammate</th>
                          <th>Est. Acceptance (p_hat)</th>
                          <th>Est. Reliability (theta_hat)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {agent.requester.acceptance.p_hat.map((p: number, idx: number) => {
                          const targetId = idx >= agent.agent_id ? idx + 2 : idx + 1;
                          const theta = agent.requester.trust.theta_hat[idx];
                          return (
                            <tr key={idx}>
                              <td>A{targetId}</td>
                              <td>{fmt(p, 3)}</td>
                              <td>{fmt(theta, 3)}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Dynamic Curves Chart */}
          {chartData.length > 0 && (
            <div className="panel" style={{ marginBottom: "24px" }}>
              <h3>System Dynamics over Interactions</h3>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={chartData}>
                  <CartesianGrid stroke="#e1e0d9" vertical={false} />
                  <XAxis dataKey="round" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="avg_workload" stroke="#eb6834" name="Avg Workload" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="avg_resources" stroke="#2a78d6" name="Avg Resources" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="acceptance_rate" stroke="#1baf7a" name="Acceptance Rate" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Recent Interactions Log */}
          <div className="panel scroll">
            <h3>Recent Agent Interactions</h3>
            <table className="data">
              <thead>
                <tr>
                  <th>Round</th>
                  <th>Requester</th>
                  <th>Helper Contacted</th>
                  <th>Helper Action</th>
                  <th>Execution Outcome</th>
                  <th>Requester Payoff</th>
                  <th>Helper Payoff</th>
                </tr>
              </thead>
              <tbody>
                {history.slice(-15).reverse().map((r, i) => (
                  <tr key={i}>
                    <td>{r.round}</td>
                    <td>Agent A{r.requester_id + 1}</td>
                    <td>{r.asked ? `Agent A${(r.helper_id ?? 0) + 1}` : "Acted Independently"}</td>
                    <td>
                      {!r.asked ? "–" : r.accepted ? (
                        <span style={{ color: "#1baf7a", fontWeight: "bold" }}>ACCEPTED</span>
                      ) : (
                        <span style={{ color: "#e34948", fontWeight: "bold" }}>REJECTED</span>
                      )}
                    </td>
                    <td>
                      {r.success === 1 ? (
                        <span style={{ color: "#1baf7a" }}>Success</span>
                      ) : r.success === 0 ? (
                        <span style={{ color: "#e34948" }}>Failure</span>
                      ) : (
                        "–"
                      )}
                    </td>
                    <td>{fmt(r.requester_reward, 3)}</td>
                    <td>{fmt(r.helper_reward, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
