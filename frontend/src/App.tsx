import { useEffect, useMemo, useRef, useState } from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { algoColor, api, HELPER_COLOR, Preset, Round, RoundDetail, SimConfig, SimState } from "./api";
import { MultiAgentView } from "./components/MultiAgentView";

type Tab = "live" | "compare" | "multiagent" | "experiments";
const fmt = (x: number | null | undefined, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? "–" : x.toFixed(d));

export default function App() {
  const [tab, setTab] = useState<Tab>("live");
  const [presets, setPresets] = useState<Record<string, Preset>>({});
  const [algorithms, setAlgorithms] = useState<string[]>([]);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [cfg, setCfg] = useState<SimConfig>({ algorithm: "D-TS", preset: "E3", T: 2000, seed: 1, alpha_ucb: 1 });

  useEffect(() => {
    api.presets()
      .then((p) => { setPresets(p.presets); setAlgorithms(p.algorithms); })
      .catch((e) => setBackendError(String(e.message ?? e)));
  }, []);

  const preset = presets[cfg.preset];
  const effective = {
    gamma: cfg.gamma ?? preset?.gamma, lambda: cfg.lambda ?? preset?.lambda, U_self: cfg.U_self ?? preset?.U_self,
  };

  return (
    <div className="app">
      <header>
        <div>
          <h1>Learning when and whom to ask for help</h1>
          <p className="sub">Cost-aware help-seeking under changing teammate reliability · all numbers come from the Python simulator</p>
        </div>
        <nav>
          {(["live", "compare", "multiagent", "experiments"] as Tab[]).map((t) => (
            <button key={t} className={tab === t ? "tab active" : "tab"} onClick={() => setTab(t)}>
              {t === "live" ? "Live simulation" : t === "compare" ? "Compare algorithms" : t === "multiagent" ? "Decentralized Multi-Agent" : "Experiment results"}
            </button>
          ))}
        </nav>
      </header>

      {backendError && (
        <div className="error">
          Cannot reach the backend ({backendError}). Start it with{" "}
          <code>uvicorn help_seeking.api:app --app-dir backend/src --port 8000</code>
        </div>
      )}

      {tab === "multiagent" && <MultiAgentView />}

      {tab !== "experiments" && tab !== "multiagent" && (
        <section className="panel controls">
          <label>Scenario
            <select value={cfg.preset} onChange={(e) => setCfg({ ...cfg, preset: e.target.value, gamma: undefined, lambda: undefined, U_self: undefined })}>
              {Object.entries(presets).map(([k, p]) => <option key={k} value={k}>{p.label}</option>)}
            </select>
          </label>
          {tab === "live" && (
            <label>Algorithm
              <select value={cfg.algorithm} onChange={(e) => setCfg({ ...cfg, algorithm: e.target.value })}>
                {algorithms.map((a) => <option key={a}>{a}</option>)}
              </select>
            </label>
          )}
          <NumField label="γ (discount)" value={effective.gamma} step={0.005} min={0.5} max={1}
                    onChange={(v) => setCfg({ ...cfg, gamma: v })} />
          <NumField label="α (UCB)" value={cfg.alpha_ucb} step={0.05} min={0.01} max={5}
                    onChange={(v) => setCfg({ ...cfg, alpha_ucb: v })} />
          <NumField label="λ (cost mult.)" value={effective.lambda} step={0.5} min={0} max={10}
                    onChange={(v) => setCfg({ ...cfg, lambda: v })} />
          <NumField label="U_self" value={effective.U_self} step={0.05} min={0} max={1}
                    onChange={(v) => setCfg({ ...cfg, U_self: v })} />
          <NumField label="Seed" value={cfg.seed} step={1} min={0} max={100000} onChange={(v) => setCfg({ ...cfg, seed: Math.round(v) })} />
          <NumField label="Horizon T" value={cfg.T} step={100} min={10} max={20000} onChange={(v) => setCfg({ ...cfg, T: Math.round(v) })} />
          {preset && (
            <div className="presetinfo">
              θ = ({preset.theta.join(", ")}) · c = ({preset.c.join(", ")})
              {preset.reliability.type === "abrupt" && <> · after t = {String(preset.reliability.tau)}: θ = ({(preset.reliability.theta_after as number[]).join(", ")})</>}
              {preset.reliability.type === "multishift" && <> · random changes: each round with probability {String(preset.reliability.p_c)}, all θ redrawn from U[{String(preset.reliability.low)}, {String(preset.reliability.high)}]</>}
            </div>
          )}
        </section>
      )}

      {tab === "live" && <Live cfg={cfg} />}
      {tab === "compare" && <Compare cfg={cfg} algorithms={algorithms} />}
      {tab === "experiments" && <Experiments />}
    </div>
  );
}

function NumField(p: { label: string; value: number | undefined; step: number; min: number; max: number; onChange: (v: number) => void }) {
  return (
    <label>{p.label}
      <input type="number" value={p.value ?? ""} step={p.step} min={p.min} max={p.max}
             onChange={(e) => { const v = parseFloat(e.target.value); if (!Number.isNaN(v)) p.onChange(v); }} />
    </label>
  );
}

// ---------------------------------------------------------------- live
function Live({ cfg }: { cfg: SimConfig }) {
  const [state, setState] = useState<SimState | null>(null);
  const [rounds, setRounds] = useState<Round[]>([]);
  const [last, setLast] = useState<RoundDetail | null>(null);
  const [playing, setPlaying] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const busy = useRef(false);

  const start = async () => {
    setPlaying(false); setErr(null);
    try { const s = await api.start(cfg); setState(s); setRounds([]); setLast(null); }
    catch (e) { setErr(String((e as Error).message)); }
  };
  const step = async (n: number) => {
    if (!state || state.done || busy.current) return;
    busy.current = true;
    try {
      const r = await api.step(state.session_id, n);
      setState(r.state); setLast(r.last_round); setRounds((old) => old.concat(r.rounds));
      if (r.state.done) setPlaying(false);
    } catch (e) { setErr(String((e as Error).message)); setPlaying(false); }
    busy.current = false;
  };
  const reset = async () => {
    if (!state) return;
    setPlaying(false);
    const s = await api.reset(state.session_id);
    setState(s); setRounds([]); setLast(null);
  };

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => step(10), 120);
    return () => clearInterval(id);
  });

  const series = useMemo(() => {
    const out: Record<string, number>[] = [];
    let cr = 0, asks = 0;
    const win: number[] = [];
    const every = Math.max(1, Math.floor(rounds.length / 400));
    rounds.forEach((r, i) => {
      cr += r.regret; win.push(r.d); asks += r.d;
      if (win.length > 50) asks -= win.shift()!;
      if (i % every === 0 || i === rounds.length - 1) {
        const row: Record<string, number> = { t: r.t, cumRegret: +cr.toFixed(3), askRate: +(asks / win.length).toFixed(3) };
        r.theta_true.forEach((v, k) => { row[`true${k}`] = v; });
        r.theta_hat?.forEach((v, k) => { row[`hat${k}`] = +v.toFixed(3); });
        out.push(row);
      }
    });
    return out;
  }, [rounds]);

  const K = state?.theta_hat.length ?? 3;
  return (
    <>
      <section className="panel buttons">
        <button className="primary" onClick={start}>{state ? "Start new run" : "Start run"}</button>
        <button disabled={!state || state.done} onClick={() => step(1)}>Step 1</button>
        <button disabled={!state || state.done} onClick={() => step(10)}>Step 10</button>
        <button disabled={!state || state.done} onClick={() => step(100)}>Step 100</button>
        <button disabled={!state || state.done} onClick={() => setPlaying(!playing)}>{playing ? "Pause" : "Play"}</button>
        <button disabled={!state || state.done} onClick={() => step(state ? state.T - state.t : 1)}>Run to end</button>
        <button disabled={!state} onClick={reset}>Reset (same seed)</button>
        {state && <span className="progress">round {state.t} / {state.T} · {state.algorithm}</span>}
        {err && <span className="errtext">{err}</span>}
      </section>

      {!state && <p className="hint">Pick a scenario and an algorithm, then press <b>Start run</b>.</p>}

      {state && (
        <>
          <section className="grid4">
            <Stat label="Cumulative pseudo-regret" value={fmt(state.totals.regret, 2)} />
            <Stat label="Average net utility" value={fmt(state.t ? state.totals.utility / state.t : NaN)} />
            <Stat label="Ask rate" value={state.t ? `${(100 * state.totals.asks / state.t).toFixed(1)}%` : "–"} />
            <Stat label="Communication cost" value={fmt(state.totals.comm_cost, 1)} />
          </section>

          <section className="grid3">
            {Array.from({ length: K }, (_, k) => (
              <div className="panel helper" key={k}>
                <div className="helperhead"><span className="dot" style={{ background: HELPER_COLOR[k] }} />Helper {k + 1}</div>
                <table><tbody>
                  <tr><td>Trust estimate</td><td>{fmt(state.theta_hat[k])}</td></tr>
                  <tr><td>Pseudo-counts S / F</td><td>{fmt(state.S[k], 1)} / {fmt(state.F[k], 1)}</td></tr>
                  <tr><td>Cost λ·c</td><td>{fmt(state.costs[k])}</td></tr>
                  <tr className="truth"><td>Ground truth θ <small>(environment only)</small></td><td>{fmt(state.theta_true_now[k])}</td></tr>
                </tbody></table>
              </div>
            ))}
          </section>

          {last && <Inspector d={last} />}

          <section className="grid2">
            <div className="panel">
              <h3>Trust estimates vs ground truth</h3>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={series}>
                  <CartesianGrid stroke="#e1e0d9" vertical={false} />
                  <XAxis dataKey="t" type="number" domain={[0, state.T]} tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v: number) => v.toFixed(3)} />
                  {state.change_point && <ReferenceLine x={state.change_point} stroke="#898781" />}
                  {Array.from({ length: K }, (_, k) => (
                    <Line key={`h${k}`} dataKey={`hat${k}`} name={`belief, helper ${k + 1}`} stroke={HELPER_COLOR[k]} dot={false} strokeWidth={2} isAnimationActive={false} />
                  ))}
                  {Array.from({ length: K }, (_, k) => (
                    <Line key={`t${k}`} dataKey={`true${k}`} name={`true θ helper ${k + 1}`} stroke={HELPER_COLOR[k]} strokeDasharray="4 3" dot={false} strokeWidth={1.2} isAnimationActive={false} legendType="none" />
                  ))}
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                </LineChart>
              </ResponsiveContainer>
              <p className="note">Solid: agent's belief. Dashed: hidden ground truth.</p>
            </div>
            <div className="panel">
              <h3>Cumulative regret and 50-round ask rate</h3>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={series}>
                  <CartesianGrid stroke="#e1e0d9" vertical={false} />
                  <XAxis dataKey="t" type="number" domain={[0, state.T]} tick={{ fontSize: 11 }} />
                  <YAxis yAxisId="r" tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v: number) => v.toFixed(3)} />
                  {state.change_point && <ReferenceLine yAxisId="r" x={state.change_point} stroke="#898781" />}
                  <Line yAxisId="r" dataKey="cumRegret" name="cumulative regret" stroke={algoColor(state.algorithm)} dot={false} strokeWidth={2} isAnimationActive={false} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                </LineChart>
              </ResponsiveContainer>
              <ResponsiveContainer width="100%" height={110}>
                <LineChart data={series}>
                  <XAxis dataKey="t" type="number" domain={[0, state.T]} tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v: number) => `${(100 * v).toFixed(0)}%`} />
                  <Line dataKey="askRate" name="ask rate" stroke="#52514e" dot={false} strokeWidth={1.5} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>
        </>
      )}
    </>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return <div className="panel stat"><div className="statlabel">{label}</div><div className="statvalue">{value}</div></div>;
}

function Inspector({ d }: { d: RoundDetail }) {
  const K = d.belief_before.theta_hat.length;
  const extraKeys = Object.keys(d.extra).filter((k) => k !== "theta_hat");
  return (
    <section className="panel">
      <h3>Decision inspector — round {d.t}</h3>
      <div className="inspector">
        <table className="data">
          <thead><tr><th>Helper</th><th>Belief before</th>{extraKeys.map((k) => <th key={k}>{k === "bonus" ? "UCB bonus" : k === "theta_sample" ? "TS sample" : k}</th>)}<th>Score</th><th>Oracle μ</th></tr></thead>
          <tbody>
            {Array.from({ length: K }, (_, k) => (
              <tr key={k} className={d.candidate === k ? "cand" : ""}>
                <td><span className="dot" style={{ background: HELPER_COLOR[k] }} />{k + 1}{d.candidate === k ? " ★" : ""}</td>
                <td>{fmt(d.belief_before.theta_hat[k])}</td>
                {extraKeys.map((e) => <td key={e}>{fmt(d.extra[e][k])}</td>)}
                <td>{d.scores ? fmt(d.scores[k]) : "–"}</td>
                <td>{fmt(d.oracle.mu[k])}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="verdict">
          <p>Best candidate score {fmt(d.candidate_score)} {d.candidate_score !== null && (d.candidate_score > d.U_self ? ">" : "≤")} U_self = {fmt(d.U_self, 2)}</p>
          <p><b>Decision:</b> {d.d === 1 ? `ask helper ${(d.a ?? 0) + 1}` : "act alone"} → {d.d === 1 ? (d.x ? "success" : "failure") : "autonomous"}; r = {fmt(d.r)}, cost {fmt(d.cost)}</p>
          <p><b>Oracle:</b> {d.oracle.d === 1 ? `ask helper ${(d.oracle.a ?? 0) + 1}` : "act alone"} (μ* = {fmt(d.mu_star)}); regret this round {fmt(d.regret)} {d.optimal ? "✓ optimal" : ""}</p>
        </div>
      </div>
    </section>
  );
}

// ------------------------------------------------------------- compare
function Compare({ cfg, algorithms }: { cfg: SimConfig; algorithms: string[] }) {
  const [chosen, setChosen] = useState<string[]>(["Dynamic Oracle", "Greedy", "D-UCB", "D-TS", "Always-Ask + Random"]);
  const [res, setRes] = useState<Awaited<ReturnType<typeof api.compare>>["results"] | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const run = async () => {
    setLoading(true); setErr(null);
    try { setRes((await api.compare({ ...cfg, algorithms: chosen })).results); }
    catch (e) { setErr(String((e as Error).message)); }
    setLoading(false);
  };
  const data = useMemo(() => {
    if (!res || !res.length) return [];
    return res[0].t.map((t, i) => {
      const row: Record<string, number> = { t };
      res.forEach((r) => { row[r.algorithm] = r.cum_regret[i]; });
      return row;
    });
  }, [res]);

  return (
    <>
      <section className="panel buttons">
        {algorithms.map((a) => (
          <label key={a} className="check">
            <input type="checkbox" checked={chosen.includes(a)}
                   onChange={(e) => setChosen(e.target.checked ? [...chosen, a] : chosen.filter((x) => x !== a))} />
            <span className="dot" style={{ background: algoColor(a) }} />{a}
          </label>
        ))}
        <button className="primary" disabled={loading || !chosen.length} onClick={run}>{loading ? "Running…" : "Run all on same seed"}</button>
        {err && <span className="errtext">{err}</span>}
      </section>
      {res && (
        <section className="grid2">
          <div className="panel">
            <h3>Cumulative pseudo-regret (same seed, common random numbers)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={data}>
                <CartesianGrid stroke="#e1e0d9" vertical={false} />
                <XAxis dataKey="t" type="number" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v: number) => v.toFixed(1)} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {res.map((r) => (
                  <Line key={r.algorithm} dataKey={r.algorithm} stroke={algoColor(r.algorithm)} dot={false} strokeWidth={2}
                        strokeDasharray={r.algorithm.includes("Oracle") && !r.algorithm.startsWith("Initial") ? "5 3" : undefined} isAnimationActive={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
            <p className="note">One seed only: use the Experiment results tab for 30-seed means with confidence intervals.</p>
          </div>
          <div className="panel">
            <h3>Run metrics</h3>
            <table className="data">
              <thead><tr><th>Algorithm</th><th>Regret</th><th>Avg utility</th><th>Ask %</th><th>Acc_ask %</th><th>Opt. helper %</th><th>Comm. cost</th>{res[0].metrics.adaptation_latency !== undefined && <th>τ_adapt</th>}</tr></thead>
              <tbody>
                {res.map((r) => (
                  <tr key={r.algorithm}>
                    <td><span className="dot" style={{ background: algoColor(r.algorithm) }} />{r.algorithm}</td>
                    <td>{fmt(r.metrics.cumulative_regret, 1)}</td>
                    <td>{fmt(r.metrics.average_net_utility)}</td>
                    <td>{fmt(100 * (r.metrics.ask_rate ?? NaN), 1)}</td>
                    <td>{fmt(100 * (r.metrics.ask_accuracy ?? NaN), 1)}</td>
                    <td>{r.metrics.optimal_teammate_rate === null ? "–" : fmt(100 * (r.metrics.optimal_teammate_rate ?? NaN), 1)}</td>
                    <td>{fmt(r.metrics.communication_cost, 1)}</td>
                    {r.metrics.adaptation_latency !== undefined && <td>{fmt(r.metrics.adaptation_latency, 0)}</td>}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </>
  );
}

// --------------------------------------------------------- experiments
const METRICS: [string, string, number][] = [
  ["cumulative_regret", "Cumulative regret", 1], ["average_net_utility", "Average net utility", 1],
  ["ask_rate", "Ask rate (%)", 100], ["ask_accuracy", "Ask-decision accuracy (%)", 100],
  ["optimal_teammate_rate", "Optimal-teammate rate (%)", 100], ["communication_cost", "Communication cost", 1],
  ["adaptation_latency", "Adaptation latency τ_adapt (mean)", 1], ["post_change_regret", "Post-change regret", 1],
  ["share_alone", "Share acting alone (%)", 100], ["mae_theta", "MAE of trust estimates", 1],
];

function Experiments() {
  const [list, setList] = useState<{ name: string; title: string }[]>([]);
  const [name, setName] = useState("E1_stationary");
  const [metric, setMetric] = useState("cumulative_regret");
  const [data, setData] = useState<{ manifest: any; summary: any[] } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => { api.experiments().then((r) => setList(r.experiments)).catch((e) => setErr(String(e.message))); }, []);
  useEffect(() => { setErr(null); api.experiment(name).then(setData).catch((e) => { setData(null); setErr(String(e.message)); }); }, [name]);

  const scale = METRICS.find((m) => m[0] === metric)?.[2] ?? 1;
  const rows = data?.summary.filter((r) => r.metric === metric) ?? [];
  const algos = Array.from(new Set(rows.map((r) => r.algorithm as string)));
  const points: string[] = data ? data.manifest.points.map((p: any) => p.label) : [];
  const chart = points.map((p, i) => {
    const row: Record<string, number | string> = { point: p };
    rows.filter((r) => r.point === i).forEach((r) => { if (r.mean !== null) row[r.algorithm] = +(r.mean * scale).toFixed(4); });
    return row;
  });

  return (
    <>
      <section className="panel controls">
        <label>Experiment
          <select value={name} onChange={(e) => setName(e.target.value)}>
            {list.map((e) => <option key={e.name} value={e.name}>{e.title ?? e.name}</option>)}
          </select>
        </label>
        <label>Metric
          <select value={metric} onChange={(e) => setMetric(e.target.value)}>
            {METRICS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}
          </select>
        </label>
        {data && <div className="presetinfo">{data.manifest.n_runs} runs · {data.manifest.seeds.length} seeds · T = {data.manifest.config.T} · completed {data.manifest.completed_at}</div>}
      </section>
      {err && <div className="error">{err}</div>}
      {data && rows.length === 0 && <p className="hint">This metric is not defined for this experiment.</p>}
      {data && rows.length > 0 && (
        <section className="grid2">
          <div className="panel">
            <h3>{METRICS.find((m) => m[0] === metric)?.[1]} — mean over seeds</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chart}>
                <CartesianGrid stroke="#e1e0d9" vertical={false} />
                <XAxis dataKey="point" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {algos.map((a) => (
                  <Line key={a} dataKey={a} stroke={algoColor(a)} strokeWidth={2} dot={{ r: 4 }} isAnimationActive={false}
                        strokeDasharray={a.includes("Oracle") && !a.startsWith("Initial") ? "5 3" : a.includes("[alpha") ? "2 2" : undefined} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="panel scroll">
            <h3>Mean ± 95% CI</h3>
            <table className="data">
              <thead><tr><th>Algorithm</th>{points.map((p) => <th key={p}>{p}</th>)}</tr></thead>
              <tbody>
                {algos.map((a) => (
                  <tr key={a}>
                    <td><span className="dot" style={{ background: algoColor(a) }} />{a}</td>
                    {points.map((p, i) => {
                      const r = rows.find((x) => x.algorithm === a && x.point === i);
                      if (!r || r.mean === null) return <td key={p}>–</td>;
                      const half = r.ci_high === null ? 0 : (r.ci_high - r.mean) * scale;
                      return <td key={p}>{fmt(r.mean * scale, scale === 100 ? 1 : 2)}<small> ±{fmt(half, scale === 100 ? 1 : 2)}</small></td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </>
  );
}
