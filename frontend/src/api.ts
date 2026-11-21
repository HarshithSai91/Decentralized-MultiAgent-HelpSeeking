// Thin client for the FastAPI backend (backend/src/help_seeking/api.py).
// The dashboard never simulates anything itself.

export type Dict<T = unknown> = Record<string, T>;

export interface SimConfig {
  algorithm: string;
  preset: string;
  gamma?: number;
  alpha_ucb?: number;
  lambda?: number;
  U_self?: number;
  T: number;
  seed: number;
}

export interface Totals { regret: number; utility: number; comm_cost: number; asks: number; optimal_actions: number }

export interface SimState {
  session_id: string; algorithm: string; t: number; T: number; done: boolean;
  U_self: number; costs: number[]; theta_true_now: number[]; theta_hat: number[];
  S: number[]; F: number[]; totals: Totals; change_point: number | null;
}

export interface Round {
  t: number; d: number; a: number | null; x: number | null; r: number; cost: number;
  mu_chosen: number; mu_star: number; regret: number; d_star: number; optimal: boolean;
  theta_true: number[]; theta_hat: number[] | null;
}

export interface RoundDetail {
  t: number; U_self: number; theta_true: number[];
  belief_before: { theta_hat: number[]; S: number[]; F: number[] };
  scores: number[] | null; extra: Dict<number[]>; candidate: number | null; candidate_score: number | null;
  d: number; a: number | null; x: number | null; r: number; cost: number;
  mu_chosen: number; mu_star: number; regret: number;
  oracle: { d: number; a: number | null; mu: number[] }; optimal: boolean;
}

export interface Preset { label: string; theta: number[]; c: number[]; U_self: number; lambda: number; gamma: number; reliability: Dict }

export interface MultiAgentConfig {
  n_agents: number;
  requester_policy: string;
  helper_policy: string;
  U_self: number;
  lambda: number;
  B_i: number;
  resource_cost: number;
  opportunity_cost: number;
  gamma: number;
  T: number;
  seed: number;
}

export interface MultiAgentState {
  session_id: string;
  t: number;
  T: number;
  done: boolean;
  n_agents: number;
  requester_policy: string;
  helper_policy: string;
  agents: any[];
  metrics: Dict<any>;
}

export interface MultiAgentInteraction {
  round: number;
  requester_id: number;
  helper_id: number | null;
  asked: boolean;
  accepted: boolean | null;
  success: number | null;
  requester_reward: number;
  helper_reward: number;
  comm_cost: number;
}

async function call<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, body === undefined ? undefined : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { msg = (await res.json()).detail ?? msg; } catch { /* ignore */ }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  presets: () => call<{ presets: Dict<Preset>; algorithms: string[]; learning: string[] }>("/api/presets"),
  start: (cfg: SimConfig) => call<SimState>("/api/sim/start", cfg),
  step: (sid: string, n: number) =>
    call<{ state: SimState; last_round: RoundDetail | null; rounds: Round[] }>(`/api/sim/${sid}/step`, { n }),
  reset: (sid: string) => call<SimState>(`/api/sim/${sid}/reset`, {}),
  compare: (cfg: SimConfig & { algorithms: string[] }) =>
    call<{ results: { algorithm: string; metrics: Dict<number | null>; t: number[]; cum_regret: number[]; ask_rate_50: number[] }[] }>(
      "/api/compare", cfg),
  experiments: () => call<{ experiments: { name: string; title: string; points: string[]; algorithms: string[]; n_runs: number }[] }>(
    "/api/experiments"),
  experiment: (name: string) => call<{ manifest: Dict<any>; summary: Dict<any>[] }>(`/api/experiments/${name}`),
  multiAgentNew: (cfg: MultiAgentConfig) => call<{ session_id: string; state: MultiAgentState }>("/api/multiagent/sim/new", cfg),
  multiAgentStep: (sid: string, n: number) => call<{ state: MultiAgentState; last_interaction: MultiAgentInteraction | null; history: MultiAgentInteraction[] }>(`/api/multiagent/sim/${sid}/step`, { n }),
  multiAgentReset: (sid: string) => call<MultiAgentState>(`/api/multiagent/sim/${sid}/reset`, {}),
};

// Colour follows the entity: same colour for a method in every view (and in the report figures).
export const ALGO_COLOR: Dict<string> = {
  "D-UCB": "#2a78d6", "D-TS": "#eb6834", "Greedy": "#1baf7a", "Greedy (extra)": "#1baf7a",
  "Always-Ask + Random": "#eda100", "Always-Ask + Fixed-Cheapest": "#e87ba4", "Never-Ask": "#008300",
  "Always-Ask + Empirical Greedy": "#6250d6", "Initial-Knowledge Oracle": "#e34948",
  "Dynamic Oracle": "#0b0b0b", "Stationary Oracle": "#0b0b0b",
};
export const HELPER_COLOR = ["#2a78d6", "#eb6834", "#1baf7a"];
export const algoColor = (a: string) => ALGO_COLOR[a] ?? (a.startsWith("D-UCB") ? "#2a78d6" : "#52514e");
