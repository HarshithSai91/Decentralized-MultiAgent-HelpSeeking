# PROGRESS.md

_Last updated: November 22, 2026_

## Status: Fully Redesigned & Complete

The project has been transformed from a single-agent bandit model into a **fully decentralized multi-agent system (MAS)**, directly answering the evaluator feedback:
> *"The helpers are fixed and never make their own decisions, so this is really one agent choosing between options. Try to make the problem truly multiagent."*

---

## Completed Milestones

| Component | Status | Details & Artifacts |
|---|---|---|
| **Autonomous Peer Agent** | Complete | `AutonomousAgent` class with dual roles (requester & helper), internal state $s_i = (w_i, \rho_i)$, explicit activity transitions (`execute_own_task`, `help_another_agent`, `rest`, `reject_request`), independent RNGs, individual reward calculations (`backend/src/help_seeking/autonomous_agent.py`). |
| **Helper Tabular Q-Learning** | Complete | 9-state discrete Q-table ($3 \times 3$), action space $\{\text{ACCEPT}, \text{REJECT}\}$, decision-epoch semi-Markov updates with geometric discounting $\gamma_Q^{\Delta t}$ across elapsed rounds, $\epsilon$-greedy exploration with decay, individual reward with state-dependent helping cost ($r_i = B_i x - \kappa_i(w,\rho)$, $\kappa_i = c_{\text{res}}(3-\rho) + c_{\text{opp}}(1+w)$) (`helper_qlearning.py`, `autonomous_agent.py`). |
| **Decoupled Belief Tracking** | Complete | Requesters maintain separate discounted Beta--Bernoulli acceptance beliefs $\hat{p}_{ij}$ and trust beliefs $\hat{\theta}_{ij}$. All agents decay all counts once per global round; rejections update acceptance only (`acceptance.py`, `trust.py`). |
| **Multi-Agent Policies** | Complete | `MultiAgent-Greedy`, `MultiAgent-D-UCB`, and `MultiAgent-D-TS` using expected utility $\mu_{ij} = U_{\text{self}} + \hat{p}_{ij}(R \hat{\theta}_{ij} - U_{\text{self}}) - \lambda c_{ij}$ (`multiagent_policies.py`). |
| **Multi-Agent Environment & Protocol** | Complete | Sequential reciprocal interactions, full resource/workload dynamics, dynamic requester-level oracle with exact $\epsilon$-greedy helper probability computation accounting for random tie-breaking and complete optimal action sets $\mathcal{O}_t$ (`multiagent_environment.py`). |
| **Multi-Agent Simulation & Metrics** | Complete | `MultiAgentSimulationRun` with 50-round Q-table and state visit snapshots, cumulative pseudo-regret, optimal action rate, greedy-policy stability (states with $\ge 5$ visits at both checkpoints, ties excluded), state-conditioned decisions, social welfare, and undefined metric handling (`multiagent_simulation.py`). |
| **Backward Compatibility (E1–E4, S1–S5)** | Preserved | Single-agent benchmark code (`environment.py`, `algorithms.py`, `simulation.py`) intact alongside new multi-agent architecture. |
| **Multi-Agent Experiments (E5, E6, E7)** | Executed (re-run 21 Nov with state-dependent helping cost) | E5 (Q-learning helpers vs Always-Accept, Always-Reject and an informed threshold rule), E6 (Homogeneous & Heterogeneous Cooperation Dynamics), and E7 (changing reliability at $t=1000$, $\gamma \in \{1, 0.995, 0.98, 0.95\}$) run across 30 seeds, $T=2000$. Results processed in `results/processed/`. |
| **Temporal Help-Seeking (WHEN to Ask)** | Complete | `TemporalTask` with deadlines $D$, progress $p$, and solo attempts; `RequesterTimingPolicy` with 27-state tabular Q-learning; baseline timing policies (`ImmediateHelpTimingPolicy`, `FixedDelayTimingPolicy`, `UrgencyTimingPolicy`, `NeverAskTimingPolicy`); `TemporalMultiAgentEnvironment` and `compute_temporal_metrics` (`task_lifecycle.py`, `requester_timing.py`, `temporal_environment.py`, `temporal_simulation.py`). |
| **Temporal Experiment (E8)** | Executed | E8 (Temporal Help-Seeking Under Task Deadlines, $D=5$, $T=400$, 30 seeds) comparing learned (Q-learning) and planned (model-based optimal stopping, `PlannedTimingPolicy`) timing against all 4 baseline timing policies under shifting reliability at $\tau=200$. Results processed in `results/processed/E8_temporal_help_seeking_summary.csv`. |
| **Figures & Tables** | Generated | `scripts/generate_multiagent_figures_and_tables.py` writes `report/tables/e5_table.tex`, `e5_state_table.tex`, `e6_table.tex`, `e6_heterogeneous_table.tex`, `e7_table.tex`, `e8_table.tex`, the figures `figures/e5_e6_helping_behaviour.*`, `figures/e7_changing_reliability.*`, `figures/e8_temporal_help_seeking.*`, `results/processed/multiagent_key_numbers.json` (incl. paired Wilcoxon tests) and `results/MULTIAGENT_RESULTS_SUMMARY.md`. |
| **FastAPI Multi-Agent Endpoints** | Complete | `/api/multiagent/sim/new`, `/step`, `/state`, `/reset` integrated and verified (`backend/src/help_seeking/api.py`). |
| **Interactive Frontend Dashboard** | Complete | Added **Multi-Agent Simulation** tab with agent status cards, workload/resource gauges, Q-values, belief table, Recharts dynamics plots, and interaction feed (`frontend/src/components/MultiAgentView.tsx`, `App.tsx`, `api.ts`). |
| **Comprehensive Test Suite** | Complete | **198 tests** covering unit, integration, oracle tie-breaking, state visitation metrics, heterogeneous rewards, state-dependent helping cost, informed-threshold helper, activity transitions, decision-epoch transitions, global belief decay, temporal task lifecycles, and API endpoints (`backend/tests/`). |
| **Academic Report & Compiled PDF** | Complete | Report in arXiv format (`arxiv.sty`, 8 pages): Methodology (with interaction diagram and algorithm), Results (E1–E8 incl. E3 adaptation figure), E1–E8 results with paired tests, limitations, contributions; compiled with pdflatex + bibtex (`report/main.tex`, `report/main.pdf`). |

---

## Verification Summary

- **Backend tests:** `python -m pytest` passes all 198 tests.
- **Frontend types:** `npx tsc --noEmit` cleanly passes with 0 errors.
- **Data artifacts:** E1–E8 raw CSVs, summary CSVs, curve NPZs, and JSON manifests all generated and saved in `results/`.
- **Figures & Tables:** E1–E8 tables and plots generated and accessible in `report/tables/` and `figures/`.
- **Compiled Report:** `report/main.pdf` compiled; every E5–E8 number in the text checked against `results/processed/multiagent_key_numbers.json`.
- **Reproducibility:** re-running E5–E8 reproduces `results/raw/*/runs.csv` exactly.
