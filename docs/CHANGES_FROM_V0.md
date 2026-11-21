# What changed from the first version (v0, 12 Oct 2026)

The first version had a good module layout, which is kept. Several bugs made its results invalid,
and most of the experiments had not been run. The superseded v0 files were removed on 21 Oct 2026; this file records what they got wrong.

## Correctness bugs fixed

| # | Problem in v0 | Effect | Fix |
|---|---|---|---|
| 1 | Regret was computed from the *realised* reward r_t instead of the expected utility (Eq. 15 is pseudo-regret). | The Dynamic Oracle showed regret ≈ 2.7 and some Greedy runs showed **negative** regret (e.g. −6.0). Every regret number in v0 contains Bernoulli noise. | `simulation.py` logs μ of the chosen action; regret = μ*(t) − μ_chosen(t). The oracle now has exactly 0 regret (tested). |
| 2 | Environment and policy RNGs were both `default_rng(seed)` with the **same integer**. | The Random baseline's helper choice and the outcome came from identical bit streams, so they were correlated. | Independent streams from `SeedSequence(seed).spawn(3)`. A test checks independence. |
| 3 | No common random numbers. Outcomes depended on how many random numbers each policy had drawn. | Comparisons between methods were noisier than needed. | One uniform u[t,k] per round and helper; x_t = 1[u[t,a_t] < θ]. All methods see the same draws, so comparisons are seed-paired. |
| 4 | Always-Ask + Fixed-Cheapest used `argmin(c)`, which returns index 0 on ties. | In E1 (equal costs) helper 1 is the best helper, so this "baseline" matched the oracle. The v0 report claimed it had "same performance as oracle". | Ties among the cheapest helpers are broken uniformly at random **once per run**. |
| 5 | Initial-Knowledge Oracle was hard-coded to θ = [0.8, 0.7, 0.9], U_self = 0.5. Never-Ask was hard-coded to U_self = 0.35. | Only correct for E3. | Both take values from the configuration. |
| 6 | `configs/e1_stationary.json` listed "Stationary Oracle", which did not exist in the policy map. | `run_all_experiments.py` crashed on E1. | Alias table in `policies.py`. |
| 7 | `adaptation_latency()` was `pass`. Optimal-teammate rate, MAE_θ, selection shares and post-change regret were never computed. | H3/H4 could not be evaluated. | All metrics implemented in `metrics.py` and tested against hand calculations. |
| 8 | No tie handling in the oracle. In E2 with U_self = 0.65, asking helper 1 (0.75 − 0.1) and acting alone are exactly equal. | Ask accuracy at that point would depend on floating-point luck. | Tie-aware oracle: both decisions count as correct, and both have zero regret. |
| 9 | Act-alone rounds discounted beliefs by reaching into `policy.trust` from the simulation loop. | Fragile; the decision inspector could not show it. | `Policy.update(d, a, x)` handles both cases (Algorithm 1, lines 22–23 and 27). |

## Experiments: v0 vs now

| | v0 | now |
|---|---|---|
| E1 | Δ = 0.10 only, 30 seeds (results invalid, see bugs 1–4) | all Δ ∈ {0.02, 0.05, 0.10, 0.15}, 30 seeds |
| E2 | 5 seeds, T = 500, U_self = 0.65 only | all 4 U_self values, 30 seeds, T = 2000 |
| E3 | 5 seeds, **T = 500 with τ = 1000**, so the change never happened | all 5 γ values, 30 seeds, T = 2000, τ = 1000 |
| E4 | 5 seeds, T = 500, λ = 2.5 only | all 5 λ values, 30 seeds |
| Extra | – | S1 (H2: γ in a stationary world), S2 (α_ucb sensitivity), S3 (optional drift), S4 (ask accuracy vs γ), S5 (multi-shift setting of proposal Sec. 2.3, which v0 never implemented) |

## Other deliverables

- **Report:** v0 `main.tex` was one page with invalid numbers. It is now a full 18-page report with
  auto-generated tables, 10 figures, hypothesis verdicts, discussion, limitations and the 11 references.
  Note that the proposal's reference [8] (Garivier & Moulines, ALT 2011) is published as "...for
  *Switching* Bandit Problems". The bib entry uses the published title.
- **Dashboard:** the v0 `App.tsx` drew its regret chart from `Math.random()` (fabricated values) and its
  buttons only showed `alert()` boxes. It now has a working FastAPI backend (`api.py`) and a 3-tab
  React dashboard that uses only real simulation output.
- **Tests:** v0 tests mostly asserted `d in (0, 1)`, so they always passed. There are now 52 tests with
  hand-calculated expectations.
- **Audit:** v0 `FINAL_AUDIT.md` marked items PASS that were not true (e.g. "regret correctly defined").
  It has been rewritten.

# Final multi-agent version (21–22 Nov 2026)

Changes relative to the first multi-agent version (the redesign made after the TA's feedback):

| # | Problem | Effect | Fix |
|---|---|---|---|
| 1 | The helper reward `B_i * x - resource_cost - opportunity_cost` and the execution reliability did not depend on the helper's workload or resources. | The best helper decision was the same in every state, so the 9-state Q-table had nothing to learn about the state; the reported "state-sensitive thresholds" (87.7% → 68.0%) were learning noise (31% of visited helper states ended preferring REJECT although ACCEPT paid more). | Helping cost now grows with workload and resource depletion (DECISIONS #19). E5 now shows 88.5% / 32.7% / 25.7% acceptance in the idle / medium / busy states. |
| 2 | No reference for what good helping looks like. | Learned helpers could only be compared with Always-Accept/Reject. | Informed-threshold helper added to E5 (DECISIONS #20). |
| 3 | E7 text reported that γ = 0.98 helps MA-D-TS but not that it hurts MA-D-UCB. | One-sided conclusion. | E7 now sweeps γ ∈ {1, 0.995, 0.98, 0.95}; the report explains when discounting helps (DECISIONS #22). |
| 4 | Report used the plain `article` class; TA asked for the arXiv format and much shorter writing. | — | Report rewritten with `arxiv.sty` (8 pages: Methodology, Results, Contributions). |
| 5 | Table generator hid confidence intervals smaller than 0.01 or 0.1. | Inconsistent tables. | New generator computes everything from per-seed raw results and writes `results/processed/multiagent_key_numbers.json` and `results/MULTIAGENT_RESULTS_SUMMARY.md`. |
| 6 | README linked a non-existent `E5_helper_qlearning_summary.csv` and quoted outdated numbers. | — | README updated. |
| 7 | `frontend/package.json` listed `@rollup/rollup-darwin-arm64` (a Mac-only binary) as a dependency. | `npm install` failed on Windows and Linux (EBADPLATFORM), so the dashboard could not be installed there. | Dependency removed (rollup installs the right binary itself); lockfile regenerated; build verified. |
| 8 | Live dashboard metrics were computed over the whole pre-allocated horizon T instead of the rounds played so far. | Mid-run values were wrong (e.g. ask rate 2% and average resources 0.04 after 50 rounds). | `api._record_so_far` restricts the record to the rounds played; regression test added. Metric cards styled; new screenshots. |
| 9 | Learned (Q-learning) timing lost to the hand-designed Fixed-delay and Urgency rules. | Diagnosis: the whether-rule compares asking now with never asking, so the timing learner must learn the option value of waiting; with ~100 tasks per agent it does not (training 5× longer or evaluating greedily left the gap in place; `scripts/diagnose_e8_timing.py`). | Added model-based `PlannedTimingPolicy` (DECISIONS #26). It matches the two rules on requester reward (0.288 vs 0.277 / 0.297, n.s.) with 44% of tasks asking instead of 68–79%. The Q-learning result is still reported. |

| 10 | `PlannedTimingPolicy` cached solo values only up to the environment's default deadline. | `run_task(deadline=7)` with default 5 crashed (`KeyError: 6`). E8 (deadline 5) was not affected. | Solo values are computed lazily and cached per deadline; regression test added. |
| 11 | Before the first round the dashboard computed metrics on the empty pre-allocated record. | At t = 0 it showed 100 total rounds and average resources 0. | Empty record at t = 0; averages describe the agents' initial state; test added. |
| 12 | The report called planned timing an optimal-stopping solution. | Overstated: it freezes the current posterior means and the final decision uses posterior samples. | Described as an approximate, frozen-belief planner (report, docs, docstring). |
| 13 | Packaging: unused `nicefrac` package; two screenshots stored with permission 600; dashboard needed two servers. | Portability. | Removed `nicefrac`; all files 644 / folders 755 in the zip; the API now also serves `frontend/dist` (status endpoint moved to `/api/status`). |

E5–E8 were re-run with all 30 seeds after these changes; E1–E4 and S1–S5 are unchanged.
