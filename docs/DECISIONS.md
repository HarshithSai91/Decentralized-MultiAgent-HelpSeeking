# DECISIONS.md: implementation choices not fixed by the proposal

Every choice below can be reversed. Each is also described in the report (Sec. 4.1) and implemented where noted.

| # | Topic | Decision | Where |
|---|---|---|---|
| 1 | Prior | Uniform Beta(1,1): θ̂ = (S+1)/(S+F+2) (implied by Eq. 5). | `trust.py` |
| 2 | Update order | Discount all counts, then add the new outcome for the queried helper (Eqs. 3–4, Algorithm 1 lines 22–23). Act-alone rounds only discount. | `trust.py`, `algorithms.Policy.update` |
| 3 | Ties in argmax | Uniformly at random among scores within 1e-12 (Algorithm 1, line 6). | `algorithms.argmax_random` |
| 4 | Ask rule | Strict inequality: ask iff best score > U_self (Eq. 9). | `algorithms.IndexPolicy.decide` |
| 5 | D-UCB constants | α_ucb = 1.0 (UCB1 constant) in E1–E4, ε = 1e-6. S2 studies α ∈ {0.05,…,2}. | `configs/*.json` |
| 6 | Autonomous outcome | r_t = U_self (its expectation). The proposal fixes only E[x_self] = U_self. This affects realised utility, not regret. | `environment.realized_utility` |
| 7 | Regret | Pseudo-regret Σ(μ*(t) − μ_chosen(t)), using expected utilities and no Bernoulli noise (Eq. 15). | `simulation.py`, `metrics.py` |
| 8 | Oracle ties | If max_k μ_k = U_self (E2, U_self = 0.65), both decisions are optimal: zero regret, and both count as correct in Acc_ask. The reported d* follows Eq. 11 (strict >). | `utilities.dynamic_oracle` |
| 9 | a* when d* = 0 | a* = argmax_k μ_k is still used to score *whom* the agent asked (the optimal-teammate rate is conditional on asking). | `metrics.optimal_teammate_selection_rate` |
| 10 | τ_adapt | First s ≥ 1 such that, at round τ+s, the post-change optimal action was chosen in > 50% of the trailing 50 rounds. The window may include pre-change rounds, as the definition says, so a perfect switch scores 26. Runs that never meet it are censored at T − τ; medians and the censored count are reported. | `metrics.adaptation_latency` |
| 11 | MAE_θ | Time-averaged over all rounds, using the belief held *when deciding*. The final-round value is also stored. Not reported for oracles or Never-Ask. | `metrics.py` |
| 12 | Seeds | Run i uses `SeedSequence(i)` split into environment / policy / drift streams. Seeds 1–30. | `experiments.build_run` |
| 13 | Common random numbers | One uniform u[t,k] per (round, helper); x_t = 1[u[t,a_t] < θ_{a_t}(t)]. The same draws are used for every algorithm and sweep point. | `environment.py` |
| 14 | Fixed-Cheapest ties | A random helper among the cheapest, chosen once per run. | `baselines.AlwaysAskFixedCheapest` |
| 15 | Confidence intervals | 95% Student-t over 30 independent runs (run-level, not time-step-level). Comparisons are seed-paired differences. | `stats.py` |
| 16 | Drift (optional) | Implemented (Eq. 2) and evaluated in S3 with σ = 0.05, U_self = 0.35, with a new path per seed. | `reliability.DriftProfile` |
| 16b | Multi-shift (proposal Sec. 2.3) | Before each round t ≥ 2 a change occurs with probability p_c; on a change, every θ_k is redrawn from U[0.2, 0.95]. The proposal leaves the post-change distribution open. The path is precomputed per seed. Evaluated in S5 with p_c ∈ {0.001, 0.0025, 0.005}. | `reliability.MultiShiftProfile` |
| 17 | E2 extra | Discounted Greedy (γ = 0.98) is added to E2 as "Greedy (extra)", marked † in the report. | `configs/e2_when_to_ask.json` |
| 18 | Supplementary studies | S1–S5 were designed after the main results, to test H2 directly, to explain failure regimes and (S5) to cover the multi-shift setting. The report labels them as such. | `configs/s*.json` |

## Multi-agent experiments (E5–E8)

| # | Topic | Decision | Where |
|---|---|---|---|
| 19 | Helping cost | The helper's cost depends on its state when it accepts: `cost(w, rho) = c_res * (3 - rho) + c_opp * (1 + w)` (0.15 idle, 0.30 at (1,1), 0.45 busy with the defaults). With a state-independent cost the optimal helper policy would ignore (w, rho). | `AutonomousAgent.helping_cost` |
| 20 | Informed reference helper | E5 adds `informed_threshold`: accept iff `B_j * theta_j - cost(w, rho) > 0`, using the helper's true theta. It is a benchmark, not a learner. | `AutonomousAgent.informed_accept`, `configs/e5_helper_learning.json` |
| 21 | Belief decay | All agents decay all belief counts once per global round; the requester then adds its observation without a second discount. | `multiagent_environment.py`, `temporal_environment.py` |
| 22 | E7 discount grid | γ ∈ {1.00, 0.995, 0.98, 0.95}. 0.995 is included because 0.995⁴ ≈ 0.98: with round-robin over N = 4 agents it gives each requester the same memory per own decision as γ = 0.98 in the single-agent E3. | `configs/e7_changing_reliability.json` |
| 23 | MA-Greedy | Kept as the no-exploration baseline. With uninformative priors its estimate 0.5·U_self + 0.15 is below U_self for U_self > 0.3, so it never asks in E5–E7; it is omitted from the report tables and this is stated in the text. | `multiagent_policies.py` |
| 24 | Statistics (E5–E8) | Seeds 1–30 are shared across settings, so comparisons are paired by seed (two-sided Wilcoxon signed-rank, SciPy). Random draws are not event-aligned across policies in the multi-agent code, so this is seed pairing, not strict common random numbers. | `scripts/generate_multiagent_figures_and_tables.py` |
| 25 | Policy stability | Share of states, visited ≥ 5 times at both of two consecutive 50-step checkpoints and not tied, whose greedy action is unchanged. It measures settling, not optimality. | `multiagent_simulation.py`, `temporal_simulation.py` |
| 26 | Planned timing (E8) | `PlannedTimingPolicy`: *approximate* optimal stopping with frozen beliefs (posterior means held fixed during planning, although beliefs decay every round; the final ask decision still uses MA-D-TS samples), by backward induction on a 401-point progress grid (41-point midpoint quadrature for the uniform progress increment), using the requester's posterior means q_j = p̂_j·θ̂_j and λc_j; solo values are cached per deadline, so any task deadline is supported. It evaluates help iff Ask(p, τ) ≥ Wait(p, τ). Added because the whether-rule compares "ask now" with "never ask" and so ignores the option value of asking later; the model-free Q-learning timing policy has to learn that value from ~100 tasks per agent. The other five E8 policies are unaffected (identical per-seed results). | `requester_timing.py`, `configs/e8_temporal_help_seeking.json` |
