# REQUIREMENTS.md

## Project Title
Learning When and Whom to Ask for Help Under Changing Teammate Reliability: A Decentralized Multi-Agent Formulation

## Team (Group 24 — IIT Gandhinagar CS329 MAS)
- Cherukuri Harshith Sai (Roll 24110091) — Multi-Agent Architecture, Environment Formulation, Dynamic Oracle & E1/E7 Lead
- Atragada Venkata Naga Sai Ganesh (Roll 24110060) — Decentralized Learning Algorithms, Trust & Acceptance Estimation, E5 Lead
- Ambati Chaitanya Ram (Roll 24110035) — Autonomous Agent Policies, Helper Q-Learning Dynamics & E2/E3/E6 Lead
- Dasari Lohith (Roll 24110100) — Temporal Task Lifecycle, Timing Policies, Evaluation & E4/E8 Lead

---

## Sources of Truth
1. **Proposal Document:** `docs/Project_Idea.pdf` (14 pages, initial single-agent MAB proposal).
2. **Revised Project Specification:** `docs/REVISED_PROJECT_IDEA.md` (decentralized multi-agent architecture and emergent cooperation).
3. **Formal Mathematical Specification:** `docs/mathematical_model.md` (definitive mathematical formulation of agent states, semi-Markov helper Q-learning, decoupled beliefs, dynamic oracle, ex-ante pseudo-regret, and temporal timing policies).

---

## System Architecture and Key Mathematical Definitions

### 1. Decentralized Multi-Agent Architecture
- $N$ autonomous peer agents $\mathcal{A} = \lbrace A_1, \dots, A_N \rbrace$ operating without a centralized controller or shared team reward.
- **Fluid Dual Roles:** Agents dynamically alternate between **Requester** ($d_t \in \lbrace 0, 1 \rbrace$) and **Helper** ($a_t \in \lbrace \text{ACCEPT}, \text{REJECT} \rbrace$).
- **Physical Internal State:** $s_i(t) = (w_i(t), \rho_i(t)) \in \lbrace 0, 1, 2 \rbrace^2$ (9 discrete states spanning Workload and Resource levels).
- **Activity State Transitions:**
  - Own Execution / Fallback: $w \leftarrow \min(2, w+1)$, $\rho \leftarrow \max(0, \rho-1)$
  - Accepted Helping: $w \leftarrow \min(2, w+1)$, $\rho \leftarrow \max(0, \rho-1)$
  - Idle / Offloaded: $w \leftarrow \max(0, w-1)$, $\rho \leftarrow \min(2, \rho+1)$
  - Rejected Helping: $w \leftarrow \max(0, w-1)$, $\rho \leftarrow \rho$ (preserves resources)

### 2. Decision-Epoch Helper Q-Learning
- Tabular Q-function: $Q_j: \mathcal{S} \times \lbrace \text{ACCEPT}, \text{REJECT} \rbrace \to \mathbb{R}$.
- Semi-Markov decision-epoch update accounting for variable elapsed rounds $\Delta t = \max(1, t - t_{\text{prev}})$:

$$Q_j(s, a) \leftarrow Q_j(s, a) + \alpha \left[ r + \gamma_Q^{\Delta t} \max_{a'} Q_j(s', a') - Q_j(s, a) \right]$$

- **State-Dependent Helping Cost:**

$$\kappa_j(w, \rho) = c_{\mathrm{res}, j} (1 + (2 - \rho)) + c_{\mathrm{opp}, j} (1 + w)$$

- **Helper Payoff:** $r_j(t) = B_j \cdot x_t - \kappa_j(w_j, \rho_j)$ if accepted; $0$ if rejected.

### 3. Decoupled Requester Belief Tracking
- **Acceptance Probability Belief:** Discounted Beta-Bernoulli with pseudo-counts $A_{ij}, R_{ij}$:

$$\hat{p}_{ij}(t) = \frac{1 + A_{ij}(t)}{2 + A_{ij}(t) + R_{ij}(t)}$$

- **Conditional Execution Reliability Belief:** Discounted Beta-Bernoulli with pseudo-counts $S_{ij}, F_{ij}$:

$$\hat{\theta}_{ij}(t) = \frac{1 + S_{ij}(t)}{2 + S_{ij}(t) + F_{ij}(t)}$$

- **Expected Net Utility:**

$$\hat{\mu}_{ij}(t) = U_i^{\mathrm{self}} + \hat{p}_{ij}(t) \left( R \hat{\theta}_{ij}(t) - U_i^{\mathrm{self}} \right) - \lambda c_{ij}$$

### 4. Dynamic Oracle and Ex-Ante Pseudo-Regret
- Oracle incorporates exact helper $\epsilon$-greedy acceptance probabilities $p_{ij}(t) = (1 - \epsilon)\mathbb{P}_{\text{greedy}}(\text{ACCEPT}) + \frac{\epsilon}{2}$.
- Complete set of optimal actions $\mathcal{O}_t$ with tie-breaking:

$$\mathcal{O}_t = \left\lbrace (d, a) \;\middle|\; |\mu_{d, a}(t) - \mu^\star(t)| < 10^{-9} \right\rbrace$$

- Ex-ante pseudo-regret evaluated on expected choice utility prior to stochastic outcome realization:

$$\mathrm{Regret}_t = \max\left(0,\, \mu^\star(t) - \mu_{\mathrm{chosen}}(t)\right)$$

### 5. Temporal Task Lifecycle (WHEN to Ask)
- Task tuple $\mathcal{T} = (p, D, \tau_{\mathrm{rem}}, a_{\mathrm{solo}}, \text{status})$ under hard deadline $D \in \mathbb{N}^+$.
- Irwin-Hall exact continuation value of solo finishing $V_{\mathrm{solo}}(p, \tau_{\mathrm{rem}})$.
- State space: 27 discrete states $s_{\mathrm{timing}} = (p_{\mathrm{lvl}}, \tau_{\mathrm{lvl}}, a_{\mathrm{lvl}}) \in \lbrace 0, 1, 2 \rbrace^3$.
- Planned Timing policy computed via backward induction optimal stopping.

---

## Experimental Suite (E1–E8 & S1–S5)

### E1: Stationary Reliability Benchmark
- **Goal:** Validate requester learning in stationary peer environments with constant competences.
- **Independent Variable:** Competence gap $\Delta \in \lbrace 0.02, 0.05, 0.10, 0.15 \rbrace$.
- **Parameters:** $\theta = (0.60+\Delta, 0.60, 0.50)$, $c = (0.10, 0.10, 0.10)$, $K=3$, $T=2000$, $R=1.0$, $U_{\mathrm{self}}=0.35$, $\lambda=1.0$, $\gamma=1.0$.
- **Policies:** Stationary Oracle, Random, Fixed-Cheapest, Greedy, D-UCB, D-TS.
- **Metrics:** Cumulative regret, optimal teammate selection rate, MAE of beliefs, ask rate.

### E2: When to Ask (Autonomy vs Delegation)
- **Goal:** Analyze whether-to-ask adaptation as agent autonomous self-capability varies.
- **Independent Variable:** Autonomous utility $U_{\mathrm{self}} \in \lbrace 0.30, 0.50, 0.65, 0.80 \rbrace$.
- **Parameters:** $\theta = (0.75, 0.65, 0.50)$, $c = (0.10, 0.10, 0.10)$, $K=3$, $T=2000$, $R=1.0$, $\lambda=1.0$, $\gamma=0.98$.
- **Policies:** Dynamic Oracle, Always-Ask+Greedy, Never-Ask, D-UCB, D-TS.
- **Metrics:** Ask rate, ask decision accuracy ($\text{Acc}_{\mathrm{ask}}$), average net utility, total communication cost.

### E3: Abrupt Reliability Shift
- **Goal:** Evaluate non-stationary adaptation speed and regret recovery following teammate degradation.
- **Independent Variable:** Discount factor $\gamma \in \lbrace 1.0, 0.995, 0.98, 0.95, 0.90 \rbrace$.
- **Parameters:** Initial $\theta(0) = (0.80, 0.70, 0.90)$, post-shift $\theta(1000) = (0.80, 0.70, 0.40)$, $U_{\mathrm{self}}=0.50$, $\lambda=1.0$, $T=2000$.
- **Policies:** Dynamic Oracle, Initial-Knowledge Oracle, Greedy, D-UCB, D-TS.
- **Metrics:** Adaptation latency ($\tau_{\mathrm{adapt}}$), post-change regret, average net utility, optimal teammate selection.

### E4: Communication Cost Variation
- **Goal:** Test rational policy shift from delegation to autonomy under increasing communication penalty.
- **Independent Variable:** Cost multiplier $\lambda \in \lbrace 0, 1.0, 2.5, 4.0, 6.0 \rbrace$.
- **Parameters:** $\theta = (0.60, 0.75, 0.90)$, $c = (0.05, 0.10, 0.20)$, $K=3$, $T=2000$, $R=1.0$, $U_{\mathrm{self}}=0.45$, $\gamma=0.98$.
- **Policies:** Dynamic Oracle, Fixed-Cheapest, Greedy, D-UCB, D-TS.
- **Metrics:** Teammate selection share, ask rate, total communication cost, average net utility.

### E5: Autonomous Helper Policy Learning
- **Goal:** Evaluate decentralized helper tabular Q-learning vs non-adaptive and informed baselines.
- **Policies Compared:** Independent Tabular Q-Learning, Always-Accept, Always-Reject, Informed Threshold.
- **Parameters:** $N=4$ autonomous agents, 9-state $(w, \rho)$ space, state-dependent helping cost $\kappa(w, \rho)$, $T=2000$ rounds, 30 seeds.
- **Metrics:** State-dependent acceptance rate (Idle vs Medium vs Busy), helper payoff, requester regret, team social welfare.

### E6: Multi-Agent Cooperation Dynamics
- **Goal:** Investigate emergent cooperation, load shifting, and congestion balancing in heterogeneous teams.
- **Parameters:** Cooperation incentive $B_i \in \lbrace 0.20, 0.50, 0.80 \rbrace$, heterogeneous abilities $\theta = (0.70, 0.60, 0.50, 0.80)$, varying base resource and opportunity costs.
- **Metrics:** Per-agent acceptance rate, physical workload distribution, resource depletion rate, social welfare $W = U_{\mathrm{req}} + U_{\mathrm{help}}$.

### E7: Changing Teammate Reliability in Decentralized Teams
- **Goal:** Characterize belief adaptation and regret recovery in 4-agent peer networks under abrupt competence drops.
- **Parameters:** $N=4$ agents, $\gamma \in \lbrace 1.00, 0.995, 0.98, 0.95 \rbrace$, Agent 4 reliability drops $0.80 \to 0.30$ at round $t=1000$.
- **Key Finding:** Round-robin scheduling spreads interactions across $N$ agents, making per-round $\gamma = 0.995$ functionally equivalent to $\gamma^4 \approx 0.98$ on an agent's personal decision clock.
- **Metrics:** Cumulative pseudo-regret, adaptation latency, tracking MAE of $\hat{p}$ and $\hat{\theta}$, team social welfare.

### E8: Sequential Help-Seeking Under Deadlines
- **Goal:** Resolve **WHEN** during a multi-step task lifecycle to seek help under hard deadlines ($D=5$).
- **Timing Policies Compared:**
  1. Never-Ask (Pure Solo Execution)
  2. Ask-Immediate ($u_1 = 1$, asks at inception)
  3. Ask-Urgent ($u_k = 1$ only when $\tau_{\mathrm{rem}} \le 1$)
  4. Fixed-Delay ($u_k = 1$ on second attempt)
  5. Tabular Q-Learning (Learns $Q(s_{\mathrm{timing}}, u)$ across 27 compact states)
  6. Planned Timing (Model-based backward induction optimal stopping)
- **Environment:** Multi-step progression with Irwin-Hall distribution; helper changes reliability at task 200 ($0.85 \to 0.35$).
- **Metrics:** Task completion rate under deadline, cumulative requester reward, timing distribution of requests.

### Sensitivity Analysis Suite (S1–S5)
- **S1: Gamma Sweep in Stationary Environments** — $\gamma \in \lbrace 0.90, 0.95, 0.98, 0.99, 1.00 \rbrace$ under constant reliability.
- **S2: UCB Exploration Parameter Sweep** — $\alpha_{\mathrm{ucb}} \in \lbrace 0.5, 1.0, 1.5, 2.0, 3.0 \rbrace$.
- **S3: Non-Stationary Reliability Drift** — Continuous sinusoidal/linear drift in teammate reliability.
- **S4: When-to-Ask Under Varying Discounting** — Interaction of $U_{\mathrm{self}}$ and $\gamma \in \lbrace 0.95, 0.98, 1.00 \rbrace$.
- **S5: Multi-Shift Non-Stationary Regime** — Multiple abrupt shifts occurring at rounds 500, 1000, and 1500.

---

## Algorithms and Baselines Required

### Requester Policies:

1. **MultiAgent Greedy (MA-Greedy):** Queries candidate helper with highest expected net utility:

$$j^\star = \arg\max_j \hat{\mu}_{ij}(t)$$

$$\text{Decision: query } j^\star \text{ if } \hat{\mu}_{ij^\star}(t) > U_i^{\mathrm{self}}, \quad \text{else act alone } (d_t = 0)$$

2. **MultiAgent Discounted UCB (MA-D-UCB):** Uses optimistic upper confidence bounds on both acceptance and reliability.

3. **MultiAgent Discounted Thompson Sampling (MA-D-TS):** Beta posterior sampling on both acceptance and reliability:

$$j^\star = \arg\max_j \tilde{\mu}_{ij}(t)$$

$$\text{Decision: query } j^\star \text{ if } \tilde{\mu}_{ij^\star}(t) > U_i^{\mathrm{self}}, \quad \text{else act alone } (d_t = 0)$$

4. **Dynamic Oracle:** Full information benchmark with exact helper $\epsilon$-greedy tie-breaking probabilities.

### Helper Policies:
1. **Independent Tabular Q-Learning:** Learns state-dependent acceptance over $(w, \rho)$ states.
2. **Always-Accept:** Unconditional compliance baseline.
3. **Always-Reject:** Purely selfish baseline.
4. **Informed Threshold:** Optimal non-learning threshold baseline knowing true $\theta_j$.

### Timing Policies (E8):
1. **Solo / Never-Ask**
2. **Ask-Immediate**
3. **Ask-Urgent**
4. **Fixed-Delay**
5. **Tabular Q-Learning (27 States)**
6. **Planned Timing (Backward Induction Optimal Stopping)**

---

## Evaluation Metrics

- **Cumulative Dynamic Pseudo-Regret:**

$$\mathrm{Regret}(T) = \sum_{t=1}^T \left(\mu^\star(t) - \mu_{\mathrm{chosen}}(t)\right)$$

- **Average Net Utility:**

$$U_{\mathrm{net}} = \frac{1}{T} \sum_{t=1}^T r_{\mathrm{requester}, t}$$

- **Social Welfare:**

$$W = U_{\mathrm{req}} + U_{\mathrm{help}}$$

- **Ask Rate:** Proportion of rounds where assistance is requested ($d_t = 1$)
- **Ask Decision Accuracy ($\text{Acc}_{\mathrm{ask}}$):** Consistency of binary whether-to-ask decisions with dynamic oracle
- **Optimal-Teammate Selection Rate:** Accuracy of whom-to-ask conditional on requesting help
- **Adaptation Latency ($\tau_{\mathrm{adapt}}$):** Number of rounds post-shift required to switch to the new optimal peer
- **Belief MAE (`MAE_p`, `MAE_θ`):** Mean absolute error between beliefs and true parameters
- **State-Dependent Acceptance Rate:** Helper compliance segregated by physical state (Idle vs Medium vs Busy)
- **Deadline Task Completion Rate:** Proportion of multi-step tasks completed before deadline expiration ($p \ge 1.0$)

---

## Verification and Success Criteria
1. **198 automated unit and integration tests** passing with 0 failures (`pytest backend/tests/`).
2. **All 13 experiment configurations** (E1–E8, S1–S5) successfully executed across 30 random seeds (9,240 total simulation runs).
3. **Full interactive visualization:** React + TypeScript dashboard with live telemetry, state inspectors, round-by-round logs, and policy comparisons.
4. **Publication-grade LaTeX manuscript:** arXiv-formatted report (`report/main.pdf`) with automated tables (`report/tables/`) and figures (`figures/`).
