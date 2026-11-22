# IMPLEMENTATION PLAN: Decentralized Multi-Agent System Redesign

> Historical planning document. In the final version the helper's cost of helping depends on its state:
> `B_i * x - cost(w, rho)` with `cost = resource_cost * (3 - rho) + opportunity_cost * (1 + w)`.
> See `docs/mathematical_model.md` for the model as implemented.

## STAGE 1: Repository Inspection (COMPLETED)
✅ Inspected complete repository structure:
- Backend: 23 Python modules in `backend/src/help_seeking/`
- Frontend: React + TypeScript dashboard with live telemetry, state inspectors, round-by-round logs, and policy comparisons
- Report: Publication-grade LaTeX arXiv-format report with automated result tables and vector figures
- Configs: 13 JSON experiment configurations (E1–E8, S1–S5)
- Tests: 198 unit and integration tests (100% passing)
- Scripts: Experiment runners, diagnostic scripts, figure and table generators

---

## STAGE 2: Autonomous Agent Implementation

### 2.1 New Core Classes to Create

**File: `backend/src/help_seeking/autonomous_agent.py`**
```python
class AutonomousAgent:
    """An agent that can act as both requester and helper."""
    
    # Requester-side (existing algorithms adapted)
    - requester_policy: One of DiscountedGreedy, D-UCB, D-TS
    - acceptance_beliefs: Discounted pseudo-counts A_ij, R_ij for each potential helper j
    - trust_beliefs: Discounted Beta-Bernoulli S_ij, F_ij for each potential helper j
    
    # Helper-side (NEW)
    - helper_q_table: Q-table of shape (9, 2) for states (workload, resources) x actions (ACCEPT, REJECT)
    - helper_rng: Independent random generator for epsilon-greedy
    - workload: Current workload level (0=Low, 1=Medium, 2=High)
    - resources: Current resource level (0=Low, 1=Medium, 2=High)
    
    # Individual rewards
    - B_i: Individual cooperation reward
    - resource_cost_i: Cost of helping
    - opportunity_cost_i: Value of own task progress sacrificed
    
    # Methods
    - decide_request(U_self): Requester decision (which helper to ask, or act alone)
    - decide_help(requester_id): Helper decision (ACCEPT/REJECT) using Q-learning
    - calculate_individual_reward(x, accepted): Compute helper's reward
    - update_helping_policy(state, action, reward, next_state): Q-learning update
    - update_beliefs(helper_id, accepted, success): Update acceptance & trust beliefs
    - reset(): Reset all beliefs and Q-table
```

### 2.2 Helper State Discretization
- Workload: 3 levels (0=Low, 1=Medium, 2=High)
- Resources: 3 levels (0=Low, 1=Medium, 2=High)
- Total states: 9
- Actions: 2 (ACCEPT=0, REJECT=1)
- Q-table: np.zeros((9, 2))

### 2.3 Q-Learning Parameters
- Learning rate (alpha): 0.1
- Discount factor (gamma_Q): 0.9
- Epsilon: 0.1 (decay to 0.02 minimum)
- Epsilon-greedy with random tie-breaking
- Independent RNG per agent

---

## STAGE 3: Acceptance Belief Implementation

### 3.1 New Belief Structure
For each ordered pair (i, j) where i is requester, j is helper:
- `A_ij`: Discounted count of accepted requests
- `R_ij`: Discounted count of rejected requests
- Posterior: `p_hat_ij = (1 + A_ij) / (2 + A_ij + R_ij)`
- Update: `A_ij(t+1) = gamma * A_ij(t) + I[accepted]`
- Update: `R_ij(t+1) = gamma * R_ij(t) + I[rejected]`
- Discount all beliefs every round (like trust beliefs)

### 3.2 Integration with Trust Model
- Keep existing `TrustState` for execution reliability (theta)
- Add new `AcceptanceState` class for acceptance probability (p)
- Both discounted with same gamma

---

## STAGE 4: Requester Algorithm Modifications

### 4.1 Modified Utility Calculation
For each helper j:
- `p_hat_ij`: Estimated acceptance probability
- `theta_hat_ij`: Estimated conditional execution reliability
- `mu_hat_ij = U_self + p_hat_ij * (R * theta_hat_ij - U_self) - lambda * c_ij`

### 4.2 Algorithm Updates
**Discounted Greedy:**
- Use posterior means for both p and theta
- Select helper with max expected net utility

**Discounted UCB:**
- Confidence bounds for both p and theta
- `p_upper = min(1, p_hat + beta_p)`
- `theta_upper = min(1, theta_hat + beta_theta)`
- `mu_UCB = U_self + p_upper * max(R * theta_upper - U_self, 0) - lambda * c`

**Discounted Thompson Sampling:**
- Sample `p_sample ~ Beta(1 + A_ij, 1 + R_ij)`
- Sample `theta_sample ~ Beta(1 + S_ij, 1 + F_ij)`
- `mu_TS = U_self + p_sample * (R * theta_sample - U_self) - lambda * c`

### 4.3 Key Rules
- Rejection updates acceptance beliefs ONLY
- Rejection is NOT an execution failure
- Acceptance + success updates both acceptance and reliability
- Acceptance + failure updates both acceptance and reliability

---

## STAGE 5: Multi-Agent Environment

### 5.1 New Environment Class
**File: `backend/src/help_seeking/multiagent_environment.py`**
```python
class MultiAgentEnvironment:
    """Environment managing N autonomous agents interacting."""
    
    - agents: List[AutonomousAgent]
    - reliability_profile: How each agent's execution reliability changes
    - interaction_protocol: Sequential (for v1) or parallel
    - task_arrival_process: Which agent gets a task each round
    
    Methods:
    - step(): Execute one round of interactions
    - run(T): Run T rounds
    - reset(): Reset all agents and environment
```

### 5.2 Interaction Protocol (Sequential for v1)
For each round:
1. Select a requesting agent (round-robin or random)
2. Requester evaluates autonomous execution utility
3. Requester evaluates potential helpers
4. Requester decides whether to ask and whom
5. If asking: send request to selected helper
6. Helper observes own workload/resources
7. Helper independently chooses ACCEPT/REJECT via Q-learning
8. If REJECTED:
   - Requester performs autonomous execution
   - Communication cost incurred
   - Acceptance belief updated (rejection)
   - Execution reliability NOT updated
   - Helper receives zero reward
9. If ACCEPTED:
   - Helper executes task
   - Success/failure generated using helper's current execution reliability
   - Requester receives task reward minus communication cost
   - Helper receives individual reward: B_i * x - resource_cost - opportunity_cost
   - Helper updates Q-table
   - Requester updates acceptance and execution reliability beliefs
10. Update workloads and resources for all agents
11. Continue to next interaction

### 5.3 Agent State Dynamics
- Workload increases when helping, decreases over time
- Resources decrease when helping, recover over time
- Each agent has its own execution reliability profile (can change over time)

---

## STAGE 6: Simulation Integration

### 6.1 New Simulation Classes
**File: `backend/src/help_seeking/multiagent_simulation.py`**
```python
class MultiAgentSimulationRun:
    """Runs multi-agent simulation with logging."""
    
    - environment: MultiAgentEnvironment
    - T: Horizon
    - record_beliefs: Whether to record per-agent beliefs
    
    Recording per round:
    - For each agent: role (requester/helper/idle), action, outcome, reward
    - Per-agent beliefs (acceptance, trust, Q-table)
    - Global metrics
```

### 6.2 Metrics for Multi-Agent
Add new metrics:
- Acceptance rate (per agent, global)
- Rejection rate (per agent, global)
- Individual reward per agent
- Average helper reward
- Agent request frequency
- Agent cooperation frequency
- Task completion rate
- Communication cost (per agent, global)
- Requester net utility
- Helper policy convergence (Q-table stability)

---

## STAGE 7: Baselines, Oracle, and Metrics

### 7.1 Helper Baselines
- AlwaysAccept: Helper always accepts
- AlwaysReject: Helper always rejects
- IndependentQLearning: The new Q-learning helper policy

### 7.2 Multi-Agent Oracle
- Requester-level oracle: knows true p_ij and theta_ij for all pairs
- Computes optimal requester decisions given true helper policies
- Does NOT claim global optimality (helpers act independently)

### 7.3 Metrics Implementation
Extend `metrics.py` with multi-agent metrics.

---

## STAGE 8: Experiments (E1–E8 & S1–S5)

### 8.1 Foundational Requester Benchmarks (E1–E4)
Run with multi-agent interaction model and MAB baselines:
- **E1: Stationary Reliability** — Validate learning with stationary teammates across gap $\Delta \in \{0.02, 0.05, 0.10, 0.15\}$.
- **E2: When to Ask** — Analyze whether-to-ask adaptation as self-capability $U_{\mathrm{self}} \in \{0.30, 0.50, 0.65, 0.80\}$ varies.
- **E3: Abrupt Reliability Shift** — Adaptation latency $\tau_{\mathrm{adapt}}$ and recovery across $\gamma \in \{1.0, 0.995, 0.98, 0.95, 0.90\}$.
- **E4: Communication Cost Variation** — Rational policy shift from asking to autonomy as $\lambda \in \{0, 1.0, 2.5, 4.0, 6.0\}$ scales.

### 8.2 Decentralized Multi-Agent Experiments (E5–E7)
- **E5: Autonomous Helper Policy Learning**
  - Independent Tabular Q-learning vs Always-Accept vs Always-Reject vs Informed Threshold.
  - State-dependent helping cost $\kappa_j(w, \rho) = c_{\mathrm{res}, j}(1+(2-\rho)) + c_{\mathrm{opp}, j}(1+w)$ across Idle, Medium, and Busy states.
- **E6: Multi-Agent Cooperation Dynamics**
  - Emergent cooperation and congestion load shifting under varying cooperation benefits $B_i \in \{0.20, 0.50, 0.80\}$.
  - Heterogeneous agent teams with varying baseline costs and reliabilities.
- **E7: Changing Teammate Reliability in Decentralized Teams**
  - Abrupt drop in teammate competence in a 4-agent peer network at round $t=1000$.
  - Evaluation of discount factors $\gamma \in \{1.00, 0.995, 0.98, 0.95\}$.
  - Analysis of interaction frequency: per-round $\gamma = 0.995$ is equivalent to $\gamma^4 \approx 0.98$ on personal decision clocks.

### 8.3 Sequential Temporal Help-Seeking (E8)
- **E8: When to Ask Under Hard Deadlines ($D=5$)**
  - Multi-step task lifecycles with continuous progress $p \in [0, 1]$ and deadline $\tau_{\mathrm{rem}} \in \{0, \dots, D\}$.
  - Comparison of 6 timing policies: Never-Ask, Ask-Immediate, Ask-Urgent, Fixed-Delay, Tabular Q-Learning (27 compact states), and Planned Timing (model-based backward induction optimal stopping).
  - Helper reliability drop at task 200 ($0.85 \to 0.35$).

### 8.4 Sensitivity Suite (S1–S5)
- **S1:** Discount factor $\gamma$ sweep in stationary conditions ($\gamma \in \{0.90, 0.95, 0.98, 0.99, 1.00\}$).
- **S2:** UCB exploration parameter $\alpha_{\mathrm{ucb}} \in \{0.5, 1.0, 1.5, 2.0, 3.0\}$.
- **S3:** Continuous reliability drift (sinusoidal and linear).
- **S4:** Cross-variation of $U_{\mathrm{self}}$ and $\gamma \in \{0.95, 0.98, 1.00\}$.
- **S5:** Multi-shift non-stationary regimes (abrupt shifts at $t \in \{500, 1000, 1500\}$).

### 8.5 Experiment Configurations and Execution
- 13 JSON experiment configs (`configs/e1_*.json` to `configs/e8_*.json`, `configs/s1_*.json` to `configs/s5_*.json`).
- 30 independent seeds per configuration (9,240 total simulation runs).
- Independent RNGs per agent, proper belief and Q-table resets between runs.

---

## STAGE 9: Dashboard Updates

### 9.1 Frontend Changes
- "Multi-Agent Live" interactive view (`frontend/src/components/MultiAgentView.tsx`).
- Real-time display of all 4 peer agents: physical workload, resource meters, Q-values ($Q(\text{ACCEPT})$ vs $Q(\text{REJECT})$), and Beta-Bernoulli beliefs ($\hat{p}_{ij}$, $\hat{\theta}_{ij}$).
- Round-by-round telemetry: task arrival, requester decision, selected helper, outcome, and individualized payoffs.
- Full interactive control: Step 1, Step 50, Step 500, and Reset.

### 9.2 API Extensions
- Fast simulation endpoints in `backend/src/help_seeking/api.py`.
- Step-by-step telemetry serialization for frontend state inspection.

---

## STAGE 10: Research Report Updates

### 10.1 Mathematical Formulation Updates
- Complete formal mathematical model (`docs/mathematical_model.md`).
- Decision-epoch semi-Markov helper Q-learning.
- Decoupled Beta-Bernoulli beliefs for willingness and competence.
- Dynamic oracle with $\epsilon$-greedy tie-breaking and ex-ante pseudo-regret.
- Irwin-Hall continuation value and planned timing optimal stopping.

### 10.2 Architecture Section
- Full multi-agent interaction sequence and dual-role transitions.
- Internal state dynamics $(w, \rho)$.
- State-dependent helping cost function $\kappa(w, \rho)$.

### 10.3 Experimental Results
- Comprehensive results across all 8 experiments (E1–E8) and sensitivity suite (S1–S5).
- Automated LaTeX tables in `report/tables/` (`table1_e1_metrics.tex` to `table8_e8_metrics.tex`).
- Publication figures in `figures/` (`e1_regret_curves.pdf` through `e8_temporal_help_seeking.png`).
- Analysis of emergent cooperation, congestion relief, and clock divergence.

---

## FILES CREATED AND MAINTAINED

### Core Multi-Agent Backend (`backend/src/help_seeking/`):
1. `autonomous_agent.py` — Fluid dual-role agent class (internal state, Q-table, beliefs, decision logic)
2. `acceptance.py` — Beta-Bernoulli acceptance belief tracker
3. `helper_qlearning.py` — Decision-epoch semi-Markov helper Q-learning
4. `multiagent_environment.py` — Multi-agent environment managing round-robin task arrivals and state transitions
5. `multiagent_simulation.py` — Multi-agent simulation runner with telemetry logging
6. `multiagent_policies.py` — Multi-agent requester policies (MA-Greedy, MA-D-UCB, MA-D-TS, Dynamic Oracle)
7. `task_lifecycle.py` — Multi-step task lifecycle state and progression
8. `requester_timing.py` — Timing policies (Immediate, Urgent, Fixed-Delay, Tabular Q-learning, Planned Timing)
9. `temporal_environment.py` — Sequential operational environment under deadlines
10. `temporal_simulation.py` — Sequential simulation runner for E8

### Experiment Configurations (`configs/`):
1. `e1_stationary.json` to `e8_temporal_help_seeking.json` (all 8 experiments)
2. `s1_gamma_stationary.json` to `s5_multishift.json` (all 5 sensitivity experiments)

### Test Suite (`backend/tests/`):
17 test suites with 198 unit and integration tests covering multi-agent mechanics, helper Q-learning, timing policies, and oracle regret.

---

## SUCCESS CRITERIA (ALL MET)

1. **198 automated tests pass** (`pytest backend/tests/` -> 100% passing).
2. **E1–E4 reproduce baseline MAB dynamics** (H1–H5 confirmed).
3. **E5 demonstrates helper Q-learning convergence** (learns state-dependent selectivity matching informed threshold).
4. **E6 reveals emergent cooperation** ($B_i$ scaling, congestion load-shifting).
5. **E7 shows peer network adaptation** (clock divergence between global rounds and personal decisions).
6. **E8 demonstrates superior deadline completion** (Planned Timing and Tabular Q-learning outperform static heuristics).
7. **Interactive Dashboard operational** (serves telemetry and simulation views).
8. **Publication manuscript complete** (`report/main.pdf` compiled with full E1–E8 tables and figures).