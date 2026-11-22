# REVISED PROJECT PROPOSAL / IDEA DOCUMENT
## CS329 – Foundations of AI: Multi-Agent Systems

**Original Title:** *Learning When and Whom to Ask for Help Under Changing Teammate Reliability*  
**Revised Title:** **Learning When and Whom to Ask for Help in Decentralized Multi-Agent Systems Under Changing Teammate Reliability**

---

## 1. Motivation: Addressing the Evaluator Critique

### Evaluator Critique
> *"The helpers are fixed and never make their own decisions, so this is really one agent choosing between options. Try to make the problem truly multiagent."*

### The Core Flaw in the Original Idea
In the original proposal, the system was formulated as a **Single-Agent Non-Stationary Multi-Armed Bandit**:
- There was only **one** decision-maker (the requester).
- Teammates were passive static stochastic functions (arms) that unconditionally executed tasks when called.
- Teammates had no internal state, no policy, no independent learning, no goals, and no actions.
- Game-theoretically, this was an individual decision problem against nature, not a multi-agent system.

---

## 2. The Revised Multi-Agent Idea: 5 Core Changes

To make the problem fundamentally multi-agent, we transform the system into an **ad-hoc decentralized cooperative network** of $N$ symmetric, autonomous agents $\mathcal{A} = \lbrace A_1, A_2, \dots, A_N \rbrace$.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                           AUTONOMOUS AGENT A_i                                │
├───────────────────────────────────────────────────────────────────────────────┤
│  ROLE 1: REQUESTER (When holding a task)                                      │
│  • Solves online exploration-exploitation dilemma (D-Greedy, D-UCB, D-TS)     │
│  • Learns teammate availability: p_ij = P(A_j accepts request from A_i)       │
│  • Learns teammate execution capability: theta_ij = P(Success | Accepted)     │
│  • Evaluates utility of asking vs. autonomous execution                       │
├───────────────────────────────────────────────────────────────────────────────┤
│  ROLE 2: HELPER (When receiving a request from peer A_k)                      │
│  • Observes internal state: s_i(t) = (workload_i(t), resources_i(t)) ∈ [3x3]  │
│  • Independent Action: ACCEPT (0) or REJECT (1)                               │
│  • Learns policy via private Tabular Q-Learning (Bellman optimality)          │
│  • Selfish/Individual payoff: r_i = B_i * x - cost(w_i, rho_i)                │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Change 1: Dynamic Dual Roles & Reciprocal Peer Interactions
- There is **no permanent requester** and **no permanently designated helper pool**.
- Every agent receives tasks over time and acts as a requester.
- Every agent receives requests from peers and acts as a helper.
- Role switching is fluid: Agent $A_1$ may ask $A_2$ in round 1, and $A_2$ may ask $A_1$ in round 2.

### Change 2: Autonomous Helper Decisions with Physical State Dynamics
- Helpers are **autonomous agents** that can say **NO**.
- Each agent maintains an internal state:

$$s_i(t) = (\text{workload}_i(t),\, \text{resources}_i(t)) \in \lbrace 0, 1, 2 \rbrace \times \lbrace 0, 1, 2 \rbrace$$

  - $\text{workload} \in \lbrace \text{Low}, \text{Medium}, \text{High} \rbrace$: increases by $+1$ when helping; cools down when idle.
  - $\text{resources} \in \lbrace \text{Low}, \text{Medium}, \text{High} \rbrace$: depletes by $-1$ when helping; regenerates when idle.
- Helper action space: $\mathcal{A}_i = \lbrace \text{ACCEPT}, \text{REJECT} \rbrace$.

### Change 3: Decentralized Helper Q-Learning with Individual Payoffs
- Helpers learn when to accept assistance requests using **independent Tabular Q-Learning**:

$$Q_i(s, a) \leftarrow Q_i(s, a) + \alpha_{\mathrm{RL}} \left[ r_i + \gamma_Q \max_{a'} Q_i(s', a') - Q_i(s, a) \right]$$

- Helpers optimize their **individual reward** (no shared team reward, no central arbiter):

$$
r_i = \begin{cases} B_i \cdot x - \kappa_i(w_i, \rho_i), & \text{if } \text{ACCEPT} \\\\ 0, & \text{if } \text{REJECT} \end{cases}
$$

$$\kappa_i(w, \rho) = c_{\mathrm{res},i}(3 - \rho) + c_{\mathrm{opp},i}(1 + w)$$

where $B_i$ is cooperation compensation, $x \in \lbrace 0, 1 \rbrace$ is task success, and the helping cost $\kappa_i$ (resource use plus foregone progress on the agent's own queue) grows with the helper's workload and resource depletion at the moment it accepts. (Final version, Nov 2026: an earlier draft used a state-independent cost, under which the optimal helping decision did not depend on the helper's state.)

### Change 4: Decoupled Acceptance Beliefs vs. Execution Trust
A requester must now track two independent latent variables for each teammate $A_j$:
1. **Acceptance Probability** $p_{ij} = P(A_j \text{ accepts } A_i)$:  
   Estimated via discounted pseudo-counts $A_{ij}$ (accepted) and $R_{ij}$ (rejected):

$$\hat{p}_{ij} = \frac{1 + A_{ij}}{2 + A_{ij} + R_{ij}}$$

2. **Conditional Execution Reliability** $\theta_{ij} = P(\text{Success} \mid \text{Accepted})$:  
   Estimated via discounted Beta-Bernoulli trust pseudo-counts $S_{ij}$ (success) and $F_{ij}$ (failure):

$$\hat{\theta}_{ij} = \frac{1 + S_{ij}}{2 + S_{ij} + F_{ij}}$$

*Critical Rule:* **Rejection is NOT an execution failure.** Rejection updates acceptance beliefs only; execution reliability is untouched.

### Change 5: Autonomous Return on Rejection & Expected Net Utility
If a helper rejects a request, the task returns immediately to the requester for autonomous execution in the same round.
The expected net utility of querying helper $j$ becomes:

$$\mu_{ij} = p_{ij}(R\theta_j - \lambda c_{ij}) + (1 - p_{ij})(U_i^{\mathrm{self}} - \lambda c_{ij}) = U_i^{\mathrm{self}} + p_{ij}(R\theta_j - U_i^{\mathrm{self}}) - \lambda c_{ij}$$

Estimated utility:

$$\hat{\mu}_{ij} = U_i^{\mathrm{self}} + \hat{p}_{ij}(R\hat{\theta}_{ij} - U_i^{\mathrm{self}}) - \lambda c_{ij}$$

The requester queries candidate helper:

$$j^\star = \arg\max_j \hat{\mu}_{ij}$$

$$\text{Decision: query } j^\star \text{ if } \hat{\mu}_{ij^\star} > U_i^{\mathrm{self}}, \quad \text{else act alone } (d_t = 0)$$

---

## 3. Why This Makes the System Truly Multi-Agent

| Feature | Original Proposal | Revised Multi-Agent Redesign |
|---|---|---|
| **System Paradigm** | 1-agent MAB against passive arms | $N$ decentralized autonomous agents |
| **Agent Roles** | Fixed Requester + Fixed Helpers | Fluid dual roles (Requester $\leftrightarrow$ Helper) |
| **Helper Behavior** | 100% unconditional compliance | Autonomous ACCEPT / REJECT decisions |
| **Helper State** | None (stateless) | Dynamic physical state: $(\text{workload}, \text{resources})$ |
| **Helper Learning** | None (fixed exogenous parameter) | Tabular Q-learning over individual rewards |
| **Helper Motivation** | Free labor | Self-interested payoff: `r_i = B_i · x - κ_i(w_i, ρ_i)` (cost grows with load) |
| **Learned Beliefs** | Execution reliability $\theta$ only | Decoupled: Willingness `p_ij` and competence `θ_ij` |
| **Emergent Dynamics** | Simple arm convergence | Congestion-driven load shifting, cooperation emergence |

---

## 4. Emergent Multi-Agent Dynamics

1. **Endogenous Congestion Balancing:**  
   If agent $A_3$ has the highest execution reliability ($\theta_3 = 0.9$), all other agents prefer to ask $A_3$. This floods $A_3$'s queue, increasing its workload to High ($w_3 = 2$) and exhausting its resources ($\rho_3 = 0$). Under its Q-learning policy, $A_3$ begins rejecting requests. Requesters observe the drop in $\hat{p}_{i3}$ and adaptively shift their requests to second-best helpers (e.g., $A_1$ or $A_2$).
2. **Cooperation Incentives:**  
   If cooperation reward $B_i$ is low ($B_i = 0.20$), agents protect their resources and reject requests (31% rejection rate). If $B_i$ is high ($B_i = 0.80$), helping is profitable and sustainable cooperation emerges (95% acceptance rate).
3. **No Centralized Point of Failure:**  
   Agents operate entirely on local observations, local Q-tables, and local communication without needing a global coordinator.

---

## 5. Experimental Roadmap (E1--E8 & S1--S5)

### Primary Experimental Suite (E1--E8)
- **E1: Stationary Reliability** under reciprocal peer interactions.
- **E2: Whether-to-Ask** as autonomous self-capability $U_{\mathrm{self}}$ varies.
- **E3: Abrupt Reliability Drop** demonstrating fast adaptation via discounting ($\gamma = 0.98$).
- **E4: Communication Cost Scaling** showing rational transition from asking to autonomy.
- **E5: Helper Policy Learning (NEW):** Empirical comparison of Tabular Q-Learning vs. Always-Accept vs. Always-Reject and Informed Threshold.
- **E6: Multi-Agent Cooperation Dynamics (NEW):** Analysis of emergent cooperation, workload congestion, and acceptance rates across reward incentives ($B_i \in \lbrace 0.2, 0.5, 0.8 \rbrace$) and heterogeneous teams.
- **E7: Changing Teammate Reliability in Teams (NEW):** Abrupt reliability shift in decentralized 4-agent teams; clock divergence between per-decision clocks and global rounds ($\gamma=0.995$ vs $\gamma=0.98$).
- **E8: Temporal Help-Seeking & Timing under Deadlines (NEW):** Multi-step task lifecycle ($D=5$); tabular 27-state Q-learning timing, heuristic baselines (Immediate, Fixed Delay, Urgency), and belief-based Planned Timing (optimal stopping).

### Robustness & Sensitivity Suite (S1--S5)
- **S1: Discounting in Stationary Worlds** — Parametric sweep over $\gamma \in \lbrace 0.90, 0.95, 0.98, 0.99, 1.00 \rbrace$ confirming retention of evidence without premature forgetting.
- **S2: UCB Exploration Intensity** — Calibration of the upper-confidence parameter $\alpha_{\mathrm{ucb}} \in \lbrace 0.5, 1.0, 1.5, 2.0, 3.0 \rbrace$.
- **S3: Continuous Reliability Drift** — Tracking gradual sinusoidal and linear non-stationary drift in peer competence.
- **S4: Whether-to-Ask Under Varying Discounting** — Cross-sensitivity analysis of $U_{\mathrm{self}}$ thresholds under different memory horizons.
- **S5: Multi-Shift Non-Stationary Regimes** — Stress-testing policy recovery across multiple recurring competence disruptions (rounds 500, 1000, 1500).
