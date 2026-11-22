# Decentralized Multi-Agent Mathematical Model

## 1. Overview and Agent Autonomy

The system consists of $N$ autonomous peer agents:

$$\mathcal{A} = \{A_1, A_2, \dots, A_N\}$$

Every agent can dynamically alternate between two roles:
1. **Requester Role:** Receives an incoming task, determines whether to execute autonomously ($d_t = 0$) or query a teammate ($d_t = 1$), and selects which peer $j \in \mathcal{A} \setminus \{i\}$ to contact.
2. **Helper Role:** Receives help requests from peers, observes its own internal state, and independently decides whether to accept ($a_t = 0$) or reject ($a_t = 1$).

There is **no centralized controller**, no shared team reward, and no permanent requester or helper designation. Exogenous task arrivals are scheduled by the environment, but all decisions, beliefs, and Q-tables are completely private and decentralized.

---

## 2. Agent State Space and Activity Transitions

Each agent $A_i$ maintains an internal state:

$$s_i(t) = (w_i(t), \rho_i(t)) \in \{0, 1, 2\}^2$$

discretized into 9 states ($w \in \{0, 1, 2\}$: Low, Med, High workload; $\rho \in \{0, 1, 2\}$: Low, Med, High resources).

At round $t$, internal states transition according to four mutually exclusive activities:
1. **Executes Own Task (`execute_own_task`):**
   - Condition: Requester acts alone ($d_t = 0$) or executes fallback after a peer rejection ($a_t = \text{REJECT}$).
   - Dynamics: $w_i \leftarrow \min(2, w_i + 1)$, $\rho_i \leftarrow \max(0, \rho_i - 1)$.
2. **Accepts and Helps Peer (`help_another_agent`):**
   - Condition: Helper accepts an incoming request ($a_t = \text{ACCEPT}$).
   - Dynamics: $w_j \leftarrow \min(2, w_j + 1)$, $\rho_j \leftarrow \max(0, \rho_j - 1)$.
3. **Idle / Rest (`rest`):**
   - Condition: Uninvolved agents, or a requester who successfully offloaded its task to a helper.
   - Dynamics: $w_k \leftarrow \max(0, w_k - 1)$, $\rho_k \leftarrow \min(2, \rho_k + 1)$.
4. **Rejects Request (`reject_request`):**
   - Condition: Helper declines an incoming help request ($a_t = \text{REJECT}$).
   - Dynamics: Avoids task execution. Workload relieves toward baseline while resources are preserved without depletion: $w_j \leftarrow \max(0, w_j - 1)$, $\rho_j \leftarrow \rho_j$.

---

## 3. Decision-Epoch Helper Q-Learning

Each agent maintains an independent Q-table for helping decisions:

$$Q_i: \mathcal{S} \times \{\text{ACCEPT}, \text{REJECT}\} \to \mathbb{R}$$

Because helping decisions occur asynchronously at discrete request rounds rather than every global simulation step, helper learning is formulated as **decision-epoch semi-Markov Q-learning**.

When agent $A_j$ receives a request at round $t$:

- **Step 1 (State Update):** If a previous transition $(s_{\text{prev}}, a_{\text{prev}}, r_{\text{prev}}, t_{\text{prev}})$ is pending, it is updated using the helper's current state $s_j(t)$ as the true successor state $s'$:

$$Q_j(s_{\text{prev}}, a_{\text{prev}}) \leftarrow Q_j(s_{\text{prev}}, a_{\text{prev}}) + \alpha \left[ r_{\text{prev}} + \gamma_Q^{\Delta t} \max_{a'} Q_j(s', a') - Q_j(s_{\text{prev}}, a_{\text{prev}}) \right]$$

where $\Delta t = \max(1, t - t_{\text{prev}})$ accounts for geometric discounting over elapsed rounds.

- **Step 2 (Action Selection):** The agent selects an action using $\epsilon$-greedy exploration with random tie-breaking.
- **Step 3 (Store Transition):** The new decision transition $(s_j(t), a_t, r_j(t), t)$ is stored as pending until the next decision epoch or episode termination.

### Helper Individual Reward Function (state-dependent helping cost)

When helper $A_j$ accepts in state $(w_j, \rho_j)$ and executes a task with binary outcome $x_t \in \{0, 1\}$:

$$r_j(t) = B_j \cdot x_t - \kappa_j(w_j, \rho_j), \qquad \kappa_j(w, \rho) = c_{\mathrm{res}, j}\,(1 + (2 - \rho)) + c_{\mathrm{opp}, j}\,(1 + w)$$

where $B_j$ is the individual cooperation benefit, $c_{\mathrm{res}, j}$ the base resource cost and $c_{\mathrm{opp}, j}$ the base opportunity cost.
The cost is evaluated in the state **in which the helper accepted** (before its own state transition).
With the default $c_{\mathrm{res}} = 0.10$, $c_{\mathrm{opp}} = 0.05$: $\kappa(0,2) = 0.15$ (idle), $\kappa(1,1) = 0.30$, $\kappa(2,0) = 0.45$ (busy).
If $A_j$ rejects, $r_j(t) = 0.0$.

Why state-dependent: if the helping cost did not depend on $(w, \rho)$, the reward of every action would be the same in every state and
the optimal helper policy would ignore the state entirely (the 9-state Q-table would be decorative). With $\kappa_j(w,\rho)$ the sign of
the expected immediate payoff $B_j\theta_j - \kappa_j(w,\rho)$ changes with the helper's load, and the load depends on how often *other*
agents ask it, which couples the agents' learning problems (implemented in `AutonomousAgent.helping_cost` / `calculate_helper_reward`).

### Reference helper: informed threshold

For E5 we add a non-learning reference helper (`helper_policy_type="informed_threshold"`) that knows its own reliability $\theta_j$ and
accepts iff $B_j \theta_j - \kappa_j(w_j, \rho_j) > 0$. The oracle uses $p_{ij}(t) \in \{0, 1\}$ for it.

---

## 4. Requester Belief Tracking and Policies

Requesters decouple teammate acceptance from execution reliability. For each peer $j$, requester $i$ maintains two discounted Beta--Bernoulli belief models:

**Acceptance Probability Belief:**

$$\hat{p}_{ij}(t) = \frac{1 + A_{ij}(t)}{2 + A_{ij}(t) + R_{ij}(t)}$$

Every global round **all agents** decay all their counts once ($A_{ij} \leftarrow \gamma A_{ij}$, $R_{ij} \leftarrow \gamma R_{ij}$); the requester then adds the new observation without a second discount: $A_{ij} \leftarrow A_{ij} + \mathbf{1}_{\{\text{accepted}\}}$, $R_{ij} \leftarrow R_{ij} + \mathbf{1}_{\{\text{rejected}\}}$.

**Conditional Execution Reliability Belief:**

$$\hat{\theta}_{ij}(t) = \frac{1 + S_{ij}(t)}{2 + S_{ij}(t) + F_{ij}(t)}$$

Decayed every global round like the acceptance counts; incremented only when a request was accepted: $S_{ij} \leftarrow S_{ij} + x_t$, $F_{ij} \leftarrow F_{ij} + (1 - x_t)$. A rejection adds nothing to $(S, F)$.

Note on memory: because decay is applied every *global* round while each requester acts only every $N$-th round (round-robin) and spreads its requests over $N-1$ teammates, the effective memory per teammate is much shorter than in the single-agent setting. A per-round $\gamma = 0.995$ corresponds to $\approx 0.98$ per own decision for $N = 4$ ($0.995^4 \approx 0.980$); E7 therefore sweeps $\gamma \in \{1.00, 0.995, 0.98, 0.95\}$.

### Expected Net Utility

$$\hat{\mu}_{ij}(t) = U_i^{\mathrm{self}} + \hat{p}_{ij}(t) \left( R \hat{\theta}_{ij}(t) - U_i^{\mathrm{self}} \right) - \lambda c_{ij}$$

### Requester Policies

- **MultiAgent Greedy (MA-Greedy):** Queries $\arg\max_j \hat{\mu}_{ij}(t)$ if $\max_j \hat{\mu}_{ij}(t) > U_i^{\mathrm{self}}$; else acts alone.
- **MultiAgent Discounted UCB (MA-D-UCB):** Uses optimistic upper confidence bounds on both acceptance and reliability.
- **MultiAgent Discounted Thompson Sampling (MA-D-TS):** Posterior sampling from $\tilde{p}_{ij} \sim \text{Beta}(1 + A_{ij}, 1 + R_{ij})$ and $\tilde{\theta}_{ij} \sim \text{Beta}(1 + S_{ij}, 1 + F_{ij})$.

---

## 5. Dynamic Oracle, Tie-Breaking, and Pseudo-Regret

### Exact Helper Acceptance Probability

Helper action selection uses $\epsilon$-greedy selection with random tie-breaking. The exact theoretical acceptance probability is:

$$p_{ij}(t) = (1 - \epsilon)\,\mathbb{P}_{\text{greedy}}(\text{ACCEPT}) + \frac{\epsilon}{2}$$

where:

$$\mathbb{P}_{\text{greedy}}(\text{ACCEPT}) = \begin{cases} 1.0, & \text{if } Q(\text{ACCEPT}) > Q(\text{REJECT}) \\ 0.0, & \text{if } Q(\text{REJECT}) > Q(\text{ACCEPT}) \\ 0.5, & \text{if } |Q(\text{ACCEPT}) - Q(\text{REJECT})| \le 10^{-12} \end{cases}$$

### Complete Set of Optimal Actions

$$\mu^*(t) = \max\left( U_i^{\mathrm{self}},\, \max_j \mu^*_{ij}(t) \right)$$

$$\mathcal{O}_t = \left\{ (d, a) \mid |\mu_{d, a}(t) - \mu^*(t)| < 10^{-9} \right\}$$

An action is optimal (`opt_action = 1`) if it belongs to $\mathcal{O}_t$, correctly handling multiple equally optimal helpers and ties with autonomous execution.

### Ex-Ante Pseudo-Regret

Evaluated on expected choice utility prior to stochastic outcome realization:

$$\mu_{\mathrm{chosen}}(t) = \begin{cases} U_i^{\mathrm{self}}, & \text{if } d_t = 0 \\ U_i^{\mathrm{self}} + p_{ij}(t)\left(R\theta_j(t) - U_i^{\mathrm{self}}\right) - \lambda c_{ij}, & \text{if } d_t = 1 \end{cases}$$

$$\mathrm{Regret}_t = \max\left(0,\, \mu^*(t) - \mu_{\mathrm{chosen}}(t)\right)$$

---

## 6. Social Welfare and Team Efficiency

To distinguish individual requester payoff from overall team efficiency without double counting communication costs:
- **Requester Utility:** $U_{\mathrm{req}} = \frac{1}{T} \sum_{t=1}^T r_{\mathrm{requester}, t}$
- **Helper Utility:** $U_{\mathrm{help}} = \frac{1}{T} \sum_{t=1}^T r_{\mathrm{helper}, t}$
- **Social Welfare:** $W = U_{\mathrm{req}} + U_{\mathrm{help}}$
Communication costs $\lambda c_{ij}$ are incurred exclusively by the requester channel and are reflected directly in $r_{\mathrm{requester}, t}$.


---

## 7. Temporal Help-Seeking (WHEN to Ask)

In sequential operational environments, tasks do not resolve in a single instantaneous step; they possess multi-step lifecycles subject to hard completion deadlines $D \in \mathbb{N}^+$. An autonomous agent must determine **when** during execution to seek external assistance versus continuing solo effort.

### 7.1 Task Lifecycle Formulation
Each task $\mathcal{T}$ is characterized by a tuple:

$$\mathcal{T} = \left(p, D, \tau_{\mathrm{rem}}, a_{\mathrm{solo}}, \text{status}\right)$$

- Normalized cumulative progress $p \in [0, 1]$, initialized at $p = 0.0$.
- Deadline $D \in \mathbb{N}^+$ and remaining time $\tau_{\mathrm{rem}} \in \{0, \dots, D\}$, initialized at $\tau_{\mathrm{rem}} = D$.
- Number of previous solo attempts $a_{\mathrm{solo}} \in \mathbb{N}$, initialized at $0$.
- Single-request constraint: at most one help request is permitted per task lifecycle (`help_requested` $\in \{\text{True}, \text{False}\}$).

At each task decision epoch $k \in \{1, \dots, D\}$:
1. If $p \ge 1.0$, the task is completed with success reward $+R$.
2. If $\tau_{\mathrm{rem}} = 0$ and $p < 1.0$, the task fails with deadline penalty $-c_{\mathrm{fail}}$.
3. Otherwise, the agent selects a timing action $u_k \in \{0, 1\}$ (`Continue_Self` = 0, `Evaluate_Help` = 1).

### 7.2 Solo Execution Progression
Under **Continue_Self** ($u_k = 0$):

$$p \leftarrow \min(1.0,\, p + \Delta p),\qquad \Delta p \sim \mathrm{Uniform}(0.20,\, 0.45) \cdot \kappa_{\mathrm{solo}}$$

$$\tau_{\mathrm{rem}} \leftarrow \tau_{\mathrm{rem}} - 1,\qquad a_{\mathrm{solo}} \leftarrow a_{\mathrm{solo}} + 1$$

incurring step cost $c_{\mathrm{step}} = 0.02$.

### 7.3 Compact State Discretization
To ensure efficient tabular Q-learning convergence without function approximation, each agent maps continuous task attributes into a 27-state discrete space $s_{\mathrm{timing}} = (p_{\mathrm{lvl}},\, \tau_{\mathrm{lvl}},\, a_{\mathrm{lvl}}) \in \{0, 1, 2\}^3$:
- **Progress Level ($p_{\mathrm{lvl}}$):**
  - $p_{\mathrm{lvl}} = 0$ (Low) if $p < 0.34$
  - $p_{\mathrm{lvl}} = 1$ (Med) if $0.34 \le p < 0.67$
  - $p_{\mathrm{lvl}} = 2$ (High) if $p \ge 0.67$
- **Time Left Level ($\tau_{\mathrm{lvl}}$):**
  - $\tau_{\mathrm{lvl}} = 0$ (Urgent) if $\tau_{\mathrm{rem}} \le 1$
  - $\tau_{\mathrm{lvl}} = 1$ (Moderate) if $2 \le \tau_{\mathrm{rem}} \le 3$
  - $\tau_{\mathrm{lvl}} = 2$ (Plentiful) if $\tau_{\mathrm{rem}} > 3$
- **Solo Attempts Level ($a_{\mathrm{lvl}}$):**
  - $a_{\mathrm{lvl}} = 0$ (None) if $a_{\mathrm{solo}} = 0$
  - $a_{\mathrm{lvl}} = 1$ (One) if $a_{\mathrm{solo}} = 1$
  - $a_{\mathrm{lvl}} = 2$ (Multiple) if $a_{\mathrm{solo}} \ge 2$

State index: $\mathrm{idx}(s) = 9 \cdot p_{\mathrm{lvl}} + 3 \cdot \tau_{\mathrm{lvl}} + a_{\mathrm{lvl}} \in \{0, \dots, 26\}$.

### 7.4 State-Dependent Continuation Utility and Decision Coupling
Under **Evaluate_Help** ($u_k = 1$):
- The agent computes the exact discounted value of finishing alone, $V_{\mathrm{solo}}(p, \tau_{\mathrm{rem}})$, from the Irwin--Hall distribution of the summed uniform progress increments (completion on attempt $k$ is worth $\gamma_{\mathrm{timing}}^{k-1} R$ minus the discounted step costs of the $k-1$ earlier steps; failure is worth $-\gamma_{\mathrm{timing}}^{\tau-1} c_{\mathrm{fail}}$ minus the discounted step costs).
- For each teammate it computes (with MA-D-TS samples $\tilde p, \tilde\theta$):

$$V_{\mathrm{ask},j} = -\lambda c_{ij} + \tilde p_{ij}\tilde\theta_{ij} R + (1 - \tilde p_{ij}\tilde\theta_{ij})\left[-c_{\mathrm{step}} + \gamma_{\mathrm{timing}} V_{\mathrm{solo}}(p, \tau_{\mathrm{rem}} - 1)\right]$$

for $\tau_{\mathrm{rem}} > 1$; for $\tau_{\mathrm{rem}} = 1$ the bracket evaluates to $-c_{\mathrm{fail}}$.
- If $\max_j V_{\mathrm{ask},j} > V_{\mathrm{solo}}(p, \tau_{\mathrm{rem}})$, a request is transmitted to candidate helper $j^*$; otherwise the agent takes a solo step.
- Helper $j^*$ accepts or rejects via independent tabular Q-learning on private $(w_j, \rho_j)$ states; its reward uses the same state-dependent helping cost $\kappa_j(w_j, \rho_j)$ as in Section 3.
- If accepted and successful, progress advances to $1.0$ ($p \leftarrow 1.0$), completing the task.
- If rejected or failed, exactly 1 deadline unit is consumed. If $\tau_{\mathrm{rem}} > 0$, the task enters a deterministic post-request solo phase until completion or deadline failure, without corrupting the timing policy on forced actions.

Q-value updates:

$$Q_{\mathrm{timing}}(s, u) \leftarrow Q_{\mathrm{timing}}(s, u) + \alpha_{\mathrm{timing}} \left[ r_k + \gamma_{\mathrm{timing}} \max_{u'} Q_{\mathrm{timing}}(s', u') - Q_{\mathrm{timing}}(s, u) \right]$$

with $\alpha_{\mathrm{timing}} = 0.1$, $\gamma_{\mathrm{timing}} = 0.95$, terminal completion reward $+R$, and deadline failure penalty $-c_{\mathrm{fail}}$.

### 7.5 Model-based alternative: planned timing (`PlannedTimingPolicy`, approximate)
The whether-rule above compares asking **now** with **never** asking ($V_{\mathrm{solo}}$), so it ignores the option of asking later; the
tabular timing policy has to learn that option value. The planned policy computes it from the requester's current posterior means
$q_j = \hat p_{ij}\hat\theta_{ij}$:

$$\mathrm{Ask}(p,\tau) = \max_j\big[-\lambda c_{ij} + q_j R + (1 - q_j) F(p,\tau)\big],\qquad F(p,\tau) = \begin{cases}-c_{\mathrm{fail}} & \tau = 1\\ -c_{\mathrm{step}} + \gamma_{\mathrm{timing}} V_{\mathrm{solo}}(p, \tau - 1) & \tau > 1\end{cases}$$

$$\mathrm{Wait}(p,\tau) = \mathbb{E}_{\Delta p}\Big[R\,\mathbf{1}_{\{p + \Delta p \ge 1\}} + \mathbf{1}_{\{p + \Delta p < 1\}}\big(-c_{\mathrm{step}} + \gamma_{\mathrm{timing}} V^*(p + \Delta p, \tau - 1)\big)\Big]$$

(where the second term inside the expectation evaluates to $-c_{\mathrm{fail}}$ if $\tau = 1$).

$$V^*(p,\tau) = \max\{\mathrm{Ask}(p,\tau), \mathrm{Wait}(p,\tau)\}$$

computed by backward induction on a 401-point progress grid (41-point quadrature for $\Delta p$). The policy chooses **Evaluate_Help** iff
$\mathrm{Ask}(p,\tau) \ge \mathrm{Wait}(p,\tau)$; the request itself is then decided by the usual whether/whom rule.
This is an approximation of the optimal policy for the full system: the beliefs are frozen at their current posterior means during
planning (in the environment they decay every global round and change with each observation), and the final ask decision uses
MA-D-TS posterior samples rather than the means.

