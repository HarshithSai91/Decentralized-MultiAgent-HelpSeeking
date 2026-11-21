# IMPLEMENTATION PLAN: Decentralized Multi-Agent System Redesign

> Historical planning document. In the final version the helper's cost of helping depends on its state:
> `B_i * x - cost(w, rho)` with `cost = resource_cost * (3 - rho) + opportunity_cost * (1 + w)`.
> See `docs/mathematical_model.md` for the model as implemented.

## STAGE 1: Repository Inspection (COMPLETED)
✅ Inspected complete repository structure:
- Backend: 20 Python modules in `backend/src/help_seeking/`
- Frontend: React dashboard with 3 tabs (Live, Compare, Experiments)
- Report: LaTeX arXiv-format report with 9 sections
- Configs: 9 JSON experiment configurations (E1-E4, S1-S5)
- Tests: 52 unit/integration tests
- Scripts: Experiment runner, analysis, figure generation

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

## STAGE 8: Experiments

### 8.1 Preserve Original E1-E4
Run with new multi-agent interaction model:
- E1: Stationary Reliability (all agents have stationary reliability)
- E2: When to Ask (vary U_self)
- E3: Abrupt Reliability Changes (some agents change reliability)
- E4: Communication Cost Variation (vary lambda)

### 8.2 New Experiments
- **E5: Autonomous Helper Learning**
  - Compare: Always Accept vs Always Reject vs Independent Q-learning
  
- **E6: Multi-Agent Cooperation**
  - Investigate how workload, resources, rewards influence interactions
  - Test reciprocity emergence

### 8.3 Experiment Configurations
Create new JSON configs for multi-agent experiments.
- 30 seeds minimum
- Fair initial conditions
- Independent RNGs per agent
- Reset Q-tables and beliefs between runs

---

## STAGE 9: Dashboard Updates

### 9.1 Frontend Changes
- New "Multi-Agent Live" tab
- Visualize multiple agents' states simultaneously
- Show requester/helper role switches
- Display per-agent Q-tables, beliefs, rewards
- Network graph of interactions
- New metrics panels

### 9.2 API Extensions
- New endpoints for multi-agent simulation
- WebSocket for real-time multi-agent updates
- Experiment comparison for new experiments

---

## STAGE 10: Research Report Updates

### 10.1 Mathematical Formulation Updates
- Decentralized multi-agent model
- Individual reward functions
- Helper Q-learning formulation
- Acceptance probability estimation
- Modified utility calculations

### 10.2 Architecture Section
- AutonomousAgent class diagram
- Interaction protocol sequence diagram
- State transition diagrams

### 10.3 Experimental Results
- E1-E6 results with new architecture
- Comparison with original results
- Analysis of emergent cooperation

### 10.4 Limitations
- Sequential interaction assumption
- No negotiation
- Fixed ACCEPT/REJECT action space
- No effort allocation

---

## FILES TO MODIFY/CREATE

### New Files:
1. `backend/src/help_seeking/autonomous_agent.py` - Core agent class
2. `backend/src/help_seeking/acceptance.py` - Acceptance belief model
3. `backend/src/help_seeking/multiagent_environment.py` - Multi-agent environment
4. `backend/src/help_seeking/multiagent_simulation.py` - Multi-agent simulation runner
5. `backend/src/help_seeking/helper_qlearning.py` - Helper Q-learning policy
6. `configs/e5_helper_learning.json` - E5 experiment config
7. `configs/e6_multiagent_cooperation.json` - E6 experiment config
8. Frontend components for multi-agent visualization

### Modified Files:
1. `backend/src/help_seeking/algorithms.py` - Update requester algorithms for acceptance+reliability
2. `backend/src/help_seeking/trust.py` - Keep, ensure compatibility
3. `backend/src/help_seeking/policies.py` - Add multi-agent policy factory
4. `backend/src/help_seeking/metrics.py` - Add multi-agent metrics
5. `backend/src/help_seeking/experiments.py` - Support multi-agent experiments
6. `backend/src/help_seeking/api.py` - Add multi-agent API endpoints
7. `frontend/src/App.tsx` - Add multi-agent tabs
8. `frontend/src/api.ts` - Add multi-agent API types and calls
9. `report/main.tex` - Update with new formulation and results

---

## TESTING STRATEGY

### Unit Tests (add to backend/tests/):
1. `test_autonomous_agent.py` - Agent initialization, reset, dual roles
2. `test_acceptance.py` - Acceptance belief updates, discounting
3. `test_helper_qlearning.py` - Q-table updates, epsilon-greedy, convergence
4. `test_multiagent_environment.py` - Protocol execution, state transitions
5. `test_multiagent_simulation.py` - Full run, logging, reproducibility

### Integration Tests:
1. `test_multiagent_integration.py` - End-to-end multi-agent runs
2. `test_experiment_compatibility.py` - Original E1-E4 still work with new model

### Regression Tests:
- Run all existing 52 tests - must pass
- Run new tests - must pass

---

## IMPLEMENTATION ORDER

1. **Create core data structures** (acceptance.py, helper_qlearning.py)
2. **Implement AutonomousAgent class** (autonomous_agent.py)
3. **Update requester algorithms** (algorithms.py - integrate acceptance beliefs)
4. **Create multi-agent environment** (multiagent_environment.py)
5. **Create multi-agent simulation** (multiagent_simulation.py)
6. **Add multi-agent metrics** (metrics.py)
7. **Update experiment runner** (experiments.py)
8. **Add API endpoints** (api.py)
9. **Create experiment configs** (E5, E6 JSON)
10. **Run experiments and verify**
11. **Update frontend** (App.tsx, api.ts)
12. **Update LaTeX report** (main.tex)
13. **Run full test suite**

---

## CRITICAL DESIGN DECISIONS

1. **Sequential vs Parallel Interactions**: Use sequential for v1 to avoid concurrency complexity. One task per round, assigned to one agent.

2. **Task Assignment**: Round-robin or random selection of requester each round. Document clearly.

3. **Helper State Dynamics**: Workload increases by 1 level when helping, decays by 1 level per round when idle. Resources similar.

4. **Execution Reliability**: Each agent has its own reliability profile that can change (for E3, E5, E6).

5. **Individual Rewards**: 
   - Requester: R * x - lambda * c if accepted, U_self if rejected/acted alone
   - Helper: B_i * x - resource_cost - opportunity_cost if accepted, 0 if rejected

6. **No Shared State**: Each agent has completely independent beliefs, Q-table, RNG.

7. **Backward Compatibility**: Original single-requester experiments should still run by creating N=1 agent + N-1 fixed helpers.

---

## RISKS AND MITIGATIONS

| Risk | Mitigation |
|------|------------|
| Breaking existing experiments | Keep original SimulationRun, Environment classes; add new MultiAgent* classes alongside |
| Complexity of multi-agent debugging | Add detailed logging, inspector mode, step-by-step execution |
| Q-learning non-convergence | Use small state space (9 states), track convergence metrics |
| Performance with N agents | Start with N=3-4, optimize later |
| Frontend complexity | Add multi-agent tab incrementally, reuse existing components |

---

## SUCCESS CRITERIA

1. All 52 existing tests pass
2. New unit tests pass (>20 new tests)
3. E1-E4 reproduce qualitatively similar results with new architecture
4. E5 shows Q-learning helpers learn sensible policies
5. E6 shows emergent cooperation patterns
6. Dashboard shows multi-agent interactions correctly
7. Report updated with new formulation and results
8. Code is clean, documented, and follows existing patterns