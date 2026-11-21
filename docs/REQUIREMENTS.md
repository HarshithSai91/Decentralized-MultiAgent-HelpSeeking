# REQUIREMENTS.md

## Project Title
Learning When and Whom to Ask for Help Under Changing Teammate Reliability

## Team
- Cherukuri Harshith Sai (Roll 24110091) — Problem Formulation & Simulation Environment; Lead E1
- Atragada Venkata Naga Sai Ganesh (Roll 24110060) — Learning Algorithms & Trust Estimation
- Ambati Chaitanya Ram (Roll 24110035) — Baselines, Dynamics & Adaptation Analysis; Lead E2, E3
- Dasari Lohith (Roll 24110100) — Communication Cost, Evaluation & Visualization; Lead E4

## Source of Truth
PDF: Project_Idea.pdf (14 pages, proposal finalized)

## Key Mathematical Definitions
- K teammates, T rounds
- R > 0 (normalized to 1.0)
- U_self(t) autonomous utility
- d_t ∈ {0,1} ask/no-ask
- a_t ∈ H selected teammate
- θ_k(t) true reliability
- x_t ∈ {0,1} outcome
- c_k communication cost
- λ cost multiplier
- r_t = R·x_t - λ·c_at if d_t=1; else x_self_t with E=U_self
- μ_k(t) = R·θ_k(t) - λ·c_k
- μ_self(t) = U_self(t)
- S_k(t+1) = γ·S_k(t) + I·x_t
- F_k(t+1) = γ·F_k(t) + I·(1-x_t)
- θ̂_k(t) = (1 + S_k(t)) / (2 + S_k(t) + F_k(t))
- Dynamic Oracle: d* = I[max μ_k(t) > U_self]; a* = argmax μ_k
- Regret Reg_π(T) = Σ (μ*(t) - E_π[r_t])

## Algorithms Required
1. Discounted Greedy (D-Greedy)
2. Discounted UCB (D-UCB)
3. Discounted Thompson Sampling (D-TS)

## Baselines Required
1. Never-Ask / Always-Act-Alone
2. Always-Ask + Random Selection
3. Always-Ask + Fixed-Cheapest
4. Always-Ask + Empirical Greedy (γ=1, undiscounted)
5. Dynamic Oracle (full info)
6. Initial-Knowledge Oracle (secondary)

## Experiments (Exact Configurations from PDF)
### E1: Stationary Learning
- Independent variable: Δ ∈ {0.02, 0.05, 0.10, 0.15}
- θ = (0.60+Δ, 0.60, 0.50), c = (0.10, 0.10, 0.10)
- K=3, T=2000, R=1.0, U_self=0.35, λ=1.0, γ=1.0
- Algorithms: Stationary Oracle, Random, Fixed-Cheapest, Greedy, D-UCB, D-TS
- Metrics: Cumulative regret, Optimal-teammate selection rate (given d_t=1), MAE_θ, Ask Rate

### E2: When to Ask
- Independent variable: U_self ∈ {0.30, 0.50, 0.65, 0.80}
- θ = (0.75, 0.65, 0.50), c = (0.10, 0.10, 0.10)
- K=3, T=2000, R=1.0, λ=1.0, γ=0.98
- Algorithms: Dynamic Oracle, Always-Ask+Greedy, Never-Ask, D-UCB, D-TS
- Metrics: Ask Rate, Ask Decision Accuracy (Acc_ask), Average Net Utility, Total Communication Cost

### E3: Abrupt Reliability Change
- Independent variable: γ ∈ {1.0, 0.995, 0.98, 0.95, 0.90}
- Initial θ(0) = (0.80, 0.70, 0.90), after τ=1000: θ = (0.80, 0.70, 0.40)
- c = (0.10, 0.10, 0.10), U_self=0.50, λ=1.0, T=2000
- Algorithms: Dynamic Oracle, Initial-Knowledge Oracle, Greedy, D-UCB, D-TS
- Metrics: Adaptation latency τ_adapt, post-change regret, average net utility, optimal-teammate selection rate

### E4: Communication Cost Variation
- Independent variable: λ ∈ {0, 1.0, 2.5, 4.0, 6.0}
- θ = (0.60, 0.75, 0.90), c = (0.05, 0.10, 0.20)
- K=3, T=2000, R=1.0, U_self=0.45, γ=0.98
- Algorithms: Dynamic Oracle, Fixed-Cheapest, Greedy, D-UCB, D-TS
- Metrics: Teammate selection share, Ask Rate, total communication cost, average net utility

## Evaluation Metrics
- Cumulative Dynamic Regret
- Average Net Utility
- Ask Rate
- Ask Decision Accuracy (Acc_ask)
- Optimal-Teammate Selection Rate (conditional on d_t=1)
- Adaptation Latency (τ_adapt)
- Mean Absolute Estimation Error (MAE_θ)
- Total Communication Cost

## Hypotheses
- H1: D-UCB and D-TS > Random and Fixed-Cheapest (stationary, when asking beneficial)
- H2: γ=1.0 retains evidence; excessive discounting (γ < 1.0) increases steady-state regret in stationary settings
- H3: Discounted methods (γ < 1.0) have lower τ_adapt than non-discounted (γ=1.0) after abrupt change
- H4: Increasing λ shifts preference from Helper 3 → Helper 2 → autonomous action
- H5: Adaptive policies distinguish advantageous vs. disadvantageous help-seeking regimes (Acc_ask > 90%)

## Success Criteria
1. Learning-based selection improves over non-adaptive baselines in E1.
2. High Ask Decision Accuracy (>90%) and lower unnecessary cost vs Always-Ask.
3. Quantitative γ vs τ_adapt relationship shown.
4. Cost scaling λ changes selection, including autonomous transition.
5. At least one failure/regime identified with technical explanation.

## Limitations (from PDF)
- Simplified autonomous benchmark (parameterized U_self, not MDP)
- Synthetic Bernoulli reliability (not hardware physics)
- Binary feedback only
- Scalar reliability per teammate
- Fixed, exogenous costs; scripted helpers unaffected by load
