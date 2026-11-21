# backend/src/help_seeking/policies.py
"""Name -> policy factory used by the experiment runner and the API."""
from __future__ import annotations

from .algorithms import DiscountedGreedy, DiscountedUCB, DiscountedThompson, Policy
from .baselines import (AlwaysAskEmpiricalGreedy, AlwaysAskFixedCheapest, AlwaysAskRandom,
                        DynamicOracle, InitialKnowledgeOracle, NeverAsk)
from .multiagent_policies import (MultiAgentDiscountedGreedy, MultiAgentDiscountedUCB,
                                 MultiAgentDiscountedThompson)

CANONICAL = {
    "Dynamic Oracle": DynamicOracle,
    "Initial-Knowledge Oracle": InitialKnowledgeOracle,
    "Never-Ask": NeverAsk,
    "Always-Ask + Random": AlwaysAskRandom,
    "Always-Ask + Fixed-Cheapest": AlwaysAskFixedCheapest,
    "Always-Ask + Empirical Greedy": AlwaysAskEmpiricalGreedy,
    "Greedy": DiscountedGreedy,
    "D-UCB": DiscountedUCB,
    "D-TS": DiscountedThompson,
    "MultiAgent-Greedy": MultiAgentDiscountedGreedy,
    "MultiAgent-D-UCB": MultiAgentDiscountedUCB,
    "MultiAgent-D-TS": MultiAgentDiscountedThompson,
}

# Names used in the proposal's Table 2 and in the first version of the configs.
ALIASES = {
    "Stationary Oracle": "Dynamic Oracle",
    "Oracle": "Dynamic Oracle",
    "Random": "Always-Ask + Random",
    "Fixed-Cheapest": "Always-Ask + Fixed-Cheapest",
    "Always-Ask + Greedy": "Always-Ask + Empirical Greedy",
    "Empirical Greedy": "Always-Ask + Empirical Greedy",
    "D-Greedy": "Greedy",
    "Discounted Greedy": "Greedy",
    "Discounted UCB": "D-UCB",
    "Discounted Thompson Sampling": "D-TS",
    "MultiAgent Greedy": "MultiAgent-Greedy",
    "MultiAgent D-UCB": "MultiAgent-D-UCB",
    "MultiAgent D-TS": "MultiAgent-D-TS",
}

LEARNING_METHODS = ("Greedy", "D-UCB", "D-TS")


def canonical_name(name: str) -> str:
    base = name.split("[")[0].strip()      # allow labels like "D-UCB [alpha=0.25]"
    base = ALIASES.get(base, base)
    if base not in CANONICAL:
        raise ValueError(f"Unknown algorithm {name!r}. Known: {sorted(CANONICAL)}")
    return base


def make_policy(name: str, *, K: int, R: float, lambda_mult: float, c, gamma: float,
                seed=None, alpha_ucb: float = 1.0, theta_initial=None, U_self=None,
                overrides: dict | None = None) -> Policy:
    """Instantiate a policy by (alias-tolerant) name.

    ``overrides`` may change gamma / alpha_ucb for this algorithm only.
    """
    overrides = overrides or {}
    gamma = float(overrides.get("gamma", gamma))
    alpha_ucb = float(overrides.get("alpha_ucb", alpha_ucb))
    cls = CANONICAL[canonical_name(name)]
    kwargs = dict(K=K, R=R, lambda_mult=lambda_mult, c=c, gamma=gamma, seed=seed)
    if cls is DiscountedUCB:
        kwargs["alpha_ucb"] = alpha_ucb
    if cls is InitialKnowledgeOracle:
        kwargs["theta_initial"] = theta_initial
        kwargs["U_self"] = U_self
    return cls(**kwargs)
