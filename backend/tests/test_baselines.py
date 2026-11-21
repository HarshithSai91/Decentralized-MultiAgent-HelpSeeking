"""Baselines and the dynamic oracle (Sec. 4.2, Eqs. 11-13)."""
import numpy as np

from help_seeking.baselines import (AlwaysAskEmpiricalGreedy, AlwaysAskFixedCheapest,
                                    AlwaysAskRandom, DynamicOracle, InitialKnowledgeOracle,
                                    NeverAsk)
from help_seeking.utilities import dynamic_oracle


def test_oracle_asks_best_helper():
    o = dynamic_oracle([0.75, 0.65, 0.50], U_self=0.35, c=[0.1, 0.1, 0.1])
    assert (o.d, o.a) == (1, 0)
    assert np.isclose(o.mu_star, 0.65)


def test_oracle_acts_alone_when_helpers_not_worth_it():
    o = dynamic_oracle([0.75, 0.65, 0.50], U_self=0.80, c=[0.1, 0.1, 0.1])
    assert (o.d, o.a) == (0, None)
    assert np.isclose(o.mu_star, 0.80)


def test_oracle_tie_counts_both_decisions_as_optimal():
    # E2 with U_self = 0.65: R*0.75 - 0.1 = 0.65 exactly.
    o = dynamic_oracle([0.75, 0.65, 0.50], U_self=0.65, c=[0.1, 0.1, 0.1])
    assert o.d == 0                          # Eq. 11 uses a strict inequality
    assert o.ask_ok and o.alone_ok
    assert o.is_optimal_action(1, 0) and o.is_optimal_action(0, None)
    assert not o.is_optimal_action(1, 1)


def test_e4_cost_switches_preferences():
    th, c, u = [0.60, 0.75, 0.90], [0.05, 0.10, 0.20], 0.45
    assert dynamic_oracle(th, u, lambda_mult=0.0, c=c).a == 2
    assert dynamic_oracle(th, u, lambda_mult=1.0, c=c).a == 2
    assert dynamic_oracle(th, u, lambda_mult=2.5, c=c).a == 1
    assert dynamic_oracle(th, u, lambda_mult=4.0, c=c).d == 0
    assert dynamic_oracle(th, u, lambda_mult=6.0, c=c).d == 0


def test_never_ask():
    assert NeverAsk(3).decide(0.1).d == 0


def test_always_ask_random_is_uniform():
    pol = AlwaysAskRandom(3, seed=1)
    picks = np.bincount([pol.decide(0.9).a for _ in range(6000)], minlength=3)
    assert all(abs(p / 6000 - 1 / 3) < 0.03 for p in picks)


def test_fixed_cheapest_unique_minimum():
    pol = AlwaysAskFixedCheapest(3, c=[0.05, 0.10, 0.20], seed=3)
    assert {pol.decide(0.0).a for _ in range(20)} == {0}


def test_fixed_cheapest_breaks_cost_ties_once_per_run():
    picks = set()
    for s in range(40):
        pol = AlwaysAskFixedCheapest(3, c=[0.1, 0.1, 0.1], seed=s)
        within = {pol.decide(0.0).a for _ in range(10)}
        assert len(within) == 1              # fixed within a run
        picks |= within
    assert picks == {0, 1, 2}                # but not always helper 1


def test_empirical_greedy_always_asks_and_ignores_gamma():
    pol = AlwaysAskEmpiricalGreedy(3, c=[0.1] * 3, gamma=0.5, seed=0)
    assert pol.trust.gamma == 1.0
    assert pol.decide(U_self=0.99).d == 1


def test_dynamic_oracle_policy_uses_truth():
    pol = DynamicOracle(3, c=[0.1] * 3)
    assert pol.uses_truth
    dec = pol.decide(0.5, theta_true=np.array([0.8, 0.7, 0.4]))
    assert (dec.d, dec.a) == (1, 0)


def test_initial_knowledge_oracle_keeps_initial_choice():
    pol = InitialKnowledgeOracle(3, c=[0.1] * 3, theta_initial=[0.8, 0.7, 0.9], U_self=0.5)
    assert not pol.uses_truth
    dec = pol.decide(0.5, theta_true=np.array([0.8, 0.7, 0.4]))
    assert (dec.d, dec.a) == (1, 2)
