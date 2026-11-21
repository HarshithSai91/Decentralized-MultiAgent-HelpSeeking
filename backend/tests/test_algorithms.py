"""Learning methods: score formulas (Eqs. 8, 16, 17) and the two-stage rule."""
import numpy as np

from help_seeking.algorithms import (DiscountedGreedy, DiscountedThompson, DiscountedUCB,
                                     argmax_random)

C = [0.1, 0.1, 0.1]


def test_greedy_scores_and_decision():
    pol = DiscountedGreedy(3, c=C, seed=0)
    pol.trust.S[:] = [8, 1, 0]
    pol.trust.F[:] = [2, 1, 0]
    dec = pol.decide(U_self=0.35)
    expected = (np.array([8, 1, 0]) + 1) / (np.array([10, 2, 0]) + 2) - 0.1
    assert np.allclose(dec.scores, expected)
    assert (dec.d, dec.a) == (1, 0)


def test_greedy_acts_alone_when_best_estimate_below_self_utility():
    # Prior mean 0.5 - cost 0.1 = 0.4 < 0.5: greedy never even tries a helper.
    pol = DiscountedGreedy(3, c=C, seed=0)
    dec = pol.decide(U_self=0.5)
    assert (dec.d, dec.a) == (0, None)
    assert dec.candidate in (0, 1, 2)          # candidate still computed


def test_strict_inequality_for_asking():
    pol = DiscountedGreedy(1, c=[0.1], seed=0)
    assert pol.decide(U_self=0.4).d == 0          # 0.4 is not > 0.4


def test_ucb_bonus_matches_eq16():
    pol = DiscountedUCB(3, c=C, alpha_ucb=0.5, seed=0)
    pol.trust.S[:] = [6, 2, 0]
    pol.trust.F[:] = [2, 2, 0]
    N_k = np.array([8.0, 4.0, 0.0])
    bonus = 1.0 * 0.5 * np.sqrt(2 * np.log(12.0) / (N_k + 1e-6))
    assert np.allclose(pol.bonus(), bonus)
    dec = pol.decide(U_self=0.3)
    assert dec.a == 2                             # unexplored arm has the largest index


def test_ucb_uses_ln2_floor_for_small_N():
    pol = DiscountedUCB(2, c=[0, 0], alpha_ucb=1.0, seed=0)
    pol.trust.S[:] = [0.5, 0.0]
    b = pol.bonus()
    assert np.isclose(b[0], np.sqrt(2 * np.log(2.0) / (0.5 + 1e-6)))


def test_ucb_bonus_grows_when_helper_is_not_queried():
    pol = DiscountedUCB(2, c=[0, 0], gamma=0.9, alpha_ucb=1.0, seed=0)
    for _ in range(20):
        pol.update(1, 0, 1)
    before = pol.bonus()[0]
    for _ in range(20):
        pol.update(0, None, None)                 # act alone: counts decay
    assert pol.bonus()[0] > before


def test_thompson_samples_from_beta_posterior():
    pol = DiscountedThompson(1, c=[0.0], seed=123)
    pol.trust.S[:] = [30.0]
    pol.trust.F[:] = [10.0]
    draws = np.array([pol.helper_scores()[0][0] for _ in range(20000)])
    a, b = 31.0, 11.0
    assert abs(draws.mean() - a / (a + b)) < 0.005
    assert abs(draws.var() - a * b / ((a + b) ** 2 * (a + b + 1))) < 0.0005


def test_thompson_explores_under_equal_beliefs():
    pol = DiscountedThompson(3, c=C, seed=7)
    picks = {pol.decide(U_self=0.0).a for _ in range(200)}
    assert picks == {0, 1, 2}


def test_argmax_random_breaks_ties_uniformly():
    rng = np.random.default_rng(0)
    counts = np.bincount([argmax_random(np.array([1.0, 1.0, 0.5]), rng) for _ in range(4000)],
                         minlength=3)
    assert counts[2] == 0
    assert abs(counts[0] / 4000 - 0.5) < 0.04


def test_policy_reset_reproduces_decisions():
    pol = DiscountedThompson(3, c=C, seed=11)
    first = [pol.decide(0.0).a for _ in range(30)]
    pol.reset()
    assert [pol.decide(0.0).a for _ in range(30)] == first
