"""Discounted Beta-Bernoulli belief model (Eqs. 3-5) against hand calculations."""
import numpy as np
import pytest

from help_seeking.trust import TrustState


def test_uniform_prior_gives_one_half():
    assert np.allclose(TrustState(3).posterior_mean(), 0.5)


def test_single_success_and_failure():
    s = TrustState(3, gamma=1.0)
    s.update(0, 1)
    s.update(1, 0)
    assert np.isclose(s.posterior_mean()[0], 2 / 3)     # (1+1)/(1+0+2)
    assert np.isclose(s.posterior_mean()[1], 1 / 3)     # (0+1)/(0+1+2)
    assert np.isclose(s.posterior_mean()[2], 1 / 2)


def test_discount_then_add_order_matches_eq3():
    # S(t+1) = gamma * S(t) + x  -> two successes with gamma = 0.9 give S = 0.9 + 1 = 1.9
    s = TrustState(1, gamma=0.9)
    s.update(0, 1)
    s.update(0, 1)
    assert np.isclose(s.S[0], 1.9)
    assert np.isclose(s.F[0], 0.0)
    assert np.isclose(s.posterior_mean()[0], 2.9 / 3.9)


def test_unqueried_teammates_decay_every_round():
    s = TrustState(2, gamma=0.5)
    s.update(0, 1)          # S = [1, 0]
    s.update(1, 0)          # S = [0.5, 0], F = [0, 1]
    s.decay()               # acting alone: S = [0.25, 0], F = [0, 0.5]
    assert np.allclose(s.S, [0.25, 0.0])
    assert np.allclose(s.F, [0.0, 0.5])


def test_long_decay_returns_to_prior():
    s = TrustState(1, gamma=0.9)
    for _ in range(10):
        s.update(0, 1)
    for _ in range(300):
        s.decay()
    assert np.isclose(s.posterior_mean()[0], 0.5, atol=1e-6)


def test_gamma_one_counts_never_decay():
    s = TrustState(1, gamma=1.0)
    s.update(0, 1)
    for _ in range(100):
        s.decay()
    assert s.S[0] == 1.0


def test_effective_sample_size_bounded_by_geometric_series():
    s = TrustState(1, gamma=0.98)
    for _ in range(2000):
        s.update(0, 1)
    assert np.isclose(s.effective_sample_size()[0], 1 / (1 - 0.98), rtol=1e-6)


def test_invalid_gamma_rejected():
    with pytest.raises(ValueError):
        TrustState(2, gamma=0.0)
    with pytest.raises(ValueError):
        TrustState(2, gamma=1.5)
