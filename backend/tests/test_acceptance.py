"""Tests for the AcceptanceState belief model."""
import numpy as np
import pytest

from help_seeking.acceptance import AcceptanceState


def test_uniform_prior_gives_one_half():
    """Initial posterior mean should be 0.5 with uniform Beta(1,1) prior."""
    state = AcceptanceState(3)
    assert np.allclose(state.posterior_mean(), 0.5)


def test_single_accept_and_reject():
    """Test posterior after one accept and one reject."""
    state = AcceptanceState(3, gamma=1.0)
    state.update_accept(0)
    state.update_reject(1)

    # Helper 0: A=1, R=0 -> (1+1)/(2+1+0) = 2/3
    assert np.isclose(state.posterior_mean()[0], 2/3)
    # Helper 1: A=0, R=1 -> (1+0)/(2+0+1) = 1/3
    assert np.isclose(state.posterior_mean()[1], 1/3)
    # Helper 2: A=0, R=0 -> 1/2
    assert np.isclose(state.posterior_mean()[2], 1/2)


def test_discount_then_add_order_matches_eq():
    """A(t+1) = gamma * A(t) + I[accepted]"""
    state = AcceptanceState(1, gamma=0.9)
    state.update_accept(0)  # A = 1
    state.update_accept(0)  # A = 0.9 * 1 + 1 = 1.9
    assert np.isclose(state.A[0], 1.9)
    assert np.isclose(state.R[0], 0.0)
    # p_hat = (1 + 1.9) / (2 + 1.9 + 0) = 2.9 / 3.9
    assert np.isclose(state.posterior_mean()[0], 2.9 / 3.9)


def test_unqueried_helpers_decay_every_round():
    """Unqueried helpers' counts decay when decay() is called."""
    state = AcceptanceState(2, gamma=0.5)
    state.update_accept(0)   # A = [1, 0]
    state.update_reject(1)   # A = [0.5, 0], R = [0, 1]
    state.decay()            # acting alone: A = [0.25, 0], R = [0, 0.5]
    assert np.allclose(state.A, [0.25, 0.0])
    assert np.allclose(state.R, [0.0, 0.5])


def test_long_decay_returns_to_prior():
    """After many decay steps, beliefs return to prior 0.5."""
    state = AcceptanceState(1, gamma=0.9)
    for _ in range(10):
        state.update_accept(0)
    for _ in range(300):
        state.decay()
    assert np.isclose(state.posterior_mean()[0], 0.5, atol=1e-6)


def test_gamma_one_counts_never_decay():
    """With gamma=1, counts never decay."""
    state = AcceptanceState(1, gamma=1.0)
    state.update_accept(0)
    for _ in range(100):
        state.decay()
    assert state.A[0] == 1.0
    assert state.R[0] == 0.0


def test_effective_sample_size():
    """Effective sample size = A + R."""
    state = AcceptanceState(1, gamma=1.0)  # No discount for this test
    for _ in range(10):
        state.update_accept(0)
    for _ in range(5):
        state.update_reject(0)
    # With gamma=1.0, no decay, so A=10, R=5, total=15
    assert np.isclose(state.effective_sample_size()[0], 15.0)


def test_posterior_params():
    """Beta parameters should be (1+A, 1+R)."""
    state = AcceptanceState(2, gamma=1.0)
    state.update_accept(0)
    state.update_accept(0)
    state.update_reject(1)
    alpha, beta = state.posterior_params()
    assert np.allclose(alpha, [3, 1])   # helper 0: A=2 -> 1+2=3, helper 1: A=0 -> 1
    assert np.allclose(beta, [1, 2])    # helper 0: R=0 -> 1, helper 1: R=1 -> 2


def test_invalid_gamma_rejected():
    with pytest.raises(ValueError):
        AcceptanceState(2, gamma=0.0)
    with pytest.raises(ValueError):
        AcceptanceState(2, gamma=1.5)


def test_snapshot():
    state = AcceptanceState(2, gamma=0.9)
    state.update_accept(0)
    snap = state.snapshot()
    assert "A" in snap
    assert "R" in snap
    assert "p_hat" in snap
    assert len(snap["A"]) == 2