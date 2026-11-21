"""Tests for the HelperQLearning module."""
import numpy as np
import pytest

from help_seeking.helper_qlearning import HelperQLearning, encode_state, decode_state


def test_encode_decode_state():
    """Test state encoding/decoding."""
    for w in range(3):
        for r in range(3):
            s = encode_state(w, r)
            assert 0 <= s <= 8
            w2, r2 = decode_state(s)
            assert w2 == w and r2 == r


def test_invalid_state_raises():
    with pytest.raises(ValueError):
        encode_state(3, 0)
    with pytest.raises(ValueError):
        encode_state(0, 3)
    with pytest.raises(ValueError):
        encode_state(-1, 0)


def test_q_table_initialization():
    """Q-table should be zeros initially."""
    ql = HelperQLearning(seed=0)
    assert ql.q_table.shape == (9, 2)
    assert np.all(ql.q_table == 0.0)


def test_choose_action_returns_valid():
    """Actions should be 0 (ACCEPT) or 1 (REJECT)."""
    ql = HelperQLearning(seed=0)
    for w in range(3):
        for r in range(3):
            a = ql.choose_action(w, r)
            assert a in (0, 1)


def test_epsilon_greedy_explores():
    """With epsilon=1, should explore uniformly."""
    ql = HelperQLearning(seed=0, epsilon=1.0, epsilon_min=1.0, epsilon_decay=1.0)
    counts = np.zeros(2)
    for _ in range(10000):
        a = ql.choose_action(0, 0)
        counts[a] += 1
    # Should be roughly uniform
    assert abs(counts[0] / 10000 - 0.5) < 0.02
    assert abs(counts[1] / 10000 - 0.5) < 0.02


def test_greedy_exploits():
    """With epsilon=0, should exploit best action."""
    ql = HelperQLearning(seed=0, epsilon=0.0, epsilon_min=0.0)
    # Set Q-values: ACCEPT better in state (0,0)
    ql.q_table[encode_state(0, 0), 0] = 1.0  # ACCEPT
    ql.q_table[encode_state(0, 0), 1] = 0.0  # REJECT
    assert ql.choose_action(0, 0) == 0
    assert ql.greedy_action(0, 0) == 0


def test_q_learning_update():
    """Test Q-learning update formula."""
    ql = HelperQLearning(seed=0, alpha=0.5, gamma_q=0.9, epsilon=0.0)
    # Initial Q = 0
    # Update: Q(s,a) += alpha * (r + gamma * max Q(s') - Q(s,a))
    # s=(0,0), a=0, r=1.0, s'=(1,0)
    # Q_new = 0 + 0.5 * (1.0 + 0.9 * 0 - 0) = 0.5
    ql.update(0, 0, 0, reward=1.0, next_workload=1, next_resources=0)
    assert np.isclose(ql.q_table[encode_state(0, 0), 0], 0.5)


def test_q_learning_update_with_future_value():
    """Test update with non-zero future Q-value."""
    ql = HelperQLearning(seed=0, alpha=0.5, gamma_q=0.9, epsilon=0.0)
    # Set future state Q-value
    ql.q_table[encode_state(1, 0), 0] = 10.0  # ACCEPT is great in next state
    # Update current state
    ql.update(0, 0, 0, reward=1.0, next_workload=1, next_resources=0)
    # Q_new = 0 + 0.5 * (1.0 + 0.9 * 10.0 - 0) = 0.5 * 10.0 = 5.0
    assert np.isclose(ql.q_table[encode_state(0, 0), 0], 5.0)


def test_epsilon_decay():
    """Epsilon should decay over updates."""
    ql = HelperQLearning(seed=0, epsilon=0.5, epsilon_min=0.01, epsilon_decay=0.9)
    initial_eps = ql.epsilon
    ql.update(0, 0, 0, 1.0, 0, 0)
    assert ql.epsilon < initial_eps
    # After many updates, should approach epsilon_min
    for _ in range(100):
        ql.update(0, 0, 0, 1.0, 0, 0)
    assert np.isclose(ql.epsilon, 0.01, atol=0.001)


def test_independent_rng_per_agent():
    """Different seeds should give different action sequences."""
    ql1 = HelperQLearning(seed=1, epsilon=1.0, epsilon_min=1.0, epsilon_decay=1.0)
    ql2 = HelperQLearning(seed=2, epsilon=1.0, epsilon_min=1.0, epsilon_decay=1.0)
    actions1 = [ql1.choose_action(0, 0) for _ in range(100)]
    actions2 = [ql2.choose_action(0, 0) for _ in range(100)]
    # Very unlikely to be identical with different seeds
    assert actions1 != actions2


def test_reset_preserves_seed():
    """Reset with same seed should reproduce behavior."""
    ql = HelperQLearning(seed=42, epsilon=1.0, epsilon_min=1.0, epsilon_decay=1.0)
    actions1 = [ql.choose_action(0, 0) for _ in range(20)]
    ql.reset(seed=42)
    actions2 = [ql.choose_action(0, 0) for _ in range(20)]
    assert actions1 == actions2


def test_get_q_values():
    """Test getting Q-values for a state."""
    ql = HelperQLearning(seed=0)
    ql.q_table[encode_state(1, 2), 0] = 3.5
    ql.q_table[encode_state(1, 2), 1] = -1.2
    vals = ql.get_q_values(1, 2)
    assert np.allclose(vals, [3.5, -1.2])


def test_get_policy():
    """Test getting greedy policy for all states."""
    ql = HelperQLearning(seed=0)
    # Make ACCEPT better in all states
    ql.q_table[:, 0] = 1.0
    ql.q_table[:, 1] = 0.0
    policy = ql.get_policy()
    assert policy.shape == (9,)
    assert np.all(policy == 0)  # All ACCEPT


def test_snapshot():
    """Test snapshot includes all relevant info."""
    ql = HelperQLearning(seed=0, alpha=0.1, gamma_q=0.9, epsilon=0.1)
    snap = ql.snapshot()
    assert "q_table" in snap
    assert "alpha" in snap
    assert "gamma_q" in snap
    assert "epsilon" in snap
    assert "policy" in snap
    assert len(snap["q_table"]) == 9
    assert len(snap["policy"]) == 9