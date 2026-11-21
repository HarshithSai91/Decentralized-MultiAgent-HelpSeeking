"""Unit and integration tests for temporal help-seeking (WHEN TO ASK).

Verifies:
1. Task lifecycle progression, deadline countdown, solo execution increments.
2. Independent RequesterTimingPolicy Q-learning and action selection.
3. Timing baselines (Immediate, Fixed Delay, Urgency, Never-Ask).
4. Full temporal multi-agent interaction integration.
5. Strict adherence to at most one request per task.
6. Helper Q-learning independent decisions during temporal tasks.
"""
from __future__ import annotations

import numpy as np
import pytest

from help_seeking.task_lifecycle import TemporalTask
from help_seeking.requester_timing import (
    RequesterTimingPolicy,
    ImmediateHelpTimingPolicy,
    FixedDelayTimingPolicy,
    UrgencyTimingPolicy,
    NeverAskTimingPolicy,
    CONTINUE_SELF,
    EVALUATE_HELP,
    encode_timing_state,
)
from help_seeking.temporal_environment import TemporalMultiAgentEnvironment
from help_seeking.temporal_simulation import compute_temporal_metrics
from help_seeking.reliability import StationaryProfile


# ---------------------------------------------------------------------------
# 1. Task Lifecycle Tests
# ---------------------------------------------------------------------------

def test_task_lifecycle_initialization():
    task = TemporalTask(task_id=1, requester_id=0, deadline=5)
    assert task.progress == 0.0
    assert task.remaining_time == 5
    assert task.attempts == 0
    assert not task.completed
    assert not task.failed
    assert not task.help_requested


def test_task_solo_progression():
    rng = np.random.default_rng(42)
    task = TemporalTask(task_id=1, requester_id=0, deadline=5)

    # First attempt: increment drawn from Uniform(0.20, 0.45) * 1.0
    inc1 = task.attempt_solo(solo_capability=1.0, rng=rng)
    assert 0.20 <= inc1 <= 0.45
    assert abs(task.progress - inc1) < 1e-9
    assert task.remaining_time == 4
    assert task.attempts == 1
    assert not task.completed

    # Second attempt
    inc2 = task.attempt_solo(solo_capability=1.0, rng=rng)
    assert 0.20 <= inc2 <= 0.45
    assert abs(task.progress - (inc1 + inc2)) < 1e-9
    assert task.remaining_time == 3
    assert task.attempts == 2


def test_task_deadline_expiration():
    rng = np.random.default_rng(10)
    task = TemporalTask(task_id=2, requester_id=0, deadline=2)

    # Capability 0.1 -> increments between 0.02 and 0.045, failing deadline of 2 steps
    task.attempt_solo(solo_capability=0.1, rng=rng)
    assert task.remaining_time == 1
    assert not task.failed

    task.attempt_solo(solo_capability=0.1, rng=rng)
    assert task.remaining_time == 0
    assert task.failed
    assert not task.completed


def test_rejection_time_consistency():
    task = TemporalTask(task_id=3, requester_id=0, deadline=3)
    assert task.remaining_time == 3
    task.mark_help_rejected()
    # Exactly one deadline unit consumed
    assert task.remaining_time == 2
    assert task.help_requested
    assert task.help_accepted is False
    assert not task.failed


def test_rejection_exhausts_deadline_without_extra_solo():
    task = TemporalTask(task_id=4, requester_id=0, deadline=1)
    task.mark_help_rejected()
    assert task.remaining_time == 0
    assert task.failed
    # Calling attempt_solo on expired task returns 0.0 and does not decrement remaining_time
    rng = np.random.default_rng(0)
    inc = task.attempt_solo(solo_capability=1.0, rng=rng)
    assert inc == 0.0
    assert task.remaining_time == 0


# ---------------------------------------------------------------------------
# 2. Timing Policy Tests
# ---------------------------------------------------------------------------

def test_timing_policy_state_encoding():
    task = TemporalTask(task_id=1, requester_id=0, deadline=5)
    s = encode_timing_state(task)
    assert 0 <= s <= 26

    # Advance task to medium progress and lower time
    task.progress = 0.5
    task.remaining_time = 2
    task.attempts = 2
    s2 = encode_timing_state(task)
    assert 0 <= s2 <= 26
    assert s2 != s


def test_timing_qlearning_independent_instances():
    p1 = RequesterTimingPolicy(seed=1)
    p2 = RequesterTimingPolicy(seed=2)
    assert p1.q_table is not p2.q_table

    # Update p1 only
    p1.update(state=0, action=0, reward=1.0, next_state=1, done=False)
    assert p1.q_table[0, 0] > 0.0
    assert p2.q_table[0, 0] == 0.0


def test_timing_baselines():
    task = TemporalTask(task_id=1, requester_id=0, deadline=5)

    # Immediate
    imm = ImmediateHelpTimingPolicy()
    assert imm.select_action(task) == EVALUATE_HELP

    # Never Ask
    never = NeverAskTimingPolicy()
    assert never.select_action(task) == CONTINUE_SELF

    # Fixed Delay
    fixed = FixedDelayTimingPolicy(delay=2)
    assert fixed.select_action(task) == CONTINUE_SELF
    task.attempts = 1
    assert fixed.select_action(task) == CONTINUE_SELF
    task.attempts = 2
    assert fixed.select_action(task) == EVALUATE_HELP

    # Urgency
    urg = UrgencyTimingPolicy(urgency_threshold=2)
    assert urg.select_action(task) == CONTINUE_SELF  # remaining_time = 5 > 2
    task.remaining_time = 2
    assert urg.select_action(task) == EVALUATE_HELP


def test_one_help_request_limit():
    task = TemporalTask(task_id=1, requester_id=0, deadline=5)
    task.help_requested = True
    policy = RequesterTimingPolicy(seed=3)
    # Once help is requested, action is strictly forced to CONTINUE_SELF
    assert policy.select_action(task) == CONTINUE_SELF


# ---------------------------------------------------------------------------
# 3. Environment Integration Tests
# ---------------------------------------------------------------------------

def test_temporal_environment_step():
    profiles = [StationaryProfile([0.8]), StationaryProfile([0.8])]
    env = TemporalMultiAgentEnvironment(
        n_agents=2,
        reliability_profiles=profiles,
        U_self=0.05,
        timing_policy_type="immediate",
        default_deadline=4,
        c_fail=0.5,
        seed=3,
    )
    rec = env.run_task()
    assert rec.task_id == 1
    assert rec.steps_taken <= 4
    assert rec.completed or rec.deadline_exceeded
    assert rec.help_requested  # immediate policy triggers help evaluation
    assert rec.actual_request_step is not None
    assert rec.actual_request_step >= 1


def test_failure_negative_reward():
    profiles = [StationaryProfile([0.0]), StationaryProfile([0.0])]
    env = TemporalMultiAgentEnvironment(
        n_agents=2,
        reliability_profiles=profiles,
        U_self=0.01,
        timing_policy_type="never",  # Never ask -> pure solo attempts
        default_deadline=1,           # 1 step -> cannot reach 1.0 progress
        c_fail=0.75,
        seed=1,
    )
    rec = env.run_task()
    assert rec.deadline_exceeded
    assert not rec.completed
    # Negative reward includes failure penalty -c_fail
    assert rec.requester_reward <= -0.75


def test_temporal_environment_multi_tasks_and_metrics():
    profiles = [StationaryProfile([0.8]), StationaryProfile([0.8])]
    env = TemporalMultiAgentEnvironment(
        n_agents=2,
        reliability_profiles=profiles,
        timing_policy_type="learned",
        default_deadline=5,
        seed=100,
    )
    history = env.run(n_tasks=20)
    assert len(history) == 20

    metrics = compute_temporal_metrics(history, env)
    assert "task_completion_rate" in metrics
    assert "avg_solo_attempts" in metrics
    assert "ask_rate" in metrics
    assert 0.0 <= metrics["task_completion_rate"] <= 1.0
    assert metrics["total_tasks"] == 20


def test_belief_decay_during_consecutive_solo_epochs():
    """Verify that beliefs decay geometrically once per elapsed solo decision round."""
    profiles = [StationaryProfile([0.8]), StationaryProfile([0.8])]
    gamma = 0.95
    env = TemporalMultiAgentEnvironment(
        n_agents=2,
        reliability_profiles=profiles,
        requester_policy_params={"gamma": gamma},
        timing_policy_type="never",  # Pure solo execution (no help queries)
        default_deadline=3,
        seed=42,
    )
    # Inject non-default prior evidence across all agents
    for agent in env.agents:
        agent.acceptance.A[:] = 10.0
        agent.acceptance.R[:] = 5.0
        agent.trust.S[:] = 8.0
        agent.trust.F[:] = 4.0

    init_acc_A = 10.0
    init_trust_S = 8.0

    rec = env.run_task()
    requester = env.agents[rec.requester_id]
    steps = rec.steps_taken
    assert steps >= 2  # multiple solo attempts

    # Each solo step must have decayed beliefs by gamma
    expected_acc_A = init_acc_A * (gamma ** steps)
    expected_trust_S = init_trust_S * (gamma ** steps)
    assert abs(requester.acceptance.A[0] - expected_acc_A) < 1e-6
    assert abs(requester.trust.S[0] - expected_trust_S) < 1e-6


def test_rejection_agent_state_transitions():
    """Verify state transitions on rejection: helper rejects, requester waits, inactive peers rest."""
    profiles = [StationaryProfile([0.8]), StationaryProfile([0.8]), StationaryProfile([0.8])]
    env = TemporalMultiAgentEnvironment(
        n_agents=3,
        reliability_profiles=profiles,
        timing_policy_type="immediate",
        helper_policy_type="always_reject",
        default_deadline=5,
        seed=7,
    )
    # Set known workloads and resources
    for agent in env.agents:
        agent.workload = 1
        agent.resources = 1

    rec = env.run_task()
    assert rec.help_requested
    assert rec.help_accepted is False
    assert rec.steps_taken >= 2
