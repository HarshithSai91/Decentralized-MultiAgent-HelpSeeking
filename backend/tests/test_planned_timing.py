"""Tests for the model-based (planned) timing policy used in E8."""
import numpy as np
import pytest

from help_seeking.requester_timing import (PlannedTimingPolicy, make_timing_policy,
                                           CONTINUE_SELF, EVALUATE_HELP)
from help_seeking.task_lifecycle import TemporalTask
from help_seeking.temporal_environment import TemporalMultiAgentEnvironment
from help_seeking.temporal_simulation import compute_temporal_metrics
from help_seeking.reliability import StationaryProfile


def _env(timing="planned", n=3, theta=0.8, seed=5):
    return TemporalMultiAgentEnvironment(n_agents=n, reliability_profiles=[StationaryProfile([theta]) for _ in range(n)],
                                         timing_policy_type=timing, default_deadline=5, step_cost=0.02,
                                         c_fail=0.5, solo_capability=0.6, seed=seed)


def _task(progress, remaining, attempts=0):
    t = TemporalTask(task_id=1, requester_id=0, progress=progress, required_progress=1.0,
                     deadline=5, remaining_time=remaining)
    t.attempts = attempts
    return t


def test_factory_and_binding():
    assert isinstance(make_timing_policy("planned"), PlannedTimingPolicy)
    env = _env()
    for agent in env.agents:
        assert agent.timing_policy.agent is agent and agent.timing_policy.env is env


def test_wait_value_without_help_equals_exact_solo_value():
    """With q = 0 (help never succeeds) and prohibitive cost, waiting forever is optimal,
    so Wait must reproduce the exact Irwin-Hall solo value."""
    env = _env()
    pol = env.agents[0].timing_policy
    for tau in (1, 2, 3, 5):
        ask, wait = pol.ask_and_wait(np.array([0.0]), np.array([10.0]), tau)
        for p in (0.0, 0.3, 0.6, 0.85):
            exact = env.compute_v_solo(p, tau, env.solo_capability)
            assert abs(np.interp(p, pol.grid, wait) - exact) < 0.01


def test_waits_when_solo_is_almost_done_and_asks_when_it_cannot_finish():
    env = _env()
    agent = env.agents[0]
    pol = agent.timing_policy
    # make the requester confident that teammates accept and succeed
    for j in range(2):
        for _ in range(30):
            agent.requester_policy.record_acceptance_observation(j, True)
            agent.requester_policy.record_trust_observation(j, 1)
    assert pol.select_action(_task(0.95, 3)) == CONTINUE_SELF      # one solo step almost surely finishes
    assert pol.select_action(_task(0.0, 2)) == EVALUATE_HELP       # solo cannot reach 1.0 in 2 steps


def test_option_value_makes_planner_wait_where_one_shot_rule_would_ask():
    """At task start the one-shot rule (ask now vs. never ask) prefers asking, but the planner
    sees that one solo step first keeps the option to ask and is worth more."""
    env = _env()
    agent = env.agents[0]
    pol = agent.timing_policy
    for j in range(2):
        for _ in range(30):
            agent.requester_policy.record_acceptance_observation(j, True)
        for _ in range(10):
            agent.requester_policy.record_trust_observation(j, 1)
            agent.requester_policy.record_trust_observation(j, 0)
    q = agent.requester_policy.acceptance_means() * agent.requester_policy.trust_means()
    ask, wait = pol.ask_and_wait(q, agent.requester_policy.costs, 5)
    v_solo = env.compute_v_solo(0.0, 5, env.solo_capability)
    assert np.interp(0.0, pol.grid, ask) > v_solo          # one-shot rule would ask immediately
    assert np.interp(0.0, pol.grid, wait) > np.interp(0.0, pol.grid, ask)  # planner prefers to wait
    assert pol.select_action(_task(0.0, 5)) == CONTINUE_SELF


def test_planned_policy_runs_in_environment():
    env = _env(n=4, theta=0.7, seed=11)
    recs = env.run(n_tasks=40)
    m = compute_temporal_metrics(recs, env)
    assert len(recs) == 40
    assert 0.0 <= m["task_completion_rate"] <= 1.0
    assert m["ask_rate"] > 0.0


def test_planned_policy_supports_deadlines_beyond_the_default():
    env = _env(n=4, theta=0.7, seed=3)          # default_deadline = 5
    for _ in range(15):
        rec = env.run_task(deadline=7)
        assert rec.steps_taken <= 7
    pol = env.agents[0].timing_policy
    ask, wait = pol.ask_and_wait(np.array([0.5, 0.5, 0.5]), np.array([0.1, 0.1, 0.1]), 7)
    assert ask.shape == wait.shape == pol.grid.shape
