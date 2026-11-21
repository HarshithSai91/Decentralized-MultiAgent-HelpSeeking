# backend/src/help_seeking/task_lifecycle.py
"""Task lifecycle and solo-execution progression model for temporal help-seeking.

Models a task with an explicit temporal horizon, progress tracking, and deadline.
The requester can perform independent work attempts (consuming resources and workload),
observe progress, and decide when to seek help from a peer before the deadline expires.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class TemporalTask:
    """Explicit task state for temporal multi-agent help-seeking.

    State representation:
        - task_id: Unique task identifier.
        - requester_id: ID of the agent owning the task.
        - progress: Current cumulative progress in [0.0, 1.0].
        - required_progress: Target completion threshold (default 1.0).
        - deadline: Total allowed time steps / attempts before expiry.
        - remaining_time: Time steps remaining until deadline expires.
        - attempts: Number of solo execution attempts performed.
        - help_requested: Whether assistance has already been requested (at most 1 request per task).
        - help_accepted: Whether requested assistance was accepted.
        - completed: Whether the task reached required_progress.
        - failed: Whether deadline expired without completion.
    """
    task_id: int
    requester_id: int
    progress: float = 0.0
    required_progress: float = 1.0
    deadline: int = 5
    remaining_time: Optional[int] = None
    attempts: int = 0
    help_requested: bool = False
    help_accepted: Optional[bool] = None
    completed: bool = False
    failed: bool = False

    def __post_init__(self):
        if self.remaining_time is None:
            self.remaining_time = int(self.deadline)
        else:
            self.remaining_time = int(self.remaining_time)

    @property
    def progress_fraction(self) -> float:
        """Fraction of required work completed in [0.0, 1.0]."""
        return min(1.0, max(0.0, self.progress / self.required_progress))

    @property
    def slack(self) -> int:
        """Remaining time steps before deadline."""
        return max(0, self.remaining_time)

    def attempt_solo(self, solo_capability: float, rng: np.random.Generator) -> float:
        """Execute one unit of solo work on the task.

        Progress increment model:
            Each solo attempt consumes 1 unit of task time.
            The continuous stochastic progress increment Delta_p is scaled by the
            physical solo capability parameter kappa_solo:
                Delta_p ~ Uniform(0.20, 0.45) * kappa_solo

        Args:
            solo_capability: Physical solo execution scaling factor (kappa_solo).
            rng: Agent's private random number generator.

        Returns:
            The progress increment gained in this attempt.
        """
        if self.completed or self.failed:
            return 0.0

        self.attempts += 1
        self.remaining_time -= 1

        # Mathematical progress increment:
        # Uniform stochastic progress scaled by agent solo capability:
        # Delta_p ~ Uniform(0.20, 0.45) * solo_capability
        increment = float(rng.uniform(0.20, 0.45) * solo_capability)

        self.progress = min(self.required_progress, self.progress + increment)

        if self.progress >= self.required_progress - 1e-9:
            self.completed = True
            self.failed = False
        elif self.remaining_time <= 0:
            self.failed = True
            self.completed = False

        return increment

    def apply_helper_execution(self, success: int) -> bool:
        """Apply the outcome of an accepted helper execution.

        If helper succeeds (success=1), the task is immediately resolved as completed.
        If helper fails (success=0), the task remains at its current progress, consumes 1 time unit,
        and can continue via solo attempts if remaining time permits.
        """
        self.help_requested = True
        self.help_accepted = True
        self.remaining_time = max(0, self.remaining_time - 1)

        if success == 1:
            self.progress = self.required_progress
            self.completed = True
            self.failed = False
            return True
        else:
            if self.remaining_time <= 0 and not self.completed:
                self.failed = True
            return False

    def mark_help_rejected(self) -> None:
        """Record that a help request was rejected by the candidate helper.

        Rejection consumes 1 communication round. If remaining time > 0, the requester
        can resume solo attempts.
        """
        self.help_requested = True
        self.help_accepted = False
        self.remaining_time = max(0, self.remaining_time - 1)
        if self.remaining_time <= 0 and not self.completed:
            self.failed = True

    def snapshot(self) -> dict:
        """Serialize task state for logging and inspection."""
        return {
            "task_id": self.task_id,
            "requester_id": self.requester_id,
            "progress": round(self.progress, 3),
            "progress_fraction": round(self.progress_fraction, 3),
            "deadline": self.deadline,
            "remaining_time": self.remaining_time,
            "attempts": self.attempts,
            "help_requested": self.help_requested,
            "help_accepted": self.help_accepted,
            "completed": self.completed,
            "failed": self.failed,
        }
