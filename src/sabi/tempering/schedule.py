"""Tempering schedules: choose the tempering state at each round of the loop.

A schedule emits `(state, final)` where `state` is an opaque PyTree
whose shape / type matches the paired `TemperingScheme` (a scalar β
for likelihood tempering, a subset id for data tempering, etc.).
`final=True` signals that this round is the terminal target; the
schedule should not advance past it.

The optional `terminal_state()` method returns the schedule's final
state (the one corresponding to the actual target distribution) — used
by `AcquisitionTarget.TERMINAL` to build the look-ahead SP at the
terminal target. The default implementation probes
`next(round_idx, None)` at a large index; built-in schedules override
with closed-form returns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TemperingSchedule(ABC):
    @abstractmethod
    def next(self, round_idx: int, loop_state: Any) -> tuple[Any, bool]:
        """Return `(tempering_state, final)` for the given round."""

    def terminal_state(self) -> Any:
        """Return the schedule's terminal state (the one with `final=True`).

        Default: probe `next(round_idx, None)` at a large index. Both
        built-in schedules clamp past-the-end to the terminal entry, so
        this works. Adaptive schedules (e.g., ESS-adaptive) can't
        precompute the terminal without running the loop and should
        override (or raise `NotImplementedError`) accordingly.
        """
        state, final = self.next(10**9, None)
        if not final:
            raise NotImplementedError(
                f"{type(self).__name__}.terminal_state: schedule did not "
                "converge to a terminal state at a large round index. "
                "Override `terminal_state()` explicitly for adaptive "
                "schedules."
            )
        return state


@dataclass
class UntemperedSchedule(TemperingSchedule):
    """Always returns `(None, True)`. Default for untempered loops."""

    def next(self, round_idx: int, loop_state: Any) -> tuple[Any, bool]:
        return None, True

    def terminal_state(self) -> Any:
        return None


@dataclass
class FixedSchedule(TemperingSchedule):
    """Iterate through a pre-computed sequence of tempering states.

    The schedule is strategy-agnostic: `states` may be floats
    (likelihood tempering), ints (data tempering subset indices), or
    anything the paired `TemperingScheme.intermediate_target` knows how
    to consume. The last entry is marked `final=True`; the user is
    responsible for ensuring it represents the terminal target.
    """

    states: tuple[Any, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.states) == 0:
            raise ValueError("FixedSchedule requires at least one state.")

    def next(self, round_idx: int, loop_state: Any) -> tuple[Any, bool]:
        idx = min(round_idx, len(self.states) - 1)
        return self.states[idx], idx == len(self.states) - 1

    def terminal_state(self) -> Any:
        return self.states[-1]
