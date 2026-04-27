"""Tempering schedules: choose the tempering state at each round of the loop.

A schedule emits `(state, final)` where `state` is an opaque PyTree whose
shape / type matches the paired `Tempering` strategy (a scalar β for
`LikelihoodTempering`, a subset id for `DataTempering`, etc.). `final=True`
signals that this round is the terminal target; the schedule should not
advance past it.
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


@dataclass
class UntemperedSchedule(TemperingSchedule):
    """Always returns `(None, True)`. Default for v0."""

    def next(self, round_idx: int, loop_state: Any) -> tuple[Any, bool]:
        return None, True


@dataclass
class FixedSchedule(TemperingSchedule):
    """Iterate through a pre-computed sequence of tempering states.

    The schedule is strategy-agnostic: `states` may be floats (likelihood
    tempering), ints (data tempering subset indices), or anything the paired
    `Tempering.apply` knows how to consume. The last entry is marked
    `final=True`; the user is responsible for ensuring it represents the
    terminal target.
    """

    states: tuple[Any, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.states) == 0:
            raise ValueError("FixedSchedule requires at least one state.")

    def next(self, round_idx: int, loop_state: Any) -> tuple[Any, bool]:
        idx = min(round_idx, len(self.states) - 1)
        return self.states[idx], idx == len(self.states) - 1
