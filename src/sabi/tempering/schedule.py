"""Tempering schedules: assign a tempering state to each round of the loop.

Round indexing convention
-------------------------

A `round_idx` is the integer index of a round of target-map
evaluations. Conventions:

- **Round 0 is the initial-design round.** The loop's pre-loop block
  draws ``n_initial`` design points from the initial sampler and
  evaluates the target on them. This is round 0.
- **Round 1, 2, ..., n_rounds-1 are acquisition rounds.** Each adds
  ``q`` new evaluations chosen by the acquisition. The loop body
  iterates over these.
- **`Algorithm.n_rounds` is the total number of rounds of target-map
  evaluations**, including round 0. Total evaluations across a run:
  ``n_initial + (n_rounds - 1) * q``.

`TemperingSchedule.at(round_idx)` returns ``(state, is_terminal)`` for
the given round. The schedule is responsible for clamping past-the-end
indices to the terminal state (the built-in schedules do).

`schedule.terminal_state()` returns the schedule's final state directly
— used by `AcquisitionTarget.TERMINAL` to build a look-ahead SP at
the terminal target.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TemperingSchedule(ABC):
    @abstractmethod
    def at(self, round_idx: int) -> tuple[Any, bool]:
        """Return ``(tempering_state, is_terminal_state)`` for the given round.

        Round 0 corresponds to the initial-design round; subsequent
        round indices are acquisition rounds. The schedule should clamp
        past-the-end indices to the terminal state (the built-in
        schedules do).
        """

    def terminal_state(self) -> Any:
        """Return the schedule's terminal state.

        Default: probe `at(round_idx)` at a large index. Both built-in
        schedules clamp past-the-end to the terminal entry, so this
        works. Adaptive schedules (e.g., ESS-adaptive) can't precompute
        the terminal without running the loop and should override (or
        raise `NotImplementedError`) accordingly.
        """
        state, is_terminal = self.at(10**9)
        if not is_terminal:
            raise NotImplementedError(
                f"{type(self).__name__}.terminal_state: schedule did not "
                "converge to a terminal state at a large round index. "
                "Override `terminal_state()` explicitly for adaptive "
                "schedules."
            )
        return state


@dataclass
class UntemperedSchedule(TemperingSchedule):
    """Always returns ``(None, True)``. Default for untempered loops."""

    def at(self, round_idx: int) -> tuple[Any, bool]:
        return None, True

    def terminal_state(self) -> Any:
        return None


@dataclass
class FixedSchedule(TemperingSchedule):
    """Iterate through a pre-computed sequence of tempering states.

    The schedule is strategy-agnostic: ``states`` may be floats
    (likelihood tempering), ints (data tempering subset indices), or
    anything the paired `TemperingScheme.intermediate_target` knows how
    to consume. The last entry is marked ``is_terminal_state=True``;
    the user is responsible for ensuring it represents the terminal
    target.
    """

    states: tuple[Any, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.states) == 0:
            raise ValueError("FixedSchedule requires at least one state.")

    def at(self, round_idx: int) -> tuple[Any, bool]:
        idx = min(round_idx, len(self.states) - 1)
        return self.states[idx], idx == len(self.states) - 1

    def terminal_state(self) -> Any:
        return self.states[-1]
