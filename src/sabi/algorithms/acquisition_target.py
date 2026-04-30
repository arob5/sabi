"""`AcquisitionTarget` — which tempering state the acquisition optimizes against.

In a tempered loop, three natural choices exist for the state at which
the acquisition's `SurrogatePosterior` is built:

- **CURRENT**: state of the current round (`state_t`). Simplest;
  default. The acquisition sees the SP at the round's intermediate
  distribution.
- **NEXT**: state of the next round (`state_{t+1}`, clamped to terminal
  on the last round). Standard SMC-flavor look-ahead — acquisition
  picks points to inform the *next* intermediate. Typical when the
  loop is sampling toward a sequence of harder targets.
- **TERMINAL**: the schedule's terminal state. Acquisition optimizes
  toward the final target throughout, regardless of where the schedule
  is. Useful when intermediate distributions are scaffolding only and
  the final target is what matters.

For untempered loops (`NoTempering` + `UntemperedSchedule`), all three
collapse to the same state and the choice has no effect.

For richer policies (e.g., ESS-adaptive look-ahead, custom callable
that depends on loop state), see issue #5 — the enum is a starting
point; a `Callable[..., state]` policy generalizes it.
"""

from __future__ import annotations

from enum import Enum


class AcquisitionTarget(Enum):
    """Which tempering state the acquisition optimizes against."""

    CURRENT = "current"
    NEXT = "next"
    TERMINAL = "terminal"
