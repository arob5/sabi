"""Declarative `EmulatorUpdate` types for the cheap-update dispatch.

An `EmulatorUpdate` is a *value* that names what changed between the
current emulator and a desired-next emulator. The
`sabi.emulators.dispatch.update_emulator` entry point reads the type
and decides whether a registered handler can realize the update
cheaply, falling back to a full refit otherwise.

Refit is intentionally **not** a first-class op here. ``Emulator.fit``
is the canonical "build a fresh emulator on this data" API, and the
dispatcher reaches for it as the universal fallback when no cheap
handler is feasible. So the op surface here is exactly the cheap-path
operations that arise naturally in the sequential-acquisition loop:

- `AppendRows`: append ``(X_new, Y_new)`` to the training set. Triggered
  by the round's acquisition step. ``Y_new`` is the values *at the
  emulator's current state* (i.e., already in the same coordinate
  system as the existing training Y).
- `RescaleOutputs`: multiply all training outputs by a scalar.
  Triggered by a tempering-state change for `Rescale`-shaped output
  transforms (e.g., `LikelihoodTemperingViaTarget`). A factor of
  ``1.0`` means "no change" — handlers may treat it as a no-op.
- `RescaleThenAppend`: composite — rescale existing outputs by
  ``factor``, then append ``(X_new, Y_new)`` already at the new
  state. The natural unit for a round of the loop that combines
  tempering with new acquisition rows.

Each update is an immutable dataclass; the loop builds a single op per
dispatch call. Composition deeper than ``RescaleThenAppend`` is left
to the fallback path (refit).

When ProbPipe ships a native emulator-update mechanism, the migration
is mostly mechanical: the op types either move into ProbPipe or stay
sabi-side, and the dispatcher (`sabi.emulators.dispatch`) rebases its
registry onto ProbPipe's.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from jax import Array


class EmulatorUpdate(ABC):
    """Sealed-style base for declarative emulator update ops.

    Subclasses are immutable dataclasses describing one cheap-path
    operation. The dispatcher (`update_emulator`) reads the runtime
    type to look up a registered handler.
    """


@dataclass(frozen=True)
class AppendRows(EmulatorUpdate):
    """Append ``(X_new, Y_new)`` to the existing training set.

    ``Y_new`` is given in the same coordinate system as the current
    emulator's training Y (i.e., already at the current state — the
    loop materializes it via ``output_transform.apply(state, X_new,
    Y_raw_new)`` before constructing this op).
    """

    X_new: Array
    Y_new: Array


@dataclass(frozen=True)
class RescaleOutputs(EmulatorUpdate):
    """Multiply all training outputs by ``factor``.

    Triggered by a tempering-state change for `Rescale`-shaped output
    transforms — e.g., going from ``Y_train_a = beta_a * Y_raw`` to
    ``Y_train_b = beta_b * Y_raw`` is a rescale by ``beta_b / beta_a``.

    A factor of ``1.0`` is a no-op; handlers are free to short-circuit.
    """

    factor: float


@dataclass(frozen=True)
class RescaleThenAppend(EmulatorUpdate):
    """Composite: rescale existing outputs by ``factor``, then append new rows.

    ``Y_new`` is at the *new* state (post-rescale coordinate system),
    so combining the two ops yields a training set whose existing
    rows have been rescaled and whose new rows are added at the new
    state — equivalent to refitting on the full materialized data.

    This composite is its own dataclass (rather than a generic
    sequence of ops) because backends often fuse the two operations
    into a single Cholesky update, beating two separate dispatch calls.
    """

    factor: float
    X_new: Array
    Y_new: Array
