"""`OutputTransform` — value-typed transforms from raw to training outputs.

An `IntermediateTarget`'s ``output_transform`` is the rule for turning
cached raw evaluations of the *base* target ``f`` into training values
``Y_train`` for the round's tempered target ``f_state``. Historically
this was a bare callable ``(state, X, Y_raw) -> Y_train``; this module
replaces that with a small structured value-type hierarchy.

A structured `OutputTransform` knows two things:

1. **How to apply itself.** ``apply(state, X, Y_raw) -> Y_train`` (and
   ``__call__`` delegating, so existing call sites read unchanged).
   This is the correctness ground truth for materializing ``Y_train``
   at a given state.
2. **How to derive a fast-path update across states.**
   ``diff(state_a, state_b) -> EmulatorUpdate | None`` returns a
   structured update op describing how the existing emulator's Y
   values would change going from state_a to state_b. ``None`` means
   "no fast path expressible here; loop should fall back to refit."

Together with `sabi.emulators.dispatch.update_emulator`, this gives
the loop a single source of truth: the transform object describes
both *correctness* (apply) and *performance* (diff). Schemes pick
the appropriate `OutputTransform` subtype based on the math; the
emulator-side dispatch decides whether a cheap path is available.

Concrete subtypes:

- `Identity`: ``Y_train == Y_raw`` at any state. ``diff`` returns
  `RescaleOutputs(factor=1.0)` — a no-op update that handlers can
  short-circuit, and that combines cleanly with new-row appends.
- `Rescale`: ``Y_train = state * Y_raw`` (state is a scalar). ``diff``
  returns ``RescaleOutputs(factor=state_b/state_a)``.
- `Generic`: arbitrary callable, no structural promises. ``diff``
  defaults to ``None``; loop always falls back to refit.

Schemes that want a transform whose structural shape depends on the
state are free to construct different subtypes per state — the
transform object lives on the `IntermediateTarget`, so it's a
per-state value.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax import Array

from sabi.emulators.updates import EmulatorUpdate, RescaleOutputs


class OutputTransform(ABC):
    """Structured transform from ``(state, X, Y_raw)`` to training outputs.

    Subclasses implement :meth:`apply`. :meth:`__call__` delegates to
    ``apply`` so call sites read as plain function calls. Override
    :meth:`diff` to advertise a closed-form fast-path update across
    state changes; the default returns ``None`` (no fast path).
    """

    @abstractmethod
    def apply(self, state: Any, X: Array, Y_raw: Array) -> Array:
        """Return ``Y_train`` at ``state``."""

    def __call__(self, state: Any, X: Array, Y_raw: Array) -> Array:
        """Convenience: same as :meth:`apply`."""
        return self.apply(state, X, Y_raw)

    def diff(self, state_a: Any, state_b: Any) -> EmulatorUpdate | None:
        """Optional structured update describing the ``state_a → state_b`` transition.

        Returns an `EmulatorUpdate` describing how the existing
        emulator's training Y values would change to be at ``state_b``,
        or ``None`` if no closed-form fast path exists (caller should
        fall back to refit).

        Default implementation returns ``None``. Subtypes with a
        structural shape (e.g., `Rescale`, `Identity`) override.
        """
        return None


@dataclass(frozen=True)
class Identity(OutputTransform):
    """``Y_train == Y_raw`` regardless of state.

    Used for tempering schemes whose state change is captured entirely
    in the form (`LikelihoodTemperingViaForm`) or where there is no
    tempering at all (`NoTempering`). ``diff`` returns the no-op
    rescale ``RescaleOutputs(factor=1.0)`` — combining cleanly with
    new-row appends in the loop's plan-building, and short-circuited
    by handlers that special-case ``factor == 1.0``.
    """

    def apply(self, state: Any, X: Array, Y_raw: Array) -> Array:
        return Y_raw

    def diff(self, state_a: Any, state_b: Any) -> EmulatorUpdate | None:
        # Y is invariant in state; the update is a structural no-op.
        return RescaleOutputs(factor=1.0)


@dataclass(frozen=True)
class Rescale(OutputTransform):
    """``Y_train = state * Y_raw``. State is a scalar.

    The natural transform for `LikelihoodTemperingViaTarget`, where the
    emulator target is ``f_state(x) = state * f(x)``. ``diff(a, b)``
    returns ``RescaleOutputs(factor=b/a)``; if ``a == 0`` the update
    is undefined (existing Y is identically zero, so we can't recover
    Y_raw to rescale it — caller falls back to refit).
    """

    def apply(self, state: Any, X: Array, Y_raw: Array) -> Array:
        return jnp.asarray(state) * Y_raw

    def diff(self, state_a: Any, state_b: Any) -> EmulatorUpdate | None:
        a = float(jnp.asarray(state_a))
        b = float(jnp.asarray(state_b))
        if a == 0.0:
            # Can't recover Y_raw from state_a == 0 (the existing Y is
            # identically zero); no closed-form update.
            return None
        return RescaleOutputs(factor=b / a)


@dataclass(frozen=True)
class Generic(OutputTransform):
    """Black-box callable: arbitrary ``(state, X, Y_raw) -> Y_train``.

    Use when the transform doesn't fit one of the structural subtypes
    above. ``diff`` defaults to ``None``, so all state changes fall
    back to refit. If the transform happens to be a closed-form rescale
    or identity in disguise, prefer the structural subtype to enable
    the fast path.
    """

    fn: Callable[[Any, Array, Array], Array]

    def apply(self, state: Any, X: Array, Y_raw: Array) -> Array:
        return self.fn(state, X, Y_raw)
