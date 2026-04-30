"""Cheap-update dispatch for `Emulator`.

Sabi-side dispatcher built on ProbPipe's generic priority-based method
registry (`probpipe.core._registry.MethodRegistry`). Sharing the
registry shape now means future migration to a ProbPipe-native
emulator-update mechanism is mostly mechanical — register sabi's
methods onto whatever ProbPipe ships, drop the local singleton.

Architecture
------------

- `EmulatorUpdateMethod` — base class for handlers. Subclasses pin a
  specific ``(emulator_type, plan_type)`` pair via
  ``supported_types()`` (the emulator type, used for fast pre-filter)
  and ``check(emulator, plan)`` (further plan-type filtering returned
  via `MethodInfo.feasible`).

- `emulator_update_registry` — singleton `MethodRegistry`. Methods
  register at import time. Tests can install / remove handlers via
  the registry's `register` / `set_priorities` API.

- `update_emulator(current, plan, *, factory, X_full, Y_full)` —
  the entry point. Tries registered handlers in priority order; falls
  back to ``factory().fit(X_full, Y_full)`` when none is feasible. The
  caller is responsible for materializing ``(X_full, Y_full)`` (the
  data at the new state with all rows present), which guarantees the
  fallback path is always correct.

The fallback (``Emulator.fit``) is **not** a registered handler. It's
the universal correctness floor that runs when the registry has
nothing to offer. Plans like `RefitFromScratch` would conflate "I need
to fit from scratch" with "no cheap path available," which is exactly
the indirection we want to avoid.
"""

from __future__ import annotations

from collections.abc import Callable

from jax import Array
from probpipe.core._registry import Method, MethodInfo, MethodRegistry

from sabi.emulators.base import Emulator
from sabi.emulators.updates import EmulatorUpdate

__all__ = [
    "EmulatorUpdateMethod",
    "MethodInfo",  # re-export for handler authors
    "emulator_update_registry",
    "update_emulator",
]


class EmulatorUpdateMethod(Method):
    """Base class for emulator-update handlers.

    Subclasses implement:

    - ``name`` (unique string id),
    - ``supported_types() -> (EmulatorSubclass,)`` (fast pre-filter
      on the emulator type — `MethodRegistry` caches this),
    - ``check(emulator, plan) -> MethodInfo`` (further plan-type
      filtering; cheap, must not run the actual update),
    - ``execute(emulator, plan) -> Emulator`` (the cheap update).

    Optionally override ``priority`` (default 0; higher tried first).
    """


emulator_update_registry: MethodRegistry[EmulatorUpdateMethod] = MethodRegistry()


def update_emulator(
    current: Emulator,
    plan: EmulatorUpdate | None,
    *,
    factory: Callable[[], Emulator],
    X_full: Array,
    Y_full: Array,
) -> Emulator:
    """Apply `plan` to `current`; fall back to ``factory().fit(X_full, Y_full)``.

    The dispatcher consults `emulator_update_registry` for a feasible
    handler and runs it on success. On failure (no plan, or no
    registered handler reports ``feasible=True``), it materializes the
    full data and refits — guaranteed to be correct, even if slow.

    Args:
        current: emulator at its previous-fit state.
        plan: structured update description, or ``None`` to skip the
            registry and refit immediately. ``None`` is the explicit
            "I have no cheap path to offer" signal from the caller.
        factory: zero-arg producer of a fresh emulator (used for the
            refit fallback). Typically ``algorithm.emulator_factory``.
        X_full: training inputs after the update is applied.
        Y_full: training outputs after the update is applied (i.e.,
            at the new state, with all rows present). The fallback
            calls ``factory().fit(X_full, Y_full)`` to realize the
            update universally.

    Returns:
        Emulator equivalent to ``factory().fit(X_full, Y_full)`` — via
        the cheap path when a handler accepts; via refit otherwise.
    """
    if plan is None:
        return factory().fit(X_full, Y_full)
    info = emulator_update_registry.check(current, plan)
    if info.feasible:
        return emulator_update_registry.execute(current, plan)
    return factory().fit(X_full, Y_full)
