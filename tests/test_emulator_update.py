"""Tests for `EmulatorUpdate` types and the `update_emulator` dispatch.

Coverage:
- `EmulatorUpdate` dataclasses construct + are immutable.
- `update_emulator(plan=None)` falls back to `factory().fit(X_full, Y_full)`.
- `update_emulator(plan=...)` with no registered handler falls back to refit.
- `update_emulator` with a registered fast-path handler dispatches to it.
- The registry honors priority and `MethodInfo.feasible`.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from probpipe.core._registry import MethodInfo

from sabi.emulators import TinyGPEmulator
from sabi.emulators.dispatch import (
    EmulatorUpdateMethod,
    emulator_update_registry,
    update_emulator,
)
from sabi.emulators.updates import (
    AppendRows,
    EmulatorUpdate,
    RescaleOutputs,
    RescaleThenAppend,
)


# -------------------------------------------------------------------------
# EmulatorUpdate dataclasses
# -------------------------------------------------------------------------


def test_append_rows_constructs_and_is_immutable():
    op = AppendRows(X_new=jnp.zeros((2, 2)), Y_new=jnp.zeros((2,)))
    assert op.X_new.shape == (2, 2)
    assert op.Y_new.shape == (2,)
    with pytest.raises(Exception):
        op.X_new = jnp.zeros((3, 2))  # frozen


def test_rescale_outputs_constructs_and_is_immutable():
    op = RescaleOutputs(factor=0.5)
    assert op.factor == 0.5
    with pytest.raises(Exception):
        op.factor = 1.0


def test_rescale_then_append_constructs_and_is_immutable():
    op = RescaleThenAppend(
        factor=0.5, X_new=jnp.zeros((2, 2)), Y_new=jnp.zeros((2,))
    )
    assert op.factor == 0.5
    assert op.X_new.shape == (2, 2)
    with pytest.raises(Exception):
        op.factor = 1.0


def test_all_op_types_subclass_emulator_update():
    """Sealed-style hierarchy: every concrete op IS an EmulatorUpdate."""
    assert issubclass(AppendRows, EmulatorUpdate)
    assert issubclass(RescaleOutputs, EmulatorUpdate)
    assert issubclass(RescaleThenAppend, EmulatorUpdate)


# -------------------------------------------------------------------------
# Fallback path: no registered handler → refit
# -------------------------------------------------------------------------


def _gp_factory():
    return TinyGPEmulator(input_shape=(2,))


def _fit_emulator():
    """Small fitted GP emulator for reuse."""
    X = jnp.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, -1.0]])
    Y = jnp.asarray([0.0, 1.0, 0.5])
    return _gp_factory().fit(X, Y), X, Y


def test_update_emulator_with_plan_none_falls_back_to_fit():
    em, _X, _Y = _fit_emulator()
    X_full = jnp.asarray([[0.5, 0.5], [1.5, 1.5]])
    Y_full = jnp.asarray([2.0, 3.0])
    out = update_emulator(em, None, factory=_gp_factory, X_full=X_full, Y_full=Y_full)
    # Fallback returns a freshly-fit emulator on (X_full, Y_full); that's
    # equivalent to `factory().fit(X_full, Y_full)`.
    expected = _gp_factory().fit(X_full, Y_full)
    test_X = jnp.asarray([[0.0, 0.0], [1.0, 0.0]])
    assert jnp.allclose(out.predict_mean(test_X), expected.predict_mean(test_X))


def test_update_emulator_with_unfitted_emulator_falls_back_to_fit():
    """An unfitted emulator has no cache, so the registered handlers
    report ``feasible=False`` (no Cholesky to update) and dispatch
    falls back to ``factory().fit(X_full, Y_full)``."""
    em_unfit = _gp_factory()  # not yet fit; _predict_cache is None.
    X_full = jnp.asarray([[0.5, 0.5], [1.5, 1.5]])
    Y_full = jnp.asarray([2.0, 3.0])
    plan = AppendRows(X_new=jnp.asarray([[1.5, 1.5]]), Y_new=jnp.asarray([3.0]))
    out = update_emulator(em_unfit, plan, factory=_gp_factory, X_full=X_full, Y_full=Y_full)
    expected = _gp_factory().fit(X_full, Y_full)
    test_X = jnp.asarray([[0.0, 0.0], [1.0, 0.0]])
    assert jnp.allclose(out.predict_mean(test_X), expected.predict_mean(test_X))


# -------------------------------------------------------------------------
# Registered fast-path: dispatch reaches the handler
# -------------------------------------------------------------------------


class _RecordingRescaleHandler(EmulatorUpdateMethod):
    """Test-only handler claiming RescaleOutputs(factor=1.0) as a no-op fast path.

    Returns the current emulator unchanged (factor=1.0 is a true no-op),
    while incrementing a call counter so we can verify the dispatch path.
    """

    def __init__(self):
        self.calls = 0

    @property
    def name(self) -> str:
        return "_test_rescale_noop"

    def supported_types(self) -> tuple[type, ...]:
        return (TinyGPEmulator,)

    def check(self, emulator, plan):
        feasible = isinstance(plan, RescaleOutputs) and plan.factor == 1.0
        return MethodInfo(feasible=feasible, method_name=self.name)

    def execute(self, emulator, plan):
        self.calls += 1
        return emulator  # factor=1.0 is structurally a no-op

    @property
    def priority(self) -> int:
        return 100  # ahead of any future default-priority handler


def test_registered_handler_is_dispatched_and_short_circuits_refit(monkeypatch):
    """When a handler reports `feasible=True`, `update_emulator` runs it and skips refit."""
    handler = _RecordingRescaleHandler()
    emulator_update_registry.register(handler)
    try:
        em, X, Y = _fit_emulator()
        plan = RescaleOutputs(factor=1.0)
        # Call factory: would raise if invoked unexpectedly. The handler's
        # short-circuit means factory shouldn't be called.

        def _exploding_factory():
            raise AssertionError("factory should not be called when fast path is feasible")

        out = update_emulator(em, plan, factory=_exploding_factory, X_full=X, Y_full=Y)
        assert handler.calls == 1
        assert out is em  # handler returns the input unchanged for factor=1.0
    finally:
        # Clean up the registry to keep tests independent.
        emulator_update_registry._methods.remove(handler)
        del emulator_update_registry._name_index[handler.name]
        emulator_update_registry._sort_methods()


def test_registered_handler_check_returning_infeasible_falls_back_to_refit():
    """A handler whose ``check`` reports ``feasible=False`` must not
    run; dispatch falls back to refit. We use an *unfitted* emulator
    so all GPEmulator-typed handlers also report infeasible — the
    only feasible candidate would be a custom one, and our test
    custom handler always reports infeasible too. With everyone
    infeasible, dispatch falls back to ``factory().fit(...)``.
    """

    class _NeverFeasible(EmulatorUpdateMethod):
        def __init__(self):
            self.execute_calls = 0

        @property
        def name(self) -> str:
            return "_test_never_feasible"

        def supported_types(self):
            return (TinyGPEmulator,)

        def check(self, emulator, plan):
            return MethodInfo(feasible=False, method_name=self.name, description="never")

        def execute(self, emulator, plan):
            self.execute_calls += 1
            return emulator

    handler = _NeverFeasible()
    emulator_update_registry.register(handler)
    try:
        # Unfitted emulator → built-in GP handlers also report
        # infeasible (cache absent), so the only feasibility decision
        # left is the custom handler's, which is always False →
        # dispatch falls back to refit.
        em_unfit = _gp_factory()
        X = jnp.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, -1.0]])
        Y = jnp.asarray([0.0, 1.0, 0.5])
        plan = RescaleOutputs(factor=0.5)
        out = update_emulator(em_unfit, plan, factory=_gp_factory, X_full=X, Y_full=Y)
        assert handler.execute_calls == 0
        # Fallback gives the same predictions as a direct refit.
        expected = _gp_factory().fit(X, Y)
        test_X = jnp.asarray([[0.0, 0.0], [1.0, 0.0]])
        assert jnp.allclose(out.predict_mean(test_X), expected.predict_mean(test_X))
    finally:
        emulator_update_registry._methods.remove(handler)
        del emulator_update_registry._name_index[handler.name]
        emulator_update_registry._sort_methods()
