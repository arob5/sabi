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


# -------------------------------------------------------------------------
# TinyGP cheap-update equivalence — mirrors the DSPGPEmulator coverage
# in tests/test_dsp_gp.py. Handlers are typed against the abstract
# `GPEmulator` base, so the math is shared with DSPGPEmulator; covering
# it for TinyGPEmulator here verifies the dispatch lands on the right
# handler and the resulting predictions match a fresh fit.
# -------------------------------------------------------------------------


def _fitted_tinygp(seed: int = 401, n: int = 12):
    import jax.random as jr

    key = jr.key(seed)
    X = jr.uniform(key, (n, 2))
    Y = jnp.sin(X[:, 0])
    return TinyGPEmulator(input_shape=(2,)).fit(X, Y), X, Y


def test_tinygp_rescale_outputs_factor_one_is_exact_no_op():
    """`RescaleOutputs(factor=1.0)` is structurally a no-op: the
    handler short-circuits and returns the input emulator unchanged.
    Predictions should be bit-identical (no recomputation)."""
    em, X, Y = _fitted_tinygp(seed=411)

    def _exploding_factory():
        raise AssertionError("factor=1.0 short-circuit; factory should not be called")

    out = update_emulator(
        em,
        RescaleOutputs(factor=1.0),
        factory=_exploding_factory,
        X_full=X,
        Y_full=Y,
    )
    # The handler returns the original instance for factor=1.0.
    assert out is em


def test_tinygp_rescale_then_append_dispatch_matches_compose_of_steps():
    """`RescaleThenAppend(factor=β, X_new, Y_new)` should produce the
    same predictions as: rescale by β, then condition_on `Y_new` in
    new-state units. Mirrors `test_dspgp_rescale_then_append_dispatch_matches_compose_of_steps`
    for the TinyGP backend — the handler is typed against the abstract
    `GPEmulator` base, so coverage here verifies the right handler is
    picked and produces the right predictions on the tinygp cache."""
    import jax.random as jr

    em, X, Y = _fitted_tinygp(seed=421)

    X_new = jr.uniform(jr.key(422), (3, 2))
    beta = 2.5
    Y_new_at_new_state = beta * jnp.sin(X_new[:, 0])

    def _exploding_factory():
        raise AssertionError("cheap path should fire; factory should not be called")

    plan = RescaleThenAppend(factor=beta, X_new=X_new, Y_new=Y_new_at_new_state)
    em_after = update_emulator(
        em,
        plan,
        factory=_exploding_factory,
        X_full=jnp.concatenate([X, X_new], axis=0),
        Y_full=jnp.concatenate([beta * Y, Y_new_at_new_state], axis=0),
    )

    # Two-step path: rescale (cache untouched), then condition_on the new rows.
    em_rescaled = update_emulator(
        em,
        RescaleOutputs(factor=beta),
        factory=_exploding_factory,
        X_full=X,
        Y_full=beta * Y,
    )
    em_two_step = em_rescaled.condition_on(X_new, Y_new_at_new_state)

    X_test = jr.uniform(jr.key(423), (6, 2))
    assert jnp.allclose(
        em_after.predict_mean(X_test),
        em_two_step.predict_mean(X_test),
        rtol=1e-10,
        atol=1e-12,
    )
    assert jnp.allclose(
        em_after.predict_variance(X_test),
        em_two_step.predict_variance(X_test),
        rtol=1e-10,
        atol=1e-12,
    )


def test_tinygp_rescale_then_append_rejects_nonpositive_factor():
    """Factor ≤ 0 is infeasible (z-scoring requires positive scale);
    dispatch falls back to refit on the user-supplied (X_full, Y_full)."""
    import jax.random as jr

    em, X, Y = _fitted_tinygp(seed=431, n=8)
    X_new = jr.uniform(jr.key(432), (2, 2))
    Y_new = -jnp.sin(X_new[:, 0])

    factory_calls = {"n": 0}

    def _counting_factory():
        factory_calls["n"] += 1
        return TinyGPEmulator(input_shape=(2,))

    out = update_emulator(
        em,
        RescaleThenAppend(factor=-1.0, X_new=X_new, Y_new=Y_new),
        factory=_counting_factory,
        X_full=jnp.concatenate([X, X_new], axis=0),
        Y_full=jnp.concatenate([-Y, Y_new], axis=0),
    )
    assert factory_calls["n"] == 1
    assert out._predict_cache is not None
