"""Tests for `OutputTransform` value-types.

Coverage:
- `apply` and `__call__` produce identical results for each subtype.
- `Identity.diff` returns the no-op rescale `RescaleOutputs(1.0)`.
- `Rescale.diff` returns `RescaleOutputs(state_b/state_a)`; ``state_a==0``
  returns ``None``.
- `Generic.diff` returns ``None`` (no structural promise).
- The base `OutputTransform.diff` defaults to ``None``.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from sabi.emulators.updates import EmulatorUpdate, RescaleOutputs
from sabi.tempering.output_transform import (
    Generic,
    Identity,
    OutputTransform,
    Rescale,
)


# -------------------------------------------------------------------------
# apply == __call__
# -------------------------------------------------------------------------


def test_identity_apply_returns_y_raw_unchanged():
    Y_raw = jnp.asarray([1.0, 2.0, 3.0])
    out = Identity().apply(state=0.5, X=jnp.zeros((3, 2)), Y_raw=Y_raw)
    assert jnp.array_equal(out, Y_raw)


def test_identity_call_matches_apply():
    Y_raw = jnp.asarray([1.0, 2.0, 3.0])
    t = Identity()
    assert jnp.array_equal(t(0.5, jnp.zeros((3, 2)), Y_raw), t.apply(0.5, jnp.zeros((3, 2)), Y_raw))


def test_rescale_apply_multiplies_by_state():
    Y_raw = jnp.asarray([1.0, 2.0, 3.0])
    out = Rescale().apply(state=0.5, X=jnp.zeros((3, 2)), Y_raw=Y_raw)
    assert jnp.allclose(out, 0.5 * Y_raw)


def test_rescale_call_matches_apply():
    Y_raw = jnp.asarray([1.0, 2.0, 3.0])
    t = Rescale()
    assert jnp.allclose(t(0.5, jnp.zeros((3, 2)), Y_raw), t.apply(0.5, jnp.zeros((3, 2)), Y_raw))


def test_generic_apply_uses_callable():
    Y_raw = jnp.asarray([1.0, 2.0, 3.0])
    t = Generic(fn=lambda s, X, Y: s * Y * Y)
    out = t.apply(state=2.0, X=jnp.zeros((3, 2)), Y_raw=Y_raw)
    assert jnp.allclose(out, 2.0 * Y_raw * Y_raw)


# -------------------------------------------------------------------------
# diff
# -------------------------------------------------------------------------


def test_identity_diff_returns_unit_rescale():
    """Identity is invariant in state — diff is the no-op `RescaleOutputs(1.0)`."""
    diff = Identity().diff(0.1, 0.9)
    assert isinstance(diff, RescaleOutputs)
    assert diff.factor == 1.0


def test_identity_diff_unit_rescale_for_any_state_pair():
    diff_a = Identity().diff("anything", "anything_else")
    diff_b = Identity().diff(None, None)
    assert isinstance(diff_a, RescaleOutputs) and diff_a.factor == 1.0
    assert isinstance(diff_b, RescaleOutputs) and diff_b.factor == 1.0


def test_rescale_diff_returns_factor_ratio():
    diff = Rescale().diff(state_a=0.4, state_b=0.8)
    assert isinstance(diff, RescaleOutputs)
    assert diff.factor == pytest.approx(2.0)


def test_rescale_diff_at_equal_states_is_unit_factor():
    diff = Rescale().diff(state_a=0.4, state_b=0.4)
    assert isinstance(diff, RescaleOutputs)
    assert diff.factor == pytest.approx(1.0)


def test_rescale_diff_from_zero_returns_none():
    """state_a == 0 means existing Y is identically 0; can't recover Y_raw to rescale."""
    assert Rescale().diff(state_a=0.0, state_b=0.5) is None


def test_generic_diff_returns_none():
    """Generic transforms make no structural promise about state changes."""
    t = Generic(fn=lambda s, X, Y: Y)
    assert t.diff(0.1, 0.9) is None


def test_base_output_transform_diff_defaults_to_none():
    """Subclasses that don't override `diff` get the no-fast-path default."""

    class CustomTransform(OutputTransform):
        def apply(self, state, X, Y_raw):
            return Y_raw

    assert CustomTransform().diff(0.1, 0.9) is None


# -------------------------------------------------------------------------
# Diff is structurally-typed (sealed-style EmulatorUpdate)
# -------------------------------------------------------------------------


def test_diff_returns_emulator_update_subtype_or_none():
    """Diff's return type is structurally an EmulatorUpdate or None — not a callable."""
    diff_identity = Identity().diff(0.1, 0.9)
    diff_rescale = Rescale().diff(0.1, 0.9)
    assert diff_identity is None or isinstance(diff_identity, EmulatorUpdate)
    assert diff_rescale is None or isinstance(diff_rescale, EmulatorUpdate)


# -------------------------------------------------------------------------
# diff apply-equivalence: applying the structured diff to Y_train_a yields
# Y_train_b. Catches sign / inverse-direction bugs in `diff`.
# -------------------------------------------------------------------------


def test_identity_diff_apply_equivalence():
    """Identity transform: Y_train is invariant in state, so applying the
    diff (a unit rescale) to Y_train_a yields Y_train_b == Y_train_a."""
    transform = Identity()
    X = jnp.zeros((4, 2))
    Y_raw = jnp.asarray([1.0, -2.0, 3.5, 0.0])
    state_a, state_b = 0.3, 0.8
    Y_train_a = transform.apply(state_a, X, Y_raw)
    Y_train_b = transform.apply(state_b, X, Y_raw)

    diff = transform.diff(state_a, state_b)
    assert isinstance(diff, RescaleOutputs)
    # Apply the structured update: multiply existing Y_train by factor.
    Y_train_a_updated = diff.factor * Y_train_a
    assert jnp.allclose(Y_train_a_updated, Y_train_b)


def test_rescale_diff_apply_equivalence():
    """Rescale transform: Y_train_a = state_a * Y_raw, Y_train_b = state_b *
    Y_raw, and diff returns RescaleOutputs(factor=state_b/state_a). Applying
    the factor to Y_train_a recovers Y_train_b exactly."""
    transform = Rescale()
    X = jnp.zeros((4, 2))
    Y_raw = jnp.asarray([1.0, -2.0, 3.5, 0.5])
    state_a, state_b = 0.4, 0.8
    Y_train_a = transform.apply(state_a, X, Y_raw)
    Y_train_b = transform.apply(state_b, X, Y_raw)

    diff = transform.diff(state_a, state_b)
    assert isinstance(diff, RescaleOutputs)
    # Apply the structured update to Y_train_a; should equal a fresh apply
    # at state_b.
    Y_train_a_updated = diff.factor * Y_train_a
    assert jnp.allclose(Y_train_a_updated, Y_train_b)


def test_rescale_diff_apply_equivalence_reversed_direction():
    """Same equivalence but going from a larger state to a smaller one —
    catches sign / inverse-direction bugs (e.g., a/b vs b/a)."""
    transform = Rescale()
    X = jnp.zeros((3, 2))
    Y_raw = jnp.asarray([2.0, -1.5, 0.25])
    state_a, state_b = 0.9, 0.3
    Y_train_a = transform.apply(state_a, X, Y_raw)
    Y_train_b = transform.apply(state_b, X, Y_raw)

    diff = transform.diff(state_a, state_b)
    assert isinstance(diff, RescaleOutputs)
    Y_train_a_updated = diff.factor * Y_train_a
    assert jnp.allclose(Y_train_a_updated, Y_train_b)
