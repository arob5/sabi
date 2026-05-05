"""Tests for FantasyImputer strategies."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from probpipe import mean

from sabi.acquisitions.fantasize import (
    ConstantLiar,
    KrigingBeliever,
)

from tests.conftest import make_acquisition_state


def _state(n: int = 20, seed: int = 0):
    """Local alias for the shared `make_acquisition_state` fixture builder."""
    return make_acquisition_state(n=n, seed=seed)


def test_kriging_believer_returns_predictive_mean():
    state = _state()
    x_pending = jnp.asarray([[0.0, 0.0], [1.0, -0.5]])

    imputer = KrigingBeliever()
    y = imputer.impute(x_pending, state)
    expected = jnp.asarray(mean(state.surrogate_distribution.emulator(x_pending)))
    assert jnp.allclose(y, expected, atol=1e-5)
    assert y.shape == (2,) + state.problem.output_shape


def test_constant_liar_min_max_mean_yield_constants():
    state = _state()
    x_pending = jnp.asarray([[0.0, 0.0], [1.0, -0.5], [-1.0, 1.0]])

    for value, expected_constant in [
        ("min", float(jnp.min(state.Y_train))),
        ("max", float(jnp.max(state.Y_train))),
        ("mean", float(jnp.mean(state.Y_train))),
    ]:
        imputer = ConstantLiar(value=value)
        y = imputer.impute(x_pending, state)
        assert y.shape == (3,) + state.problem.output_shape
        assert jnp.all(y == expected_constant)


def test_constant_liar_explicit_float():
    state = _state()
    x_pending = jnp.asarray([[0.0, 0.0]])
    imputer = ConstantLiar(value=-7.5)
    y = imputer.impute(x_pending, state)
    assert float(y[0]) == pytest.approx(-7.5)


def test_constant_liar_invalid_string_raises():
    state = _state()
    x_pending = jnp.asarray([[0.0, 0.0]])
    imputer = ConstantLiar(value="median")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="value="):
        imputer.impute(x_pending, state)
