"""Tests for FantasyImputer strategies."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from probpipe import mean

from sabi.acquisitions.base import AcquisitionState
from sabi.acquisitions.fantasize import (
    ConstantLiar,
    KrigingBeliever,
)
from sabi.posterior.surrogate_posterior import SurrogatePosterior
from sabi.problems.gaussian2d import gaussian2d
from sabi.surrogates.gp import GPSurrogate


def _state(n: int = 20, seed: int = 0):
    problem = gaussian2d()
    key = jax.random.key(seed)
    lower, upper = problem.support.low, problem.support.high
    X = lower + (upper - lower) * jax.random.uniform(
        key, shape=(n,) + problem.input_shape
    )
    Y = problem.target_function(X)
    surrogate = GPSurrogate(input_shape=problem.input_shape).fit(X, Y)
    sp = SurrogatePosterior(
        surrogate=surrogate,
        log_density_form=problem.log_density_form,
        support=problem.support,
        input_shape=problem.input_shape,
        prior=problem.prior,
    )
    return AcquisitionState(
        problem=problem,
        surrogate_posterior=sp,
        X=X,
        Y=Y,
        tempering_state=None,
    )


def test_kriging_believer_returns_predictive_mean():
    state = _state()
    x_pending = jnp.asarray([[0.0, 0.0], [1.0, -0.5]])

    imputer = KrigingBeliever()
    y = imputer.impute(x_pending, state)
    expected = jnp.asarray(mean(state.surrogate_posterior.surrogate(x_pending)))
    assert jnp.allclose(y, expected, atol=1e-5)
    assert y.shape == (2,) + state.problem.output_shape


def test_constant_liar_min_max_mean_yield_constants():
    state = _state()
    x_pending = jnp.asarray([[0.0, 0.0], [1.0, -0.5], [-1.0, 1.0]])

    for value, expected_constant in [
        ("min", float(jnp.min(state.Y))),
        ("max", float(jnp.max(state.Y))),
        ("mean", float(jnp.mean(state.Y))),
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
