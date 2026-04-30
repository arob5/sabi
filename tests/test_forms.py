"""Tests for `LogDensityForm` subclasses and the shape contract.

Public ``__call__(X, Y, prior)`` is batched: ``X.shape == (n,) + input_shape``,
``Y.shape == (n,) + output_shape``; returns shape ``(n,)``. Subclasses
implement ``_call_single(x, y, prior) -> scalar`` (the per-point hook;
underscore by convention).
"""

import jax.numpy as jnp
import pytest
from probpipe.distributions.continuous import Normal

from sabi._probpipe_compat import independent_uniform
from sabi.problems.forms import ForwardModel, Identity, LogLikPlusPrior


# -------------------------------------------------------------------------
# Identity
# -------------------------------------------------------------------------


def test_identity_per_point_passes_y_through():
    form = Identity()
    x = jnp.asarray([0.3])
    y = jnp.asarray(-1.5)
    assert float(form._call_single(x, y)) == -1.5


def test_identity_batched_returns_y_unchanged():
    """Identity.__call__ is overridden to return Y directly (skipping vmap)."""
    form = Identity()
    X = jnp.asarray([[0.3], [0.7], [-1.0]])
    Y = jnp.asarray([1.0, -2.0, 0.5])
    out = form(X, Y)
    assert jnp.allclose(out, Y)


# -------------------------------------------------------------------------
# LogLikPlusPrior
# -------------------------------------------------------------------------


def test_log_lik_plus_prior_per_point_adds_prior():
    """Standard normal prior with x=2 → log_prior(2) = -0.5*4 - 0.5*log(2π)."""
    prior = Normal(loc=jnp.asarray(0.0), scale=jnp.asarray(1.0), name="p")
    form = LogLikPlusPrior()
    x = jnp.asarray(2.0)  # scalar; matches Normal's event_shape == ()
    y = jnp.asarray(1.5)
    expected = 1.5 + (-0.5 * 4.0 - 0.5 * float(jnp.log(2 * jnp.pi)))
    assert float(form._call_single(x, y, prior=prior)) == pytest.approx(
        expected, abs=1e-5
    )


def test_log_lik_plus_prior_batched_matches_per_point():
    """Public batched __call__ vmaps the per-point hook over X, Y."""
    prior = Normal(loc=jnp.asarray(0.0), scale=jnp.asarray(1.0), name="p")
    form = LogLikPlusPrior()
    X = jnp.asarray([0.0, 1.0, -2.0])
    Y = jnp.asarray([0.5, -1.0, 2.0])
    out = form(X, Y, prior=prior)
    expected = jnp.asarray([
        float(form._call_single(x, y, prior=prior)) for x, y in zip(X, Y)
    ])
    assert jnp.allclose(out, expected, atol=1e-6)


def test_log_lik_plus_prior_errors_without_prior():
    form = LogLikPlusPrior()
    with pytest.raises(ValueError, match="prior"):
        form._call_single(jnp.asarray([0.0]), jnp.asarray(0.0), prior=None)


# -------------------------------------------------------------------------
# ForwardModel
# -------------------------------------------------------------------------


def test_forward_model_per_point_combines_likelihood_and_prior():
    """Multivariate-event uniform prior on [-2, 2]^2 (density 1/16 →
    log = -log(16)). Likelihood is -0.5 * sum((y - 1)^2) with y=[2, 2]
    → -0.5 * 2 = -1.0."""
    prior = independent_uniform(
        low=jnp.full((2,), -2.0), high=jnp.full((2,), 2.0), name="p"
    )
    form = ForwardModel(
        log_lik_from_outputs=lambda x, y: -0.5 * jnp.sum((y - 1.0) ** 2)
    )
    x = jnp.asarray([1.0, 1.0])
    y = jnp.asarray([2.0, 2.0])
    expected = -1.0 + (-jnp.log(16.0))
    assert float(form._call_single(x, y, prior=prior)) == pytest.approx(
        float(expected), abs=1e-5
    )


def test_forward_model_batched_matches_per_point():
    prior = independent_uniform(
        low=jnp.full((2,), -2.0), high=jnp.full((2,), 2.0), name="p"
    )
    form = ForwardModel(
        log_lik_from_outputs=lambda x, y: -0.5 * jnp.sum((y - 1.0) ** 2)
    )
    X = jnp.asarray([[0.5, 0.5], [-1.0, 1.0], [1.5, -0.5]])
    Y = jnp.asarray([[1.5, 1.5], [0.0, 0.5], [2.0, 0.0]])
    out = form(X, Y, prior=prior)
    expected = jnp.asarray([
        float(form._call_single(x, y, prior=prior)) for x, y in zip(X, Y)
    ])
    assert jnp.allclose(out, expected, atol=1e-6)
