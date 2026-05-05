"""Tests for `TargetDistribution`.

Coverage:
- Construction (single-point callable; batched view derived via `jax.vmap`).
- `_unnormalized_log_prob` accepts both single-point and batched input.
- Distribution-protocol satisfaction (`SupportsUnnormalizedLogProb`).
- `condition_on(target)` dispatches to NUTS (smoke).
- Convenience accessors mirror the constructor args.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from probpipe import condition_on
from probpipe import unnormalized_log_prob as pp_unnormalized_log_prob
from probpipe.core._distribution_base import Distribution
from probpipe.core.protocols import SupportsUnnormalizedLogProb

from sabi._probpipe_compat import independent_uniform
from sabi.problems.forms import Identity, LogLikPlusPrior
from sabi.target_distribution import TargetDistribution


def _gaussian_target_single(x):
    """Quadratic log-density: -0.5 * sum(x**2). Used as a smooth target."""
    return -0.5 * jnp.sum(x * x)


def _box_prior():
    return independent_uniform(
        low=jnp.full((2,), -5.0), high=jnp.full((2,), 5.0), name="box_prior"
    )


def _gaussian_target() -> TargetDistribution:
    return TargetDistribution(
        target_single=_gaussian_target_single,
        name="quadratic_test",
        input_shape=(2,),
        output_shape=(),
        log_density_form=Identity(),
        prior=_box_prior(),
    )


def test_target_map_derived_via_vmap():
    td = _gaussian_target()
    X = jnp.asarray([[0.0, 0.0], [1.0, -1.0], [2.0, 0.5]])
    Y_batched = td.target_map(X)
    Y_single = jnp.asarray([td.target_single(x) for x in X])
    assert jnp.allclose(Y_batched, Y_single)


def test_unnormalized_log_prob_single_point():
    td = _gaussian_target()
    x = jnp.asarray([0.5, -0.3])
    expected = float(_gaussian_target_single(x))
    assert float(td._unnormalized_log_prob(x)) == pytest.approx(expected, abs=1e-6)


def test_unnormalized_log_prob_batched():
    td = _gaussian_target()
    X = jnp.asarray([[0.5, -0.3], [1.0, 1.0], [-2.0, 0.0]])
    out = td._unnormalized_log_prob(X)
    expected = jax.vmap(_gaussian_target_single)(X)
    assert out.shape == (3,)
    assert jnp.allclose(out, expected, atol=1e-6)


def test_unnormalized_log_prob_dispatches_by_ndim_not_shape_equality():
    """A `(n=2, d=2)` batch with `input_shape=(2,)` is unambiguously a
    batch (rank 2, not rank 1), even though `(n, d) == input_shape` is
    True for the leading-axis match. Previous shape-equality detection
    misidentified this as a single point. Now dispatch is by `ndim`."""
    td = _gaussian_target()
    # Two 2-D points, in batched form. Both `value.shape` (= (2, 2))
    # and `input_shape` (= (2,)) start with `2`, so a shape-equality
    # check `value.shape == input_shape` would be False here, but a
    # single-rank check `value.ndim == 1` is also False. Either way the
    # batched path should fire.
    X = jnp.asarray([[1.0, 0.5], [-0.5, 1.0]])
    out = td._unnormalized_log_prob(X)
    expected = jax.vmap(_gaussian_target_single)(X)
    assert out.shape == (2,)
    assert jnp.allclose(out, expected, atol=1e-6)


def test_unnormalized_log_prob_rejects_wrong_input_shape():
    """A rank-1 array of the wrong size raises (single-point branch
    expects `input_shape`)."""
    td = _gaussian_target()
    with pytest.raises(ValueError, match="single-point input expects"):
        td._unnormalized_log_prob(jnp.asarray([1.0, 2.0, 3.0]))


def test_unnormalized_log_prob_rejects_wrong_batched_trailing_shape():
    """A rank-2 array whose trailing dims don't match `input_shape`
    raises (batched branch validates the suffix)."""
    td = _gaussian_target()
    with pytest.raises(ValueError, match="batched input expects shape"):
        td._unnormalized_log_prob(jnp.zeros((4, 3)))


def test_unnormalized_log_prob_rejects_unsupported_ndim():
    """Inputs neither single-point nor batched (e.g. rank-3) raise."""
    td = _gaussian_target()
    with pytest.raises(ValueError, match="expected ndim"):
        td._unnormalized_log_prob(jnp.zeros((2, 3, 2)))


def test_satisfies_supports_unnormalized_log_prob():
    td = _gaussian_target()
    assert isinstance(td, SupportsUnnormalizedLogProb)
    assert isinstance(td, Distribution)


def test_unnormalized_log_prob_op_dispatch():
    """ProbPipe's `unnormalized_log_prob` op dispatches to the
    distribution's `_unnormalized_log_prob`."""
    td = _gaussian_target()
    x = jnp.asarray([0.5, -0.3])
    out = pp_unnormalized_log_prob(td, x)
    expected = float(_gaussian_target_single(x))
    assert float(jnp.asarray(out)) == pytest.approx(expected, abs=1e-6)


def test_event_shape_matches_input_shape():
    td = _gaussian_target()
    assert td.event_shape == (2,)
    assert td.input_shape == (2,)


def test_log_lik_plus_prior_form_uses_prior():
    """With LogLikPlusPrior + multivariate-event Uniform prior on a
    box, log p = -0.5*x^2 + log_prior(x)."""
    prior = _box_prior()
    td = TargetDistribution(
        target_single=_gaussian_target_single,
        name="quadratic_with_prior",
        input_shape=(2,),
        output_shape=(),
        log_density_form=LogLikPlusPrior(),
        prior=prior,
    )
    x = jnp.asarray([0.5, -0.3])
    out = float(td._unnormalized_log_prob(x))
    # _unnormalized_log_prob = log_lik(x) + log_prior(x)
    # log_lik = _gaussian_target_single(x); log_prior = log(1/100) for the box.
    log_lik = float(_gaussian_target_single(x))
    log_prior = -2.0 * jnp.log(10.0)  # log(1/10) per dim, summed across two dims
    assert out == pytest.approx(log_lik + float(log_prior), abs=1e-5)


def test_condition_on_dispatches_mcmc_smoke():
    """`condition_on(target)` should produce an ApproximateDistribution
    via auto-dispatched NUTS (since the target satisfies
    SupportsUnnormalizedLogProb)."""
    td = _gaussian_target()
    approx = condition_on(td, num_results=50, num_warmup=50, num_chains=2, random_seed=0)
    # Just verify we got something back and the chains have the right shape.
    chains = jnp.stack([jnp.asarray(c) for c in approx.chains], axis=0)
    assert chains.shape == (2, 50, 2)
