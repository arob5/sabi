"""Tests for `TargetDistribution`.

Coverage:
- Construction (math-only identity: name, input_shape, support, optional
  analytical density).
- ``_unnormalized_log_prob`` accepts both single-point and batched input
  when an analytical callable is provided.
- ``NotImplementedError`` short-circuit when no analytical density was
  supplied (user inverse problems).
- Distribution-protocol satisfaction (``SupportsUnnormalizedLogProb``).
- ``condition_on(target)`` dispatches to NUTS (smoke).
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
from sabi.target_distribution import TargetDistribution


def _gaussian_log_density(x):
    """Quadratic log-density: -0.5 * sum(x**2)."""
    return -0.5 * jnp.sum(x * x)


def _box_support():
    return independent_uniform(
        low=jnp.full((2,), -5.0),
        high=jnp.full((2,), 5.0),
        name="box_support",
    ).support


def _benchmark_target() -> TargetDistribution:
    """Analytical-density target."""
    return TargetDistribution(
        name="quadratic_test",
        input_shape=(2,),
        support=_box_support(),
        unnormalized_log_prob=_gaussian_log_density,
    )


def _opaque_target() -> TargetDistribution:
    """User-inverse-problem target: no analytical density."""
    return TargetDistribution(
        name="opaque_test",
        input_shape=(2,),
        support=_box_support(),
    )


def test_unnormalized_log_prob_single_point():
    td = _benchmark_target()
    x = jnp.asarray([0.5, -0.3])
    expected = float(_gaussian_log_density(x))
    assert float(td._unnormalized_log_prob(x)) == pytest.approx(expected, abs=1e-6)


def test_unnormalized_log_prob_batched():
    td = _benchmark_target()
    X = jnp.asarray([[0.5, -0.3], [1.0, 1.0], [-2.0, 0.0]])
    out = td._unnormalized_log_prob(X)
    expected = jax.vmap(_gaussian_log_density)(X)
    assert out.shape == (3,)
    assert jnp.allclose(out, expected, atol=1e-6)


def test_unnormalized_log_prob_dispatches_by_ndim_not_shape_equality():
    """A `(n=2, d=2)` batch with `input_shape=(2,)` is unambiguously a
    batch (rank 2, not rank 1)."""
    td = _benchmark_target()
    X = jnp.asarray([[1.0, 0.5], [-0.5, 1.0]])
    out = td._unnormalized_log_prob(X)
    expected = jax.vmap(_gaussian_log_density)(X)
    assert out.shape == (2,)
    assert jnp.allclose(out, expected, atol=1e-6)


def test_unnormalized_log_prob_rejects_wrong_input_shape():
    td = _benchmark_target()
    with pytest.raises(ValueError, match="single-point input expects"):
        td._unnormalized_log_prob(jnp.asarray([1.0, 2.0, 3.0]))


def test_unnormalized_log_prob_rejects_wrong_batched_trailing_shape():
    td = _benchmark_target()
    with pytest.raises(ValueError, match="batched input expects shape"):
        td._unnormalized_log_prob(jnp.zeros((4, 3)))


def test_unnormalized_log_prob_rejects_unsupported_ndim():
    td = _benchmark_target()
    with pytest.raises(ValueError, match="expected ndim"):
        td._unnormalized_log_prob(jnp.zeros((2, 3, 2)))


def test_opaque_target_raises_not_implemented():
    """A target without analytical density raises NotImplementedError."""
    td = _opaque_target()
    with pytest.raises(NotImplementedError, match="no analytical"):
        td._unnormalized_log_prob(jnp.asarray([1.0, 2.0]))


def test_satisfies_supports_unnormalized_log_prob():
    td = _benchmark_target()
    assert isinstance(td, SupportsUnnormalizedLogProb)
    assert isinstance(td, Distribution)


def test_unnormalized_log_prob_op_dispatch():
    td = _benchmark_target()
    x = jnp.asarray([0.5, -0.3])
    out = pp_unnormalized_log_prob(td, x)
    expected = float(_gaussian_log_density(x))
    assert float(jnp.asarray(out)) == pytest.approx(expected, abs=1e-6)


def test_event_shape_matches_input_shape():
    td = _benchmark_target()
    assert td.event_shape == (2,)
    assert td.input_shape == (2,)


def test_support_round_trips():
    box = _box_support()
    td = TargetDistribution(
        name="round_trip",
        input_shape=(2,),
        support=box,
    )
    assert td.support is box


def test_constructor_rejects_none_support():
    with pytest.raises(ValueError, match="non-None `support`"):
        TargetDistribution(
            name="bad",
            input_shape=(2,),
            support=None,  # type: ignore[arg-type]
        )


def test_condition_on_dispatches_mcmc_smoke():
    """`condition_on(target)` produces an ApproximateDistribution via NUTS."""
    td = _benchmark_target()
    approx = condition_on(td, num_results=50, num_warmup=50, num_chains=2, random_seed=0)
    chains = jnp.stack([jnp.asarray(c) for c in approx.chains], axis=0)
    assert chains.shape == (2, 50, 2)
