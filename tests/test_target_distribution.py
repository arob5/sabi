"""Tests for ``TargetDistribution`` (abstract math identity).

Coverage:
- Subclasses with analytical density satisfy ``SupportsUnnormalizedLogProb``.
- Subclasses without (no ``_unnormalized_log_prob`` override) do not.
- ``unnormalized_log_prob(target, X)`` op handles batching.
- ``condition_on(target)`` dispatches to NUTS smoke.
- Constructor rejects None support.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from probpipe import condition_on
from probpipe import unnormalized_log_prob as pp_unnormalized_log_prob
from probpipe.core._distribution_base import Distribution
from probpipe.core.protocols import SupportsUnnormalizedLogProb

from sabi.target_distribution import TargetDistribution

from tests._targets import (
    OpaqueTarget,
    QuadraticTarget,
    box_support,
    quadratic_log_density,
)


def test_subclass_with_density_satisfies_protocol():
    target = QuadraticTarget()
    assert isinstance(target, SupportsUnnormalizedLogProb)
    assert isinstance(target, Distribution)


def test_subclass_without_density_does_not_satisfy_protocol():
    """Targets that don't override ``_unnormalized_log_prob`` are not
    ``SupportsUnnormalizedLogProb`` — exactly what we want for user
    inverse problems."""
    target = OpaqueTarget()
    assert not isinstance(target, SupportsUnnormalizedLogProb)


def test_unnormalized_log_prob_single_point_via_op():
    target = QuadraticTarget()
    x = jnp.asarray([0.5, -0.3])
    expected = float(quadratic_log_density(x))
    out = float(jnp.asarray(pp_unnormalized_log_prob(target, x)))
    assert out == pytest.approx(expected, abs=1e-6)


def test_unnormalized_log_prob_batched_via_op():
    target = QuadraticTarget()
    X = jnp.asarray([[0.5, -0.3], [1.0, 1.0], [-2.0, 0.0]])
    out = jnp.asarray(pp_unnormalized_log_prob(target, X))
    expected = jnp.asarray([float(quadratic_log_density(x)) for x in X])
    assert out.shape == (3,)
    assert jnp.allclose(out, expected, atol=1e-6)


def test_event_shape_matches_input_shape():
    target = QuadraticTarget()
    assert target.event_shape == (2,)
    assert target.input_shape == (2,)


def test_support_round_trips():
    box = box_support()
    target = QuadraticTarget()
    # box_support and target.support are constructed from the same factory;
    # the equality test would hold for the same instance only.
    assert isinstance(target.support, type(box))


def test_constructor_rejects_none_support():
    with pytest.raises(ValueError, match="non-None `support`"):

        class _Bad(TargetDistribution):
            def _unnormalized_log_prob(self, x):
                return jnp.asarray(0.0)

        _Bad(name="bad", input_shape=(2,), support=None)


def test_condition_on_dispatches_mcmc_smoke():
    """``condition_on(target)`` produces an ApproximateDistribution via NUTS."""
    target = QuadraticTarget()
    approx = condition_on(target, num_results=50, num_warmup=50, num_chains=2, random_seed=0)
    chains = jnp.stack([jnp.asarray(c) for c in approx.chains], axis=0)
    assert chains.shape == (2, 50, 2)
