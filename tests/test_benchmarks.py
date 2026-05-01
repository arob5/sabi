"""Tests for the validated benchmark suite (`sabi.problems.benchmarks`)."""

from __future__ import annotations

import jax.numpy as jnp
from probpipe.core._empirical import NumericEmpiricalDistribution

from sabi.problems.base import BenchmarkProblem, Problem
from sabi.problems.benchmarks import banana_2d, banana_10d


def test_banana_2d_is_a_benchmark_problem():
    bp = banana_2d()
    assert isinstance(bp, BenchmarkProblem)
    assert isinstance(bp, Problem)
    assert bp.name == "banana_2d"
    assert bp.artifact_version == "v1"
    assert bp.input_shape == (2,)
    assert bp.reference_distribution is not None
    assert isinstance(bp.reference_distribution, NumericEmpiricalDistribution)


def test_banana_10d_is_a_benchmark_problem():
    bp = banana_10d()
    assert isinstance(bp, BenchmarkProblem)
    assert bp.name == "banana_10d"
    assert bp.input_shape == (10,)
    assert isinstance(bp.reference_distribution, NumericEmpiricalDistribution)


def test_banana_benchmarks_have_distinct_names():
    """Each named benchmark is its own identity (posteriordb invariant)."""
    assert banana_2d().name != banana_10d().name


def test_banana_2d_log_posterior_finite_at_typical_points():
    """A few sanity-check log-densities should be finite."""
    bp = banana_2d()
    for x in [
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([1.0, -0.5]),
        jnp.asarray([-2.0, -3.0]),
    ]:
        lp = float(bp.log_posterior(x))
        assert jnp.isfinite(lp)


def test_banana_10d_reference_first_two_dims_match_2d():
    """The (x_1, x_2) marginal of banana_10d is the same banana as in
    banana_2d (the filler dims are independent N(0, 1))."""
    bp_2d = banana_2d()
    bp_10d = banana_10d()
    s_2d = jnp.asarray(bp_2d.reference_distribution.samples)
    s_10d = jnp.asarray(bp_10d.reference_distribution.samples)[:, :2]
    # Same underlying analytic distribution → moments match within MC error.
    assert jnp.allclose(jnp.var(s_2d, axis=0), jnp.var(s_10d, axis=0), atol=0.15)
    assert jnp.allclose(jnp.mean(s_2d, axis=0), jnp.mean(s_10d, axis=0), atol=0.15)


def test_benchmark_factories_are_pure():
    """Calling the factory twice yields equivalent benchmarks (modulo
    object identity) — no hidden state across calls."""
    a, b = banana_2d(), banana_2d()
    x = jnp.asarray([0.5, -0.5])
    assert float(a.log_posterior(x)) == float(b.log_posterior(x))
    assert a.input_shape == b.input_shape
    assert a.name == b.name
