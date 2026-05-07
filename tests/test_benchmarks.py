"""Tests for the validated benchmark suite (`sabi.problems.benchmarks`)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from probpipe import sample
from probpipe.core._empirical import NumericEmpiricalDistribution
from probpipe.distributions.multivariate import MultivariateNormal

from sabi.problems.base import BenchmarkProblem, Problem
from sabi.problems.benchmarks import (
    banana_2d,
    banana_10d,
    gaussian_2d,
    gaussian_10d,
    neals_funnel_3d,
)


def test_banana_2d_is_a_benchmark_problem():
    bp = banana_2d()
    assert isinstance(bp, BenchmarkProblem)
    assert isinstance(bp, Problem)
    assert bp.name == "banana_2d"
    assert bp.artifact_version == "v1"
    assert bp.target_distribution.input_shape == (2,)
    assert bp.reference_distribution is not None
    assert isinstance(bp.reference_distribution, NumericEmpiricalDistribution)


def test_banana_10d_is_a_benchmark_problem():
    bp = banana_10d()
    assert isinstance(bp, BenchmarkProblem)
    assert bp.name == "banana_10d"
    assert bp.target_distribution.input_shape == (10,)
    assert isinstance(bp.reference_distribution, NumericEmpiricalDistribution)


def test_banana_benchmarks_have_distinct_names():
    """Each named benchmark is its own identity (posteriordb invariant)."""
    assert banana_2d().name != banana_10d().name


def test_banana_2d_unnormalized_log_prob_finite_at_typical_points():
    """A few sanity-check log-densities should be finite."""
    target = banana_2d().target_distribution
    for x in [
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([1.0, -0.5]),
        jnp.asarray([-2.0, -3.0]),
    ]:
        lp = float(target._unnormalized_log_prob(x))
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
    a_logp = a.target_distribution._unnormalized_log_prob(x)
    b_logp = b.target_distribution._unnormalized_log_prob(x)
    assert float(a_logp) == float(b_logp)
    assert a.target_distribution.input_shape == b.target_distribution.input_shape
    assert a.name == b.name


# ---------------------------------------------------------------------------
# Gaussian benchmarks
# ---------------------------------------------------------------------------


def test_gaussian_2d_is_a_benchmark_problem():
    bp = gaussian_2d()
    assert isinstance(bp, BenchmarkProblem)
    assert isinstance(bp, Problem)
    assert bp.name == "gaussian_2d"
    assert bp.artifact_version == "v1"
    assert bp.target_distribution.input_shape == (2,)
    assert isinstance(bp.reference_distribution, MultivariateNormal)


def test_gaussian_2d_preserves_historical_defaults():
    """gaussian_2d() must reproduce the pre-refactor gaussian2d defaults
    (mean=0, cov=[[1, 0.5], [0.5, 1]]). Sampling from the analytic
    reference reproduces those moments."""
    bp = gaussian_2d()
    samples = jnp.asarray(
        sample(bp.reference_distribution, key=jax.random.key(0), sample_shape=(4096,))
    )
    emp_mean = jnp.mean(samples, axis=0)
    emp_cov = jnp.cov(samples, rowvar=False)
    assert jnp.allclose(emp_mean, jnp.zeros(2), atol=0.05)
    assert jnp.allclose(emp_cov, jnp.asarray([[1.0, 0.5], [0.5, 1.0]]), atol=0.1)


def test_gaussian_10d_is_a_benchmark_problem():
    bp = gaussian_10d()
    assert isinstance(bp, BenchmarkProblem)
    assert bp.name == "gaussian_10d"
    assert bp.target_distribution.input_shape == (10,)
    assert isinstance(bp.reference_distribution, MultivariateNormal)


def test_gaussian_10d_is_isotropic():
    """gaussian_10d() defaults to mean=0, cov=I_10 — sample variances
    cluster around 1 in every dim."""
    bp = gaussian_10d()
    samples = jnp.asarray(
        sample(bp.reference_distribution, key=jax.random.key(0), sample_shape=(4096,))
    )
    emp_var = jnp.var(samples, axis=0)
    assert jnp.allclose(emp_var, jnp.ones(10), atol=0.1)
    assert jnp.allclose(jnp.mean(samples, axis=0), jnp.zeros(10), atol=0.1)


# ---------------------------------------------------------------------------
# Neal's funnel benchmark
# ---------------------------------------------------------------------------


def test_neals_funnel_3d_is_a_benchmark_problem():
    bp = neals_funnel_3d()
    assert isinstance(bp, BenchmarkProblem)
    assert bp.name == "neals_funnel_3d"
    # 1 v dim + 2 x dims = 3 total.
    assert bp.target_distribution.input_shape == (3,)
    assert isinstance(bp.reference_distribution, NumericEmpiricalDistribution)


def test_neals_funnel_3d_reference_loads_from_cache():
    """The committed parquet artifact under reference_posteriors/
    neals_funnel/ should hydrate without triggering NUTS regeneration."""
    bp = neals_funnel_3d()
    # 4 chains × 2000 results = 8000.
    samples = jnp.asarray(bp.reference_distribution.samples)
    assert samples.shape == (8000, 3)


# ---------------------------------------------------------------------------
# All benchmarks share the same identity invariants
# ---------------------------------------------------------------------------


def test_all_benchmark_names_distinct():
    names = {
        banana_2d().name,
        banana_10d().name,
        gaussian_2d().name,
        gaussian_10d().name,
        neals_funnel_3d().name,
    }
    assert len(names) == 5


def test_benchmark_problem_rejects_empty_name():
    """A `BenchmarkProblem` constructed with `name=""` must raise — the
    name is the benchmark's identity (posteriordb invariant), and a
    nameless validated benchmark cannot be addressed in the suite.

    Built from `gaussian_2d()` so we have a concretely valid
    target/reference distribution pair to work with.
    """
    bp = gaussian_2d()
    with pytest.raises(ValueError, match="non-empty name"):
        BenchmarkProblem(
            target_distribution=bp.target_distribution,
            reference_distribution=bp.reference_distribution,
            name="",
        )
