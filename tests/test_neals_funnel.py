"""Tests for the Neal's funnel benchmark.

These tests assume the on-disk reference artifact has been generated and
committed at `reference_posteriors/neals_funnel/`. Run
`./scripts/regenerate_references --problem neals_funnel` to refresh.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from probpipe.core._distribution_base import Distribution
from probpipe.core._empirical import NumericEmpiricalDistribution
from probpipe.core.constraints import Constraint
from probpipe.core.protocols import SupportsSampling

from sabi.problems.neals_funnel import neals_funnel


def test_neals_funnel_shapes_and_types():
    """Construct neals_funnel; reference loads from cache. Verify the
    core problem-level invariants."""
    problem = neals_funnel(d=2)
    assert problem.input_shape == (3,)
    assert problem.output_shape == ()
    assert isinstance(problem.prior, Distribution)
    assert isinstance(problem.support, Constraint)
    assert isinstance(problem.reference_distribution, NumericEmpiricalDistribution)
    assert isinstance(problem.reference_distribution, SupportsSampling)


def test_neals_funnel_target_log_density_known_values():
    """Spot-check the joint log-density at a few exact points."""
    problem = neals_funnel(d=2, sigma_v=3.0)
    log_p = problem.target_function

    # At (v=0, x=0): log_p_v = -0.5*log(2π·9), log_p_x|v = 2 * (-0.5*log(2π·1)) = -log(2π)
    # Total = -0.5*log(2π·9) - log(2π) = -0.5 log(2π·9) - log(2π)
    out00 = float(log_p(jnp.asarray([0.0, 0.0, 0.0])))
    expected = (
        -0.5 * float(jnp.log(2 * jnp.pi * 9.0))
        + 2 * (-0.5 * float(jnp.log(2 * jnp.pi)) - 0.5 * 0.0)  # v=0 → log p(x|v)
    )
    assert out00 == pytest.approx(expected, abs=1e-6)


def test_neals_funnel_reference_marginal_v():
    """Reference samples' v marginal should be ~ N(0, sigma_v²).

    The funnel's vanilla-NUTS ESS is small (~50 effective samples), so
    we use loose tolerances reflecting the actual mixing — the standard
    error of the v marginal mean is roughly σ_v / sqrt(ESS) ~ 0.4 with
    these settings, so a 1σ-class tolerance lands at ~0.5–1.0.
    """
    problem = neals_funnel(d=2, sigma_v=3.0)
    samples = jnp.asarray(problem.reference_distribution.samples)
    v = samples[:, 0]
    assert float(jnp.mean(v)) == pytest.approx(0.0, abs=1.5)
    assert float(jnp.var(v)) == pytest.approx(9.0, rel=0.6)


def test_neals_funnel_reference_funnel_geometry():
    """At low v, x marginals should be tight (var ≈ exp(v) ≈ small).
    At high v, x marginals should be wide. This is the funnel signature."""
    problem = neals_funnel(d=2, sigma_v=3.0)
    samples = jnp.asarray(problem.reference_distribution.samples)
    v = samples[:, 0]
    x1 = samples[:, 1]

    # Bottom 20% of v.
    v_threshold_low = jnp.quantile(v, 0.2)
    low_v_x1_var = float(jnp.var(x1[v < v_threshold_low]))

    # Top 20% of v.
    v_threshold_high = jnp.quantile(v, 0.8)
    high_v_x1_var = float(jnp.var(x1[v > v_threshold_high]))

    # var(x | v=high) >> var(x | v=low)
    assert high_v_x1_var > 5 * low_v_x1_var


def test_neals_funnel_log_posterior_method():
    """`Problem.log_posterior` (which composes target_function + Identity form)
    matches `target_function` directly for this benchmark."""
    problem = neals_funnel(d=2)
    x = jnp.asarray([0.5, 1.0, -1.0])
    assert float(problem.log_posterior(x)) == pytest.approx(
        float(problem.target_function(x)), abs=1e-6
    )
