"""Tests for the Neal's funnel benchmark.

These tests assume the on-disk reference artifact has been generated and
committed at `reference_posteriors/neals_funnel/`.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from probpipe import unnormalized_log_prob
from probpipe.core._empirical import NumericEmpiricalDistribution
from probpipe.core.constraints import Constraint
from probpipe.core.protocols import SupportsSampling

from sabi.density_decomposition import LogProbTarget
from sabi.problems.neals_funnel import neals_funnel


def test_neals_funnel_shapes_and_types():
    """Construct neals_funnel; reference loads from cache."""
    problem = neals_funnel(d=2)
    target = problem.target_distribution
    assert target.input_shape == (3,)
    assert isinstance(target.support, Constraint)
    assert isinstance(problem.reference_distribution, NumericEmpiricalDistribution)
    assert isinstance(problem.reference_distribution, SupportsSampling)


def test_neals_funnel_target_log_density_known_values():
    """Spot-check the joint log-density at a few exact points."""
    problem = neals_funnel(d=2, sigma_v=3.0)
    out00 = float(jnp.asarray(unnormalized_log_prob(
        problem.target_distribution, jnp.asarray([0.0, 0.0, 0.0])
    )))
    expected = (
        -0.5 * float(jnp.log(2 * jnp.pi * 9.0))
        + 2 * (-0.5 * float(jnp.log(2 * jnp.pi)) - 0.5 * 0.0)
    )
    assert out00 == pytest.approx(expected, abs=1e-6)


def test_neals_funnel_reference_marginal_v():
    """Reference samples' v marginal should be ~ N(0, sigma_v²)."""
    problem = neals_funnel(d=2, sigma_v=3.0)
    samples = jnp.asarray(problem.reference_distribution.samples)
    v = samples[:, 0]
    assert float(jnp.mean(v)) == pytest.approx(0.0, abs=1.5)
    assert float(jnp.var(v)) == pytest.approx(9.0, rel=0.6)


def test_neals_funnel_reference_funnel_geometry():
    """At low v, x marginals tight. At high v, x marginals wide."""
    problem = neals_funnel(d=2, sigma_v=3.0)
    samples = jnp.asarray(problem.reference_distribution.samples)
    v = samples[:, 0]
    x1 = samples[:, 1]

    v_threshold_low = jnp.quantile(v, 0.2)
    low_v_x1_var = float(jnp.var(x1[v < v_threshold_low]))

    v_threshold_high = jnp.quantile(v, 0.8)
    high_v_x1_var = float(jnp.var(x1[v > v_threshold_high]))

    assert high_v_x1_var > 5 * low_v_x1_var


def test_neals_funnel_unnormalized_log_prob_matches_decomposition():
    """The wrapped target's analytical density agrees with the
    ``LogProbTarget`` decomposition's at the same x — both go through
    the ProbPipe op and yield the same value."""
    target = neals_funnel(d=2).target_distribution
    decomposition = LogProbTarget(target)
    x = jnp.asarray([0.5, 1.0, -1.0])
    target_val = float(jnp.asarray(unnormalized_log_prob(target, x)))
    decomp_val = float(jnp.asarray(unnormalized_log_prob(decomposition, x)))
    assert target_val == pytest.approx(decomp_val, abs=1e-6)
