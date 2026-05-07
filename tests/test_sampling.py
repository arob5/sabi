"""Tests for the BatchSampler abstraction.

`PriorSampler` is the only concrete sampler in v1.4.1; coverage focuses on:
- Output shape matches `(n,) + problem.target_distribution.input_shape`.
- Samples lie in the problem's support.
- Independent keys produce different draws; same key reproduces the draw.
- Missing-prior construction raises (since `prior` is required on
  `TargetDistribution`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from sabi.problems.forms import Identity
from sabi.problems.benchmarks import gaussian_2d
from sabi.target_distribution import TargetDistribution
from sabi.sampling import PriorSampler


def test_prior_sampler_shape_and_support():
    problem = gaussian_2d()
    target = problem.target_distribution
    sampler = PriorSampler()
    X = sampler.sample(problem, jax.random.key(0), n=32)
    assert X.shape == (32,) + target.input_shape
    assert jnp.all(jnp.asarray(target.support.check(X)))


def test_prior_sampler_seed_determinism_and_independence():
    problem = gaussian_2d()
    sampler = PriorSampler()
    X_a = sampler.sample(problem, jax.random.key(7), n=8)
    X_b = sampler.sample(problem, jax.random.key(7), n=8)
    X_c = sampler.sample(problem, jax.random.key(8), n=8)
    # Same key → same samples.
    assert jnp.allclose(X_a, X_b)
    # Different key → different samples.
    assert not jnp.allclose(X_a, X_c)


def test_target_distribution_requires_prior():
    """`TargetDistribution(prior=None)` raises with a clear pointer.

    `prior` is required for two reasons: (1) it defines the support of
    the parameter space, and (2) it acts as the design distribution
    for sampling. See the `TargetDistribution` module docstring for
    the full role description.
    """
    with pytest.raises(ValueError, match="prior"):
        TargetDistribution(
            name="no_prior_target",
            input_shape=(2,),
            output_shape=(),
            target_single=lambda x: jnp.sum(x),
            log_density_form=Identity(),
            prior=None,
        )
