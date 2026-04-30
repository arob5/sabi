"""Tests for the BatchSampler abstraction.

`PriorSampler` is the only concrete sampler in v1.4.1; coverage focuses on:
- Output shape matches `(n,) + problem.input_shape`.
- Samples lie in the problem's support.
- Independent keys produce different draws; same key reproduces the draw.
- Missing prior raises `ValueError` with a clear pointer.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from probpipe.core.constraints import interval

from sabi.problems.base import Problem
from sabi.problems.forms import Identity
from sabi.problems.gaussian2d import gaussian2d
from sabi.sampling import PriorSampler


def test_prior_sampler_shape_and_support():
    problem = gaussian2d()
    sampler = PriorSampler()
    X = sampler.sample(problem, jax.random.key(0), n=32)
    assert X.shape == (32,) + problem.input_shape
    assert jnp.all(jnp.asarray(problem.support.check(X)))


def test_prior_sampler_seed_determinism_and_independence():
    problem = gaussian2d()
    sampler = PriorSampler()
    X_a = sampler.sample(problem, jax.random.key(7), n=8)
    X_b = sampler.sample(problem, jax.random.key(7), n=8)
    X_c = sampler.sample(problem, jax.random.key(8), n=8)
    # Same key → same samples.
    assert jnp.allclose(X_a, X_b)
    # Different key → different samples.
    assert not jnp.allclose(X_a, X_c)


def test_prior_sampler_missing_prior_raises():
    """A Problem with `prior=None` must surface a clear error."""
    problem = Problem.from_target_single(
        name="no_prior",
        input_shape=(2,),
        output_shape=(),
        target_single=lambda x: jnp.sum(x),
        prior=None,
        support=interval(low=jnp.zeros(2), high=jnp.ones(2)),
        log_density_form=Identity(),
    )
    with pytest.raises(ValueError, match="prior"):
        PriorSampler().sample(problem, jax.random.key(0), n=4)
