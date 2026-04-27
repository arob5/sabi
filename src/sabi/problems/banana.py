"""Rosenbrock "banana" posterior — analytic marginals via change-of-variables.

Standard 2-D benchmark:
    log p(x) ∝ -0.5 * (x₁² / a² + b * (x₂ + x₁² - a²)²)

With z = x₂ + x₁² - a², the posterior factors as
(x₁, z) ~ N(0, a²) × N(0, 1/b), so exact reference samples come for free.
We pre-compute a sample bank and wrap it in a ProbPipe
`NumericEmpiricalDistribution` as the reference.

`target_function` keeps the inline analytic log-density — no clean ProbPipe
distribution describes a banana directly. Expressing the banana posterior
as a `TransformedDistribution(Normal(...), Bijector)` is a v1.5+ exercise.

Shapes: `input_shape=(2,)`, `output_shape=()` (scalar log-density).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from probpipe.core._empirical import NumericEmpiricalDistribution
from probpipe.core.constraints import interval
from probpipe.distributions.continuous import Uniform

from sabi.problems.base import Problem
from sabi.problems.forms import Identity


def banana(
    a: float = 1.0,
    b: float = 4.0,
    bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (-4.0, 4.0),
        (-10.0, 4.0),
    ),
    n_reference_samples: int = 4096,
) -> Problem:
    """Build the banana-shaped posterior benchmark.

    Args:
        a: controls x₁ marginal scale.
        b: controls curvature (larger = tighter banana).
        bounds: per-dimension `(lower, upper)` box used as the design distribution
            (`Uniform`) and the support metadata.
        n_reference_samples: size of the reference empirical distribution.
    """
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive.")

    log_norm = -jnp.log(2 * jnp.pi) - jnp.log(a) + 0.5 * jnp.log(b)

    def log_prob(x: Array) -> Array:
        x1, x2 = x[0], x[1]
        z = x2 + x1 * x1 - a * a
        return log_norm - 0.5 * (x1 * x1 / (a * a) + b * z * z)

    def sample_reference(key: Array, n: int) -> Array:
        key1, key2 = jax.random.split(key)
        x1 = a * jax.random.normal(key1, shape=(n,))
        z = jax.random.normal(key2, shape=(n,)) / jnp.sqrt(b)
        x2 = z - x1 * x1 + a * a
        return jnp.stack([x1, x2], axis=-1)

    key_ref = jax.random.key(0)
    ref_samples = sample_reference(key_ref, n_reference_samples)
    reference = NumericEmpiricalDistribution(samples=ref_samples, name="banana_ref")

    lower = jnp.asarray([bounds[0][0], bounds[1][0]], dtype=jnp.float64)
    upper = jnp.asarray([bounds[0][1], bounds[1][1]], dtype=jnp.float64)
    prior = Uniform(low=lower, high=upper, name=f"banana_design_a{a}_b{b}")

    return Problem(
        name="banana",
        input_shape=(2,),
        output_shape=(),
        target_function=log_prob,
        log_density_form=Identity(),
        prior=prior,
        support=interval(lower, upper),
        reference_distribution=reference,
    )
