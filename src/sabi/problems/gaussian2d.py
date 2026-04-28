"""2-D Gaussian posterior — analytic reference.

Target is a 2-D Gaussian with user-specified mean and covariance.
`target_function` routes through the ProbPipe `MultivariateNormal`'s
`log_prob`, so the analytic posterior IS the `reference_distribution`
(the same ProbPipe object), exercising the abstractions end-to-end.

Shapes: `input_shape=(2,)`, `output_shape=()` (scalar log-density).

The `prior` field doubles as the design distribution for initial-design /
random acquisition (`Uniform` over a symmetric box around the mean).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from probpipe import log_prob
from probpipe.core.constraints import interval
from probpipe.distributions.continuous import Uniform
from probpipe.distributions.multivariate import MultivariateNormal

from sabi.problems.base import Problem
from sabi.problems.forms import Identity


def gaussian2d(
    mean: tuple[float, float] = (0.0, 0.0),
    cov: tuple[tuple[float, float], tuple[float, float]] = ((1.0, 0.5), (0.5, 1.0)),
    bounds_radius: float = 5.0,
) -> Problem:
    """Build a 2-D Gaussian benchmark.

    Args:
        mean: posterior mean.
        cov: posterior covariance matrix (2×2, must be PD).
        bounds_radius: symmetric bounds (|x_i - mean_i| ≤ r) for the design
            distribution and the support metadata.
    """
    mu = jnp.asarray(mean, dtype=jnp.float64)
    Sigma = jnp.asarray(cov, dtype=jnp.float64)
    if Sigma.shape != (2, 2):
        raise ValueError(f"cov must be 2x2, got {Sigma.shape}.")
    sign, _ = jnp.linalg.slogdet(Sigma)
    if sign <= 0:
        raise ValueError("cov must be positive-definite.")

    posterior = MultivariateNormal(loc=mu, cov=Sigma, name=f"gaussian2d_{id(mu)}")

    def target(x: Array) -> Array:
        # Extract the scalar from ProbPipe's NumericRecord wrapper.
        return jnp.asarray(log_prob(posterior, x))

    lower = mu - bounds_radius
    upper = mu + bounds_radius
    prior = Uniform(low=lower, high=upper, name=f"gaussian2d_design_{id(mu)}")

    return Problem(
        name="gaussian2d",
        input_shape=(2,),
        output_shape=(),
        target_function=target,  # emulator learns the log-posterior directly
        log_density_form=Identity(),
        prior=prior,
        support=interval(lower, upper),
        reference_distribution=posterior,
    )
