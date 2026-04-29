r"""2-D Gaussian posterior — analytic reference.

Target is a 2-D Gaussian :math:`\mathcal{N}(\mu, \Sigma)` with user-specified
mean and covariance:

.. math::

    \log p(x) = -\tfrac{1}{2} (x - \mu)^\top \Sigma^{-1} (x - \mu)
                - \tfrac{1}{2} \log\!\big((2\pi)^2 |\Sigma|\big),
    \quad x \in \mathbb{R}^2.

``target_function`` routes through the ProbPipe ``MultivariateNormal``'s
``log_prob``, so the analytic posterior IS the ``reference_distribution``
(the same ProbPipe object), exercising the abstractions end-to-end.

Shapes: ``input_shape=(2,)``, ``output_shape=()`` (scalar log-density). See
``docs/notation.md`` for sabi's shape conventions.

The ``prior`` field doubles as the design distribution for initial-design /
random acquisition: ``Uniform`` over a symmetric box
:math:`[\mu - r, \mu + r]^2` for ``bounds_radius`` :math:`r`.
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

    def target_single(x: Array) -> Array:
        # Extract the scalar from ProbPipe's NumericRecord wrapper.
        return jnp.asarray(log_prob(posterior, x))

    lower = mu - bounds_radius
    upper = mu + bounds_radius
    prior = Uniform(low=lower, high=upper, name=f"gaussian2d_design_{id(mu)}")

    return Problem.from_target_single(
        target_single=target_single,
        name="gaussian2d",
        input_shape=(2,),
        output_shape=(),
        log_density_form=Identity(),
        prior=prior,
        support=interval(lower, upper),
        reference_distribution=posterior,
    )
