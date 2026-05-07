r"""`d`-dimensional Gaussian posterior — analytic reference.

Target is :math:`\mathcal{N}(\mu, \Sigma)` on :math:`\mathbb{R}^d`:

.. math::

    \log p(x) = -\tfrac{1}{2} (x - \mu)^\top \Sigma^{-1} (x - \mu)
                - \tfrac{1}{2} \log\!\big((2\pi)^d |\Sigma|\big).

Shapes: ``input_shape=(d,)``. The factory returns a `Problem` with an
analytical ``_unnormalized_log_prob`` on the ``target_distribution``;
the algorithm-side decomposition is constructed by the user.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array
from probpipe import log_prob
from probpipe.distributions.multivariate import MultivariateNormal

from sabi._probpipe_compat import independent_uniform
from sabi.problems.base import Problem
from sabi.target_distribution import TargetDistribution


def gaussian(
    d: int = 2,
    mean: Sequence[float] | None = None,
    cov: Sequence[Sequence[float]] | None = None,
    bounds_radius: float = 5.0,
) -> Problem:
    """Build a `d`-dimensional Gaussian benchmark.

    Args:
        d: parameter dimension. Must be ≥ 1.
        mean: posterior mean; defaults to :math:`0 \\in \\mathbb{R}^d`.
        cov: posterior covariance (`d × d`, must be PD); defaults to
            :math:`I_d`.
        bounds_radius: symmetric per-dim half-width
            (:math:`|x_i - \\mu_i| \\le r`) for the target support.

    Returns:
        `Problem` with ``input_shape=(d,)``, an analytical
        ``_unnormalized_log_prob``, and the analytic
        `MultivariateNormal` itself as the reference distribution.
    """
    if d < 1:
        raise ValueError(f"d must be ≥ 1, got {d}.")
    if bounds_radius <= 0:
        raise ValueError(f"bounds_radius must be positive, got {bounds_radius}.")

    mu = jnp.asarray(mean if mean is not None else [0.0] * d, dtype=jnp.float64)
    if mu.shape != (d,):
        raise ValueError(f"mean must have shape ({d},), got {mu.shape}.")

    Sigma = (
        jnp.asarray(cov, dtype=jnp.float64)
        if cov is not None
        else jnp.eye(d, dtype=jnp.float64)
    )
    if Sigma.shape != (d, d):
        raise ValueError(f"cov must be {d}x{d}, got {Sigma.shape}.")
    sign, _ = jnp.linalg.slogdet(Sigma)
    if sign <= 0:
        raise ValueError("cov must be positive-definite.")

    posterior = MultivariateNormal(loc=mu, cov=Sigma, name=f"gaussian_d{d}_{id(mu)}")

    def target_single(x: Array) -> Array:
        return jnp.asarray(log_prob(posterior, x))

    lower = mu - bounds_radius
    upper = mu + bounds_radius
    support = independent_uniform(
        low=lower, high=upper, name=f"gaussian_d{d}_support_{id(mu)}"
    ).support

    target = TargetDistribution(
        name=f"gaussian_d{d}_target_{id(mu)}",
        input_shape=(d,),
        support=support,
        unnormalized_log_prob=target_single,
    )
    return Problem(
        target_distribution=target,
        reference_distribution=posterior,
        name="gaussian",
    )
