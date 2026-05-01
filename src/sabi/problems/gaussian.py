r"""`d`-dimensional Gaussian posterior — analytic reference.

Target is :math:`\mathcal{N}(\mu, \Sigma)` on :math:`\mathbb{R}^d`:

.. math::

    \log p(x) = -\tfrac{1}{2} (x - \mu)^\top \Sigma^{-1} (x - \mu)
                - \tfrac{1}{2} \log\!\big((2\pi)^d |\Sigma|\big).

``target_function`` routes through the ProbPipe ``MultivariateNormal``'s
``log_prob``, so the analytic posterior IS the ``reference_distribution``
(the same ProbPipe object) — exercising the abstractions end-to-end.

Shapes: ``input_shape=(d,)``, ``output_shape=()``. See
``docs/notation.md`` for sabi's shape conventions.

The ``prior`` field doubles as the design distribution for initial-design
/ random acquisition: ``Uniform`` over a symmetric box
:math:`[\mu - r, \mu + r]^d` for ``bounds_radius`` :math:`r`.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array
from probpipe import log_prob
from probpipe.distributions.multivariate import MultivariateNormal

from sabi._probpipe_compat import independent_uniform
from sabi.problems.base import Problem
from sabi.problems.forms import Identity
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
            (:math:`|x_i - \\mu_i| \\le r`) for the design distribution
            and the support metadata.

    Returns:
        `Problem` with ``input_shape=(d,)``, ``Identity`` log-density
        form, and the analytic `MultivariateNormal` itself as the
        reference distribution.
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
    prior = independent_uniform(
        low=lower, high=upper, name=f"gaussian_d{d}_design_{id(mu)}"
    )

    target = TargetDistribution(
        target_single=target_single,
        name=f"gaussian_d{d}_target_{id(mu)}",
        input_shape=(d,),
        output_shape=(),
        log_density_form=Identity(),
        prior=prior,
    )
    return Problem(
        target_distribution=target,
        reference_distribution=posterior,
        name="gaussian",
    )


def gaussian2d(
    mean: tuple[float, float] = (0.0, 0.0),
    cov: tuple[tuple[float, float], tuple[float, float]] = ((1.0, 0.5), (0.5, 1.0)),
    bounds_radius: float = 5.0,
) -> Problem:
    """Build a 2-D Gaussian benchmark — historical entry point.

    Thin wrapper around `gaussian(d=2, ...)` preserving the original
    correlated-covariance default. Kept for back-compat with existing
    configs and tests; new code should call `gaussian(d=…)` directly or
    use `sabi.problems.benchmarks.gaussian_2d()` for the validated
    `BenchmarkProblem` form.
    """
    p = gaussian(d=2, mean=mean, cov=cov, bounds_radius=bounds_radius)
    return Problem(
        target_distribution=p.target_distribution,
        reference_distribution=p.reference_distribution,
        name="gaussian2d",
    )
