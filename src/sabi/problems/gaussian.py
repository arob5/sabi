r"""`d`-dimensional Gaussian posterior — analytic reference.

Target is :math:`\mathcal{N}(\mu, \Sigma)` on :math:`\mathbb{R}^d`.

Public surface:

- :func:`gaussian` — factory returning a ``Problem`` whose
  ``target_distribution`` is a :class:`GaussianTarget`.
- :class:`GaussianTarget` — :class:`TargetDistribution` subclass with
  the analytical ``_unnormalized_log_prob`` (delegated to a ProbPipe
  ``MultivariateNormal``).
- :class:`GaussianLogProbDecomposition` —
  :class:`LogProbTermTarget` subclass that emulates the full
  unnormalized log-density.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array
from probpipe import log_prob as pp_log_prob
from probpipe.core.constraints import Constraint
from probpipe.distributions.multivariate import MultivariateNormal

from sabi._probpipe_compat import independent_uniform
from sabi.density_decomposition import LogProbTermTarget
from sabi.problems.base import Problem
from sabi.target_distribution import TargetDistribution


# ---------------------------------------------------------------------------
# Private helper — single-event evaluation of the multivariate normal density.
# ---------------------------------------------------------------------------


def _gaussian_log_density(x: Array, *, mvn: MultivariateNormal) -> Array:
    return jnp.asarray(pp_log_prob(mvn, x))


# ---------------------------------------------------------------------------
# Subclasses
# ---------------------------------------------------------------------------


class GaussianTarget(TargetDistribution):
    """``TargetDistribution`` for the multivariate Gaussian benchmark."""

    def __init__(
        self,
        *,
        name: str,
        input_shape: tuple[int, ...],
        support: Constraint,
        mvn: MultivariateNormal,
    ):
        self._mvn = mvn
        super().__init__(name=name, input_shape=input_shape, support=support)

    @property
    def mvn(self) -> MultivariateNormal:
        return self._mvn

    def _unnormalized_log_prob(self, x: Array) -> Array:
        return _gaussian_log_density(x, mvn=self._mvn)


class GaussianLogProbDecomposition(LogProbTermTarget):
    """Decomposition for the Gaussian benchmark — full log-density emulation.

    ``link = Identity``, ``shift = None``: ``target_map(x)`` is the
    full unnormalized log-density.
    """

    def __init__(
        self,
        *,
        name: str,
        input_shape: tuple[int, ...],
        support: Constraint,
        mvn: MultivariateNormal,
    ):
        self._mvn = mvn
        super().__init__(
            name=name, input_shape=input_shape, support=support, prior=None
        )

    def target_map(self, x: Array) -> Array:
        return _gaussian_log_density(x, mvn=self._mvn)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def gaussian(
    d: int = 2,
    mean: Sequence[float] | None = None,
    cov: Sequence[Sequence[float]] | None = None,
    bounds_radius: float = 5.0,
) -> Problem:
    """Build a `d`-dimensional Gaussian benchmark.

    Returns a ``Problem`` whose ``target_distribution`` is a
    :class:`GaussianTarget`. Reference is the analytic
    ``MultivariateNormal``.
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

    lower = mu - bounds_radius
    upper = mu + bounds_radius
    support = independent_uniform(
        low=lower, high=upper, name=f"gaussian_d{d}_support_{id(mu)}"
    ).support

    target = GaussianTarget(
        name=f"gaussian_d{d}_target_{id(mu)}",
        input_shape=(d,),
        support=support,
        mvn=posterior,
    )

    return Problem(
        target_distribution=target,
        reference_distribution=posterior,
        name="gaussian",
    )
