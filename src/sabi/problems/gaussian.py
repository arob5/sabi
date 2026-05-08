r"""`d`-dimensional Gaussian posterior — analytic reference.

Target is :math:`\mathcal{N}(\mu, \Sigma)` on :math:`\mathbb{R}^d`.

Public surface:

- :func:`gaussian` — factory returning a ``Problem``.
- :class:`GaussianTarget` — :class:`NumericRecordDistribution` subclass
  with the analytical ``_unnormalized_log_prob`` (delegates to a
  ProbPipe ``MultivariateNormal``).
- :class:`GaussianLogProbDecomposition` —
  :class:`LogProbTermTarget` subclass that emulates the full
  unnormalized log-density.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array
from probpipe import log_prob as pp_log_prob
from probpipe.core._numeric_record_distribution import NumericRecordDistribution
from probpipe.core.constraints import Constraint
from probpipe.distributions.multivariate import MultivariateNormal

from sabi._probpipe_compat import independent_uniform
from sabi.density_decomposition import LogProbTermTarget
from sabi.problems.base import Problem


def _gaussian_log_density(x: Array, *, mvn: MultivariateNormal) -> Array:
    return jnp.asarray(pp_log_prob(mvn, x))


class GaussianTarget(NumericRecordDistribution):
    """``NumericRecordDistribution`` for the multivariate Gaussian benchmark."""

    def __init__(
        self,
        *,
        d: int,
        mvn: MultivariateNormal,
        support: Constraint,
        name: str | None = None,
    ):
        self._d = d
        self._mvn = mvn
        self._support = support
        super().__init__(name=name or f"gaussian_d{d}_target")

    @property
    def mvn(self) -> MultivariateNormal:
        return self._mvn

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self._d,)

    @property
    def support(self) -> Constraint:
        return self._support

    def _unnormalized_log_prob(self, x: Array) -> Array:
        return _gaussian_log_density(x, mvn=self._mvn)


class GaussianLogProbDecomposition(LogProbTermTarget):
    """Decomposition for the Gaussian benchmark — full log-density emulation."""

    def __init__(
        self,
        *,
        d: int,
        mvn: MultivariateNormal,
        support: Constraint,
        name: str | None = None,
    ):
        self._d = d
        self._mvn = mvn
        super().__init__(
            name=name or f"gaussian_d{d}_decomp", support=support, prior=None
        )

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self._d,)

    def target_map(self, x: Array) -> Array:
        return _gaussian_log_density(x, mvn=self._mvn)


def gaussian(
    d: int = 2,
    mean: Sequence[float] | None = None,
    cov: Sequence[Sequence[float]] | None = None,
    bounds_radius: float = 5.0,
) -> Problem:
    """Build a `d`-dimensional Gaussian benchmark."""
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

    target = GaussianTarget(d=d, mvn=posterior, support=support)

    return Problem(
        target_distribution=target,
        reference_distribution=posterior,
        name="gaussian",
    )
