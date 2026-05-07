r"""``GaussianLogLik`` and ``LogProb`` — Maps that close over a
distribution / likelihood at construction.

These are non-elementwise: ``GaussianLogLik`` reduces a
``(d,)``-dimensional input to a scalar log-likelihood; ``LogProb``
reduces an ``event_shape``-dimensional input to a scalar log-probability
under a closed-over distribution.

Both fall to the MC pushforward fallback (no closed-form registration).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from probpipe import log_prob
from probpipe.core._distribution_base import Distribution

from sabi.maps._base import Map


@dataclass(frozen=True)
class GaussianLogLik(Map):
    r"""Gaussian log-likelihood map:
    :math:`f(z) = \log\mathcal{N}(\mathrm{obs} \mid z, \mathrm{cov})`.

    Shape ``(d,) → ()`` where ``d == obs.shape[0]``. Used by
    forward-model decompositions: the emulator emits a
    ``d``-dimensional output ``z`` (a simulated observation), and this
    Map applies the Gaussian observation-model density.

    The log-density is

    .. math::

        \log\mathcal{N}(o \mid z, C) = -\tfrac{1}{2}(o - z)^\top C^{-1}
        (o - z) - \tfrac{1}{2}\log\lvert 2\pi C\rvert.

    ``cov`` must be symmetric positive-definite; the formula assumes
    a real, positive log-determinant.
    """

    obs: Array
    cov: Array

    def __post_init__(self) -> None:
        object.__setattr__(self, "obs", jnp.asarray(self.obs))
        object.__setattr__(self, "cov", jnp.asarray(self.cov))

    @property
    def event_shape_in(self) -> tuple[int, ...]:
        return tuple(self.obs.shape)

    @property
    def event_shape_out(self) -> tuple[int, ...]:
        return ()

    def __call__(self, z: Array) -> Array:
        diff = self.obs - z
        _, logdet = jnp.linalg.slogdet(self.cov)
        d = self.obs.shape[0]
        solve = jnp.linalg.solve(self.cov, diff)
        quad = jnp.einsum("...i,...i->...", diff, solve)
        return -0.5 * quad - 0.5 * (logdet + d * jnp.log(2.0 * jnp.pi))


@dataclass(frozen=True)
class LogProb(Map):
    r"""Closed-over log-probability: :math:`f(z) = \log p_\mathrm{dist}(z)`.

    Shape ``dist.event_shape → ()``. Used as the ``shift`` Map in
    `DensityDecomposition` to add a log-prior, with
    ``dist == modeling_prior``.
    """

    dist: Distribution
    event_shape_out: tuple[int, ...] = ()

    @property
    def event_shape_in(self) -> tuple[int, ...]:
        return tuple(self.dist.event_shape)

    def __call__(self, z: Array) -> Array:
        return log_prob(self.dist, z)
