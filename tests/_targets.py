"""Shared test fixtures: reusable ``NumericRecordDistribution`` and
``DensityDecomposition`` subclasses.

Tests that need a target / decomposition pair use these instead of
defining one-off subclasses inline.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from probpipe.core._distribution_base import Distribution
from probpipe.core._numeric_record_distribution import NumericRecordDistribution
from probpipe.core.constraints import Constraint

from sabi._probpipe_compat import independent_uniform
from sabi.density_decomposition import LogProbTermTarget


def box_support(d: int = 2, low: float = -5.0, high: float = 5.0) -> Constraint:
    """Per-dim ``[low, high]^d`` box support."""
    return independent_uniform(
        low=jnp.full((d,), low), high=jnp.full((d,), high), name=f"box_{d}d"
    ).support


def box_uniform(d: int = 2, low: float = -5.0, high: float = 5.0) -> Distribution:
    """Per-dim ``Uniform[low, high]^d`` distribution (used as design / prior)."""
    return independent_uniform(
        low=jnp.full((d,), low), high=jnp.full((d,), high), name=f"box_uniform_{d}d"
    )


def quadratic_log_density(x: Array) -> Array:
    """``-0.5 * sum(x * x, axis=-1)`` — vectorized quadratic log-density.

    Per the ProbPipe vectorization contract, sums over the trailing
    event axis only.
    """
    return -0.5 * jnp.sum(x * x, axis=-1)


class QuadraticTarget(NumericRecordDistribution):
    """Quadratic log-density target. Analytical density is
    :func:`quadratic_log_density`."""

    def __init__(self, *, d: int = 2, name: str = "quadratic"):
        self._d = d
        self._support = box_support(d)
        super().__init__(name=name)

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self._d,)

    @property
    def support(self) -> Constraint:
        return self._support

    def _unnormalized_log_prob(self, x: Array) -> Array:
        return quadratic_log_density(x)


class OpaqueTarget(NumericRecordDistribution):
    """User-style target: no analytical density (no ``_unnormalized_log_prob``)."""

    def __init__(self, *, d: int = 2, name: str = "opaque"):
        self._d = d
        self._support = box_support(d)
        super().__init__(name=name)

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self._d,)

    @property
    def support(self) -> Constraint:
        return self._support


class QuadraticLogProbDecomposition(LogProbTermTarget):
    """``LogProbTermTarget`` whose ``target_map`` is the quadratic
    log-density. ``link = Identity``, ``shift = LogProb(prior)`` if
    ``prior`` else ``None``."""

    def __init__(
        self,
        *,
        d: int = 2,
        name: str = "quadratic_decomp",
        prior: Distribution | None = None,
    ):
        self._d = d
        super().__init__(name=name, support=box_support(d), prior=prior)

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self._d,)

    def target_map(self, x: Array) -> Array:
        return quadratic_log_density(x)


class ConstantShiftedDecomposition(LogProbTermTarget):
    """``LogProbTermTarget`` whose ``target_map`` is the quadratic
    log-density plus a fixed scalar offset. Used for testing
    :func:`is_consistent_with`'s ``strict`` / ``strict=False`` modes."""

    def __init__(self, *, d: int = 2, offset: float = 0.0):
        self._d = d
        self._offset = offset
        super().__init__(name="quadratic_offset", support=box_support(d))

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self._d,)

    def target_map(self, x: Array) -> Array:
        return quadratic_log_density(x) + self._offset
