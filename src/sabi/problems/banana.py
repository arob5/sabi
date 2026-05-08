r"""Banana posterior — `d`-dimensional Haario "twisted Gaussian".

The 2-D form is the standard Rosenbrock-flavored banana benchmark; the
``d``-D extension keeps the banana shape on the first two coordinates
and stacks Gaussian "filler" dimensions on top — a model widely used
since Haario, Saksman & Tamminen (1999, 2001).

Public surface:

- :func:`banana` — factory returning a ``Problem``.
- :class:`BananaTarget` — :class:`NumericRecordDistribution` subclass
  with the analytical ``_unnormalized_log_prob`` (vectorized).

Callers that want to emulate the full unnormalized log-density should
wrap the target with :class:`sabi.density_decomposition.LogProbTarget`.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import Array
from probpipe.core._distribution_base import Distribution
from probpipe.core._empirical import NumericEmpiricalDistribution
from probpipe.core._numeric_record_distribution import NumericRecordDistribution
from probpipe.core.constraints import Constraint

from sabi._probpipe_compat import independent_uniform
from sabi.problems.base import Problem


# ---------------------------------------------------------------------------
# Private density helper — vectorized.
# ---------------------------------------------------------------------------


def _banana_log_density(
    x: Array, *, a: float, b: float, c: float, d: int, log_norm: Array
) -> Array:
    """Vectorized: ``x.shape == batch_shape + (d,)`` → ``batch_shape``."""
    inv_c2 = 1.0 / (c * c)
    x1, x2 = x[..., 0], x[..., 1]
    z = x2 + x1 * x1 - a * a
    head = -0.5 * (x1 * x1 / (a * a) + b * z * z)
    tail = (
        -0.5 * inv_c2 * jnp.sum(x[..., 2:] * x[..., 2:], axis=-1)
        if d > 2
        else 0.0
    )
    return log_norm + head + tail


def _default_bounds(
    d: int, a: float, b: float, c: float
) -> tuple[tuple[float, float], ...]:
    sigma_2 = 1.0 / float(jnp.sqrt(b))
    x1_radius = 4.0 * a
    x2_lower = -10.0 * sigma_2 - 4.0 * a * a
    x2_upper = 4.0 * sigma_2 + a * a
    filler_radius = 4.0 * c
    bounds = [(-x1_radius, x1_radius), (x2_lower, x2_upper)]
    bounds.extend([(-filler_radius, filler_radius)] * (d - 2))
    return tuple(bounds)


# ---------------------------------------------------------------------------
# NumericRecordDistribution subclass — analytical density
# ---------------------------------------------------------------------------


class BananaTarget(NumericRecordDistribution):
    """``NumericRecordDistribution`` for the ``d``-dimensional banana benchmark."""

    def __init__(
        self,
        *,
        d: int,
        a: float,
        b: float,
        c: float,
        support: Constraint,
        name: str | None = None,
    ):
        self._a, self._b, self._c, self._d = a, b, c, d
        log2pi = jnp.log(2.0 * jnp.pi)
        self._log_norm = (
            -0.5 * d * log2pi
            - jnp.log(a)
            + 0.5 * jnp.log(b)
            - (d - 2) * jnp.log(c)
        )
        self._support = support
        super().__init__(name=name or f"banana_d{d}_target")

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self._d,)

    @property
    def support(self) -> Constraint:
        return self._support

    def _unnormalized_log_prob(self, x: Array) -> Array:
        return _banana_log_density(
            x, a=self._a, b=self._b, c=self._c, d=self._d, log_norm=self._log_norm
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def banana(
    d: int = 2,
    a: float = 1.0,
    b: float = 4.0,
    c: float = 1.0,
    bounds: Sequence[tuple[float, float]] | None = None,
    n_reference_samples: int = 4096,
) -> Problem:
    """Build the `d`-dimensional banana benchmark."""
    if d < 2:
        raise ValueError(f"d must be ≥ 2 (banana shape needs 2 dims), got {d}.")
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError(f"a, b, c must be positive; got a={a}, b={b}, c={c}.")
    if bounds is not None and len(bounds) != d:
        raise ValueError(
            f"bounds must have length d={d}; got {len(bounds)} entries."
        )

    bnd = bounds if bounds is not None else _default_bounds(d, a, b, c)
    lower = jnp.asarray([lo for lo, _ in bnd], dtype=jnp.float64)
    upper = jnp.asarray([hi for _, hi in bnd], dtype=jnp.float64)
    support = independent_uniform(
        low=lower, high=upper, name=f"banana_d{d}_support"
    ).support

    target = BananaTarget(d=d, a=a, b=b, c=c, support=support)

    def sample_reference(key: Array, n: int) -> Array:
        keys = jax.random.split(key, 3)
        z1 = a * jax.random.normal(keys[0], shape=(n,))
        z2 = jax.random.normal(keys[1], shape=(n,)) / jnp.sqrt(b)
        x1 = z1
        x2 = z2 - z1 * z1 + a * a
        if d == 2:
            return jnp.stack([x1, x2], axis=-1)
        z_filler = c * jax.random.normal(keys[2], shape=(n, d - 2))
        return jnp.concatenate(
            [jnp.stack([x1, x2], axis=-1), z_filler], axis=-1
        )

    ref_samples = sample_reference(jax.random.key(0), n_reference_samples)
    reference: Distribution = NumericEmpiricalDistribution(
        samples=ref_samples, name=f"banana_d{d}_ref"
    )

    return Problem(
        target_distribution=target,
        reference_distribution=reference,
        name="banana",
    )
