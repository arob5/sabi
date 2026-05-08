r"""Banana posterior — `d`-dimensional Haario "twisted Gaussian".

The 2-D form is the standard Rosenbrock-flavored banana benchmark; the
``d``-D extension keeps the banana shape on the first two coordinates
and stacks Gaussian "filler" dimensions on top — a model widely used
since Haario, Saksman & Tamminen (1999, 2001).

Generative form (used to sample the reference):

.. math::

    z_1 &\sim \mathcal{N}(0, a^2), \\
    z_2 &\sim \mathcal{N}(0, 1/b), \\
    z_i &\sim \mathcal{N}(0, c^2), \quad i = 3, \dots, d.

Twist:

.. math::

    x_1 = z_1, \quad x_2 = z_2 - z_1^2 + a^2, \quad x_i = z_i \;\; (i \ge 3).

Resulting log-density on :math:`x \in \mathbb{R}^d`:

.. math::

    \log p(x) = -\tfrac{1}{2}\!\left(
        \tfrac{x_1^2}{a^2}
        + b\,(x_2 + x_1^2 - a^2)^2
        + \tfrac{1}{c^2}\!\!\sum_{i=3}^{d} x_i^2
    \right) + C.

Reference samples are exact (factored Gaussian + inverse twist).

Public surface:

- :func:`banana` — factory returning a ``Problem`` whose
  ``target_distribution`` is a :class:`BananaTarget` (analytical
  density).
- :class:`BananaTarget` — :class:`TargetDistribution` subclass with
  the analytical ``_unnormalized_log_prob``.
- :class:`BananaLogProbDecomposition` —
  :class:`LogProbTermTarget` subclass that emulates the full
  unnormalized log-density. The canonical decomposition for this
  benchmark.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import Array
from probpipe.core._distribution_base import Distribution
from probpipe.core._empirical import NumericEmpiricalDistribution
from probpipe.core.constraints import Constraint

from sabi._probpipe_compat import independent_uniform
from sabi.density_decomposition import LogProbTermTarget
from sabi.problems.base import Problem
from sabi.target_distribution import TargetDistribution


# ---------------------------------------------------------------------------
# Private density helper — shared by the target's analytical density and the
# decomposition's `target_map`. Single-event input (rank-1 ``x``).
# ---------------------------------------------------------------------------


def _banana_log_density(
    x: Array, *, a: float, b: float, c: float, d: int, log_norm: Array
) -> Array:
    """Vectorized: ``x.shape == batch_shape + (d,)`` → ``batch_shape``.

    Indexes the trailing event axis with ``x[..., k]`` so leading batch
    dims pass through naturally.
    """
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
    """Per-dim ``(lower, upper)`` covering ~4σ of the marginal."""
    sigma_2 = 1.0 / float(jnp.sqrt(b))
    x1_radius = 4.0 * a
    x2_lower = -10.0 * sigma_2 - 4.0 * a * a
    x2_upper = 4.0 * sigma_2 + a * a
    filler_radius = 4.0 * c
    bounds = [(-x1_radius, x1_radius), (x2_lower, x2_upper)]
    bounds.extend([(-filler_radius, filler_radius)] * (d - 2))
    return tuple(bounds)


# ---------------------------------------------------------------------------
# TargetDistribution subclass — analytical density
# ---------------------------------------------------------------------------


class BananaTarget(TargetDistribution):
    """``TargetDistribution`` for the ``d``-dimensional banana benchmark.

    Args:
        name: ProbPipe distribution name.
        input_shape: ``(d,)``.
        support: ``Constraint`` over the parameter space.
        a, b, c: banana-shape parameters.
        d: parameter dimension.
    """

    def __init__(
        self,
        *,
        name: str,
        input_shape: tuple[int, ...],
        support: Constraint,
        a: float,
        b: float,
        c: float,
        d: int,
    ):
        self._a, self._b, self._c, self._d = a, b, c, d
        log2pi = jnp.log(2.0 * jnp.pi)
        self._log_norm = (
            -0.5 * d * log2pi
            - jnp.log(a)
            + 0.5 * jnp.log(b)
            - (d - 2) * jnp.log(c)
        )
        super().__init__(name=name, input_shape=input_shape, support=support)

    def _unnormalized_log_prob(self, x: Array) -> Array:
        return _banana_log_density(
            x, a=self._a, b=self._b, c=self._c, d=self._d, log_norm=self._log_norm
        )


# ---------------------------------------------------------------------------
# DensityDecomposition subclass — emulator emits the full log-density
# ---------------------------------------------------------------------------


class BananaLogProbDecomposition(LogProbTermTarget):
    """Decomposition for the banana benchmark — full log-density emulation.

    ``link = Identity``, ``shift = None``: the emulator's
    ``target_map(x)`` is the full unnormalized log-density. Equivalent
    to ``LogProbTarget(banana_problem.target_distribution)`` but
    implemented directly (no ``unnormalized_log_prob`` op indirection).

    Args:
        name: ProbPipe distribution name.
        input_shape: ``(d,)``.
        support: ``Constraint`` over the parameter space.
        a, b, c: banana-shape parameters.
        d: parameter dimension.
    """

    def __init__(
        self,
        *,
        name: str,
        input_shape: tuple[int, ...],
        support: Constraint,
        a: float,
        b: float,
        c: float,
        d: int,
    ):
        self._a, self._b, self._c, self._d = a, b, c, d
        log2pi = jnp.log(2.0 * jnp.pi)
        self._log_norm = (
            -0.5 * d * log2pi
            - jnp.log(a)
            + 0.5 * jnp.log(b)
            - (d - 2) * jnp.log(c)
        )
        super().__init__(
            name=name, input_shape=input_shape, support=support, prior=None
        )

    def target_map(self, x: Array) -> Array:
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
    """Build the `d`-dimensional banana benchmark.

    Returns a ``Problem`` whose ``target_distribution`` is a
    :class:`BananaTarget` (analytical density). The user constructs a
    decomposition explicitly — typically
    :class:`BananaLogProbDecomposition` for full log-density
    emulation.
    """
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

    target = BananaTarget(
        name=f"banana_d{d}_target",
        input_shape=(d,),
        support=support,
        a=a, b=b, c=c, d=d,
    )

    # Reference samples (analytic factored draws).
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
