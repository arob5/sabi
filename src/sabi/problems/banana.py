r"""Banana posterior — `d`-dimensional Haario "twisted Gaussian".

The 2-D form is the standard Rosenbrock-flavored banana benchmark; the
``d``-D extension keeps the banana shape on the first two coordinates
and stacks Gaussian "filler" dimensions on top — a model widely used
since Haario, Saksman & Tamminen (1999, 2001) for testing adaptive
samplers in moderate dimension.

Generative form (used to sample the reference):

.. math::

    z_1 &\sim \mathcal{N}(0, a^2), \\
    z_2 &\sim \mathcal{N}(0, 1/b), \\
    z_i &\sim \mathcal{N}(0, c^2), \quad i = 3, \dots, d.

Twist (a translation along :math:`x_2`; Jacobian determinant = 1):

.. math::

    x_1 = z_1, \quad
    x_2 = z_2 - z_1^2 + a^2, \quad
    x_i = z_i \;\; (i \ge 3).

Resulting log-density on :math:`x \in \mathbb{R}^d`:

.. math::

    \log p(x) = -\tfrac{1}{2}\!\left(
        \tfrac{x_1^2}{a^2}
        + b\,(x_2 + x_1^2 - a^2)^2
        + \tfrac{1}{c^2}\!\!\sum_{i=3}^{d} x_i^2
    \right) + C,

with normalizer
:math:`C = -\tfrac{d}{2}\log(2\pi) - \log a + \tfrac{1}{2}\log b - (d-2)\log c`.

At ``d = 2`` this reduces exactly to the canonical 2-D banana (the
:math:`c` term drops out).

Reference samples are exact: draw :math:`(z_1, z_2, z_3, \dots, z_d)`
from the factored Gaussian and apply the inverse twist. No NUTS / no
on-disk artifact needed at any dimension.

Shapes: ``input_shape=(d,)``, ``output_shape=()`` (scalar log-density).
See ``docs/notation.md`` for sabi's shape conventions.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import Array
from probpipe.core._empirical import NumericEmpiricalDistribution

from sabi._probpipe_compat import independent_uniform
from sabi.problems.base import Problem
from sabi.problems.forms import Identity
from sabi.target_distribution import TargetDistribution


def _default_bounds(
    d: int, a: float, b: float, c: float
) -> tuple[tuple[float, float], ...]:
    """Per-dim ``(lower, upper)`` covering ~4σ of the marginal."""
    sigma_2 = 1.0 / float(jnp.sqrt(b))  # σ of z_2
    # x_1 ranges to ~4a; the asymmetric x_2 bound mirrors the original 2-D
    # default (lower extends with the negative parabola tail).
    x1_radius = 4.0 * a
    x2_lower = -10.0 * sigma_2 - 4.0 * a * a
    x2_upper = 4.0 * sigma_2 + a * a
    filler_radius = 4.0 * c
    bounds = [(-x1_radius, x1_radius), (x2_lower, x2_upper)]
    bounds.extend([(-filler_radius, filler_radius)] * (d - 2))
    return tuple(bounds)


def banana(
    d: int = 2,
    a: float = 1.0,
    b: float = 4.0,
    c: float = 1.0,
    bounds: Sequence[tuple[float, float]] | None = None,
    n_reference_samples: int = 4096,
) -> Problem:
    """Build the `d`-dimensional banana benchmark.

    Args:
        d: parameter dimension. Must be ≥ 2 (the banana shape lives on
            the first two coordinates).
        a: ``x_1`` marginal scale.
        b: banana curvature (larger = tighter).
        c: per-dim std of the i ≥ 3 "filler" coordinates. Ignored when
            ``d == 2``.
        bounds: optional per-dim ``(lower, upper)`` overrides for the
            design distribution and support metadata. Defaults cover
            ~4σ in each dim.
        n_reference_samples: size of the analytic reference sample
            bank.

    Returns:
        `Problem` with ``input_shape=(d,)``, an ``Identity``
        log-density form, and a `NumericEmpiricalDistribution`
        reference built from exact factored draws.
    """
    if d < 2:
        raise ValueError(f"d must be ≥ 2 (banana shape needs 2 dims), got {d}.")
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError(f"a, b, c must be positive; got a={a}, b={b}, c={c}.")
    if bounds is not None and len(bounds) != d:
        raise ValueError(
            f"bounds must have length d={d}; got {len(bounds)} entries."
        )

    log2pi = jnp.log(2.0 * jnp.pi)
    log_norm = (
        -0.5 * d * log2pi
        - jnp.log(a)
        + 0.5 * jnp.log(b)
        - (d - 2) * jnp.log(c)
    )
    inv_c2 = 1.0 / (c * c)

    def log_prob_single(x: Array) -> Array:
        x1, x2 = x[0], x[1]
        z = x2 + x1 * x1 - a * a
        head = -0.5 * (x1 * x1 / (a * a) + b * z * z)
        tail = -0.5 * inv_c2 * jnp.sum(x[2:] * x[2:]) if d > 2 else 0.0
        return log_norm + head + tail

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

    key_ref = jax.random.key(0)
    ref_samples = sample_reference(key_ref, n_reference_samples)
    reference = NumericEmpiricalDistribution(
        samples=ref_samples, name=f"banana_d{d}_ref"
    )

    bnd = bounds if bounds is not None else _default_bounds(d, a, b, c)
    lower = jnp.asarray([lo for lo, _ in bnd], dtype=jnp.float64)
    upper = jnp.asarray([hi for _, hi in bnd], dtype=jnp.float64)
    prior = independent_uniform(
        low=lower, high=upper, name=f"banana_design_d{d}_a{a}_b{b}_c{c}"
    )

    target = TargetDistribution(
        target_single=log_prob_single,
        name=f"banana_d{d}_target",
        input_shape=(d,),
        output_shape=(),
        log_density_form=Identity(),
        prior=prior,
    )
    return Problem(
        target_distribution=target,
        reference_distribution=reference,
        name="banana",
    )
