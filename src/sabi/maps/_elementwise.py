r"""Elementwise scalar Maps: ``Exp``, ``Log``, ``Softplus``,
``LogSoftplus``, ``Square``, ``LogSquare``.

Each has scalar shape ``() → ()`` and is a candidate ``link``-function
for `DensityDecomposition` (per ``docs/link_functions.md`` §3.2). The
``Log*`` variants are the numerically-stable log-of-the-base-map forms
used when the corresponding ``link`` is what the emulator targets.

Closed-form pushforward registrations (in
:mod:`sabi.maps._pushforward`):

- ``(Exp, Normal)`` → ``LogNormal``.
- ``(Log, LogNormal)`` → ``Normal``.

Everything else falls to the MC fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.nn
import jax.numpy as jnp
from jax import Array

from sabi.maps._base import Map


@dataclass(frozen=True)
class Exp(Map):
    r""":math:`f(z) = e^z`. Shape ``() → ()``."""

    event_shape_in: tuple[int, ...] = ()
    event_shape_out: tuple[int, ...] = ()

    def __call__(self, z: Array) -> Array:
        return jnp.exp(z)


@dataclass(frozen=True)
class Log(Map):
    r""":math:`f(z) = \log z`. Shape ``() → ()``.

    Defined for ``z > 0``; behavior at non-positive inputs follows
    ``jnp.log`` (returns ``-inf`` at 0, ``nan`` for negatives).
    """

    event_shape_in: tuple[int, ...] = ()
    event_shape_out: tuple[int, ...] = ()

    def __call__(self, z: Array) -> Array:
        return jnp.log(z)


@dataclass(frozen=True)
class Softplus(Map):
    r""":math:`f(z) = \log(1 + e^z)`. Shape ``() → ()``.

    Numerically stable via :func:`jax.nn.softplus`. MC pushforward only.
    """

    event_shape_in: tuple[int, ...] = ()
    event_shape_out: tuple[int, ...] = ()

    def __call__(self, z: Array) -> Array:
        return jax.nn.softplus(z)


@dataclass(frozen=True)
class LogSoftplus(Map):
    r""":math:`f(z) = \log\log(1 + e^z)`. Shape ``() → ()``.

    Numerically stable: writes
    :math:`\log\log(1 + e^z) = \log(\mathrm{softplus}(z))`. MC
    pushforward only.

    Note: undefined for ``z`` such that ``softplus(z) = 0`` (i.e.,
    very large negative inputs where softplus underflows). Callers are
    responsible for keeping inputs in range.
    """

    event_shape_in: tuple[int, ...] = ()
    event_shape_out: tuple[int, ...] = ()

    def __call__(self, z: Array) -> Array:
        return jnp.log(jax.nn.softplus(z))


@dataclass(frozen=True)
class Square(Map):
    r""":math:`f(z) = z^2`. Shape ``() → ()``.

    Falls to the MC fallback in this phase. The closed-form
    non-central-:math:`\chi^2` pushforward of ``(Square, Normal)`` is
    deferred to the square-link-end-to-end work
    (``docs/link_functions.md`` §9, Phase 7).
    """

    event_shape_in: tuple[int, ...] = ()
    event_shape_out: tuple[int, ...] = ()

    def __call__(self, z: Array) -> Array:
        return jnp.square(z)


@dataclass(frozen=True)
class LogSquare(Map):
    r""":math:`f(z) = 2\log\lvert z\rvert`. Shape ``() → ()``.

    Numerically stable form of ``Log @ Square``: avoids the explicit
    ``z^2`` intermediate. Has a singularity at ``z = 0``; the
    surrounding ``DensityDecomposition`` uses a ``NonNegative`` /
    ``Positive`` constraint to keep callers away from it.
    """

    event_shape_in: tuple[int, ...] = ()
    event_shape_out: tuple[int, ...] = ()

    def __call__(self, z: Array) -> Array:
        return 2.0 * jnp.log(jnp.abs(z))
