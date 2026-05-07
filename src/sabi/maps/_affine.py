r"""`Identity`, `Constant`, and `Affine` — the linear / constant maps.

These are the closed-form-friendly Maps: the
:func:`pushforward(map, dist) <sabi.maps.pushforward>` op handles them
analytically through Gaussians (and through arbitrary input
distributions for `Identity` / `Constant`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array

from sabi.maps._base import Map


@dataclass(frozen=True)
class Identity(Map):
    r"""Identity map: :math:`f(z) = z`. Shape ``() → ()``.

    Closed-form pushforward through any distribution: returns the input
    distribution unchanged.
    """

    event_shape_in: tuple[int, ...] = ()
    event_shape_out: tuple[int, ...] = ()

    def __call__(self, z: Array) -> Array:
        return z


@dataclass(frozen=True)
class Constant(Map):
    r"""Constant map: :math:`f(z) = c`. Ignores ``z``.

    ``event_shape_in = ()`` (declared scalar; broadcasting handles
    batches regardless of input shape — a `Constant` ignores its
    input). ``event_shape_out = c.shape``.
    """

    c: Array

    def __post_init__(self) -> None:
        # Coerce to a JAX array so c.shape is well-defined.
        object.__setattr__(self, "c", jnp.asarray(self.c))

    @property
    def event_shape_in(self) -> tuple[int, ...]:
        return ()

    @property
    def event_shape_out(self) -> tuple[int, ...]:
        return tuple(self.c.shape)

    def __call__(self, z: Array) -> Array:
        # Broadcast c against z's leading batch dims. z.shape ==
        # batch_shape + (), so broadcast_to(c, batch_shape + c.shape).
        batch_shape = z.shape
        return jnp.broadcast_to(self.c, batch_shape + self.c.shape)


@dataclass(frozen=True)
class Affine(Map):
    r"""Affine map: :math:`f(z) = \mathrm{slope} \cdot z + \mathrm{intercept}`.

    Shape ``() → ()``. ``slope`` and ``intercept`` are scalars (or
    arrays that broadcast against the input). Closed-form pushforward
    through Gaussians: see ``docs/link_functions.md`` §5.1.

    The ``event_shape_*`` fields are scalar so that scalar-Map
    composition (``Affine`` as one link in a ``Compose`` chain on
    scalar emulator outputs) typechecks without bookkeeping. The
    closed-form pushforward handlers in ``_pushforward.py`` also
    accept vector ``intercept`` against an MVN — the shape contract
    is enforced for ``Compose``, while pushforward broadcasting is
    permissive at the registered-handler level. A vector-``Affine``
    variant for first-class vector composition is left to the
    link-function refactor (#65), which integrates ``Map`` with
    ``DensityDecomposition`` and makes the layered shape semantics
    fully explicit.
    """

    slope: Array = field(default_factory=lambda: jnp.asarray(1.0))
    intercept: Array = field(default_factory=lambda: jnp.asarray(0.0))
    event_shape_in: tuple[int, ...] = ()
    event_shape_out: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "slope", jnp.asarray(self.slope))
        object.__setattr__(self, "intercept", jnp.asarray(self.intercept))

    def __call__(self, z: Array) -> Array:
        return self.slope * z + self.intercept
