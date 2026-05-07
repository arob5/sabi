r"""``Dirac`` — a degenerate point-mass Distribution shim.

ProbPipe doesn't ship a ``Dirac`` (point-mass) distribution today; the
existing ``_DiracArrayRandomFunction`` is a `RandomFunction` over ``x``,
not a `Distribution`. The pushforward dispatch
``(Constant(c), *) → Dirac(c)`` needs a Distribution, so this module
provides a minimal one as a single-atom subclass of
:class:`~probpipe.core._empirical.NumericEmpiricalDistribution`.

Migration path: when ProbPipe ships a native ``Dirac``, this module
becomes a re-export and eventually retires. Same migration shape as
:mod:`sabi.surrogate._dirac`.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array
from probpipe.core._empirical import NumericEmpiricalDistribution


class Dirac(NumericEmpiricalDistribution):
    r"""Point mass at ``c``: a single-atom :math:`\delta_c` Distribution.

    ``event_shape == c.shape``, ``mean == c``, ``variance == 0``. Used
    by :func:`sabi.maps.pushforward` for ``(Constant(c), *) → Dirac(c)``.
    """

    def __init__(self, c: Array | Any, *, name: str = "dirac") -> None:
        c_arr = jnp.asarray(c)
        # Single-atom empirical: leading axis of size 1.
        super().__init__(samples=c_arr[None, ...], name=name)
        # Cache the atom for direct access (avoids unsqueezing on every call).
        self._c = c_arr

    @property
    def c(self) -> Array:
        """The atom location."""
        return self._c
