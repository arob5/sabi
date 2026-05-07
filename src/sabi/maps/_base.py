r"""`Map` ABC and the generic `Compose` combinator.

A :class:`Map` is a single-variable measurable function with a declared
shape contract. Maps participate in the
:func:`pushforward(map, dist) <sabi.maps.pushforward>` multiple-dispatch
op — closed-form pushforwards register against `Map` subclasses; an MC
fallback handles the rest.

See ``docs/link_functions.md`` §3 for the design intent and ProbPipe
alignment, and ``docs/notation.md`` for the shape conventions.

Shape contract
--------------

- ``event_shape_in``  — shape of one input event ``z``.
- ``event_shape_out`` — shape of one output event ``f(z)``.
- **Broadcasting.** Given input of shape
  ``batch_shape + event_shape_in``, ``__call__`` returns
  ``batch_shape + event_shape_out``. Same convention as
  ``ArrayRandomFunction``.
- **Composition.** ``f @ g`` requires
  ``f.event_shape_in == g.event_shape_out``. Result has
  ``event_shape_in == g.event_shape_in``,
  ``event_shape_out == f.event_shape_out``.

No ``inverse`` / ``log_det_jacobian`` slots — sabi's Maps are not
required to be bijective. Bijectors will land in ProbPipe as a `Map`
subclass that adds those slots.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from jax import Array


class Map(ABC):
    """Single-variable callable participating in pushforward dispatch.

    Subclasses implement :meth:`__call__` and declare ``event_shape_in``
    / ``event_shape_out`` (typically as dataclass fields or properties).
    See the module docstring for the shape contract.
    """

    event_shape_in: tuple[int, ...]
    event_shape_out: tuple[int, ...]

    @abstractmethod
    def __call__(self, z: Array) -> Array:
        """Apply the Map. ``z.shape == batch_shape + event_shape_in``;
        returns ``batch_shape + event_shape_out``.
        """

    def __matmul__(self, other: Map) -> Map:
        r"""Composition: ``(f @ g)(z) = f(g(z))``.

        Equivalent to ``Compose(self, other)``. Raises ``ValueError`` at
        construction time if ``self.event_shape_in != other.event_shape_out``.
        """
        return Compose(self, other)


@dataclass(frozen=True)
class Compose(Map):
    r"""Composition of two Maps: ``Compose(f, g)(z) = f(g(z))``.

    Shape contract:

    - ``event_shape_in  = g.event_shape_in``
    - ``event_shape_out = f.event_shape_out``
    - requires ``f.event_shape_in == g.event_shape_out`` at construction.
    """

    f: Map
    g: Map

    def __post_init__(self) -> None:
        if self.f.event_shape_in != self.g.event_shape_out:
            raise ValueError(
                f"Compose: shape mismatch — f.event_shape_in="
                f"{self.f.event_shape_in} but g.event_shape_out="
                f"{self.g.event_shape_out}."
            )

    @property
    def event_shape_in(self) -> tuple[int, ...]:
        return self.g.event_shape_in

    @property
    def event_shape_out(self) -> tuple[int, ...]:
        return self.f.event_shape_out

    def __call__(self, z: Array) -> Array:
        return self.f(self.g(z))
