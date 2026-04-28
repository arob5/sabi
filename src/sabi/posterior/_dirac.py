"""Sabi-local Dirac shims.

A `_DiracDistribution` is a degenerate `Distribution[Array]` concentrated at
a single value; a `_DiracArrayRandomFunction` is a degenerate
`RandomFunction` whose marginals are Diracs at deterministic per-input
values. They support the protocols sabi's v1.2 random-measure plumbing
needs (sampling, mean, single-point evaluation) without committing to a
density representation, since a Dirac on a continuous space has no proper
density.

ProbPipe doesn't yet ship a general Dirac abstraction. These shims live in
sabi for now; once ProbPipe lands a `Dirac[T]` (and the related
`DiracRandomFunction` / `DiracRandomMeasure`), they collapse to thin
imports. See `docs/probpipe_issues.md`: "No general `Dirac` distribution
abstraction".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import jax.numpy as jnp
from jax import Array
from probpipe.core._distribution_base import Distribution
from probpipe.core._random_functions import RandomFunction


class _DiracDistribution(Distribution):
    """A degenerate `Distribution[Array]` concentrated at a single value.

    Implements `SupportsSampling` and `SupportsMean`. Does NOT implement
    `SupportsLogProb` — a Dirac on a continuous space has no proper
    density, and the log-prob would be a delta function, not a scalar.
    """

    _sampling_cost: ClassVar[str] = "low"
    _preferred_orchestration: ClassVar[str | None] = None

    def __init__(self, value: Array, *, name: str | None = None):
        self._value = jnp.asarray(value)
        super().__init__(name=name or "dirac")

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._value.shape

    @property
    def value(self) -> Array:
        return self._value

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Array:
        return jnp.broadcast_to(self._value, sample_shape + self._value.shape)

    def _mean(self) -> Array:
        return self._value


class _DiracArrayRandomFunction(RandomFunction):
    """A `RandomFunction` whose marginals are `_DiracDistribution`s.

    Used as the random log-density of a Dirac random measure (e.g. the
    weighted-empirical baseline). The "randomness" is degenerate: each
    `__call__(x)` returns a `_DiracDistribution` at the deterministic
    `evaluator(x)`.

    Subclasses `RandomFunction` directly (not `ArrayRandomFunction`) — we
    don't need joint-input / joint-output shape semantics for the
    point-marginal use case, and want freedom in input parsing.
    """

    _sampling_cost: ClassVar[str] = "low"
    _preferred_orchestration: ClassVar[str | None] = None

    def __init__(
        self,
        evaluator: Callable[[Array], Array],
        *,
        input_shape: tuple[int, ...] = (),
        output_shape: tuple[int, ...] = (),
        name: str | None = None,
    ):
        self._evaluator = evaluator
        self._input_shape = tuple(input_shape)
        self._output_shape = tuple(output_shape)
        super().__init__(name=name or "dirac_rf")

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._output_shape

    def __call__(self, x: Array) -> Distribution:
        return _DiracDistribution(value=self._evaluator(x))
