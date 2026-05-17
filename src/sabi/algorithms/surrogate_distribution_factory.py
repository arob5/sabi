"""Factories that build a `SurrogateDistribution` for a single round.

A `SurrogateDistributionFactory` packages the per-round
``(emulator, decomposition, support, ...)`` math primitives into a
`SurrogateDistribution`. Two factories ship today:

- `emulator_pushforward_factory` (default): wraps a fitted emulator and
  the round's ``DensityDecomposition`` into an `EmulatedDistribution`
  that pushes the emulator's predictive through ``decomposition.pushforward``.
- `weighted_empirical_factory`: the no-emulator baseline.

The decomposition's ``event_shape`` defines the inner-sample shape;
factories don't take a separate ``input_shape`` kwarg.
"""

from __future__ import annotations

from typing import Protocol

from jax import Array
from probpipe.core.constraints import Constraint

from sabi.density_decomposition import DensityDecomposition
from sabi.emulators.base import Emulator
from sabi.surrogate.surrogate_distribution import (
    EmulatedDistribution,
    SurrogateDistribution,
)
from sabi.surrogate.weighted_empirical import WeightedEmpiricalRandomMeasure


class SurrogateDistributionFactory(Protocol):
    """Builds the round's `SurrogateDistribution` from emulator + decomposition.

    Returns a `SurrogateDistribution` (the abstract base shared by
    `EmulatedDistribution` and `WeightedEmpiricalRandomMeasure`).

    ``X`` and ``Y`` are the design set; ``decomposition`` is the
    ``DensityDecomposition`` for the round (possibly per-state via
    tempering); ``support`` is the parameter-space constraint.
    The inner-sample shape is taken from ``decomposition.event_shape``.
    """

    def __call__(
        self,
        *,
        emulator: Emulator,
        X: Array,
        Y: Array,
        decomposition: DensityDecomposition,
        support: Constraint,
        problem_name: str | None = None,
    ) -> SurrogateDistribution: ...


def emulator_pushforward_factory(
    *,
    emulator: Emulator,
    X: Array,  # noqa: ARG001 — needed for protocol uniformity
    Y: Array,  # noqa: ARG001
    decomposition: DensityDecomposition,
    support: Constraint,
    problem_name: str | None = None,
) -> EmulatedDistribution:
    """Default factory: build the `EmulatedDistribution` that pushes the
    fitted emulator's predictive distribution through ``decomposition``.
    """
    return EmulatedDistribution(
        emulator=emulator,
        decomposition=decomposition,
        support=support,
        name=f"emulated_distribution_{problem_name}" if problem_name else None,
    )


def weighted_empirical_factory(
    *,
    emulator: Emulator,  # noqa: ARG001 — kept for protocol uniformity
    X: Array,
    Y: Array,
    decomposition: DensityDecomposition,
    support: Constraint,
    problem_name: str | None = None,
) -> WeightedEmpiricalRandomMeasure:
    """No-emulator baseline factory.

    Computes the per-design-point unnormalized log-density via
    ``decomposition.link(Y) + decomposition.shift(X)`` (using the
    cached evaluations ``Y = target_map(X)`` rather than re-evaluating).
    """
    log_weights = decomposition.link(Y)
    if decomposition.shift is not None:
        log_weights = log_weights + decomposition.shift(X)
    return WeightedEmpiricalRandomMeasure(
        X=X,
        log_weights=log_weights,
        support=support,
        inner_event_shape=tuple(decomposition.event_shape),
        name=f"weighted_empirical_{problem_name}" if problem_name else None,
    )
