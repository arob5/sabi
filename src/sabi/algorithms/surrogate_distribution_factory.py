"""Factories that build a `SurrogateDistribution` for a single round.

A `SurrogateDistributionFactory` packages the per-round
``(emulator, decomposition, support, ...)`` math primitives into a
`SurrogateDistribution`. Two factories ship today:

- `emulator_pushforward_factory` (default): wraps a fitted emulator and
  the round's ``DensityDecomposition`` into an `EmulatedDistribution`
  that pushes the emulator's predictive through ``decomposition.pushforward``.
- `weighted_empirical_factory`: the no-emulator baseline. Applies the
  decomposition to ``(X, Y)`` directly to compute log-weights, returning
  a `WeightedEmpiricalRandomMeasure` (a sibling of `EmulatedDistribution`
  under the abstract `SurrogateDistribution` base).

Both factories share a single signature including ``emulator``. The
weighted-empirical factory ignores its ``emulator`` argument; the loop
always has one to hand, so requiring the kwarg keeps the protocol
uniform across factories.

The ``Algorithm`` carries its choice of factory via
``Algorithm.surrogate_distribution_factory``.
"""

from __future__ import annotations

from typing import Protocol

import jax
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
    `EmulatedDistribution` and `WeightedEmpiricalRandomMeasure`). The
    default factory (`emulator_pushforward_factory`) builds an
    `EmulatedDistribution` that pushes the fitted emulator's predictive
    through the decomposition. `weighted_empirical_factory` builds the
    no-emulator baseline (`WeightedEmpiricalRandomMeasure`).

    ``X`` and ``Y`` are the design set; ``decomposition`` is the
    ``DensityDecomposition`` for the round (possibly per-state via
    tempering). ``emulator`` may be ignored by factories that don't
    need a fitted emulator (the weighted-empirical baseline).
    """

    def __call__(
        self,
        *,
        emulator: Emulator,
        X: Array,
        Y: Array,
        decomposition: DensityDecomposition,
        support: Constraint,
        input_shape: tuple[int, ...],
        problem_name: str | None = None,
    ) -> SurrogateDistribution: ...


def emulator_pushforward_factory(
    *,
    emulator: Emulator,
    X: Array,
    Y: Array,
    decomposition: DensityDecomposition,
    support: Constraint,
    input_shape: tuple[int, ...],
    problem_name: str | None = None,
) -> EmulatedDistribution:
    """Default factory: build the `EmulatedDistribution` that pushes the
    fitted emulator's predictive distribution through ``decomposition``.

    The pushforward itself lives inside `EmulatedDistribution`
    (``_random_unnormalized_log_prob`` calls ``decomposition.pushforward``);
    this factory just wires the round's emulator, decomposition, and
    problem-side primitives into a fresh `EmulatedDistribution` instance.

    Emulator-agnostic — works for any `Emulator` subclass, not just GPs.
    """
    return EmulatedDistribution(
        emulator=emulator,
        decomposition=decomposition,
        support=support,
        input_shape=input_shape,
        name=f"emulated_distribution_{problem_name}" if problem_name else None,
    )


def weighted_empirical_factory(
    *,
    emulator: Emulator,
    X: Array,
    Y: Array,
    decomposition: DensityDecomposition,
    support: Constraint,
    input_shape: tuple[int, ...],
    problem_name: str | None = None,
) -> WeightedEmpiricalRandomMeasure:
    """No-emulator baseline factory: a `WeightedEmpiricalRandomMeasure`
    at the design points.

    Applies ``decomposition`` pointwise to ``(X_i, Y_i)`` to get the
    deterministic log-density (used as ``log_weights``) at each design
    point. The ``emulator`` argument is ignored; the decomposition is
    used only here to compute the weights and is NOT carried on the
    resulting random measure.
    """
    # decomposition.__call__ supports both single-point and batched input
    # via the underlying Map broadcasting; vmap to be explicit.
    log_weights = jax.vmap(decomposition)(X, Y)
    return WeightedEmpiricalRandomMeasure(
        X=X,
        log_weights=log_weights,
        support=support,
        input_shape=input_shape,
        name=f"weighted_empirical_{problem_name}" if problem_name else None,
    )
