"""Factories that build a `SurrogateDistribution` for a single round.

A `SurrogateDistributionFactory` packages the per-round
`(emulator, form, support, prior, ...)` math primitives into a
`SurrogateDistribution`. Two factories ship today:

- `emulator_pushforward_factory` (default): wraps a fitted emulator and
  the round's log-density form into an `EmulatedDistribution` that
  pushes the emulator's predictive through the form via
  `pushforward_marginal`.
- `weighted_empirical_factory`: the no-emulator baseline. Applies the
  form to `(X, Y)` directly to compute log-weights, returning a
  `WeightedEmpiricalRandomMeasure` (a sibling of `EmulatedDistribution`
  under the abstract `SurrogateDistribution` base).

Both factories share a single signature including `emulator`. The
weighted-empirical factory ignores its `emulator` argument; the loop
always has one to hand, so requiring the kwarg keeps the protocol
uniform across factories.

The `Algorithm` carries its choice of factory via
`Algorithm.surrogate_distribution_factory`.
"""

from __future__ import annotations

from typing import Protocol

from jax import Array
from probpipe.core._distribution_base import Distribution
from probpipe.core.constraints import Constraint

from sabi.emulators.base import Emulator
from sabi.surrogate.surrogate_distribution import (
    EmulatedDistribution,
    SurrogateDistribution,
)
from sabi.surrogate.weighted_empirical import WeightedEmpiricalRandomMeasure
from sabi.problems.forms import LogDensityForm


class SurrogateDistributionFactory(Protocol):
    """Builds the round's `SurrogateDistribution` from the emulator state and
    the math primitives (support, input_shape, prior, log_density_form).

    Returns a `SurrogateDistribution` (the abstract base shared by
    `EmulatedDistribution` and `WeightedEmpiricalRandomMeasure`). The
    default factory (`emulator_pushforward_factory`) builds an
    `EmulatedDistribution` that pushes the fitted emulator's predictive
    through the form. `weighted_empirical_factory` builds the
    no-emulator baseline (`WeightedEmpiricalRandomMeasure`).

    `X` and `Y` are the design set; `log_density_form` is the form for
    the round (possibly tempered). `emulator` may be ignored by
    factories that don't need a fitted emulator (the weighted-empirical
    baseline).
    """

    def __call__(
        self,
        *,
        emulator: Emulator,
        X: Array,
        Y: Array,
        log_density_form: LogDensityForm,
        support: Constraint,
        input_shape: tuple[int, ...],
        prior: Distribution | None,
        problem_name: str | None = None,
    ) -> SurrogateDistribution: ...


def emulator_pushforward_factory(
    *,
    emulator: Emulator,
    X: Array,
    Y: Array,
    log_density_form: LogDensityForm,
    support: Constraint,
    input_shape: tuple[int, ...],
    prior: Distribution | None,
    problem_name: str | None = None,
) -> EmulatedDistribution:
    """Default factory: build the `EmulatedDistribution` that pushes the
    fitted emulator's predictive distribution through ``log_density_form``.

    The pushforward itself lives inside `EmulatedDistribution`
    (`_random_unnormalized_log_prob` / `pushforward_marginal`); this
    factory just wires the round's emulator, form, and problem-side
    primitives into a fresh `EmulatedDistribution` instance.

    Emulator-agnostic — works for any `Emulator` subclass, not just GPs.
    """
    return EmulatedDistribution(
        emulator=emulator,
        log_density_form=log_density_form,
        support=support,
        input_shape=input_shape,
        prior=prior,
        name=f"emulated_distribution_{problem_name}" if problem_name else None,
    )


def weighted_empirical_factory(
    *,
    emulator: Emulator,
    X: Array,
    Y: Array,
    log_density_form: LogDensityForm,
    support: Constraint,
    input_shape: tuple[int, ...],
    prior: Distribution | None,
    problem_name: str | None = None,
) -> WeightedEmpiricalRandomMeasure:
    """No-emulator baseline factory: a `WeightedEmpiricalRandomMeasure`
    at the design points.

    Applies `log_density_form` pointwise to `(X_i, Y_i)` to get the
    deterministic log-density (used as `log_weights`) at each design
    point. The `emulator` argument is ignored; the form is used only
    here to compute the weights and is NOT carried on the resulting
    random measure.
    """
    # log_density_form is batched: takes (X, Y) and returns shape (n,).
    log_weights = log_density_form(X, Y, prior=prior)
    return WeightedEmpiricalRandomMeasure(
        X=X,
        log_weights=log_weights,
        support=support,
        input_shape=input_shape,
        name=f"weighted_empirical_{problem_name}" if problem_name else None,
    )
