"""Factories that build a `SurrogateDistribution` for a single round.

A `SurrogateDistributionFactory` packages the per-round
`(emulator, form, support, prior, ...)` math primitives into a
`SurrogateDistribution`. Two factories ship today:

- `emulator_pushforward_factory` (default): wraps a fitted emulator and
  the round's log-density form into a `SurrogateDistribution` that
  pushes the emulator's predictive through the form via
  `pushforward_marginal`.
- `weighted_empirical_factory`: the no-emulator baseline. Applies the
  form to `(X, Y)` directly to compute log-weights, returning a
  `WeightedEmpiricalRandomMeasure` (a `SurrogateDistribution` subclass
  with `emulator=None`).

The `Algorithm` carries its choice of factory via
`Algorithm.surrogate_distribution_factory`.
"""

from __future__ import annotations

from typing import Protocol

from jax import Array
from probpipe.core._distribution_base import Distribution
from probpipe.core.constraints import Constraint

from sabi.emulators.base import Emulator
from sabi.surrogate.surrogate_distribution import SurrogateDistribution
from sabi.surrogate.weighted_empirical import WeightedEmpiricalRandomMeasure
from sabi.problems.forms import LogDensityForm


class SurrogateDistributionFactory(Protocol):
    """Builds the round's `SurrogateDistribution` from the emulator state and
    the math primitives (support, input_shape, prior, log_density_form).

    Returns a `SurrogateDistribution`. The default factory
    (`emulator_pushforward_factory`) builds an SP that pushes the
    fitted emulator's predictive through the form. The
    `weighted_empirical_factory` builds the no-emulator baseline
    (`WeightedEmpiricalRandomMeasure`, a `SurrogateDistribution` subclass
    with ``emulator=None``).

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
) -> SurrogateDistribution:
    """Default factory: build the `SurrogateDistribution` that pushes the
    fitted emulator's predictive distribution through ``log_density_form``.

    The pushforward itself lives inside `SurrogateDistribution`
    (`_random_unnormalized_log_prob` / `pushforward_marginal`); this
    factory just wires the round's emulator, form, and problem-side
    primitives into a fresh `SurrogateDistribution` instance.

    Emulator-agnostic — works for any `Emulator` subclass, not just GPs.
    """
    return SurrogateDistribution(
        emulator=emulator,
        log_density_form=log_density_form,
        support=support,
        input_shape=input_shape,
        prior=prior,
        name=f"surrogate_distribution_{problem_name}" if problem_name else None,
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
