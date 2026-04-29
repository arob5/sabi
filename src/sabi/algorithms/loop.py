"""Sequential-acquisition loop.

v1.2 composition: initial design (sampled from `problem.prior`) → surrogate →
acquisition → `SurrogatePosterior` (a `NumericRandomMeasure`) → estimator
function (`expected_target` by default) → metrics. Tempering hooks present
(`tempering_state` per round, `current_form` built each round) but v1.2 ships
with `NoTempering` + `UntemperedSchedule` defaults, so the state is `None`
every round.

The estimator is a free function with type-dispatch on `SurrogatePosterior`
subtype (`expected_target`, possibly `mean` for the Dirac case). Configurable
per `Algorithm` via the `estimator` field.

The `surrogate_posterior_factory` builds the round's `SurrogatePosterior` from
math primitives — the `SurrogatePosterior` is decoupled from `Problem`, so the
factory pulls `support` / `prior` / `input_shape` from the problem and combines
with the (possibly tempered) `LogDensityForm` for the round.

Metrics consume a `Distribution[Array]` (the estimator's output) and declare
their required ProbPipe `Supports*` protocols via the `requires` class
attribute. The loop checks each metric's `requires` against the estimator
distribution and raises `MissingProtocolError` on a mismatch.

Shape / symbol conventions: see `docs/notation.md`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import jax
import jax.numpy as jnp
from jax import Array
from probpipe.core._distribution_base import Distribution
from probpipe.core.constraints import Constraint

from probpipe.core._random_measures import NumericRandomMeasure

from sabi.acquisitions.base import Acquisition, AcquisitionState
from sabi.initial_designs.base import InitialDesign, sample_initial
from sabi.metrics.base import MissingProtocolError, PosteriorMetric
from sabi.posterior.estimators import expected_target
from sabi.posterior.surrogate_posterior import SurrogatePosterior
from sabi.posterior.weighted_empirical import WeightedEmpiricalRandomMeasure
from sabi.problems.base import Problem
from sabi.problems.forms import LogDensityForm
from sabi.surrogates.base import Surrogate
from sabi.tempering.base import NoTempering, Tempering
from sabi.tempering.schedule import TemperingSchedule, UntemperedSchedule


class SurrogatePosteriorFactory(Protocol):
    """Builds the round's posterior random measure from the surrogate state
    and the math primitives (support, input_shape, prior, log_density_form).

    Returns a `NumericRandomMeasure` — concretely either a `SurrogatePosterior`
    or a `WeightedEmpiricalRandomMeasure` (or any other random-measure
    subclass a future factory might produce).

    `X` and `Y` are the design set; `log_density_form` is the form for the
    round (possibly tempered). `surrogate` may be ignored by factories
    that don't need a fitted surrogate (e.g., the weighted-empirical baseline).
    """

    def __call__(
        self,
        *,
        surrogate: Surrogate,
        X: Array,
        Y: Array,
        log_density_form: LogDensityForm,
        support: Constraint,
        input_shape: tuple[int, ...],
        prior: Distribution | None,
        problem_name: str | None = None,
    ) -> NumericRandomMeasure: ...


def gp_pushforward_factory(
    *,
    surrogate: Surrogate,
    X: Array,
    Y: Array,
    log_density_form: LogDensityForm,
    support: Constraint,
    input_shape: tuple[int, ...],
    prior: Distribution | None,
    problem_name: str | None = None,
) -> SurrogatePosterior:
    """Default factory: build the random measure induced by the fitted
    surrogate composed with the form (a `SurrogatePosterior`)."""
    return SurrogatePosterior(
        surrogate=surrogate,
        log_density_form=log_density_form,
        support=support,
        input_shape=input_shape,
        prior=prior,
        name=f"surrogate_posterior_{problem_name}" if problem_name else None,
    )


def weighted_empirical_factory(
    *,
    surrogate: Surrogate,
    X: Array,
    Y: Array,
    log_density_form: LogDensityForm,
    support: Constraint,
    input_shape: tuple[int, ...],
    prior: Distribution | None,
    problem_name: str | None = None,
) -> WeightedEmpiricalRandomMeasure:
    """No-GP baseline factory: a `WeightedEmpiricalRandomMeasure` at the
    design points.

    Applies `log_density_form` pointwise to `(X_i, Y_i)` to get the
    deterministic log-density (used as `log_weights`) at each design point.
    The `surrogate` argument is ignored; the form is used only here to
    compute the weights and is NOT carried on the resulting random measure.
    """
    log_weights = jax.vmap(
        lambda x, y: log_density_form(x, y, prior=prior)
    )(X, Y)
    return WeightedEmpiricalRandomMeasure(
        X=X,
        log_weights=log_weights,
        support=support,
        input_shape=input_shape,
        name=f"weighted_empirical_{problem_name}" if problem_name else None,
    )


@dataclass(frozen=True)
class Algorithm:
    """Composition of the components needed to run the loop."""

    surrogate_factory: Callable[[], Surrogate]
    acquisition: Acquisition
    n_initial: int = 16
    n_rounds: int = 10
    q: int = 1
    initial_design: InitialDesign | None = None
    tempering: Tempering = field(default_factory=NoTempering)
    schedule: TemperingSchedule = field(default_factory=UntemperedSchedule)
    surrogate_posterior_factory: SurrogatePosteriorFactory = gp_pushforward_factory
    estimator: Callable[[NumericRandomMeasure], Distribution] = expected_target
    metrics: tuple[PosteriorMetric, ...] = ()


@dataclass
class RunResult:
    X: Array
    Y: Array
    surrogate: Surrogate
    tempering_states: list[Any]
    per_round_metrics: list[dict[str, Any]]
    final_estimate: Distribution | None
    final_metrics: dict[str, float]


def _check_protocols(metric: PosteriorMetric, estimate: Distribution) -> None:
    """Raise `MissingProtocolError` if `estimate` doesn't satisfy `metric.requires`."""
    missing = [p.__name__ for p in metric.requires if not isinstance(estimate, p)]
    if missing:
        raise MissingProtocolError(
            f"{type(metric).__name__} requires {missing} on the posterior "
            f"estimate, but {type(estimate).__name__} does not satisfy them."
        )


def _evaluate_metrics(
    estimate: Distribution,
    problem: Problem,
    metrics: tuple[PosteriorMetric, ...],
    key: Array,
) -> dict[str, float]:
    """Run every metric on the estimate distribution.

    Each metric:
    1. Has its `requires` checked against the estimate; mismatches raise.
    2. Is called with `(posterior, problem, key=metric_key)`; the returned dict
       is merged into the round's metric row.

    A separate PRNG key is split per metric so each metric gets independent
    randomness if it samples internally.
    """
    if not metrics:
        return {}
    keys = jax.random.split(key, len(metrics))
    merged: dict[str, float] = {}
    for metric, metric_key in zip(metrics, keys, strict=True):
        _check_protocols(metric, estimate)
        merged.update(metric(estimate, problem, key=metric_key))
    return merged


def _build_surrogate_posterior(
    factory: SurrogatePosteriorFactory,
    *,
    surrogate: Surrogate,
    X: Array,
    Y: Array,
    log_density_form: LogDensityForm,
    problem: Problem,
) -> NumericRandomMeasure:
    """Adapter: extract the math primitives from `Problem` and call the factory."""
    return factory(
        surrogate=surrogate,
        X=X,
        Y=Y,
        log_density_form=log_density_form,
        support=problem.support,
        input_shape=problem.input_shape,
        prior=problem.prior,
        problem_name=problem.name,
    )


def run(problem: Problem, algorithm: Algorithm, key: Array) -> RunResult:
    """Run the sequential surrogate-based inference loop.

    State is explicit and flat. The surrogate is re-fit each round on the full
    `(X, Y)` (no incremental updates in v1.2; design doc lists `condition_on`-
    backed updates as a v2 item).
    """
    if problem.support is None:
        raise ValueError(
            f"Problem {problem.name!r} requires a non-None `support` for v1.2 "
            "SurrogatePosterior construction."
        )
    key_init, key_loop, key_eval = jax.random.split(key, 3)

    X = sample_initial(problem, key_init, algorithm.n_initial, algorithm.initial_design)
    Y = jax.vmap(problem.target_function)(X)

    tempering_states: list[Any] = []
    per_round_metrics: list[dict[str, Any]] = []

    surrogate = algorithm.surrogate_factory()
    surrogate = surrogate.fit(X, Y)

    for round_idx in range(algorithm.n_rounds):
        tempering_state, _final = algorithm.schedule.next(round_idx, None)
        current_form = algorithm.tempering.apply(
            problem.log_density_form, tempering_state
        )

        key_acq, key_metric, key_loop = jax.random.split(key_loop, 3)
        # Pre-acquisition SP wraps the current surrogate + form. For
        # WeightedEmpiricalRandomMeasure factories this isn't a
        # SurrogatePosterior; we set `surrogate_posterior=None` in that
        # case, and acquisitions that need a surrogate raise.
        pre_round_posterior = _build_surrogate_posterior(
            algorithm.surrogate_posterior_factory,
            surrogate=surrogate,
            X=X,
            Y=Y,
            log_density_form=current_form,
            problem=problem,
        )
        acq_state = AcquisitionState(
            problem=problem,
            surrogate_posterior=(
                pre_round_posterior
                if isinstance(pre_round_posterior, SurrogatePosterior)
                else None
            ),
            X=X,
            Y=Y,
            tempering_state=tempering_state,
        )
        x_new = algorithm.acquisition.select_batch(acq_state, algorithm.q, key_acq)
        y_new = jax.vmap(problem.target_function)(x_new)

        X = jnp.concatenate([X, x_new], axis=0)
        Y = jnp.concatenate([Y, y_new], axis=0)
        surrogate = surrogate.fit(X, Y)

        sp = _build_surrogate_posterior(
            algorithm.surrogate_posterior_factory,
            surrogate=surrogate,
            X=X,
            Y=Y,
            log_density_form=current_form,
            problem=problem,
        )
        estimate = algorithm.estimator(sp)
        round_metrics = _evaluate_metrics(estimate, problem, algorithm.metrics, key_metric)
        round_metrics["round"] = round_idx
        round_metrics["tempering_state"] = tempering_state
        round_metrics["n_evals"] = int(X.shape[0])
        per_round_metrics.append(round_metrics)
        tempering_states.append(tempering_state)

    # Final evaluation at the terminal target (untempered form), regardless
    # of schedule state, so downstream tooling always has a reference row.
    final_form = problem.log_density_form
    final_sp = _build_surrogate_posterior(
        algorithm.surrogate_posterior_factory,
        surrogate=surrogate,
        X=X,
        Y=Y,
        log_density_form=final_form,
        problem=problem,
    )
    final_estimate = algorithm.estimator(final_sp)
    final_metrics = _evaluate_metrics(
        final_estimate, problem, algorithm.metrics, key_eval
    )

    return RunResult(
        X=X,
        Y=Y,
        surrogate=surrogate,
        tempering_states=tempering_states,
        per_round_metrics=per_round_metrics,
        final_estimate=final_estimate,
        final_metrics=final_metrics,
    )
