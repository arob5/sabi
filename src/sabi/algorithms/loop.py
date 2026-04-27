"""Sequential-acquisition loop.

v1.1 composition: initial design (sampled from `problem.prior`) → surrogate →
acquisition → `SurrogatePosterior` → `PosteriorEstimator` (`PlugInMean`) →
metrics. Tempering hooks present (`tempering_state` per round, `current_form`
built each round) but v1.1 ships with `NoTempering` + `UntemperedSchedule`
defaults, so the state is `None` every round.

The estimator is configurable per `Algorithm` via `surrogate_posterior_factory`
— a callable `(surrogate, X, Y, current_form, problem) -> SurrogatePosterior`.
v1.1 ships two factories: the GP-pushforward path (the v0 default) and the
weighted-empirical baseline (no-GP).

Metrics consume a `Distribution[Array]` (the estimator's output) and declare
their required ProbPipe `Supports*` protocols via the `requires` class
attribute. The loop checks each metric's `requires` against the estimator
distribution and raises `MissingProtocolError` on a mismatch.

Shape / symbol conventions: see `docs/notation.md`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from probpipe.core._distribution_base import Distribution

from sabi.acquisitions.base import Acquisition, AcquisitionState
from sabi.estimators.plug_in_mean import PlugInMean
from sabi.estimators.surrogate_posterior import (
    GPPushforwardSurrogatePosterior,
    SurrogatePosterior,
    WeightedEmpiricalSurrogatePosterior,
)
from sabi.initial_designs.base import InitialDesign, sample_initial
from sabi.metrics.base import MissingProtocolError, PosteriorMetric
from sabi.problems.base import Problem
from sabi.problems.forms import LogDensityForm
from sabi.surrogates.base import Surrogate
from sabi.tempering.base import NoTempering, Tempering
from sabi.tempering.schedule import TemperingSchedule, UntemperedSchedule


# Type alias: a factory that builds the round's `SurrogatePosterior`.
SurrogatePosteriorFactory = Callable[
    [Surrogate, Array, Array, LogDensityForm, Problem],
    SurrogatePosterior,
]


def gp_pushforward_factory(
    surrogate: Surrogate,
    X: Array,
    Y: Array,
    current_form: LogDensityForm,
    problem: Problem,
) -> GPPushforwardSurrogatePosterior:
    """Default factory: bundle the surrogate with the current form."""
    return GPPushforwardSurrogatePosterior(
        surrogate=surrogate, current_form=current_form, problem=problem
    )


def weighted_empirical_factory(
    surrogate: Surrogate,
    X: Array,
    Y: Array,
    current_form: LogDensityForm,
    problem: Problem,
) -> WeightedEmpiricalSurrogatePosterior:
    """No-GP baseline factory: weighted empirical at design points.

    Applies `current_form` to each `(X_i, Y_i)` to get the log-density at the
    design points; the resulting empirical is weighted by `softmax(log_density)`.
    Naive — doesn't de-bias against the design distribution. The `surrogate`
    argument is ignored.
    """
    log_d = jax.vmap(lambda x, y: current_form(x, y, problem))(X, Y)
    return WeightedEmpiricalSurrogatePosterior(X=X, Y=log_d, problem=problem)


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
    estimator: Callable[[SurrogatePosterior], Distribution] = PlugInMean
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

    TODO (post-ProbPipe): once the broader "distributions in, distributions out"
    rework lands (v2 with `RandomMeasure` + `PosteriorEstimator` backend dispatch),
    materialization decisions (e.g., wrapping the estimate in an `EmpiricalDistribution`
    for caching across multi-metric rounds) live here. v1.1 keeps it simple.
    """
    if not metrics:
        return {}
    keys = jax.random.split(key, len(metrics))
    merged: dict[str, float] = {}
    for metric, metric_key in zip(metrics, keys, strict=True):
        _check_protocols(metric, estimate)
        merged.update(metric(estimate, problem, key=metric_key))
    return merged


def run(problem: Problem, algorithm: Algorithm, key: Array) -> RunResult:
    """Run the sequential surrogate-based inference loop.

    State is explicit and flat. The surrogate is re-fit each round on the full
    `(X, Y)` (no incremental updates in v1.1; design doc lists `condition_on`-
    backed updates as a v2 item).
    """
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
        acq_state = AcquisitionState(
            problem=problem,
            surrogate=surrogate,
            X=X,
            Y=Y,
            current_form=current_form,
            tempering_state=tempering_state,
        )
        x_new = algorithm.acquisition.select_batch(acq_state, algorithm.q, key_acq)
        y_new = jax.vmap(problem.target_function)(x_new)

        X = jnp.concatenate([X, x_new], axis=0)
        Y = jnp.concatenate([Y, y_new], axis=0)
        surrogate = surrogate.fit(X, Y)

        sp = algorithm.surrogate_posterior_factory(
            surrogate, X, Y, current_form, problem
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
    final_sp = algorithm.surrogate_posterior_factory(surrogate, X, Y, final_form, problem)
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
