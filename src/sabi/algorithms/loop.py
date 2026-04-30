"""Sequential-acquisition loop body.

The `run` function executes one full algorithm run end-to-end given a
`Problem`, an `Algorithm`, and a PRNG key. The `Algorithm` and
`RunResult` dataclasses live in :mod:`sabi.algorithms.algorithm`;
factories that build per-round `SurrogatePosterior` instances live in
:mod:`sabi.algorithms.surrogate_posterior_factory`.

Composition: initial design (drawn via `Algorithm.initial_sampler`) →
emulator → acquisition → `SurrogatePosterior` → estimator function
(`expected_target` by default) → metrics. Tempering hooks present
(`tempering_state` per round, `current_form` built each round); the
default `NoTempering` + `UntemperedSchedule` make the state `None`
every round.

Metrics consume a `Distribution[Array]` (the estimator's output) and
declare their required ProbPipe `Supports*` protocols via the
`requires` class attribute. The loop checks each metric's `requires`
against the estimator distribution and raises `MissingProtocolError`
on a mismatch.

Shape / symbol conventions: see ``docs/notation.md``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from probpipe.core._distribution_base import Distribution

from sabi.acquisitions.base import (
    AcquisitionState,
    resolve_state,
)
from sabi.algorithms.algorithm import Algorithm, RunResult
from sabi.algorithms.surrogate_posterior_factory import SurrogatePosteriorFactory
from sabi.emulators.base import Emulator
from sabi.metrics.base import MissingProtocolError, PosteriorMetric
from sabi.posterior.surrogate_posterior import SurrogatePosterior
from sabi.problems.base import Problem
from sabi.problems.forms import LogDensityForm


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
    emulator: Emulator,
    X: Array,
    Y: Array,
    log_density_form: LogDensityForm,
    problem: Problem,
) -> SurrogatePosterior:
    """Adapter: extract the math primitives from `Problem` and call the factory."""
    return factory(
        emulator=emulator,
        X=X,
        Y=Y,
        log_density_form=log_density_form,
        support=problem.support,
        input_shape=problem.input_shape,
        prior=problem.prior,
        problem_name=problem.name,
    )


def run(problem: Problem, algorithm: Algorithm, key: Array) -> RunResult:
    """Run the sequential emulator-based inference loop.

    State is explicit and flat. The emulator is re-fit each round on the
    full `(X, Y_train)` (no incremental updates in v1.2; design doc lists
    `condition_on`-backed updates as a v2 item).

    Tempering integration: each round, the `tempering_scheme` produces
    an `IntermediateTarget` at the round's state. ``Y_train`` is derived
    from cached ``Y_raw`` via the intermediate's ``output_transform``;
    the round's form (used to build the `SurrogatePosterior`) is the
    intermediate's ``log_density_form``. Under `NoTempering` (default),
    these are identity / unchanged from the base target distribution.

    Acquisition target: ``algorithm.acquisition_target`` selects which
    state the acquisition optimizes against (`CURRENT`, `NEXT`,
    `TERMINAL`). When this differs from the round's ``current_state``,
    the loop builds a separate look-ahead `IntermediateTarget` and may
    refit the emulator on the look-ahead-state's training data before
    the acquisition runs. Issue #4 will add a cheap-update dispatch
    that avoids redundant full refits.
    """
    target = problem.target_distribution
    # `prior` is required; `target.support = prior.support` is always
    # defined (possibly unbounded — algorithms that need bounded
    # support raise where they need it, not here).
    key_init, key_loop, key_eval = jax.random.split(key, 3)

    X = algorithm.initial_sampler.sample(problem, key_init, algorithm.n_initial)
    Y_raw = problem.target_function(X)

    # Build the round-0 intermediate to derive the initial Y_train.
    state_0, _ = algorithm.schedule.next(0, None)
    target_0 = algorithm.tempering_scheme.intermediate_target(target, state_0)
    Y_train = target_0.output_transform(state_0, X, Y_raw)

    tempering_states: list[Any] = []
    per_round_metrics: list[dict[str, Any]] = []

    emulator = algorithm.emulator_factory()
    emulator = emulator.fit(X, Y_train)

    for round_idx in range(algorithm.n_rounds):
        current_state, _final = algorithm.schedule.next(round_idx, None)
        target_state = resolve_state(
            algorithm.acquisition_target,
            algorithm.schedule,
            round_idx,
            current_state,
        )
        current_intermediate = algorithm.tempering_scheme.intermediate_target(
            target, current_state
        )
        # Acquisition's intermediate may live at a different state.
        # Reuse current_intermediate when target_state == current_state
        # (both invariance flags trivially hold).
        target_invariant = (
            algorithm.tempering_scheme.is_invariant_target_function(
                current_state, target_state
            )
            and algorithm.tempering_scheme.is_invariant_form(
                current_state, target_state
            )
        )
        if target_invariant:
            target_intermediate = current_intermediate
        else:
            target_intermediate = algorithm.tempering_scheme.intermediate_target(
                target, target_state
            )

        # Y_train and emulator the acquisition sees: derived at
        # target_state. If target_state's training data differs from
        # current's (or the emulator hasn't been fit at this state
        # yet), refit. Step 5 (issue #4) will replace this with a
        # cheap-update dispatch.
        if algorithm.tempering_scheme.is_invariant_target_function(
            current_state, target_state
        ):
            Y_train_for_acq = Y_train
            emulator_for_acq = emulator
        else:
            Y_train_for_acq = target_intermediate.output_transform(
                target_state, X, Y_raw
            )
            emulator_for_acq = algorithm.emulator_factory().fit(X, Y_train_for_acq)

        key_acq, key_metric, key_loop = jax.random.split(key_loop, 3)
        # Pre-acquisition SP wraps the acquisition-state emulator +
        # form. The weighted-empirical factory produces a
        # SurrogatePosterior with ``emulator=None``; acquisitions that
        # need a real emulator check
        # ``state.surrogate_posterior.emulator is None`` and raise.
        pre_round_posterior = _build_surrogate_posterior(
            algorithm.surrogate_posterior_factory,
            emulator=emulator_for_acq,
            X=X,
            Y=Y_train_for_acq,
            log_density_form=target_intermediate.log_density_form,
            problem=problem,
        )
        acq_state = AcquisitionState(
            problem=problem,
            surrogate_posterior=pre_round_posterior,
            X=X,
            Y_raw=Y_raw,
            Y_train=Y_train_for_acq,
            tempering_state=current_state,
            target_tempering_state=target_state,
        )
        x_new = algorithm.acquisition.select_batch(acq_state, algorithm.q, key_acq)
        y_new_raw = problem.target_function(x_new)

        X = jnp.concatenate([X, x_new], axis=0)
        Y_raw = jnp.concatenate([Y_raw, y_new_raw], axis=0)
        # Round-end fit: emulator + Y_train at the round's *current*
        # state (used by round-end metrics). For the no-look-ahead
        # case this is the only emulator fit per round.
        Y_train = current_intermediate.output_transform(current_state, X, Y_raw)
        emulator = emulator.fit(X, Y_train)

        sp = _build_surrogate_posterior(
            algorithm.surrogate_posterior_factory,
            emulator=emulator,
            X=X,
            Y=Y_train,
            log_density_form=current_intermediate.log_density_form,
            problem=problem,
        )
        estimate = algorithm.estimator(sp)
        round_metrics = _evaluate_metrics(estimate, problem, algorithm.metrics, key_metric)
        round_metrics["round"] = round_idx
        round_metrics["tempering_state"] = current_state
        round_metrics["target_tempering_state"] = target_state
        round_metrics["n_evals"] = int(X.shape[0])
        per_round_metrics.append(round_metrics)
        tempering_states.append(current_state)

    # Final evaluation at the base target (un-tempered form) so
    # downstream tooling always has a reference row at the terminal
    # distribution, regardless of where the schedule ended.
    final_form = target.log_density_form
    final_sp = _build_surrogate_posterior(
        algorithm.surrogate_posterior_factory,
        emulator=emulator,
        X=X,
        Y=Y_train,
        log_density_form=final_form,
        problem=problem,
    )
    final_estimate = algorithm.estimator(final_sp)
    final_metrics = _evaluate_metrics(
        final_estimate, problem, algorithm.metrics, key_eval
    )

    return RunResult(
        X=X,
        Y_raw=Y_raw,
        Y_train=Y_train,
        emulator=emulator,
        tempering_states=tempering_states,
        per_round_metrics=per_round_metrics,
        final_estimate=final_estimate,
        final_metrics=final_metrics,
    )
