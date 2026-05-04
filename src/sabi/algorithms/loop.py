"""Sequential-acquisition loop body.

The `run` function executes one full algorithm run end-to-end given a
`Problem`, an `Algorithm`, and a PRNG key. The `Algorithm` and
`RunResult` dataclasses live in :mod:`sabi.algorithms.algorithm`;
factories that build per-round `SurrogateDistribution` instances live in
:mod:`sabi.algorithms.surrogate_distribution_factory`.

Composition: initial design (drawn via `Algorithm.initial_sampler`) →
emulator → metrics at round 0 (firing per `ScheduledMetric.every`) →
acquisition → `SurrogateDistribution` → estimator function
(`expected_target` by default) → metrics at round t. Tempering hooks
present (`tempering_state` per round, `current_form` built each round);
the default `NoTempering` + `UntemperedSchedule` make the state `None`
every round.

Per-metric scheduling: each metric in `Algorithm.metrics` is wrapped
(if not already) in a `ScheduledMetric` carrying `every` (firing
schedule), `target` (`CURRENT` vs `TERMINAL` intermediate), and
`final` (also include in the post-loop `final_metrics`). Bare
`Metric` instances auto-wrap at default config — preserves the
"every metric every round + final eval" behavior.

Metrics consume a `MetricContext` (estimate + surrogate distribution +
problem + design data + round/state metadata) and declare their
required ProbPipe `Supports*` protocols via the `requires` class
attribute. The loop checks each metric's `requires` against the
estimate distribution and raises `MissingProtocolError` on a
mismatch. Output keys are validated upfront (via
`validate_metric_keys`) before any emulator work happens; runtime
collision check at the merge step catches metrics that opted out of
upfront declaration (`keys = ()`).

Shape / symbol conventions: see ``docs/notation.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
from sabi.algorithms.surrogate_distribution_factory import SurrogateDistributionFactory
from sabi.emulators.base import Emulator
from sabi.emulators.dispatch import update_emulator
from sabi.emulators.updates import (
    AppendRows,
    EmulatorUpdate,
    RescaleOutputs,
    RescaleThenAppend,
)
from sabi.metrics.base import Metric, MetricContext, MissingProtocolError
from sabi.metrics.scheduling import (
    MetricTarget,
    ScheduledMetric,
    normalize_metrics,
    validate_metric_keys,
)
from sabi.surrogate.surrogate_distribution import SurrogateDistribution
from sabi.problems.base import Problem
from sabi.problems.forms import LogDensityForm
from sabi.tempering.output_transform import OutputTransform


def _check_protocols(metric: Metric, estimate: Distribution) -> None:
    """Raise `MissingProtocolError` if `estimate` doesn't satisfy `metric.requires`."""
    missing = [p.__name__ for p in metric.requires if not isinstance(estimate, p)]
    if missing:
        raise MissingProtocolError(
            f"{type(metric).__name__} requires {missing} on the posterior "
            f"estimate, but {type(estimate).__name__} does not satisfy them."
        )


def _merge_no_overwrite(
    dst: dict[str, Any],
    src: dict[str, float],
    source_repr: str,
) -> None:
    """Merge `src` into `dst`, raising on key collision.

    Runtime backstop for collision detection: catches metrics that
    didn't declare `keys` upfront (and so escaped
    `validate_metric_keys`'s static check) when they actually
    produce a key already in the row.
    """
    overlap = dst.keys() & src.keys()
    if overlap:
        raise ValueError(
            f"Metric key collision: keys {sorted(overlap)} produced by "
            f"{source_repr} are already present in this row. Set "
            f"`ScheduledMetric.name_suffix` or rename keys to disambiguate."
        )
    dst.update(src)


def _evaluate_scheduled_metrics(
    *,
    scheduled: Sequence[ScheduledMetric],
    current_estimate_fn: Callable[[], tuple[SurrogateDistribution, Distribution]],
    terminal_estimate_fn: Callable[[], tuple[SurrogateDistribution, Distribution]],
    problem: Problem,
    X: Array,
    Y_raw: Array,
    Y_train: Array,
    tempering_state: Any,
    round_idx: int,
    key: Array,
) -> dict[str, float]:
    """Run each scheduled metric at its target.

    Lazy: builds the current and/or terminal `(SurrogateDistribution,
    estimate)` pair only if at least one scheduled metric needs it.
    Caches each so it's built at most once per call.

    Raises on key collisions across metrics (runtime backstop;
    upfront `validate_metric_keys` should have caught most).
    """
    if not scheduled:
        return {}

    keys = jax.random.split(key, len(scheduled))
    merged: dict[str, float] = {}

    # Memoize the (SP, estimate) per target.
    cache: dict[MetricTarget, tuple[SurrogateDistribution, Distribution]] = {}

    def _get(target: MetricTarget) -> tuple[SurrogateDistribution, Distribution]:
        cached = cache.get(target)
        if cached is not None:
            return cached
        if target == MetricTarget.CURRENT:
            cache[target] = current_estimate_fn()
        elif target == MetricTarget.TERMINAL:
            cache[target] = terminal_estimate_fn()
        else:
            raise ValueError(f"Unknown MetricTarget: {target!r}")
        return cache[target]

    for s, mkey in zip(scheduled, keys, strict=True):
        sp, estimate = _get(s.target)
        _check_protocols(s.metric, estimate)
        ctx = MetricContext(
            estimate=estimate,
            surrogate_distribution=sp,
            problem=problem,
            X=X,
            Y_raw=Y_raw,
            Y_train=Y_train,
            tempering_state=tempering_state,
            round_idx=round_idx,
            metric_target=s.target,
        )
        out = s.apply_suffix(s.metric(ctx, key=mkey))
        _merge_no_overwrite(merged, out, repr(s))

    return merged


def _build_surrogate_distribution(
    factory: SurrogateDistributionFactory,
    *,
    emulator: Emulator,
    X: Array,
    Y: Array,
    log_density_form: LogDensityForm,
    problem: Problem,
) -> SurrogateDistribution:
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


def _plan_round_update(
    transform: OutputTransform,
    state_prev: Any,
    state_new: Any,
    X_new: Array | None,
    Y_new_at_new_state: Array | None,
) -> EmulatorUpdate | None:
    """Build the round's `EmulatorUpdate` plan, or ``None`` to refit.

    Combines the transform's structural diff (existing-rows update) with
    optional new-row append into a single op for `update_emulator`. The
    dispatcher reduces no-op rescales (factor=1.0 with no new rows) to
    nothing useful here, so the caller should also short-circuit on the
    invariant case before calling this.

    Args:
        transform: the round's `OutputTransform` (same shape across
            states for a given scheme; only state varies).
        state_prev: state of the emulator's last fit.
        state_new: target state for the new emulator.
        X_new: optional new rows of inputs (``None`` or empty for a
            state-only update like the look-ahead emulator).
        Y_new_at_new_state: outputs for ``X_new`` *already at*
            ``state_new`` (caller materializes via
            ``transform.apply(state_new, X_new, Y_new_raw)``).

    Returns:
        An `EmulatorUpdate` op when a fast path is expressible; ``None``
        when no closed-form diff is available (caller refits).
    """
    diff = transform.diff(state_prev, state_new)
    has_new_rows = X_new is not None and X_new.shape[0] > 0
    if diff is None:
        return None  # caller refits
    if isinstance(diff, RescaleOutputs):
        # Reduce a trivial rescale (factor=1.0) so cheap-path handlers
        # registered against `AppendRows` get a chance. Without this
        # reduction every round of a no-tempering run is dispatched as
        # `RescaleThenAppend(factor=1.0, ...)`, which only the
        # (composite-aware) handlers can match.
        if diff.factor == 1.0:
            if has_new_rows:
                return AppendRows(X_new=X_new, Y_new=Y_new_at_new_state)
            return None  # nothing to do; fall back to refit (or skip)
        if has_new_rows:
            return RescaleThenAppend(
                factor=diff.factor, X_new=X_new, Y_new=Y_new_at_new_state
            )
        return diff
    # Unknown diff shape — let the dispatcher try; if no handler claims
    # it, it'll fall back to refit on its own.
    return diff


def run(problem: Problem, algorithm: Algorithm, key: Array) -> RunResult:
    """Run the sequential emulator-based inference loop.

    State is explicit and flat. The emulator is re-fit each round on the
    full `(X, Y_train)` (no incremental updates in v1.2; design doc lists
    `condition_on`-backed updates as a v2 item).

    Tempering integration: each round, the `tempering_scheme` produces
    an `IntermediateTarget` at the round's state. ``Y_train`` is derived
    from cached ``Y_raw`` via the intermediate's ``output_transform``;
    the round's form (used to build the `SurrogateDistribution`) is the
    intermediate's ``log_density_form``. Under `NoTempering` (default),
    these are identity / unchanged from the base target distribution.

    Acquisition target: ``algorithm.acquisition_target`` selects which
    state the acquisition optimizes against (`CURRENT`, `NEXT`,
    `TERMINAL`). When this differs from the round's ``current_state``,
    the loop builds a separate look-ahead `IntermediateTarget` and may
    refit the emulator on the look-ahead-state's training data before
    the acquisition runs. Issue #4 will add a cheap-update dispatch
    that avoids redundant full refits.

    Per-round metric rows: the loop emits one `per_round_metrics` row
    per round (`0..n_rounds-1`), populated with bookkeeping fields
    (`round`, `tempering_state`, `target_tempering_state`,
    `n_evals`) plus values from any `ScheduledMetric` firing that
    round. Metrics with `final=True` also evaluate at the post-loop
    final step against the un-tempered base target; results land in
    `RunResult.final_metrics`.
    """
    target = problem.target_distribution
    # `prior` is required; `target.support = prior.support` is always
    # defined (possibly unbounded — algorithms that need bounded
    # support raise where they need it, not here).
    key_init, key_loop, key_eval = jax.random.split(key, 3)

    # Normalize and validate metrics up front — before any emulator
    # work happens. `validate_metric_keys` enumerates every round and
    # every `final=True` slot, raising on collisions among declared
    # output keys (or against bookkeeping fields).
    scheduled = normalize_metrics(algorithm.metrics)
    validate_metric_keys(scheduled, algorithm.n_rounds)

    # Round 0: initial-design round. Draw n_initial points, evaluate
    # target, fit emulator at the schedule's round-0 state.
    X = algorithm.initial_sampler.sample(problem, key_init, algorithm.n_initial)
    Y_raw = problem.target_map(X)

    state_0, _ = algorithm.schedule.at(0)
    target_0 = algorithm.tempering_scheme.intermediate_target(target, state_0)
    Y_train = target_0.output_transform(state_0, X, Y_raw)

    tempering_states: list[Any] = [state_0]
    per_round_metrics: list[dict[str, Any]] = []

    emulator = algorithm.emulator_factory()
    emulator = emulator.fit(X, Y_train)
    # Track the state of the emulator's last fit. Used as the "from"
    # state when building cheap-update plans below; updated after each
    # round-end fit to the round's current_state.
    emulator_state: Any = state_0

    # ---------- Round 0 metric eval ----------
    # Always emit a per_round_metrics row for round 0 (one row per
    # round including initial design). Lazy SP/estimate construction:
    # nothing built when no metric fires this round.
    key_metric_0, key_loop = jax.random.split(key_loop, 2)
    firing_round_0 = tuple(s for s in scheduled if s.fires_at_round(0))
    round_0_metrics = _eval_round(
        scheduled_firing=firing_round_0,
        algorithm=algorithm,
        problem=problem,
        target=target,
        current_intermediate=target_0,
        emulator=emulator,
        X=X,
        Y_raw=Y_raw,
        Y_train=Y_train,
        tempering_state=state_0,
        round_idx=0,
        key=key_metric_0,
    )
    round_0_metrics["round"] = 0
    round_0_metrics["tempering_state"] = state_0
    round_0_metrics["target_tempering_state"] = None
    round_0_metrics["n_evals"] = int(X.shape[0])
    per_round_metrics.append(round_0_metrics)

    # Loop body: rounds 1 through n_rounds-1 inclusive — the
    # acquisition rounds. Each round adds q evaluations chosen by the
    # acquisition.
    for round_idx in range(1, algorithm.n_rounds):
        current_state, _is_terminal_state = algorithm.schedule.at(round_idx)
        target_state = resolve_state(
            algorithm.acquisition_target,
            algorithm.schedule,
            round_idx,
            current_state,
        )
        current_intermediate = algorithm.tempering_scheme.intermediate_target(
            target, current_state
        )
        # Per-axis invariance between the round's state and the
        # acquisition's target state.
        invariance = algorithm.tempering_scheme.invariance(
            current_state, target_state
        )

        # Acquisition's intermediate may live at a different state.
        # Reuse current_intermediate when both axes are invariant.
        if invariance.both:
            target_intermediate = current_intermediate
        else:
            target_intermediate = algorithm.tempering_scheme.intermediate_target(
                target, target_state
            )

        # Path 1: Y_train and emulator the acquisition sees, derived at
        # target_state. When target_state == emulator_state for the Y
        # axis (`invariance.target_map`), reuse directly. Otherwise
        # dispatch a state-only cheap update (no new rows yet); the
        # dispatcher falls back to refit when no fast path is registered.
        if invariance.target_map:
            Y_train_for_acq = Y_train
            emulator_for_acq = emulator
        else:
            Y_train_for_acq = target_intermediate.output_transform(
                target_state, X, Y_raw
            )
            lookahead_plan = _plan_round_update(
                target_intermediate.output_transform,
                state_prev=emulator_state,
                state_new=target_state,
                X_new=None,
                Y_new_at_new_state=None,
            )
            emulator_for_acq = update_emulator(
                emulator,
                lookahead_plan,
                factory=algorithm.emulator_factory,
                X_full=X,
                Y_full=Y_train_for_acq,
            )

        key_acq, key_metric, key_loop = jax.random.split(key_loop, 3)
        # Pre-acquisition SP wraps the acquisition-state emulator +
        # form. The weighted-empirical factory produces a
        # SurrogateDistribution with ``emulator=None``; acquisitions that
        # need a real emulator check
        # ``state.surrogate_distribution.emulator is None`` and raise.
        pre_round_posterior = _build_surrogate_distribution(
            algorithm.surrogate_distribution_factory,
            emulator=emulator_for_acq,
            X=X,
            Y=Y_train_for_acq,
            log_density_form=target_intermediate.log_density_form,
            problem=problem,
        )
        acq_state = AcquisitionState(
            problem=problem,
            surrogate_distribution=pre_round_posterior,
            X=X,
            Y_raw=Y_raw,
            Y_train=Y_train_for_acq,
            tempering_state=current_state,
            target_tempering_state=target_state,
        )
        x_new = algorithm.acquisition.select_batch(acq_state, algorithm.q, key_acq)
        y_new_raw = problem.target_map(x_new)

        X = jnp.concatenate([X, x_new], axis=0)
        Y_raw = jnp.concatenate([Y_raw, y_new_raw], axis=0)
        # Path 2: round-end emulator at current_state with the new q
        # rows appended. Plan combines the existing-rows diff
        # (transform.diff(emulator_state, current_state)) with the new-
        # rows append into a single op; the dispatcher tries fast paths
        # and falls back to refit otherwise.
        Y_train = current_intermediate.output_transform(current_state, X, Y_raw)
        y_new_at_current = current_intermediate.output_transform(
            current_state, x_new, y_new_raw
        )
        round_plan = _plan_round_update(
            current_intermediate.output_transform,
            state_prev=emulator_state,
            state_new=current_state,
            X_new=x_new,
            Y_new_at_new_state=y_new_at_current,
        )
        emulator = update_emulator(
            emulator,
            round_plan,
            factory=algorithm.emulator_factory,
            X_full=X,
            Y_full=Y_train,
        )
        emulator_state = current_state

        # ---------- Per-round metric eval ----------
        firing = tuple(s for s in scheduled if s.fires_at_round(round_idx))
        round_metrics = _eval_round(
            scheduled_firing=firing,
            algorithm=algorithm,
            problem=problem,
            target=target,
            current_intermediate=current_intermediate,
            emulator=emulator,
            X=X,
            Y_raw=Y_raw,
            Y_train=Y_train,
            tempering_state=current_state,
            round_idx=round_idx,
            key=key_metric,
        )
        round_metrics["round"] = round_idx
        round_metrics["tempering_state"] = current_state
        round_metrics["target_tempering_state"] = target_state
        round_metrics["n_evals"] = int(X.shape[0])
        per_round_metrics.append(round_metrics)
        tempering_states.append(current_state)

    # ---------- Post-loop final eval ----------
    # Final SP/estimate at the un-tempered base form. Always built (so
    # `RunResult.final_estimate` is always populated for downstream
    # tooling), regardless of whether any metric has `final=True`.
    final_form = target.log_density_form
    final_sp = _build_surrogate_distribution(
        algorithm.surrogate_distribution_factory,
        emulator=emulator,
        X=X,
        Y=Y_train,
        log_density_form=final_form,
        problem=problem,
    )
    final_estimate = algorithm.estimator(final_sp)

    final_scheduled = tuple(s for s in scheduled if s.final)
    final_metrics: dict[str, float] = {}
    if final_scheduled:
        keys_split = jax.random.split(key_eval, len(final_scheduled))
        # `final_metrics` always evaluates at the un-tempered base —
        # `metric_target=TERMINAL` in the context regardless of the
        # ScheduledMetric's per-round `target` field.
        last_state = tempering_states[-1] if tempering_states else None
        for s, mkey in zip(final_scheduled, keys_split, strict=True):
            _check_protocols(s.metric, final_estimate)
            ctx = MetricContext(
                estimate=final_estimate,
                surrogate_distribution=final_sp,
                problem=problem,
                X=X,
                Y_raw=Y_raw,
                Y_train=Y_train,
                tempering_state=last_state,
                round_idx=algorithm.n_rounds,
                metric_target=MetricTarget.TERMINAL,
            )
            out = s.apply_suffix(s.metric(ctx, key=mkey))
            _merge_no_overwrite(final_metrics, out, repr(s))

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


def _eval_round(
    *,
    scheduled_firing: Sequence[ScheduledMetric],
    algorithm: Algorithm,
    problem: Problem,
    target,
    current_intermediate,
    emulator: Emulator,
    X: Array,
    Y_raw: Array,
    Y_train: Array,
    tempering_state: Any,
    round_idx: int,
    key: Array,
) -> dict[str, float]:
    """Evaluate a round's firing scheduled metrics against current/terminal estimates.

    Builds the SP+estimate at each target lazily: nothing constructed
    when ``scheduled_firing`` is empty; current/terminal pairs built
    independently and only when at least one firing metric needs that
    target.
    """
    if not scheduled_firing:
        return {}

    # Lazy current-state SP+estimate.
    def _current() -> tuple[SurrogateDistribution, Distribution]:
        sp = _build_surrogate_distribution(
            algorithm.surrogate_distribution_factory,
            emulator=emulator,
            X=X,
            Y=Y_train,
            log_density_form=current_intermediate.log_density_form,
            problem=problem,
        )
        return sp, algorithm.estimator(sp)

    # Lazy terminal-state SP+estimate. Uses `target.log_density_form`
    # (un-tempered base) with the round's emulator + Y_train at the
    # current state — matches the previous "final eval" semantics
    # (preserved for `MetricTarget.TERMINAL` mid-loop).
    def _terminal() -> tuple[SurrogateDistribution, Distribution]:
        sp = _build_surrogate_distribution(
            algorithm.surrogate_distribution_factory,
            emulator=emulator,
            X=X,
            Y=Y_train,
            log_density_form=target.log_density_form,
            problem=problem,
        )
        return sp, algorithm.estimator(sp)

    return _evaluate_scheduled_metrics(
        scheduled=scheduled_firing,
        current_estimate_fn=_current,
        terminal_estimate_fn=_terminal,
        problem=problem,
        X=X,
        Y_raw=Y_raw,
        Y_train=Y_train,
        tempering_state=tempering_state,
        round_idx=round_idx,
        key=key,
    )
