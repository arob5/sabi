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
from dataclasses import dataclass
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
from sabi.target_distribution import IntermediateTarget, TargetDistribution
from sabi.tempering.base import InvarianceFlags
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


@dataclass(frozen=True)
class RoundState:
    """Per-round bundle of states + intermediates + invariance flags.

    Cached once at the top of each acquisition round (via
    `_resolve_round_state`) and threaded through the helpers that make
    up the round body. Promoting these fields to a struct trims the
    helper signatures from ~6 positional state args to a single
    `round_state` plus the data tensors, and gives a single place to
    look for "what state is this round operating at".

    Attributes:
        round_idx: 1-based round index (0 is the initial-design round
            and is handled separately by `_run_initial_round`).
        current_state: the round's own tempering state, from
            ``algorithm.schedule.at(round_idx)``. The round-end emulator
            and the round's `current_intermediate` live at this state.
        target_state: state the acquisition optimizes against, from
            `resolve_state(algorithm.acquisition_target, ...)`. May
            equal `current_state` (the `CURRENT` policy or a no-op
            tempering scheme), or differ (`NEXT`, `TERMINAL`, or a
            policy that looks ahead).
        current_intermediate: `IntermediateTarget` at `current_state`.
            Owns `output_transform` (used for the round-end refit /
            cheap-update plan) and `log_density_form` (used to build
            the round-end SP for metrics).
        target_intermediate: `IntermediateTarget` at `target_state`.
            Owns the `output_transform` and `log_density_form` used to
            materialize the acquisition's view (`Y_train_for_acq` and
            the pre-round SP). Equals `current_intermediate` when
            ``invariance.both`` is True — saves a redundant rebuild.
        invariance: per-axis flags from
            `algorithm.tempering_scheme.invariance(current_state,
            target_state)`. Drives the look-ahead-vs-reuse decision in
            `_resolve_acquisition_view`.
    """

    round_idx: int
    current_state: Any
    target_state: Any
    current_intermediate: IntermediateTarget
    target_intermediate: IntermediateTarget
    invariance: InvarianceFlags


def _resolve_round_state(
    algorithm: Algorithm,
    target: TargetDistribution,
    round_idx: int,
) -> RoundState:
    """Resolve all round-level state up front.

    Queries the schedule for `current_state`, resolves the acquisition's
    `target_state` via `resolve_state`, builds both intermediates, and
    computes the per-axis invariance flags. Reuses
    `current_intermediate` for `target_intermediate` when both axes are
    invariant (the common no-tempering / `CURRENT`-target case) — this
    matches the previous inline behavior and avoids a redundant
    `intermediate_target` call.
    """
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
    invariance = algorithm.tempering_scheme.invariance(current_state, target_state)
    if invariance.both:
        target_intermediate = current_intermediate
    else:
        target_intermediate = algorithm.tempering_scheme.intermediate_target(
            target, target_state
        )
    return RoundState(
        round_idx=round_idx,
        current_state=current_state,
        target_state=target_state,
        current_intermediate=current_intermediate,
        target_intermediate=target_intermediate,
        invariance=invariance,
    )


def _resolve_acquisition_view(
    emulator: Emulator,
    emulator_state: Any,
    round_state: RoundState,
    X: Array,
    Y_raw: Array,
    Y_train: Array,
    algorithm: Algorithm,
) -> tuple[Emulator, Array]:
    """Return the `(emulator, Y_train)` pair the acquisition sees.

    Two cases:

    - ``invariance.target_map`` is True: the round-end ``Y_train`` and
      ``emulator`` already live at the acquisition's target state, so
      reuse them directly. No allocation, no dispatcher call.
    - Otherwise: materialize ``Y_train_for_acq`` at ``target_state`` via
      the target intermediate's `output_transform`, then dispatch a
      state-only cheap-update plan (no new rows yet) through
      `update_emulator`. The dispatcher falls back to a full refit when
      no fast path is registered for this transform's diff.
    """
    if round_state.invariance.target_map:
        return emulator, Y_train
    Y_train_for_acq = round_state.target_intermediate.output_transform(
        round_state.target_state, X, Y_raw
    )
    lookahead_plan = _plan_round_update(
        round_state.target_intermediate.output_transform,
        state_prev=emulator_state,
        state_new=round_state.target_state,
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
    return emulator_for_acq, Y_train_for_acq


def _run_acquisition(
    problem: Problem,
    algorithm: Algorithm,
    emulator_for_acq: Emulator,
    X: Array,
    Y_raw: Array,
    Y_train_for_acq: Array,
    round_state: RoundState,
    key: Array,
) -> tuple[Array, Array]:
    """Build the acquisition-state SP, pick the next batch, evaluate the target.

    The pre-round `SurrogateDistribution` wraps the acquisition-state
    emulator + form. The weighted-empirical factory produces an SP with
    ``emulator=None``; acquisitions that need a real emulator check
    ``state.surrogate_distribution.emulator is None`` and raise.

    Returns ``(x_new, y_new_raw)``: the new batch and its raw target
    evaluations. ``y_new_raw`` is at the *un-transformed* base target —
    output transformation to `current_state` happens in
    `_append_round_evaluations`.
    """
    pre_round_posterior = _build_surrogate_distribution(
        algorithm.surrogate_distribution_factory,
        emulator=emulator_for_acq,
        X=X,
        Y=Y_train_for_acq,
        log_density_form=round_state.target_intermediate.log_density_form,
        problem=problem,
    )
    acq_state = AcquisitionState(
        problem=problem,
        surrogate_distribution=pre_round_posterior,
        X=X,
        Y_raw=Y_raw,
        Y_train=Y_train_for_acq,
        tempering_state=round_state.current_state,
        target_tempering_state=round_state.target_state,
    )
    x_new = algorithm.acquisition.select_batch(acq_state, algorithm.q, key)
    y_new_raw = problem.target_map(x_new)
    return x_new, y_new_raw


def _append_round_evaluations(
    X: Array,
    Y_raw: Array,
    x_new: Array,
    y_new_raw: Array,
    round_state: RoundState,
) -> tuple[Array, Array, Array, Array]:
    """Append the new batch and re-materialize ``Y_train`` at ``current_state``.

    Returns ``(X, Y_raw, Y_train, y_new_at_current)``:

    - ``X``, ``Y_raw``: full design + raw evaluations after appending
      the round's ``q`` new rows.
    - ``Y_train``: the full transformed dataset at ``current_state`` —
      what the round-end emulator will be (re)fit against.
    - ``y_new_at_current``: only the new rows, transformed to
      ``current_state``. Passed into `_plan_round_update` so the
      cheap-update fast path can append rather than recomputing the
      whole transformed dataset.
    """
    X = jnp.concatenate([X, x_new], axis=0)
    Y_raw = jnp.concatenate([Y_raw, y_new_raw], axis=0)
    Y_train = round_state.current_intermediate.output_transform(
        round_state.current_state, X, Y_raw
    )
    y_new_at_current = round_state.current_intermediate.output_transform(
        round_state.current_state, x_new, y_new_raw
    )
    return X, Y_raw, Y_train, y_new_at_current


def _update_round_end_emulator(
    emulator: Emulator,
    emulator_state: Any,
    round_state: RoundState,
    X: Array,
    Y_train: Array,
    x_new: Array,
    y_new_at_current: Array,
    algorithm: Algorithm,
) -> tuple[Emulator, Any]:
    """Fit / cheap-update the round-end emulator at ``current_state``.

    Builds the `EmulatorUpdate` plan that combines the transform's
    structural diff (``emulator_state → current_state``) with the
    new-rows append, then dispatches via `update_emulator`. The
    dispatcher tries any registered fast paths and falls back to a full
    refit on the new ``(X, Y_train)``.

    Returns ``(emulator, emulator_state)``; ``emulator_state`` is now
    ``round_state.current_state``.
    """
    round_plan = _plan_round_update(
        round_state.current_intermediate.output_transform,
        state_prev=emulator_state,
        state_new=round_state.current_state,
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
    return emulator, round_state.current_state


def _build_round_metrics_row(
    *,
    scheduled: Sequence[ScheduledMetric],
    algorithm: Algorithm,
    problem: Problem,
    target: TargetDistribution,
    current_intermediate: IntermediateTarget,
    emulator: Emulator,
    X: Array,
    Y_raw: Array,
    Y_train: Array,
    current_state: Any,
    target_state: Any,
    round_idx: int,
    key: Array,
) -> dict[str, Any]:
    """Run scheduled metrics that fire this round and attach bookkeeping fields.

    SP/estimate construction inside `_eval_round` is lazy — when no
    metric fires at this round, nothing is built. Bookkeeping fields
    (``round``, ``tempering_state``, ``target_tempering_state``,
    ``n_evals``) are always populated, so every round still emits a
    row.

    `target_state=None` is the round-0 convention (no acquisition runs
    at round 0, hence no target state).
    """
    firing = tuple(s for s in scheduled if s.fires_at_round(round_idx))
    row = _eval_round(
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
        key=key,
    )
    row["round"] = round_idx
    row["tempering_state"] = current_state
    row["target_tempering_state"] = target_state
    row["n_evals"] = int(X.shape[0])
    return row


def _run_initial_round(
    problem: Problem,
    algorithm: Algorithm,
    scheduled: Sequence[ScheduledMetric],
    target: TargetDistribution,
    key_init: Array,
    key_metric: Array,
) -> tuple[Array, Array, Array, Emulator, Any, Any, dict[str, Any]]:
    """Round 0: initial design, target eval, emulator fit, round-0 metrics row.

    No acquisition runs at round 0 — the design is drawn directly via
    ``algorithm.initial_sampler``. The round operates at
    ``schedule.at(0)``'s state; ``Y_train`` is derived from ``Y_raw``
    via that state's `output_transform` (identity under no tempering).

    Returns
    -------
    ``(X, Y_raw, Y_train, emulator, emulator_state, state_0, round_0_metrics)``.
    The metrics row carries ``target_tempering_state=None`` since no
    acquisition ran.
    """
    X = algorithm.initial_sampler.sample(problem, key_init, algorithm.n_initial)
    Y_raw = problem.target_map(X)

    state_0, _ = algorithm.schedule.at(0)
    target_0 = algorithm.tempering_scheme.intermediate_target(target, state_0)
    Y_train = target_0.output_transform(state_0, X, Y_raw)

    emulator = algorithm.emulator_factory()
    emulator = emulator.fit(X, Y_train)
    # Track the state of the emulator's last fit. Used as the "from"
    # state when building cheap-update plans in subsequent rounds.
    emulator_state: Any = state_0

    round_0_metrics = _build_round_metrics_row(
        scheduled=scheduled,
        algorithm=algorithm,
        problem=problem,
        target=target,
        current_intermediate=target_0,
        emulator=emulator,
        X=X,
        Y_raw=Y_raw,
        Y_train=Y_train,
        current_state=state_0,
        target_state=None,
        round_idx=0,
        key=key_metric,
    )
    return X, Y_raw, Y_train, emulator, emulator_state, state_0, round_0_metrics


def _run_final_eval(
    *,
    problem: Problem,
    emulator: Emulator,
    X: Array,
    Y_raw: Array,
    Y_train: Array,
    target: TargetDistribution,
    algorithm: Algorithm,
    scheduled: Sequence[ScheduledMetric],
    last_state: Any,
    key: Array,
) -> tuple[Distribution, dict[str, float]]:
    """Post-loop final eval at the un-tempered base form.

    Always builds the final SP + estimate (so
    ``RunResult.final_estimate`` is populated for downstream tooling
    regardless of whether any metric has ``final=True``). Then runs
    every scheduled metric whose ``final`` flag is set, with
    ``metric_target=TERMINAL`` in the context — independent of the
    metric's per-round ``target`` field, ``final_metrics`` is always
    evaluated against the un-tempered base.
    """
    final_sp = _build_surrogate_distribution(
        algorithm.surrogate_distribution_factory,
        emulator=emulator,
        X=X,
        Y=Y_train,
        log_density_form=target.log_density_form,
        problem=problem,
    )
    final_estimate = algorithm.estimator(final_sp)

    final_scheduled = tuple(s for s in scheduled if s.final)
    final_metrics: dict[str, float] = {}
    if final_scheduled:
        keys_split = jax.random.split(key, len(final_scheduled))
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
    return final_estimate, final_metrics


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

    Round shape (paper-style outline). Each acquisition round
    (``round_idx`` 1..n-1) executes these phases, each owned by a named
    helper:

    1. ``_resolve_round_state`` — current/target tempering states,
       both `IntermediateTarget`s, and the per-axis invariance flags,
       cached in a `RoundState`.
    2. ``_resolve_acquisition_view`` — the ``(emulator, Y_train)``
       pair the acquisition sees, possibly via a state-only cheap
       update when ``target_state != current_state``.
    3. ``_run_acquisition`` — build the pre-round
       `SurrogateDistribution` + `AcquisitionState`, pick the next
       ``q`` points, evaluate the raw target on them.
    4. ``_append_round_evaluations`` — append the new rows; rebuild
       ``Y_train`` at ``current_state`` (and the new-rows-only block
       used by the round-end fast path).
    5. ``_update_round_end_emulator`` — refit / cheap-update the
       round-end emulator at ``current_state``.
    6. ``_build_round_metrics_row`` — run scheduled metrics that fire
       this round; attach bookkeeping fields. Lazy: nothing built when
       no metric fires.

    Round 0 (initial design, no acquisition) is handled by
    `_run_initial_round`; the post-loop un-tempered final eval is
    handled by `_run_final_eval`.
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

    # Round 0: initial-design round.
    key_metric_0, key_loop = jax.random.split(key_loop, 2)
    (
        X,
        Y_raw,
        Y_train,
        emulator,
        emulator_state,
        state_0,
        round_0_metrics,
    ) = _run_initial_round(
        problem, algorithm, scheduled, target, key_init, key_metric_0
    )
    tempering_states: list[Any] = [state_0]
    per_round_metrics: list[dict[str, Any]] = [round_0_metrics]

    # Acquisition rounds: 1..n_rounds-1.
    for round_idx in range(1, algorithm.n_rounds):
        round_state = _resolve_round_state(algorithm, target, round_idx)
        emulator_for_acq, Y_train_for_acq = _resolve_acquisition_view(
            emulator, emulator_state, round_state, X, Y_raw, Y_train, algorithm
        )
        key_acq, key_metric, key_loop = jax.random.split(key_loop, 3)
        x_new, y_new_raw = _run_acquisition(
            problem, algorithm, emulator_for_acq,
            X, Y_raw, Y_train_for_acq, round_state, key_acq,
        )
        X, Y_raw, Y_train, y_new_at_current = _append_round_evaluations(
            X, Y_raw, x_new, y_new_raw, round_state
        )
        emulator, emulator_state = _update_round_end_emulator(
            emulator, emulator_state, round_state, X, Y_train,
            x_new, y_new_at_current, algorithm,
        )
        per_round_metrics.append(
            _build_round_metrics_row(
                scheduled=scheduled,
                algorithm=algorithm,
                problem=problem,
                target=target,
                current_intermediate=round_state.current_intermediate,
                emulator=emulator,
                X=X,
                Y_raw=Y_raw,
                Y_train=Y_train,
                current_state=round_state.current_state,
                target_state=round_state.target_state,
                round_idx=round_idx,
                key=key_metric,
            )
        )
        tempering_states.append(round_state.current_state)

    # Post-loop: final eval at the un-tempered base form.
    final_estimate, final_metrics = _run_final_eval(
        problem=problem,
        emulator=emulator,
        X=X,
        Y_raw=Y_raw,
        Y_train=Y_train,
        target=target,
        algorithm=algorithm,
        scheduled=scheduled,
        last_state=tempering_states[-1] if tempering_states else None,
        key=key_eval,
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
