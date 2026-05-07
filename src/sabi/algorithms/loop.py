"""Sequential-acquisition loop body.

`run()` executes one full algorithm run end-to-end. The `Algorithm`
and `RunResult` dataclasses live in :mod:`sabi.algorithms.algorithm`;
SP factories live in :mod:`sabi.algorithms.surrogate_distribution_factory`.

Composition (per round): initial design via
``probpipe.sample(algorithm.initial_design_distribution, ...)``
→ emulator → acquisition → `SurrogateDistribution` → estimator (default
`expected_target`) → scheduled metrics. See ``docs/design.md`` §4.14
for the loop sketch and ``docs/notation.md`` for shape / symbol
conventions.

File layout: `RoundState` (per-round value type) → `_resolve_run_inputs`
(default-fill helper) → `run()` (public entry) → round-lifecycle
helpers (`_run_initial_round`, `_run_final_eval`) → per-round
orchestration helpers in calling order → inner machinery (metric eval,
SP construction, update planning).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from probpipe import sample as pp_sample
from probpipe.core._distribution_base import Distribution
from probpipe.core.constraints import Constraint, _Interval

from sabi._probpipe_compat import independent_uniform
from sabi.acquisitions.base import (
    AcquisitionState,
    resolve_state,
)
from sabi.algorithms.algorithm import Algorithm, RunResult
from sabi.algorithms.surrogate_distribution_factory import SurrogateDistributionFactory
from sabi.density_decomposition import DensityDecomposition
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
from sabi.target_distribution import IntermediateTarget, TargetDistribution
from sabi.tempering.base import InvarianceFlags
from sabi.tempering.output_transform import OutputTransform


# ---------------------------------------------------------------------------
# Per-round value type.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundState:
    """Per-round bundle of intermediates + per-state decomposition + invariance flags.

    Models an **acquisition round** (rounds 1..n_rounds-1). Round 0
    (initial design) has no acquisition target and is handled by
    `_run_initial_round` directly without constructing a `RoundState`.

    Cached once at the top of each acquisition round (via
    `_resolve_round_state`) and threaded through the helpers that make
    up the round body.

    Attributes:
        round_idx: 1-based round index (round 0 is handled separately).
        current_intermediate: `IntermediateTarget` at the round's own
            tempering state. Owns ``output_transform`` (used for the
            round-end refit / cheap-update plan).
        target_intermediate: `IntermediateTarget` at the state the
            acquisition optimizes against. Owns the ``output_transform``
            used to materialize the acquisition's view
            (``Y_train_for_acq``). Equals ``current_intermediate`` when
            ``invariance.both`` is True.
        current_decomposition: per-state effective ``DensityDecomposition``
            at the current state. Used to build the round-end SP for
            metrics.
        target_decomposition: per-state effective ``DensityDecomposition``
            at the acquisition's target state. Equals
            ``current_decomposition`` when ``invariance.both`` is True.
        invariance: per-axis flags from
            ``algorithm.tempering_scheme.invariance(current_state, target_state)``.
            Drives the look-ahead-vs-reuse decision in
            `_resolve_acquisition_view`.
    """

    round_idx: int
    current_intermediate: IntermediateTarget
    target_intermediate: IntermediateTarget
    current_decomposition: DensityDecomposition
    target_decomposition: DensityDecomposition
    invariance: InvarianceFlags


# ---------------------------------------------------------------------------
# Resolver for `Algorithm`'s nullable fields.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ResolvedAlgorithm:
    """Algorithm with all nullable fields resolved to concrete values."""

    decomposition: DensityDecomposition
    initial_design_distribution: Distribution
    x_support: Constraint


def _resolve_run_inputs(algorithm: Algorithm, target: TargetDistribution) -> _ResolvedAlgorithm:
    """Fill `Algorithm`'s nullable fields from the target's defaults.

    Per ``docs/density_decomposition.md`` §3.3:

    - ``x_support`` falls back to ``target.support``.
    - ``initial_design_distribution`` falls back to
      ``Uniform(x_support)`` if ``x_support`` is a bounded interval;
      otherwise raises with a pointer to both fields.
    - ``density_decomposition`` raises if ``None``.
    """
    if algorithm.density_decomposition is None:
        raise ValueError(
            "Algorithm.density_decomposition is required. Construct one "
            "via `DensityDecomposition.identity_from_target(...)` for "
            "log-density emulation, or "
            "`DensityDecomposition.likelihood_with_prior(...)` for "
            "log-likelihood + prior emulation."
        )
    x_support = (
        algorithm.x_support if algorithm.x_support is not None else target.support
    )
    if algorithm.initial_design_distribution is not None:
        initial_design_distribution = algorithm.initial_design_distribution
    elif isinstance(x_support, _Interval):
        low = jnp.asarray(x_support.low)
        high = jnp.asarray(x_support.high)
        if low.ndim == 0:
            # Promote a scalar interval to a length-1 box for the
            # 1-D parameter case. The shim expects array bounds.
            low = low[None]
            high = high[None]
        initial_design_distribution = independent_uniform(
            low=low, high=high, name="default_initial_design"
        )
    else:
        raise ValueError(
            "Algorithm.initial_design_distribution is None and the "
            "default fallback (`Uniform(x_support)`) is not available "
            f"for x_support of type {type(x_support).__name__}. Pass "
            "`Algorithm(initial_design_distribution=...)` explicitly, "
            "or set `Algorithm.x_support` to a bounded interval."
        )
    return _ResolvedAlgorithm(
        decomposition=algorithm.density_decomposition,
        initial_design_distribution=initial_design_distribution,
        x_support=x_support,
    )


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def run(problem: Problem, algorithm: Algorithm, key: Array) -> RunResult:
    """Run the sequential emulator-based inference loop.

    State is explicit and flat. The emulator is re-fit each round on
    the full ``(X, Y_train)`` unless a registered cheap-update handler
    accepts the round's plan (see :mod:`sabi.emulators.dispatch`).

    Tempering integration: each round, the ``tempering_scheme`` produces
    an :class:`IntermediateTarget` at the round's state plus a per-state
    effective :class:`DensityDecomposition`. ``Y_train`` is derived
    from cached ``Y_raw`` via the intermediate's ``output_transform``;
    the round's decomposition is the per-state effective one (used to
    build the `SurrogateDistribution`). Under `NoTempering` (default),
    these are identity / unchanged from the base.

    Acquisition target: ``algorithm.acquisition_target`` selects which
    state the acquisition optimizes against (`CURRENT`, `NEXT`,
    `TERMINAL`). When this differs from the round's ``current_state``,
    the loop builds a separate look-ahead `IntermediateTarget` /
    `DensityDecomposition` and may refit the emulator on the
    look-ahead-state's training data before the acquisition runs.

    Per-round metric rows: the loop emits one ``per_round_metrics`` row
    per round (``0..n_rounds-1``), populated with bookkeeping fields
    (``round``, ``tempering_state``, ``target_tempering_state``,
    ``n_evals``) plus values from any `ScheduledMetric` firing that
    round. Metrics with ``final=True`` also evaluate at the post-loop
    final step against the un-tempered base target; results land in
    ``RunResult.final_metrics``.
    """
    # Normalize and validate metrics up front — *before* resolving
    # algorithm defaults — so a key-collision in the metric list lands
    # ahead of the (less directly actionable) "density_decomposition is
    # required" guard.
    scheduled = normalize_metrics(algorithm.metrics)
    validate_metric_keys(scheduled, algorithm.n_rounds)

    target = problem.target_distribution
    resolved = _resolve_run_inputs(algorithm, target)
    base_decomposition = resolved.decomposition
    # Refill the algorithm with the resolved values so any downstream
    # consumer (acquisition / optimizer) reads the concrete fields
    # rather than seeing ``None``.
    algorithm = replace(
        algorithm,
        density_decomposition=resolved.decomposition,
        initial_design_distribution=resolved.initial_design_distribution,
        x_support=resolved.x_support,
    )

    key_init, key_loop, key_eval = jax.random.split(key, 3)

    # Round 0: initial-design round.
    key_metric_0, key_loop = jax.random.split(key_loop, 2)
    (
        X,
        Y_raw,
        Y_train,
        emulator,
        emulator_state,
        round_0_metrics,
    ) = _run_initial_round(
        problem=problem,
        algorithm=algorithm,
        scheduled=scheduled,
        target=target,
        base_decomposition=base_decomposition,
        initial_design_distribution=resolved.initial_design_distribution,
        x_support=resolved.x_support,
        key_init=key_init,
        key_metric=key_metric_0,
    )
    tempering_states: list[Any] = [emulator_state]
    per_round_metrics: list[dict[str, Any]] = [round_0_metrics]

    # Acquisition rounds: 1..n_rounds-1.
    for round_idx in range(1, algorithm.n_rounds):
        key_acq, key_metric, key_loop = jax.random.split(key_loop, 3)
        round_state = _resolve_round_state(
            algorithm, target, base_decomposition, round_idx
        )
        emulator_for_acq, Y_train_for_acq = _resolve_acquisition_view(
            emulator=emulator,
            emulator_state=emulator_state,
            round_state=round_state,
            X=X,
            Y_raw=Y_raw,
            Y_train=Y_train,
            algorithm=algorithm,
        )
        x_new, y_new_raw = _run_acquisition(
            problem=problem,
            algorithm=algorithm,
            base_decomposition=base_decomposition,
            x_support=resolved.x_support,
            emulator_for_acq=emulator_for_acq,
            X=X,
            Y_raw=Y_raw,
            Y_train_for_acq=Y_train_for_acq,
            round_state=round_state,
            key=key_acq,
        )
        X, Y_raw, Y_train, y_new_at_current = _append_round_evaluations(
            X, Y_raw, x_new, y_new_raw, round_state
        )
        emulator, emulator_state = _update_round_end_emulator(
            emulator=emulator,
            emulator_state=emulator_state,
            round_state=round_state,
            X=X,
            Y_train=Y_train,
            x_new=x_new,
            y_new_at_current=y_new_at_current,
            algorithm=algorithm,
        )
        per_round_metrics.append(
            _build_round_metrics_row(
                scheduled=scheduled,
                algorithm=algorithm,
                problem=problem,
                target=target,
                base_decomposition=base_decomposition,
                round_state=round_state,
                emulator=emulator,
                X=X,
                Y_raw=Y_raw,
                Y_train=Y_train,
                key=key_metric,
            )
        )
        tempering_states.append(round_state.current_intermediate.state)

    # Post-loop: final eval at the un-tempered base form.
    final_estimate, final_metrics = _run_final_eval(
        problem=problem,
        emulator=emulator,
        X=X,
        Y_raw=Y_raw,
        Y_train=Y_train,
        target=target,
        base_decomposition=base_decomposition,
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


# ---------------------------------------------------------------------------
# Lifecycle helpers (round 0, post-loop final eval).
# ---------------------------------------------------------------------------


def _run_initial_round(
    *,
    problem: Problem,
    algorithm: Algorithm,
    scheduled: Sequence[ScheduledMetric],
    target: TargetDistribution,
    base_decomposition: DensityDecomposition,
    initial_design_distribution: Distribution,
    x_support: Constraint,  # noqa: ARG001 — kept for symmetry / future use
    key_init: Array,
    key_metric: Array,
) -> tuple[Array, Array, Array, Emulator, Any, dict[str, Any]]:
    """Round 0: initial design, target eval, emulator fit, round-0 metrics row.

    Initial design is drawn directly via
    ``probpipe.sample(initial_design_distribution, ...)``. The round
    operates at ``schedule.at(0)``'s state; ``Y_train`` is derived from
    ``Y_raw`` via that state's ``output_transform`` (identity under no
    tempering).

    Returns ``(X, Y_raw, Y_train, emulator, emulator_state, round_0_metrics)``.
    """
    X = jnp.asarray(
        pp_sample(initial_design_distribution, key=key_init, sample_shape=(algorithm.n_initial,))
    )
    Y_raw = base_decomposition.target_map(X)

    state_0, _ = algorithm.schedule.at(0)
    target_0 = algorithm.tempering_scheme.intermediate_target(target, state_0)
    decomposition_0 = algorithm.tempering_scheme.intermediate_decomposition(
        base_decomposition, state_0
    )
    Y_train = target_0.output_transform(state_0, X, Y_raw)

    emulator = algorithm.emulator_factory()
    emulator = emulator.fit(X, Y_train)
    emulator_state: Any = state_0

    firing = tuple(s for s in scheduled if s.fires_at_round(0))
    round_0_metrics = _eval_round(
        scheduled_firing=firing,
        algorithm=algorithm,
        problem=problem,
        target=target,
        base_decomposition=base_decomposition,
        current_intermediate=target_0,
        current_decomposition=decomposition_0,
        emulator=emulator,
        X=X,
        Y_raw=Y_raw,
        Y_train=Y_train,
        tempering_state=state_0,
        round_idx=0,
        key=key_metric,
    )
    round_0_metrics["round"] = 0
    round_0_metrics["tempering_state"] = state_0
    round_0_metrics["target_tempering_state"] = None
    round_0_metrics["n_evals"] = int(X.shape[0])
    return X, Y_raw, Y_train, emulator, emulator_state, round_0_metrics


def _run_final_eval(
    *,
    problem: Problem,
    emulator: Emulator,
    X: Array,
    Y_raw: Array,
    Y_train: Array,
    target: TargetDistribution,
    base_decomposition: DensityDecomposition,
    algorithm: Algorithm,
    scheduled: Sequence[ScheduledMetric],
    last_state: Any,
    key: Array,
) -> tuple[Distribution, dict[str, float]]:
    """Post-loop final eval at the un-tempered base decomposition."""
    final_sp = _build_surrogate_distribution(
        algorithm.surrogate_distribution_factory,
        emulator=emulator,
        X=X,
        Y=Y_train,
        decomposition=base_decomposition,
        problem=problem,
    )
    final_estimate = algorithm.estimator(final_sp)

    final_scheduled = tuple(s for s in scheduled if s.final)
    if not final_scheduled:
        return final_estimate, {}

    final_metrics = _run_metrics_against_pair(
        metrics=final_scheduled,
        keys=jax.random.split(key, len(final_scheduled)),
        surrogate_distribution=final_sp,
        estimate=final_estimate,
        metric_target=MetricTarget.TERMINAL,
        problem=problem,
        X=X,
        Y_raw=Y_raw,
        Y_train=Y_train,
        tempering_state=last_state,
        round_idx=algorithm.n_rounds,
    )
    return final_estimate, final_metrics


# ---------------------------------------------------------------------------
# Round-orchestration helpers (called once per acquisition round, in order).
# ---------------------------------------------------------------------------


def _resolve_round_state(
    algorithm: Algorithm,
    target: TargetDistribution,
    base_decomposition: DensityDecomposition,
    round_idx: int,
) -> RoundState:
    """Resolve all round-level state up front."""
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
    current_decomposition = algorithm.tempering_scheme.intermediate_decomposition(
        base_decomposition, current_state
    )
    invariance = algorithm.tempering_scheme.invariance(current_state, target_state)
    if invariance.both:
        target_intermediate = current_intermediate
        target_decomposition = current_decomposition
    else:
        target_intermediate = algorithm.tempering_scheme.intermediate_target(
            target, target_state
        )
        target_decomposition = algorithm.tempering_scheme.intermediate_decomposition(
            base_decomposition, target_state
        )
    return RoundState(
        round_idx=round_idx,
        current_intermediate=current_intermediate,
        target_intermediate=target_intermediate,
        current_decomposition=current_decomposition,
        target_decomposition=target_decomposition,
        invariance=invariance,
    )


def _resolve_acquisition_view(
    *,
    emulator: Emulator,
    emulator_state: Any,
    round_state: RoundState,
    X: Array,
    Y_raw: Array,
    Y_train: Array,
    algorithm: Algorithm,
) -> tuple[Emulator, Array]:
    """Return the `(emulator, Y_train)` pair the acquisition sees."""
    if round_state.invariance.target_map:
        return emulator, Y_train
    target_intermediate = round_state.target_intermediate
    Y_train_for_acq = target_intermediate.output_transform(
        target_intermediate.state, X, Y_raw
    )
    lookahead_plan = _plan_round_update(
        target_intermediate.output_transform,
        state_prev=emulator_state,
        state_new=target_intermediate.state,
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
    *,
    problem: Problem,
    algorithm: Algorithm,
    base_decomposition: DensityDecomposition,
    x_support: Constraint,
    emulator_for_acq: Emulator,
    X: Array,
    Y_raw: Array,
    Y_train_for_acq: Array,
    round_state: RoundState,
    key: Array,
) -> tuple[Array, Array]:
    """Build the acquisition-state surrogate, pick the next batch, evaluate the target."""
    pre_round_posterior = _build_surrogate_distribution(
        algorithm.surrogate_distribution_factory,
        emulator=emulator_for_acq,
        X=X,
        Y=Y_train_for_acq,
        decomposition=round_state.target_decomposition,
        problem=problem,
    )
    acq_state = AcquisitionState(
        problem=problem,
        algorithm=algorithm,
        surrogate_distribution=pre_round_posterior,
        X=X,
        Y_raw=Y_raw,
        Y_train=Y_train_for_acq,
        x_support=x_support,
    )
    x_new = algorithm.acquisition.select_batch(acq_state, algorithm.q, key)
    y_new_raw = base_decomposition.target_map(x_new)
    return x_new, y_new_raw


def _append_round_evaluations(
    X: Array,
    Y_raw: Array,
    x_new: Array,
    y_new_raw: Array,
    round_state: RoundState,
) -> tuple[Array, Array, Array, Array]:
    """Append the new batch and re-materialize ``Y_train`` at ``current_state``."""
    X = jnp.concatenate([X, x_new], axis=0)
    Y_raw = jnp.concatenate([Y_raw, y_new_raw], axis=0)
    current_intermediate = round_state.current_intermediate
    Y_train = current_intermediate.output_transform(
        current_intermediate.state, X, Y_raw
    )
    y_new_at_current = current_intermediate.output_transform(
        current_intermediate.state, x_new, y_new_raw
    )
    return X, Y_raw, Y_train, y_new_at_current


def _update_round_end_emulator(
    *,
    emulator: Emulator,
    emulator_state: Any,
    round_state: RoundState,
    X: Array,
    Y_train: Array,
    x_new: Array,
    y_new_at_current: Array,
    algorithm: Algorithm,
) -> tuple[Emulator, Any]:
    """Fit / cheap-update the round-end emulator at ``current_state``."""
    current_intermediate = round_state.current_intermediate
    round_plan = _plan_round_update(
        current_intermediate.output_transform,
        state_prev=emulator_state,
        state_new=current_intermediate.state,
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
    return emulator, current_intermediate.state


def _build_round_metrics_row(
    *,
    scheduled: Sequence[ScheduledMetric],
    algorithm: Algorithm,
    problem: Problem,
    target: TargetDistribution,
    base_decomposition: DensityDecomposition,
    round_state: RoundState,
    emulator: Emulator,
    X: Array,
    Y_raw: Array,
    Y_train: Array,
    key: Array,
) -> dict[str, Any]:
    """Run scheduled metrics that fire this round and attach bookkeeping fields."""
    current_intermediate = round_state.current_intermediate
    current_decomposition = round_state.current_decomposition
    round_idx = round_state.round_idx
    firing = tuple(s for s in scheduled if s.fires_at_round(round_idx))
    row = _eval_round(
        scheduled_firing=firing,
        algorithm=algorithm,
        problem=problem,
        target=target,
        base_decomposition=base_decomposition,
        current_intermediate=current_intermediate,
        current_decomposition=current_decomposition,
        emulator=emulator,
        X=X,
        Y_raw=Y_raw,
        Y_train=Y_train,
        tempering_state=current_intermediate.state,
        round_idx=round_idx,
        key=key,
    )
    row["round"] = round_idx
    row["tempering_state"] = current_intermediate.state
    row["target_tempering_state"] = round_state.target_intermediate.state
    row["n_evals"] = int(X.shape[0])
    return row


# ---------------------------------------------------------------------------
# Inner machinery: metric eval, SP construction, update planning, utilities.
# ---------------------------------------------------------------------------


def _eval_round(
    *,
    scheduled_firing: Sequence[ScheduledMetric],
    algorithm: Algorithm,
    problem: Problem,
    target: TargetDistribution,  # noqa: ARG001 — kept for context plumbing
    base_decomposition: DensityDecomposition,
    current_intermediate: IntermediateTarget,  # noqa: ARG001
    current_decomposition: DensityDecomposition,
    emulator: Emulator,
    X: Array,
    Y_raw: Array,
    Y_train: Array,
    tempering_state: Any,
    round_idx: int,
    key: Array,
) -> dict[str, float]:
    """Evaluate a round's firing scheduled metrics against current/terminal estimates."""
    if not scheduled_firing:
        return {}

    def _current() -> tuple[SurrogateDistribution, Distribution]:
        surrogate_distribution = _build_surrogate_distribution(
            algorithm.surrogate_distribution_factory,
            emulator=emulator,
            X=X,
            Y=Y_train,
            decomposition=current_decomposition,
            problem=problem,
        )
        return surrogate_distribution, algorithm.estimator(surrogate_distribution)

    def _terminal() -> tuple[SurrogateDistribution, Distribution]:
        surrogate_distribution = _build_surrogate_distribution(
            algorithm.surrogate_distribution_factory,
            emulator=emulator,
            X=X,
            Y=Y_train,
            decomposition=base_decomposition,
            problem=problem,
        )
        return surrogate_distribution, algorithm.estimator(surrogate_distribution)

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


def _run_metrics_against_pair(
    *,
    metrics: Sequence[ScheduledMetric],
    keys: Sequence[Array],
    surrogate_distribution: SurrogateDistribution,
    estimate: Distribution,
    metric_target: MetricTarget,
    problem: Problem,
    X: Array,
    Y_raw: Array,
    Y_train: Array,
    tempering_state: Any,
    round_idx: int,
) -> dict[str, float]:
    """Run each metric in `metrics` against a fixed `(surrogate_distribution, estimate, metric_target)`."""
    if not metrics:
        return {}
    merged: dict[str, float] = {}
    for s, mkey in zip(metrics, keys, strict=True):
        _check_protocols(s.metric, estimate)
        ctx = MetricContext(
            estimate=estimate,
            surrogate_distribution=surrogate_distribution,
            problem=problem,
            X=X,
            Y_raw=Y_raw,
            Y_train=Y_train,
            tempering_state=tempering_state,
            round_idx=round_idx,
            metric_target=metric_target,
        )
        out = s.apply_suffix(s.metric(ctx, key=mkey))
        _merge_no_overwrite(merged, out, repr(s))
    return merged


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
    """Run each scheduled metric at its target."""
    if not scheduled:
        return {}

    keys = jax.random.split(key, len(scheduled))
    by_target: dict[MetricTarget, list[tuple[ScheduledMetric, Array]]] = {}
    for s, mkey in zip(scheduled, keys, strict=True):
        by_target.setdefault(s.target, []).append((s, mkey))

    factories: dict[
        MetricTarget,
        Callable[[], tuple[SurrogateDistribution, Distribution]],
    ] = {
        MetricTarget.CURRENT: current_estimate_fn,
        MetricTarget.TERMINAL: terminal_estimate_fn,
    }

    merged: dict[str, float] = {}
    for target, pairs in by_target.items():
        try:
            factory = factories[target]
        except KeyError as e:
            raise ValueError(f"Unknown MetricTarget: {target!r}") from e
        surrogate_distribution, estimate = factory()
        out = _run_metrics_against_pair(
            metrics=[s for s, _ in pairs],
            keys=[mkey for _, mkey in pairs],
            surrogate_distribution=surrogate_distribution,
            estimate=estimate,
            metric_target=target,
            problem=problem,
            X=X,
            Y_raw=Y_raw,
            Y_train=Y_train,
            tempering_state=tempering_state,
            round_idx=round_idx,
        )
        _merge_no_overwrite(merged, out, f"metrics with target={target.name}")

    return merged


def _build_surrogate_distribution(
    factory: SurrogateDistributionFactory,
    *,
    emulator: Emulator,
    X: Array,
    Y: Array,
    decomposition: DensityDecomposition,
    problem: Problem,
) -> SurrogateDistribution:
    """Adapter: extract the math primitives from `Problem` and call the factory."""
    target = problem.target_distribution
    return factory(
        emulator=emulator,
        X=X,
        Y=Y,
        decomposition=decomposition,
        support=target.support,
        input_shape=target.input_shape,
        problem_name=problem.name,
    )


def _plan_round_update(
    transform: OutputTransform,
    state_prev: Any,
    state_new: Any,
    X_new: Array | None,
    Y_new_at_new_state: Array | None,
) -> EmulatorUpdate | None:
    """Build the round's `EmulatorUpdate` plan, or ``None`` to refit."""
    diff = transform.diff(state_prev, state_new)
    has_new_rows = X_new is not None and X_new.shape[0] > 0
    if diff is None:
        return None
    if isinstance(diff, RescaleOutputs):
        if diff.factor == 1.0:
            if has_new_rows:
                return AppendRows(X_new=X_new, Y_new=Y_new_at_new_state)
            return None
        if has_new_rows:
            return RescaleThenAppend(
                factor=diff.factor, X_new=X_new, Y_new=Y_new_at_new_state
            )
        return diff
    return diff


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
    """Merge `src` into `dst`, raising on key collision."""
    overlap = dst.keys() & src.keys()
    if overlap:
        raise ValueError(
            f"Metric key collision: keys {sorted(overlap)} produced by "
            f"{source_repr} are already present in this row. Set "
            f"`ScheduledMetric.name_suffix` or rename keys to disambiguate."
        )
    dst.update(src)
