"""Tests for `ScheduledMetric`, `MetricTarget`, and the per-metric
scheduling integration in the loop.

Covers:

- Bare-metric back-compat (auto-wrapping with default schedule).
- `every` firing schedule (rounds 0, k, 2k, ...).
- `final` toggle for the post-loop terminal evaluation.
- `target=TERMINAL` mid-round produces different values than CURRENT
  under tempering.
- `MetricContext` exposes the surrogate distribution for
  surrogate-aware metrics.
- Upfront collision detection on declared keys + runtime backstop
  for metrics that opt out (`keys = ()`).
- `RunResult.final_estimate` is always populated (independent of
  whether any metric is `final=True`).
- `ScheduledMetric(every=0)` raises in `__post_init__`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import pytest
from probpipe.core._distribution_base import Distribution

from sabi.acquisitions.random import PriorSampling
from sabi.algorithms import Algorithm, run
from sabi.emulators import TinyGPEmulator
from sabi.metrics import (
    Metric,
    MetricContext,
    MetricTarget,
    ReferenceMMD,
    ScheduledMetric,
    normalize_metrics,
    validate_metric_keys,
)
from sabi.problems.gaussian import gaussian2d


def _algorithm(*, metrics, n_rounds=4, **kwargs):
    return Algorithm(
        emulator_factory=lambda: TinyGPEmulator(input_shape=(2,)),
        acquisition=PriorSampling(),
        n_initial=8,
        n_rounds=n_rounds,
        q=1,
        metrics=metrics,
        **kwargs,
    )


# -------------------------------------------------------------------------
# ScheduledMetric basics
# -------------------------------------------------------------------------


def test_scheduled_metric_every_zero_raises():
    with pytest.raises(ValueError, match="every must be >= 1"):
        ScheduledMetric(metric=ReferenceMMD(), every=0)


def test_scheduled_metric_negative_every_raises():
    with pytest.raises(ValueError, match="every must be >= 1"):
        ScheduledMetric(metric=ReferenceMMD(), every=-1)


def test_fires_at_round_default():
    s = ScheduledMetric(metric=ReferenceMMD())
    assert s.fires_at_round(0)
    assert s.fires_at_round(1)
    assert s.fires_at_round(5)


def test_fires_at_round_every_5():
    s = ScheduledMetric(metric=ReferenceMMD(), every=5)
    assert s.fires_at_round(0)
    assert not s.fires_at_round(1)
    assert not s.fires_at_round(4)
    assert s.fires_at_round(5)
    assert s.fires_at_round(10)


def test_output_keys_no_suffix():
    s = ScheduledMetric(metric=ReferenceMMD())
    assert s.output_keys() == ("mmd", "mmd2")


def test_output_keys_with_suffix():
    s = ScheduledMetric(metric=ReferenceMMD(), name_suffix="terminal")
    assert s.output_keys() == ("mmd_terminal", "mmd2_terminal")


def test_output_keys_empty_for_undeclared_metric():
    """A metric that opts out of upfront declaration (keys=()) returns
    an empty tuple — validate_metric_keys will skip it."""

    @dataclass(frozen=True)
    class Undeclared(Metric):
        def __call__(self, ctx, *, key):
            return {"x": 1.0}

    s = ScheduledMetric(metric=Undeclared(), name_suffix="suf")
    assert s.output_keys() == ()


def test_normalize_metrics_wraps_bare_metrics():
    bare = ReferenceMMD()
    sched = ScheduledMetric(metric=ReferenceMMD(), every=3)
    out = normalize_metrics([bare, sched])
    assert isinstance(out[0], ScheduledMetric)
    assert out[0].metric is bare
    assert out[0].every == 1
    assert out[0].target == MetricTarget.CURRENT
    assert out[0].final is True
    assert out[1] is sched


# -------------------------------------------------------------------------
# Upfront validation
# -------------------------------------------------------------------------


def test_validate_no_collisions_with_distinct_targets_and_no_suffix():
    """Two ScheduledMetrics on the same underlying ReferenceMMD,
    different targets, no name_suffix → keys collide upfront."""
    scheduled = normalize_metrics(
        [
            ScheduledMetric(metric=ReferenceMMD(), target=MetricTarget.CURRENT),
            ScheduledMetric(metric=ReferenceMMD(), target=MetricTarget.TERMINAL),
        ]
    )
    with pytest.raises(ValueError, match=r"key collision .* round 0.*name_suffix"):
        validate_metric_keys(scheduled, n_rounds=4)


def test_validate_collision_resolved_with_name_suffix():
    """Adding a name_suffix to one of the colliding metrics resolves the collision."""
    scheduled = normalize_metrics(
        [
            ScheduledMetric(metric=ReferenceMMD(), target=MetricTarget.CURRENT),
            ScheduledMetric(
                metric=ReferenceMMD(),
                target=MetricTarget.TERMINAL,
                name_suffix="terminal",
            ),
        ]
    )
    validate_metric_keys(scheduled, n_rounds=4)  # no raise


def test_validate_collision_with_bookkeeping_field_raises():
    """A metric whose declared key matches a bookkeeping field
    (round / tempering_state / target_tempering_state / n_evals)
    is rejected at upfront validation."""

    @dataclass(frozen=True)
    class _RoundClasher(Metric):
        keys: ClassVar[tuple[str, ...]] = ("round",)

        def __call__(self, ctx, *, key):
            return {"round": 0.0}

    scheduled = normalize_metrics([_RoundClasher()])
    with pytest.raises(ValueError, match=r"'round'.*<bookkeeping>"):
        validate_metric_keys(scheduled, n_rounds=2)


def test_validate_final_metrics_collision_raises():
    """The final_metrics upfront check is exercised in isolation here
    by passing `n_rounds=0` (per-round loop empty), leaving only the
    final-eval slot to detect the collision. In practice round 0
    always fires for any `every`, so a real-loop collision among
    `final=True` metrics with declared keys is caught by the
    per-round check first — this test is defense-in-depth for the
    final-only branch."""
    scheduled = normalize_metrics(
        [
            ScheduledMetric(metric=ReferenceMMD(), final=True),
            ScheduledMetric(metric=ReferenceMMD(), final=True),
        ]
    )
    with pytest.raises(ValueError, match=r"final_metrics key collision"):
        validate_metric_keys(scheduled, n_rounds=0)


def test_validate_skips_metrics_with_empty_keys():
    """Metrics that opt out of upfront declaration (keys=()) are
    skipped — runtime backstop catches their collisions."""

    @dataclass(frozen=True)
    class Undeclared(Metric):
        def __call__(self, ctx, *, key):
            return {}

    scheduled = normalize_metrics([Undeclared(), Undeclared()])
    validate_metric_keys(scheduled, n_rounds=4)  # no raise; runtime would catch


def test_collision_caught_before_loop_starts():
    """Upfront collision detection runs before any emulator work — verify
    by giving the loop a problem with a borked emulator factory and
    confirming the collision error fires first."""

    def bad_factory() -> TinyGPEmulator:
        raise RuntimeError("emulator factory should not be called")

    problem = gaussian2d()
    alg = Algorithm(
        emulator_factory=bad_factory,
        acquisition=PriorSampling(),
        n_initial=4,
        n_rounds=2,
        q=1,
        metrics=(
            ScheduledMetric(metric=ReferenceMMD(), target=MetricTarget.CURRENT),
            ScheduledMetric(metric=ReferenceMMD(), target=MetricTarget.TERMINAL),
        ),
    )
    with pytest.raises(ValueError, match=r"key collision"):
        run(problem, alg, jax.random.key(0))


# -------------------------------------------------------------------------
# Loop integration: round 0 firing + every-k schedule
# -------------------------------------------------------------------------


def test_round_0_row_present_with_no_metrics():
    """Even with metrics=(), the loop emits one row per round — round 0
    has bookkeeping fields but no metric values."""
    problem = gaussian2d()
    alg = _algorithm(metrics=(), n_rounds=3)
    result = run(problem, alg, jax.random.key(0))
    assert len(result.per_round_metrics) == 3
    row0 = result.per_round_metrics[0]
    assert row0["round"] == 0
    assert row0["n_evals"] == 8
    assert row0["target_tempering_state"] is None
    # No metric values.
    assert "mmd2" not in row0
    assert "mmd" not in row0


def test_bare_metric_back_compat_fires_every_round():
    """Passing a bare ReferenceMMD (auto-wrapped at default config)
    should fire on every round including round 0."""
    problem = gaussian2d()
    alg = _algorithm(metrics=(ReferenceMMD(n_estimate_samples=256, n_reference_samples=256),), n_rounds=4)
    result = run(problem, alg, jax.random.key(0))
    assert len(result.per_round_metrics) == 4
    for row in result.per_round_metrics:
        assert "mmd" in row
        assert "mmd2" in row
    assert "mmd2" in result.final_metrics


def test_every_k_fires_only_on_matching_rounds():
    """ScheduledMetric(every=3) with n_rounds=7 → fires at rounds 0, 3, 6."""
    problem = gaussian2d()
    alg = _algorithm(
        metrics=(
            ScheduledMetric(
                metric=ReferenceMMD(n_estimate_samples=256, n_reference_samples=256),
                every=3,
            ),
        ),
        n_rounds=7,
    )
    result = run(problem, alg, jax.random.key(0))
    assert len(result.per_round_metrics) == 7
    fired = [i for i, row in enumerate(result.per_round_metrics) if "mmd2" in row]
    assert fired == [0, 3, 6]


def test_final_false_excludes_from_final_metrics():
    """final=False keeps the metric out of the final eval."""
    problem = gaussian2d()
    alg = _algorithm(
        metrics=(
            ScheduledMetric(
                metric=ReferenceMMD(n_estimate_samples=256, n_reference_samples=256),
                every=1,
                final=False,
            ),
        ),
        n_rounds=3,
    )
    result = run(problem, alg, jax.random.key(0))
    # Per-round rows still get the metric.
    for row in result.per_round_metrics:
        assert "mmd2" in row
    # Final metrics empty.
    assert result.final_metrics == {}


def test_final_only_metric_via_large_every():
    """every=large, final=True: the metric appears at round 0 only
    (since only 0 % large == 0) plus final_metrics."""
    problem = gaussian2d()
    alg = _algorithm(
        metrics=(
            ScheduledMetric(
                metric=ReferenceMMD(n_estimate_samples=256, n_reference_samples=256),
                every=1000,
                final=True,
            ),
        ),
        n_rounds=4,
    )
    result = run(problem, alg, jax.random.key(0))
    assert "mmd2" in result.per_round_metrics[0]
    for row in result.per_round_metrics[1:]:
        assert "mmd2" not in row
    assert "mmd2" in result.final_metrics


# -------------------------------------------------------------------------
# Loop integration: target=TERMINAL mid-round under tempering
# -------------------------------------------------------------------------


def test_terminal_target_differs_from_current_under_tempering():
    """Under form-axis tempering with a non-trivial schedule, evaluating
    at CURRENT vs TERMINAL produces different metric values — sanity
    that the SP is built at a different log_density_form."""
    from sabi.tempering.likelihood import LikelihoodTemperingViaForm
    from sabi.tempering.schedule import FixedSchedule

    problem = gaussian2d()
    alg = Algorithm(
        emulator_factory=lambda: TinyGPEmulator(input_shape=(2,)),
        acquisition=PriorSampling(),
        n_initial=8,
        n_rounds=3,
        q=1,
        tempering_scheme=LikelihoodTemperingViaForm(),
        schedule=FixedSchedule(states=(0.1, 0.5, 1.0)),
        metrics=(
            ScheduledMetric(
                metric=ReferenceMMD(n_estimate_samples=256, n_reference_samples=256),
                target=MetricTarget.CURRENT,
            ),
            ScheduledMetric(
                metric=ReferenceMMD(n_estimate_samples=256, n_reference_samples=256),
                target=MetricTarget.TERMINAL,
                name_suffix="terminal",
            ),
        ),
    )
    result = run(problem, alg, jax.random.key(0))
    # Round 0 (state β=0.1): CURRENT and TERMINAL should differ noticeably.
    row0 = result.per_round_metrics[0]
    assert "mmd2" in row0
    assert "mmd2_terminal" in row0
    assert row0["mmd2"] != pytest.approx(row0["mmd2_terminal"], abs=1e-6)


# -------------------------------------------------------------------------
# MetricContext: surrogate-aware metric
# -------------------------------------------------------------------------


def test_surrogate_aware_metric_reads_surrogate_distribution():
    """A custom metric reading ctx.surrogate_distribution + design data
    runs cleanly inside the loop."""

    @dataclass(frozen=True)
    class DesignSizeMetric(Metric):
        keys: ClassVar[tuple[str, ...]] = ("n_design",)

        def __call__(self, ctx: MetricContext, *, key):
            # Reads ctx.X and (just to exercise it) ctx.surrogate_distribution.
            assert ctx.surrogate_distribution is not None
            return {"n_design": float(ctx.X.shape[0])}

    problem = gaussian2d()
    alg = _algorithm(metrics=(DesignSizeMetric(),), n_rounds=3)
    result = run(problem, alg, jax.random.key(0))
    # Round 0: 8 initial points; rounds 1, 2: 9, 10 (PriorSampling, q=1).
    assert [row["n_design"] for row in result.per_round_metrics] == [8.0, 9.0, 10.0]
    assert result.final_metrics["n_design"] == 10.0


# -------------------------------------------------------------------------
# RunResult.final_estimate is always populated
# -------------------------------------------------------------------------


def test_final_estimate_populated_with_no_metrics():
    """RunResult.final_estimate is built unconditionally, regardless of
    whether any metric is final=True."""
    problem = gaussian2d()
    alg = _algorithm(metrics=(), n_rounds=3)
    result = run(problem, alg, jax.random.key(0))
    assert isinstance(result.final_estimate, Distribution)


def test_final_estimate_populated_when_all_metrics_final_false():
    problem = gaussian2d()
    alg = _algorithm(
        metrics=(
            ScheduledMetric(
                metric=ReferenceMMD(n_estimate_samples=256, n_reference_samples=256),
                final=False,
            ),
        ),
        n_rounds=3,
    )
    result = run(problem, alg, jax.random.key(0))
    assert isinstance(result.final_estimate, Distribution)
    assert result.final_metrics == {}


# -------------------------------------------------------------------------
# Runtime collision backstop (for metrics with empty `keys`)
# -------------------------------------------------------------------------


def test_runtime_collision_backstop_for_undeclared_keys():
    """Two metrics with keys=() (so upfront validation skipped) that
    actually return the same key at runtime → loop raises at the
    merge step."""

    @dataclass(frozen=True)
    class UndeclaredA(Metric):
        # keys = () — opts out of upfront validation
        def __call__(self, ctx, *, key):
            return {"score": 1.0}

    @dataclass(frozen=True)
    class UndeclaredB(Metric):
        def __call__(self, ctx, *, key):
            return {"score": 2.0}

    problem = gaussian2d()
    alg = _algorithm(metrics=(UndeclaredA(), UndeclaredB()), n_rounds=2)
    with pytest.raises(ValueError, match=r"key collision.*'score'"):
        run(problem, alg, jax.random.key(0))


# -------------------------------------------------------------------------
# MetricContext.metric_target reflects the wrapper's target
# -------------------------------------------------------------------------


def test_metric_context_metric_target_reflects_wrapper():
    """The MetricContext.metric_target field tells the metric which
    target it's being evaluated against."""

    @dataclass(frozen=True)
    class TargetRecorder(Metric):
        keys: ClassVar[tuple[str, ...]] = ("target_value",)

        def __call__(self, ctx, *, key):
            return {"target_value": 1.0 if ctx.metric_target == MetricTarget.TERMINAL else 0.0}

    problem = gaussian2d()
    alg = _algorithm(
        metrics=(
            ScheduledMetric(metric=TargetRecorder(), target=MetricTarget.TERMINAL),
        ),
        n_rounds=2,
    )
    result = run(problem, alg, jax.random.key(0))
    for row in result.per_round_metrics:
        assert row["target_value"] == 1.0
    # final_metrics always evaluates at TERMINAL regardless.
    assert result.final_metrics["target_value"] == 1.0
