"""Per-metric scheduling and target selection.

`ScheduledMetric` wraps a `Metric` with a per-round firing schedule
(`every`), a target-state selection (`target`), an end-of-loop
inclusion flag (`final`), and an optional name suffix for output
keys (`name_suffix`).

`Algorithm.metrics` accepts both bare `Metric` instances (auto-wrapped
at default config — preserves the previous "every metric, every
round, against current target" behavior) and explicit
`ScheduledMetric` wrappers.

`MetricTarget.CURRENT` resolves to the round's intermediate
distribution (built via `tempering_scheme.intermediate_target` at
the round's tempering state). `MetricTarget.TERMINAL` resolves to
the un-tempered base target — i.e. `target.log_density_form` with
the emulator at its current fit, which is what the loop uses for
the post-loop final evaluation. This deliberately preserves the
existing "always have a reference row at the un-tempered base
regardless of where the schedule ended" guarantee.

Collision detection runs *before* the loop starts: the
`validate_metric_keys` helper enumerates every round and every
`final=True` slot, and raises if two scheduled metrics would
produce the same dict key. Metrics that leave `keys = ()` (default)
are skipped by the upfront check; the loop's merge step still
catches their collisions at runtime as a backstop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from sabi.metrics.base import Metric


class MetricTarget(Enum):
    """Which tempering state's intermediate distribution a metric evaluates against.

    `CURRENT` (default): the round's current state. The metric sees
    the surrogate distribution built at the round's intermediate
    target — what the existing per-round metric path uses.

    `TERMINAL`: the un-tempered base target. The metric sees a
    surrogate distribution built with `target.log_density_form` and
    the round's emulator. Useful when the user wants metric values
    comparable across rounds even when the round's intermediate is
    changing — e.g. tracking final-target MMD throughout a tempered
    run rather than only at the end.

    `NEXT` is intentionally absent — it makes sense for acquisitions
    (look-ahead) but not for metrics, which evaluate quality at a
    realised state.
    """

    CURRENT = "current"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class ScheduledMetric:
    """Wraps a `Metric` with per-round scheduling and target selection.

    Attributes:
        metric: the underlying `Metric` to invoke.
        every: fire on rounds where `round_idx % every == 0` (so
            `every=1` fires every round including round 0;
            `every=5` fires at rounds 0, 5, 10, ...). Must be `>= 1`.
        target: which tempering state's intermediate distribution to
            evaluate against — `MetricTarget.CURRENT` (default) or
            `MetricTarget.TERMINAL`.
        final: also include at the post-loop final evaluation (always
            at the un-tempered base). Default `True` — preserves the
            previous "every metric also runs at end-of-loop"
            behavior.
        name_suffix: appended as `_{name_suffix}` to each returned
            dict key when non-empty. Use this to disambiguate when
            the same metric is scheduled at multiple targets in the
            same row.
    """

    metric: Metric
    every: int = 1
    target: MetricTarget = MetricTarget.CURRENT
    final: bool = True
    name_suffix: str = ""

    def __post_init__(self) -> None:
        if self.every < 1:
            raise ValueError(
                f"ScheduledMetric.every must be >= 1, got {self.every!r}."
            )

    def fires_at_round(self, round_idx: int) -> bool:
        """True iff this scheduled metric fires at `round_idx`.

        Round 0 (initial-design round) is included by default
        (`every=1`); `every=k` fires at rounds 0, k, 2k, ....
        """
        return round_idx >= 0 and round_idx % self.every == 0

    def output_keys(self) -> tuple[str, ...]:
        """Keys this scheduled metric will produce, with `name_suffix` applied.

        Empty when the underlying metric declares `keys = ()` — i.e.
        opts out of upfront validation.
        """
        if not self.metric.keys:
            return ()
        if not self.name_suffix:
            return tuple(self.metric.keys)
        return tuple(f"{k}_{self.name_suffix}" for k in self.metric.keys)

    def apply_suffix(self, d: dict[str, float]) -> dict[str, float]:
        """Apply `name_suffix` to every key in `d`. No-op when empty."""
        if not self.name_suffix:
            return d
        return {f"{k}_{self.name_suffix}": v for k, v in d.items()}


def normalize_metrics(
    metrics: Sequence[Metric | ScheduledMetric],
) -> tuple[ScheduledMetric, ...]:
    """Wrap bare `Metric` instances in `ScheduledMetric(defaults)`. Idempotent.

    Bare metrics get the default schedule: `every=1, target=CURRENT,
    final=True, name_suffix=""` — which matches the previous
    "every metric, every round, also at final eval" behavior.
    """
    return tuple(
        m if isinstance(m, ScheduledMetric) else ScheduledMetric(metric=m)
        for m in metrics
    )


# Bookkeeping keys the loop adds to every per-round metric row. Metrics
# colliding with these are caught upfront with a clear error.
_BOOKKEEPING_ROW_KEYS = frozenset(
    {"round", "tempering_state", "target_tempering_state", "n_evals"}
)


def validate_metric_keys(
    scheduled: Sequence[ScheduledMetric],
    n_rounds: int,
) -> None:
    """Validate that no two scheduled metrics produce the same dict key.

    Two checks run before the loop starts:

    1. Per-round collision: for each round `0..n_rounds-1`, compute
       the firing scheduled metrics and verify their `output_keys()`
       don't overlap with each other or with the bookkeeping keys
       (`round`, `tempering_state`, `target_tempering_state`,
       `n_evals`) that the loop populates per-row.
    2. Final-eval collision: across all `final=True` scheduled
       metrics, verify their `output_keys()` don't overlap.

    Metrics with empty `keys` (the default) are skipped — they
    can't be validated up front. The runtime merge still detects
    their collisions and raises.

    Raises:
        ValueError: with a message naming the colliding key, the two
            metrics that produced it, and a pointer at
            `ScheduledMetric.name_suffix` for resolution.
    """
    # 1. Per-round collisions.
    for round_idx in range(n_rounds):
        seen: dict[str, str] = {k: "<bookkeeping>" for k in _BOOKKEEPING_ROW_KEYS}
        for s in scheduled:
            if not s.fires_at_round(round_idx):
                continue
            for k in s.output_keys():
                if k in seen:
                    raise ValueError(
                        f"Metric key collision at round {round_idx}: "
                        f"key {k!r} would be produced by both "
                        f"{seen[k]} and {s!r}. Set "
                        f"`ScheduledMetric.name_suffix` on one to "
                        f"disambiguate."
                    )
                seen[k] = repr(s)

    # 2. final_metrics collisions.
    seen_final: dict[str, str] = {}
    for s in scheduled:
        if not s.final:
            continue
        for k in s.output_keys():
            if k in seen_final:
                raise ValueError(
                    f"final_metrics key collision: key {k!r} would be "
                    f"produced by both {seen_final[k]} and {s!r}. Set "
                    f"`ScheduledMetric.name_suffix` on one to "
                    f"disambiguate."
                )
            seen_final[k] = repr(s)
