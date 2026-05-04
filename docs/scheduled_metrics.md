# Scheduled metrics

Recipes for `ScheduledMetric` and `MetricTarget` — for the protocol
itself and the design rationale, see `docs/design.md` §4.7.

The full API:

```python
ScheduledMetric(
    metric: Metric,
    every: int = 1,                      # fire on rounds where round_idx % every == 0
    target: MetricTarget = CURRENT,       # CURRENT or TERMINAL intermediate
    final: bool = True,                   # also include in final_metrics
    name_suffix: str = "",                # appended as `_{name_suffix}` to output keys
)
```

`Algorithm.metrics` accepts both bare `Metric` instances (auto-wrapped at
default config — preserves the pre-#6 behavior) and explicit
`ScheduledMetric` wrappers.

## Default — every metric, every round, plus final eval

```python
from sabi.algorithms import Algorithm
from sabi.metrics import ReferenceMMD

alg = Algorithm(
    emulator_factory=...,
    acquisition=...,
    metrics=(ReferenceMMD(),),  # auto-wrapped: every=1, target=CURRENT, final=True
)
```

Equivalent explicit form:

```python
from sabi.metrics import ScheduledMetric, MetricTarget

alg = Algorithm(
    ...,
    metrics=(
        ScheduledMetric(
            metric=ReferenceMMD(),
            every=1,
            target=MetricTarget.CURRENT,
            final=True,
        ),
    ),
)
```

## Cost-budgeted: expensive metric every k rounds

When a metric is expensive (e.g. a large reference-sample MMD or a
metric that runs an inner MCMC), evaluating it every round wastes
budget. `every=k` fires at rounds `0, k, 2k, ...`:

```python
alg = Algorithm(
    ...,
    n_rounds=21,  # rounds 0..20
    metrics=(
        ScheduledMetric(
            metric=ReferenceMMD(n_estimate_samples=8192, n_reference_samples=8192),
            every=10,        # fires at rounds 0, 10, 20
            final=True,      # also fires at end-of-loop final eval
        ),
    ),
)
```

The `per_round_metrics` rows for non-firing rounds still exist (one
row per round), but won't contain the metric's keys.

## Tracking the terminal target throughout a tempered run

Under tempering, `CURRENT` evaluates against the round's intermediate
distribution (β = 0.1, 0.5, 1.0, ...). That means a round-2 MMD value
isn't directly comparable to a round-5 MMD value because they're
against different targets.

To track the same target throughout, use `target=TERMINAL` (evaluates
at the un-tempered base):

```python
from sabi.tempering.likelihood import LikelihoodTemperingViaForm
from sabi.tempering.schedule import FixedSchedule

alg = Algorithm(
    ...,
    tempering_scheme=LikelihoodTemperingViaForm(),
    schedule=FixedSchedule(states=(0.1, 0.3, 0.6, 1.0)),
    metrics=(
        ScheduledMetric(
            metric=ReferenceMMD(),
            target=MetricTarget.TERMINAL,
        ),
    ),
)
```

Now every per-round MMD is against the un-tempered base, so values are
directly comparable across rounds.

## Both targets at once: name_suffix

Want both per-round views (one against the intermediate, one against
the terminal) for diagnostic purposes? Two `ScheduledMetric`s with the
same underlying metric — but their dict keys would collide. Set
`name_suffix` on one to disambiguate:

```python
alg = Algorithm(
    ...,
    metrics=(
        ScheduledMetric(
            metric=ReferenceMMD(),
            target=MetricTarget.CURRENT,
            # bare keys: "mmd", "mmd2"
        ),
        ScheduledMetric(
            metric=ReferenceMMD(),
            target=MetricTarget.TERMINAL,
            name_suffix="terminal",
            # suffixed keys: "mmd_terminal", "mmd2_terminal"
        ),
    ),
)
```

If you forget the `name_suffix`, the loop refuses to silently overwrite
— `validate_metric_keys` raises before any emulator work happens with
a message naming the colliding key and pointing at `name_suffix`.

## End-of-run only: large `every` + `final=True`

Some metrics make sense only at the end of a run:

```python
alg = Algorithm(
    ...,
    metrics=(
        ScheduledMetric(
            metric=ExpensiveCalibrationMetric(),
            every=10**9,    # only round 0 fires per-round (since 0 % anything == 0)
            final=True,     # plus the final eval
        ),
    ),
)
```

This evaluates at round 0 (which gives a baseline against the initial
design) and at end-of-loop. Nothing in between.

If you don't want the round 0 firing either, see follow-up #1
(adaptive `MetricSchedule`) — `every` always includes round 0 by
construction.

## Surrogate-aware metric

Beyond reference-comparison metrics, `MetricContext` exposes the full
`SurrogateDistribution` so metrics can inspect the emulator and form
directly:

```python
from dataclasses import dataclass
from typing import ClassVar

from sabi.metrics import Metric, MetricContext


@dataclass(frozen=True)
class DesignSize(Metric):
    """Trivial example — record the number of design points each round."""
    keys: ClassVar[tuple[str, ...]] = ("n_design",)

    def __call__(self, ctx: MetricContext, *, key):
        # ctx.surrogate_distribution carries the round's emulator + form;
        # ctx.X is the design data the emulator was fit on.
        return {"n_design": float(ctx.X.shape[0])}
```

Use `ctx.surrogate_distribution.emulator` for log-score / calibration
metrics; use `ctx.Y_raw` and `ctx.Y_train` to access raw vs
emulator-training targets.

## YAML / Hydra

Bare metric (back-compat):

```yaml
metrics:
  - name: reference_mmd
```

Scheduled metric with all fields:

```yaml
metrics:
  - name: reference_mmd
    every: 5
    target: terminal     # or "current" (default)
    final: true
    name_suffix: terminal
```

Mix of bare and scheduled in the same list works — bare entries are
auto-wrapped at run time.
