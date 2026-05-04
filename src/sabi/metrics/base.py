"""`Metric` protocol — flexible per-round measurement, distributions in,
named scalars out.

A metric is a callable taking a `MetricContext` (bundling the round's
posterior estimate, the surrogate distribution that produced it, the
problem, the design data, and the round's tempering state) plus a JAX
PRNG `key`, and returning a dict of named scalars. The dict lets
related quantities stay bundled (e.g. MMD² and its square root,
forward+reverse KL from a single sample pair) without forcing the
caller to invoke multiple metrics for one computation.

Two kinds of metrics fit naturally under this single protocol:

- **Reference-comparison metrics** (the dominant family): compute a
  distance / divergence between `ctx.estimate` and
  `ctx.problem.reference_distribution`. `MMD` is the example.
- **Surrogate-quality metrics**: read from `ctx.surrogate_distribution`
  (carrying the round's emulator + form) and possibly the design data
  to compute calibration / log-score quantities. These don't depend
  on a reference distribution.

Future `EmulatorMetric`-style metrics (per-point validation log-score)
also fit here — `ctx.surrogate_distribution.emulator` is exposed.

**Protocol declaration (`requires`).** Each metric class declares which
ProbPipe `Supports*` protocols it requires from `ctx.estimate` via the
`requires` class attribute. The loop introspects this and **raises** a
clear `MissingProtocolError` when the estimate doesn't satisfy a
metric's needs (rather than silently skipping). Example: a metric that
needs samples declares `requires = (SupportsSampling,)`. Metrics that
read from `ctx.surrogate_distribution` instead of `ctx.estimate` can
do their own internal protocol checks.

**Key declaration (`keys`).** Each metric class declares the dict keys
it produces via the `keys` class attribute. The loop uses this to
detect collisions (two metrics producing the same key in the same
row) **before the run starts**, raising with a pointer at
`ScheduledMetric.name_suffix`. Metrics with input-dependent keys can
leave `keys = ()` (the default) to opt out of upfront validation; a
runtime check then catches collisions at merge time as a backstop.

Returned-dict keys are merged into the per-round metric row;
namespace your keys when a collision is possible across metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from jax import Array
from probpipe.core._distribution_base import Distribution

from sabi.problems.base import Problem

if TYPE_CHECKING:
    from sabi.surrogate.surrogate_distribution import SurrogateDistribution


@dataclass(frozen=True)
class MetricContext:
    """Per-evaluation bundle passed to each `Metric`.

    Attributes:
        estimate: the estimator's output `Distribution[Array]` —
            satisfies every protocol in the metric's `requires`.
            Built from `surrogate_distribution` via
            `algorithm.estimator`.
        surrogate_distribution: the round's `SurrogateDistribution`
            (carries the fitted emulator and the round's
            log-density form). Available for metrics that need to
            inspect the surrogate directly. May have
            `emulator=None` when the loop is running the
            weighted-empirical baseline.
        problem: the inference problem (provides `prior`,
            `reference_distribution`, `support`, `input_shape`,
            etc.).
        X: design inputs, shape `(n,) + problem.input_shape`.
        Y_raw: raw target evaluations, shape
            `(n,) + problem.output_shape`.
        Y_train: emulator-training targets at
            `tempering_state` — what the surrogate's emulator was
            actually fit on. Equals `Y_raw` when no target-axis
            tempering is in effect.
        tempering_state: the round's tempering state (the schedule's
            state at this round). `None` for untempered loops.
        round_idx: integer round index. `0` is the initial-design
            round; `1, 2, ...` are acquisition rounds.
        metric_target: which target the surrogate distribution and
            estimate were built at — `MetricTarget.CURRENT` (the
            round's intermediate distribution) or
            `MetricTarget.TERMINAL` (the un-tempered base). Lets
            metrics record / branch on the target without the
            wrapper having to thread it separately.
    """

    estimate: Distribution
    surrogate_distribution: "SurrogateDistribution"
    problem: Problem
    X: Array
    Y_raw: Array
    Y_train: Array
    tempering_state: Any
    round_idx: int
    metric_target: Any  # `MetricTarget` — typed as Any to avoid the import cycle


class Metric(ABC):
    """Base class for sabi metrics.

    Subclasses implement `__call__(ctx, *, key)` and declare:

    - `requires`: tuple of ProbPipe `Supports*` protocol classes that
      `ctx.estimate` must satisfy. The loop checks each and raises
      `MissingProtocolError` if any are not satisfied.
    - `keys`: tuple of dict keys this metric will produce. Used for
      upfront collision detection. Default `()` opts out of upfront
      validation (runtime collision check still applies).
    """

    requires: ClassVar[tuple[type, ...]] = ()
    keys: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def __call__(
        self,
        ctx: MetricContext,
        *,
        key: Array,
    ) -> dict[str, float]:
        """Compute named scalar metrics for the round's evaluation context.

        Args:
            ctx: per-evaluation bundle (estimate, surrogate, problem,
                design data, round/state metadata). See
                `MetricContext`.
            key: JAX PRNG key for any sampling the metric does
                internally.

        Returns:
            Dict mapping metric name → scalar. Keys returned should
            be a subset of `self.keys` (when `keys` is non-empty);
            the loop validates this contract.
        """


class MissingProtocolError(TypeError):
    """Raised when a `Metric`'s required protocols aren't satisfied
    by the posterior estimate."""
