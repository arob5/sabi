"""`Algorithm` and `RunResult` — the composition object and the loop's output.

`Algorithm` is the dataclass that bundles every component the loop
needs to run end-to-end (emulator factory, acquisition, density
decomposition, tempering scheme + schedule, etc.). `RunResult` is what
`run` returns.

The loop body itself lives in :mod:`sabi.algorithms.loop`. Factories
producing `SurrogateDistribution` instances live in
:mod:`sabi.algorithms.surrogate_distribution_factory`.

Per-role distribution fields after the ``DensityDecomposition`` split
(issue #65):

- ``density_decomposition`` — required at run time. The
  ``(target_single, output_shape, link, shift)`` bundle the loop and
  surrogate-distribution factory consume.
- ``initial_design_distribution`` — sampled (via ``probpipe.sample``)
  to draw the initial design and as the default for the random
  acquisitions / pointwise-optimizer candidate sets. Falls back to
  ``Uniform(x_support)`` at run time when ``x_support`` is bounded.
- ``x_support`` — the search region for acquisition reparameterization
  bijectors. Falls back to ``problem.target_distribution.support`` at
  run time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from jax import Array
from probpipe.core._distribution_base import Distribution
from probpipe.core.constraints import Constraint

from sabi.acquisitions.base import Acquisition, AcquisitionTarget
from sabi.algorithms.surrogate_distribution_factory import (
    SurrogateDistributionFactory,
    emulator_pushforward_factory,
)
from sabi.density_decomposition import DensityDecomposition
from sabi.emulators.base import Emulator
from sabi.metrics.base import Metric
from sabi.metrics.scheduling import ScheduledMetric
from sabi.surrogate.estimators import expected_target
from sabi.surrogate.surrogate_distribution import SurrogateDistribution
from sabi.tempering.base import NoTempering, TemperingScheme
from sabi.tempering.schedule import TemperingSchedule, UntemperedSchedule


@dataclass(frozen=True)
class Algorithm:
    """Composition of the components needed to run the loop.

    Attributes:
        emulator_factory: zero-arg factory returning a fresh
            ``Emulator``. Called once per round (cheap-update paths
            avoid re-fits where possible).
        acquisition: the ``Acquisition`` that picks the next batch each
            round.
        density_decomposition: required ``DensityDecomposition`` —
            ``(target_single, output_shape, link, shift)``. The loop
            calls ``decomposition.target_map(X)`` for design
            evaluations and threads the decomposition through the
            surrogate-distribution factory. Typed ``Optional`` for
            future-proofing; ``run`` raises if ``None``.
        initial_design_distribution: ``Distribution`` over the parameter
            space, sampled (via ``probpipe.sample``) for the initial
            design and as the default for random acquisitions /
            pointwise-optimizer candidate sets. Falls back to
            ``Uniform(x_support)`` at run time when ``x_support`` is a
            bounded interval; otherwise ``run`` raises.
        x_support: ``Constraint`` defining the acquisition's search
            region. Falls back to ``problem.target_distribution.support``
            at run time. Typically equal to the target's support; a
            distinct value lets the acquisition's reparameterization
            box restrict to a sub-region of the parameter space.
        n_initial: size of the initial-design batch.
        n_rounds: number of acquisition rounds (counts round 0).
        q: per-round acquisition batch size.
        tempering_scheme: ``TemperingScheme`` producing per-state
            intermediates and per-state effective decompositions.
            Default ``NoTempering``.
        schedule: ``TemperingSchedule`` driving the per-round state.
            Default ``UntemperedSchedule``.
        acquisition_target: which tempering state the acquisition
            optimizes against (``CURRENT`` / ``NEXT`` / ``TERMINAL``).
        surrogate_distribution_factory: per-round
            ``SurrogateDistribution`` factory.
        estimator: posterior estimate constructor;
            ``surrogate_distribution -> Distribution``.
        metrics: scheduled metrics evaluated each round / at the end.
    """

    emulator_factory: Callable[[], Emulator]
    acquisition: Acquisition
    density_decomposition: DensityDecomposition | None = None
    initial_design_distribution: Distribution | None = None
    x_support: Constraint | None = None
    n_initial: int = 16
    n_rounds: int = 10
    q: int = 1
    tempering_scheme: TemperingScheme = field(default_factory=NoTempering)
    schedule: TemperingSchedule = field(default_factory=UntemperedSchedule)
    acquisition_target: AcquisitionTarget = AcquisitionTarget.CURRENT
    surrogate_distribution_factory: SurrogateDistributionFactory = emulator_pushforward_factory
    estimator: Callable[[SurrogateDistribution], Distribution] = expected_target
    metrics: tuple[ScheduledMetric | Metric, ...] = ()


@dataclass
class RunResult:
    X: Array
    Y_raw: Array
    Y_train: Array
    emulator: Emulator
    tempering_states: list[Any]
    per_round_metrics: list[dict[str, Any]]
    final_estimate: Distribution | None
    final_metrics: dict[str, float]
