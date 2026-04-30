"""`Algorithm` and `RunResult` — the composition object and the loop's output.

`Algorithm` is the dataclass that bundles every component the loop
needs to run end-to-end (emulator factory, acquisition, sampler,
tempering scheme + schedule, etc.). `RunResult` is what `run` returns.

The loop body itself lives in :mod:`sabi.algorithms.loop`. Factories
producing `SurrogatePosterior` instances live in
:mod:`sabi.algorithms.surrogate_posterior_factory`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from jax import Array
from probpipe.core._distribution_base import Distribution

from sabi.acquisitions.base import Acquisition, AcquisitionTarget
from sabi.algorithms.surrogate_posterior_factory import (
    SurrogatePosteriorFactory,
    emulator_pushforward_factory,
)
from sabi.emulators.base import Emulator
from sabi.metrics.base import PosteriorMetric
from sabi.posterior.estimators import expected_target
from sabi.posterior.surrogate_posterior import SurrogatePosterior
from sabi.sampling import BatchSampler, PriorSampler
from sabi.tempering.base import NoTempering, TemperingScheme
from sabi.tempering.schedule import TemperingSchedule, UntemperedSchedule


@dataclass(frozen=True)
class Algorithm:
    """Composition of the components needed to run the loop."""

    emulator_factory: Callable[[], Emulator]
    acquisition: Acquisition
    n_initial: int = 16
    n_rounds: int = 10
    q: int = 1
    initial_sampler: BatchSampler = field(default_factory=PriorSampler)
    tempering_scheme: TemperingScheme = field(default_factory=NoTempering)
    schedule: TemperingSchedule = field(default_factory=UntemperedSchedule)
    acquisition_target: AcquisitionTarget = AcquisitionTarget.CURRENT
    surrogate_posterior_factory: SurrogatePosteriorFactory = emulator_pushforward_factory
    estimator: Callable[[SurrogatePosterior], Distribution] = expected_target
    metrics: tuple[PosteriorMetric, ...] = ()


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
