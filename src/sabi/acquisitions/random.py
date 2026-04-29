"""Sampling-based acquisitions.

`PriorSampling` draws the next batch directly from a `BatchSampler`
(default `PriorSampler`, which samples from `problem.prior`). It does not
have a score; it is not a `PointwiseScoredAcquisition`. Useful as a
baseline and as one component of a future `MixtureSampling` acquisition
(v1.5+).

v1.5 will add `PosteriorThompsonSampling` and `MixtureSampling` here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jax import Array

from sabi.acquisitions.base import Acquisition, AcquisitionState
from sabi.sampling import BatchSampler, PriorSampler


@dataclass(frozen=True)
class PriorSampling(Acquisition):
    """Draw `q` samples from a `BatchSampler` (default ``PriorSampler``).

    With the default sampler this returns ``X ~ problem.prior^q`` of shape
    ``(q,) + problem.input_shape``. Swap ``sampler`` to use Sobol, LHS,
    or any other `BatchSampler` strategy.
    """

    sampler: BatchSampler = field(default_factory=PriorSampler)

    def select_batch(self, state: AcquisitionState, q: int, key: Array) -> Array:
        return self.sampler.sample(state.problem, key, q)
