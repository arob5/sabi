"""Sampling-based acquisitions.

`PriorSampling` draws the next batch directly from the problem's design
distribution (`problem.prior`). It does not have a score; it is not a
`PointwiseScoredAcquisition`. Useful as a baseline and as one component
of a future `MixtureSampling` acquisition (v1.5+).

v1.5 will add `PosteriorThompsonSampling` and `MixtureSampling` here.
"""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array

from sabi.acquisitions.base import Acquisition, AcquisitionState
from sabi.initial_designs.base import sample_initial


@dataclass(frozen=True)
class PriorSampling(Acquisition):
    """Draw `q` i.i.d. samples from ``problem.prior`` (the design distribution).

    For ``problem.prior`` a ``Distribution`` over ``problem.input_shape``,
    returns ``X ~ problem.prior^q`` of shape ``(q,) + problem.input_shape``.
    """

    def select_batch(self, state: AcquisitionState, q: int, key: Array) -> Array:
        return sample_initial(state.problem, key, q)
