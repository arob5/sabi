"""Sampling-based acquisitions.

`PriorSampling` (alias `Random`) draws the next batch directly from the
problem's design distribution (`problem.prior`). It does not have a
score; it is not a `PointwiseScoredAcquisition`. Useful as a baseline and
as one component of a future `MixtureSampling` acquisition (v1.5+).

v1.5 will add `PosteriorThompsonSampling` and `MixtureSampling` here.
"""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array

from sabi.acquisitions.base import Acquisition, AcquisitionState
from sabi.initial_designs.base import sample_initial


@dataclass(frozen=True)
class PriorSampling(Acquisition):
    """Draw `q` samples from `problem.prior` (the design distribution).

    Equivalent to the v1.x `Random` acquisition. `Random` is retained as
    an alias for config / import-path backwards compatibility.
    """

    def select_batch(self, state: AcquisitionState, q: int, key: Array) -> Array:
        return sample_initial(state.problem, key, q)


# Backwards-compatibility alias.
Random = PriorSampling
