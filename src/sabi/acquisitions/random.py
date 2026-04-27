"""Random acquisition: draw `q` samples uniformly from the acquisition space.

Uses `problem.sampling_bounds`; prior-based sampling is selected if available
(mirrors the initial-design resolution rules).
"""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array

from sabi.acquisitions.base import Acquisition, AcquisitionState
from sabi.initial_designs.base import sample_initial


@dataclass(frozen=True)
class Random(Acquisition):
    def select_batch(self, state: AcquisitionState, q: int, key: Array) -> Array:
        return sample_initial(state.problem, key, q)
