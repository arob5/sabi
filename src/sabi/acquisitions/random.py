"""Sampling-based acquisitions.

`DistributionSampling` draws the next batch directly from a ``Distribution``
chosen per call by ``distribution_from_state`` (default: the
algorithm's ``initial_design_distribution``). It does not have a score;
it is not a `PointwiseScoredAcquisition`. Useful as a baseline and as
one component of a future `MixtureSampling` acquisition.

`PosteriorThompsonSampling` and `MixtureSampling` are tracked as
follow-ups.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array
from probpipe import sample as pp_sample
from probpipe.core._distribution_base import Distribution

from sabi.acquisitions.base import Acquisition, AcquisitionState


def _initial_design_from_state(state: AcquisitionState) -> Distribution:
    """Default ``distribution_from_state``: read the algorithm's initial design."""
    dist = state.algorithm.initial_design_distribution
    if dist is None:
        raise ValueError(
            "DistributionSampling: state.algorithm.initial_design_distribution "
            "is None. Pass an explicit `distribution_from_state=...` callable "
            "or set `Algorithm.initial_design_distribution` (or `x_support`)."
        )
    return dist


@dataclass(frozen=True)
class DistributionSampling(Acquisition):
    """Sample ``q`` points from a ``Distribution`` resolved per call.

    ``distribution_from_state`` is called per round and returns the
    ``Distribution`` to sample from — typically the algorithm's
    ``initial_design_distribution``, the current surrogate, or a
    mixture. For complex strategies (clustering, Stein thinning, ...)
    implement ``Acquisition`` directly.

    Args:
        distribution_from_state: callable mapping ``AcquisitionState`` to
            the ``Distribution`` to sample from. Default reads
            ``state.algorithm.initial_design_distribution``.
    """

    distribution_from_state: Callable[[AcquisitionState], Distribution] = field(
        default=_initial_design_from_state,
    )

    def select_batch(self, state: AcquisitionState, q: int, key: Array) -> Array:
        dist = self.distribution_from_state(state)
        return jnp.asarray(pp_sample(dist, key=key, sample_shape=(q,)))
