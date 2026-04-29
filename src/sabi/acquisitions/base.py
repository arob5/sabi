"""Acquisition interface.

A `PosteriorMetric`-style class that picks the next batch of `(q,) +
input_shape` evaluation points each round of the loop. Concrete acquisitions
fall into three families:

1. **Pointwise-scored** (`PointwiseScoredAcquisition`) — provides a scalar
   `score(x, state)` per single point of shape `input_shape`. Optimization
   is delegated to a pluggable `PointwiseOptimizer` (candidate-set,
   continuous multi-start, or greedy multi-point with fantasy imputation).
2. **Sampling-style** (`SamplingAcquisition`) — picks the batch by drawing
   from a distribution (the prior, the current posterior estimate, a
   mixture, etc.). v1.4 ships only `PriorSampling` (the v1.x `Random`).
   Posterior Thompson sampling and mixture sampling land in v1.5+.
3. **Batch-scored** (`BatchScoredAcquisition`) — provides a scalar
   `score_batch(X, state)` over a joint `q`-batch. q-EI, max-min entropy,
   etc. Not implemented in v1.4; the abstraction is structured to slot in
   a parallel `BatchOptimizer` hierarchy when this lands (v1.5+).

The acquisition sees the current `Surrogate` plus an `AcquisitionState`
bundling everything it might need: the problem, the current design set
`(X, Y)`, the post-tempering `current_form`, and the current
`tempering_state` (opaque PyTree — see `docs/notation.md`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jax import Array

from sabi.problems.base import Problem
from sabi.problems.forms import LogDensityForm
from sabi.surrogates.base import Surrogate

if TYPE_CHECKING:
    from sabi.acquisitions.optim import PointwiseOptimizer


@dataclass(frozen=True)
class AcquisitionState:
    """Per-round bundle passed to acquisitions. All fields are read-only."""

    problem: Problem
    surrogate: Surrogate
    X: Array  # (n,) + input_shape
    Y: Array  # (n,) + output_shape
    current_form: LogDensityForm  # form after tempering.apply
    tempering_state: Any  # opaque PyTree; None for untempered


class Acquisition(ABC):
    """Top-level acquisition: picks the next batch of evaluation points."""

    @abstractmethod
    def select_batch(
        self,
        state: AcquisitionState,
        q: int,
        key: Array,
    ) -> Array:
        """Return `(q,) + problem.input_shape` parameter locations to evaluate next."""


class PointwiseScoredAcquisition(Acquisition, ABC):
    """Acquisition that scores single points and uses a `PointwiseOptimizer`.

    Subclasses implement `score(x, state) -> scalar`. The `optimizer` field
    chooses how the score is searched: candidate-set (default; cheap),
    continuous multi-start (gradient-based, more accurate), or greedy
    multi-point (for `q > 1` with in-batch diversity via a `FantasyImputer`).

    `select_batch` is implemented by delegation: the acquisition asks its
    optimizer to find `q` points maximizing `score`.
    """

    optimizer: "PointwiseOptimizer"

    @abstractmethod
    def score(self, x: Array, state: AcquisitionState) -> Array:
        """Score at a single point of shape `input_shape`. Returns scalar.
        Higher is more desirable to evaluate next."""

    def select_batch(self, state: AcquisitionState, q: int, key: Array) -> Array:
        return self.optimizer.optimize(self, state, q, key)
