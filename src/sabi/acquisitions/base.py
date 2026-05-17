"""Acquisition interface.

An `Acquisition` picks the next batch of `(q,) + input_shape` evaluation
points each round of the loop. Concrete acquisitions fall into three
families:

1. **Pointwise-scored** (`PointwiseScoredAcquisition`) — provides a
   scalar `_score_single(x, state)` per single point of shape
   `input_shape`. Optimization is delegated to a pluggable
   `PointwiseOptimizer` (candidate-set, continuous multi-start, or
   greedy multi-point with fantasy imputation).
2. **Sampling-style** (e.g. `DistributionSampling`) — picks the batch
   by drawing from a distribution (the algorithm's
   ``initial_design_distribution``, the current posterior estimate, a
   mixture, etc.).
3. **Batch-scored** (`BatchScoredAcquisition`) — provides a scalar
   `score_batch(X, state)` over a joint `q`-batch. Not implemented yet.

The acquisition sees the current `SurrogateDistribution` (carrying the
round's emulator and decomposition) plus an `AcquisitionState`
bundling the design data, the algorithm, and the round's search-region
support.

Shape and symbol conventions (`x`, `X`, `Y`, `q`, `n`, etc.) follow
``docs/notation.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import jax
from jax import Array
from probpipe.core.constraints import Constraint

from sabi.surrogate.surrogate_distribution import SurrogateDistribution
from sabi.problems.base import Problem

if TYPE_CHECKING:
    from sabi.acquisitions.optim import PointwiseOptimizer
    from sabi.algorithms.algorithm import Algorithm
    from sabi.tempering.schedule import TemperingSchedule


class AcquisitionTarget(Enum):
    """Which tempering state the acquisition optimizes against.

    In a tempered loop, three natural choices exist for the state at
    which the acquisition's `SurrogateDistribution` is built:

    - ``CURRENT``: state of the current round (``state_t``). Simplest;
      default. The acquisition sees the SP at the round's intermediate
      distribution.
    - ``NEXT``: state of the next round (``state_{t+1}``, clamped to
      terminal on the last round). Standard SMC-flavor look-ahead —
      acquisition picks points to inform the *next* intermediate.
    - ``TERMINAL``: the schedule's terminal state. Acquisition
      optimizes toward the final target throughout, regardless of
      where the schedule is.

    For untempered loops (``NoTempering`` + ``UntemperedSchedule``),
    all three collapse to the same state and the choice has no effect.
    """

    CURRENT = "current"
    NEXT = "next"
    TERMINAL = "terminal"


def resolve_state(
    acquisition_target: AcquisitionTarget,
    schedule: "TemperingSchedule",
    round_idx: int,
    current_state: Any,
) -> Any:
    """Map an `AcquisitionTarget` value to a concrete tempering state."""
    if acquisition_target == AcquisitionTarget.CURRENT:
        return current_state
    if acquisition_target == AcquisitionTarget.NEXT:
        next_state, _ = schedule.at(round_idx + 1)
        return next_state
    if acquisition_target == AcquisitionTarget.TERMINAL:
        return schedule.terminal_state()
    raise ValueError(f"Unknown AcquisitionTarget: {acquisition_target!r}")


@dataclass(frozen=True)
class AcquisitionState:
    """Per-round bundle passed to acquisitions. All fields are read-only.

    Attributes:
        problem: the inference problem; reach through
            ``problem.target_distribution`` for ``support`` and
            ``input_shape``.
        algorithm: the run-time ``Algorithm`` composition. Exposes
            ``density_decomposition``, ``initial_design_distribution``,
            and ``x_support`` (when set on the algorithm) for
            acquisition consumption.
        surrogate_distribution: round's surrogate random measure built
            at the acquisition's target tempering state.
        X: design inputs, shape
            ``(n,) + problem.target_distribution.input_shape``.
        Y_raw: raw target evaluations from
            ``decomposition.target_map(X)``.
        Y_train: emulator-training targets at the acquisition's target
            tempering state. Equal to ``Y_raw`` when no target-side
            tempering is in effect.
        x_support: the resolved acquisition search region —
            ``algorithm.x_support`` if set, otherwise
            ``problem.target_distribution.support``. Acquisitions read
            this directly rather than re-resolving from the algorithm
            field, so the loop's resolver runs once.
    """

    problem: Problem
    algorithm: "Algorithm"
    surrogate_distribution: SurrogateDistribution
    X: Array
    Y_raw: Array
    Y_train: Array
    x_support: Constraint


class Acquisition(ABC):
    """Top-level acquisition: picks the next batch of evaluation points."""

    @abstractmethod
    def select_batch(
        self,
        state: AcquisitionState,
        q: int,
        key: Array,
    ) -> Array:
        """Return ``(q,) + problem.target_distribution.input_shape`` parameter locations to evaluate next."""


class PointwiseScoredAcquisition(Acquisition, ABC):
    """Acquisition that scores single points and uses a `PointwiseOptimizer`.

    **Scoring contract.** Subclasses implement ``_score_single(x, state)``,
    which returns a scalar where **higher is more desirable to evaluate
    next**. ``x`` has shape ``state.problem.target_distribution.input_shape``
    (a single point). The default ``score(X, state)`` implementation
    ``jax.vmap``s ``_score_single`` over the leading axis of ``X``.

    **JAX-traceability.** ``_score_single`` (or ``score``) must be
    JAX-traceable for ``ContinuousMultiStartOptimizer`` to take its
    gradient. The candidate-set optimizer needs only forward evaluation.

    **Optimizer.** The ``optimizer`` field selects how ``score`` is
    searched. Default is ``CandidateSetOptimizer()``.
    """

    optimizer: "PointwiseOptimizer"

    def _score_single(self, x: Array, state: AcquisitionState) -> Array:
        """Score at a single point of shape `state.problem.target_distribution.input_shape`.
        Returns scalar. Higher is more desirable.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement either `_score_single` "
            "(single-point) or `score` (batched)."
        )

    def score(self, X: Array, state: AcquisitionState) -> Array:
        """Batched score: shape `(n,) + input_shape` → `(n,)`."""
        return jax.vmap(lambda x: self._score_single(x, state))(X)

    def select_batch(self, state: AcquisitionState, q: int, key: Array) -> Array:
        return self.optimizer.optimize(self, state, q, key)
