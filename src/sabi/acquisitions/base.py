"""Acquisition interface.

An `Acquisition` picks the next batch of `(q,) + input_shape` evaluation
points each round of the loop. Concrete acquisitions fall into three
families:

1. **Pointwise-scored** (`PointwiseScoredAcquisition`) — provides a
   scalar `_score_single(x, state)` per single point of shape
   `input_shape`. Optimization is delegated to a pluggable
   `PointwiseOptimizer` (candidate-set, continuous multi-start, or
   greedy multi-point with fantasy imputation).
2. **Sampling-style** (e.g. `PriorSampling`) — picks the batch by
   drawing from a distribution (the prior, the current posterior
   estimate, a mixture, etc.). v1.4 ships only `PriorSampling`;
   posterior Thompson sampling and mixture sampling land in v1.5+.
3. **Batch-scored** (`BatchScoredAcquisition`) — provides a scalar
   `score_batch(X, state)` over a joint `q`-batch. q-EI, max-min
   entropy, etc. Not implemented in v1.4; the abstraction is
   structured to slot in a parallel `BatchOptimizer` hierarchy when
   this lands (v1.5+).

The acquisition sees the current `SurrogatePosterior` (carrying the
round's surrogate and log-density form) plus an `AcquisitionState`
bundling the design data and tempering state. See the
`PointwiseScoredAcquisition` docstring for the scoring contract.

Shape and symbol conventions (`x`, `X`, `Y`, `q`, `n`, etc.) follow
``docs/notation.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
from jax import Array

from sabi.posterior.surrogate_posterior import SurrogatePosterior
from sabi.problems.base import Problem

if TYPE_CHECKING:
    from sabi.acquisitions.optim import PointwiseOptimizer


@dataclass(frozen=True)
class AcquisitionState:
    """Per-round bundle passed to acquisitions. All fields are read-only.

    The round's `SurrogatePosterior` carries the surrogate fit on the
    current design data plus the log-density form for the round.
    Acquisitions that need a real (non-degenerate) surrogate should check
    ``state.surrogate_posterior.surrogate is None`` — this is the case
    when the loop is running the weighted-empirical baseline (a
    `WeightedEmpiricalRandomMeasure`, which is a `SurrogatePosterior`
    subclass with ``surrogate=None``).

    Attributes:
        problem: the inference problem (provides `prior`, `support`,
            `input_shape`, etc.).
        surrogate_posterior: round's surrogate-posterior random measure
            (always set; ``surrogate_posterior.surrogate`` may be
            ``None`` for the weighted-empirical baseline).
        X: design inputs, shape `(n,) + problem.input_shape`.
        Y: design outputs, shape `(n,) + problem.output_shape`.
        tempering_state: opaque PyTree from the `TemperingSchedule`;
            `None` for untempered loops. Provided so acquisitions that
            care about the round's tempering state can read it directly
            without inferring from the SP.
    """

    problem: Problem
    surrogate_posterior: SurrogatePosterior
    X: Array
    Y: Array
    tempering_state: Any


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

    **Scoring contract.** Subclasses implement `_score_single(x, state)`,
    which returns a scalar where **higher is more desirable to evaluate
    next**. `x` has shape `state.problem.input_shape` (a single point).
    The default `score(X, state)` implementation `jax.vmap`s
    `_score_single` over the leading axis of `X`; override `score`
    directly when a true batched implementation is more efficient (e.g.,
    one surrogate call over the whole batch instead of `n` per-point
    queries).

    **JAX-traceability.** `_score_single` (or `score`) must be
    JAX-traceable for `ContinuousMultiStartOptimizer` to take its
    gradient. The candidate-set optimizer needs only forward evaluation.

    **Surrogate-protocol assumptions.** The score function reads from
    `state.surrogate_posterior.surrogate` (an `ArrayRandomFunction`)
    whatever predictive moments / samples it needs. See subclass
    docstrings for specifics. ProbPipe's `mean` and `variance` ops
    require `SupportsMean` / `SupportsVariance` — they do **not**
    auto-fall-back to MC at the op level. ProbPipe distributions wrapping
    TFP (`Normal`, `MultivariateNormal`, etc.) implement these by
    construction; custom non-TFP distributions need to opt in
    explicitly (e.g. via `@compute_expectation` on `_mean`).

    **Optimizer.** The `optimizer` field selects how `score` is searched.
    Default is `CandidateSetOptimizer()` (random candidates + top-q;
    cheap, gradient-free). Use `ContinuousMultiStartOptimizer()` for
    BFGS-refined argmaxes; wrap with `GreedyMultiPointOptimizer(inner=...)`
    for `q > 1` with in-batch diversity via a `FantasyImputer`.
    """

    optimizer: PointwiseOptimizer

    def _score_single(self, x: Array, state: AcquisitionState) -> Array:
        """Score at a single point of shape `state.problem.input_shape`.
        Returns scalar. Higher is more desirable.

        Subclasses override this to get a vmapped `score` for free, OR
        override `score` directly for a batched implementation. At least
        one of the two must be implemented.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement either `_score_single` "
            "(single-point) or `score` (batched)."
        )

    def score(self, X: Array, state: AcquisitionState) -> Array:
        """Batched score: shape `(n,) + input_shape` → `(n,)`.

        Default: `jax.vmap(_score_single)`. Override directly when a
        batched evaluation is more efficient (e.g., one surrogate call
        for the whole batch).
        """
        return jax.vmap(lambda x: self._score_single(x, state))(X)

    def select_batch(self, state: AcquisitionState, q: int, key: Array) -> Array:
        return self.optimizer.optimize(self, state, q, key)
