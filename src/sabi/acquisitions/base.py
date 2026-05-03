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

The acquisition sees the current `SurrogateDistribution` (carrying the
round's emulator and log-density form) plus an `AcquisitionState`
bundling the design data and tempering state. See the
`PointwiseScoredAcquisition` docstring for the scoring contract.

Shape and symbol conventions (`x`, `X`, `Y`, `q`, `n`, etc.) follow
``docs/notation.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import jax
from jax import Array

from sabi.surrogate.surrogate_distribution import SurrogateDistribution
from sabi.problems.base import Problem

if TYPE_CHECKING:
    from sabi.acquisitions.optim import PointwiseOptimizer
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
      Typical when the loop is sampling toward a sequence of harder
      targets.
    - ``TERMINAL``: the schedule's terminal state. Acquisition
      optimizes toward the final target throughout, regardless of
      where the schedule is. Useful when intermediate distributions
      are scaffolding only and the final target is what matters.

    For untempered loops (``NoTempering`` + ``UntemperedSchedule``),
    all three collapse to the same state and the choice has no effect.

    For richer policies (e.g., ESS-adaptive look-ahead, custom callable
    that depends on loop state), see issue #5 — the enum is a starting
    point; a ``Callable[..., state]`` policy generalizes it.
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
    """Map an `AcquisitionTarget` value to a concrete tempering state.

    Used by the algorithm loop to determine which tempering state the
    acquisition's `SurrogateDistribution` should be built at:

    - `CURRENT`: returns `current_state` (the round's state).
    - `NEXT`: returns `schedule.at(round_idx + 1)[0]`. Built-in
      schedules clamp at the terminal state past the end.
    - `TERMINAL`: returns `schedule.terminal_state()`.

    Lives next to `AcquisitionTarget` because it's the canonical
    consumer of the enum — `target_tempering_state` on
    `AcquisitionState` is what acquisitions actually see.
    """
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

    The round's `SurrogateDistribution` carries the emulator fit on the
    current design data plus the log-density form for the round.
    Acquisitions that need a real (non-degenerate) emulator should check
    ``state.surrogate_distribution.emulator is None`` — this is the case
    when the loop is running the weighted-empirical baseline (a
    `WeightedEmpiricalRandomMeasure`, which is a `SurrogateDistribution`
    subclass with ``emulator=None``).

    Two `Y` arrays are exposed:

    - ``Y_raw``: the un-transformed evaluations of
      ``problem.target_map``. Always present, regardless of any
      tempering scheme.
    - ``Y_train``: the values the round's emulator was actually trained
      on. Under no tempering this equals ``Y_raw``. Under the upcoming
      tempering schemes (Step 3+) this may be a state-dependent
      transformation (e.g., ``lambda * Y_raw`` for likelihood tempering
      via target).

    Most acquisitions only care about ``Y_train`` (it's what's
    consistent with ``state.surrogate_distribution.emulator``'s training
    data). Diagnostic / logging code can use ``Y_raw`` to access the
    raw evaluations.

    Attributes:
        problem: the inference problem (provides `prior`, `support`,
            `input_shape`, etc.).
        surrogate_distribution: round's surrogate-posterior random measure
            built at ``target_tempering_state`` (the state the
            acquisition optimizes against — see `AcquisitionTarget`).
            Always set; ``surrogate_distribution.emulator`` may be
            ``None`` for the weighted-empirical baseline.
        X: design inputs, shape `(n,) + problem.input_shape`.
        Y_raw: raw target evaluations, shape `(n,) + problem.output_shape`.
        Y_train: emulator-training targets *at the
            target_tempering_state*, same shape as Y_raw. Consistent
            with the SP's emulator. Equal to Y_raw when no
            target-side tempering is in effect.
        tempering_state: round's current state from the schedule.
            Available for acquisitions that want to introspect the
            round's intermediate distribution independently of the
            target.
        target_tempering_state: state at which the
            ``surrogate_distribution`` is built — the state the
            acquisition optimizes against. Equal to ``tempering_state``
            when ``Algorithm.acquisition_target == CURRENT`` (default).
    """

    problem: Problem
    surrogate_distribution: SurrogateDistribution
    X: Array
    Y_raw: Array
    Y_train: Array
    tempering_state: Any
    target_tempering_state: Any


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
    one emulator call over the whole batch instead of `n` per-point
    queries).

    **JAX-traceability.** `_score_single` (or `score`) must be
    JAX-traceable for `ContinuousMultiStartOptimizer` to take its
    gradient. The candidate-set optimizer needs only forward evaluation.

    **Emulator-protocol assumptions.** The score function reads from
    `state.surrogate_distribution.emulator` (an `ArrayRandomFunction`)
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
        batched evaluation is more efficient (e.g., one emulator call
        for the whole batch).
        """
        return jax.vmap(lambda x: self._score_single(x, state))(X)

    def select_batch(self, state: AcquisitionState, q: int, key: Array) -> Array:
        return self.optimizer.optimize(self, state, q, key)
