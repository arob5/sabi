"""`SurrogateDistribution` and concrete subtypes.

`SurrogateDistribution` is sabi's abstract base for surrogate posteriors —
random measures over `Distribution[Array]`s on the parameter space. Two
concrete realizations live in this package, both inheriting from this base:

- `EmulatedDistribution` (this module) — emulator-backed; pushes the
  emulator's predictive through a :class:`sabi.density_decomposition.DensityDecomposition`.
  Every emulator function realization defines a deterministic posterior;
  the random measure is the distribution over these.
- `WeightedEmpiricalRandomMeasure` (in `weighted_empirical.py`) —
  degenerate / Dirac at a weighted empirical of design points. No
  emulator, no decomposition.

The two subtypes are **siblings**, not parent/child. Code that needs to
operate on either uses `SurrogateDistribution` as the type. Code that
needs an emulator narrows to `EmulatedDistribution` via `isinstance` and
raises if the runtime type is the no-emulator baseline.

See ``docs/design.md §4.5`` for the architectural framing and
``docs/notation.md`` for the "emulator" vs. "surrogate" naming
convention.

Protocol opt-ins
----------------

`EmulatedDistribution` opts into:

- `SupportsRandomUnnormalizedLogProb`: returns a thin `RandomFunction`
  that pushes the emulator's predictive at `X` through the
  decomposition via :meth:`DensityDecomposition.pushforward`.

It does NOT opt into `SupportsSampling`, `SupportsMean`, or
`SupportsRandomLogProb` on the base class — function-trajectory
sampling, unbiased expected-posterior, and normalized random log-prob
require machinery (MC backends, normalization estimates) deferred to
later milestones.

`WeightedEmpiricalRandomMeasure` opts into all four protocols via the
underlying `NumericEmpiricalDistribution` and Dirac shims.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

from jax import Array
from probpipe.core._distribution_base import Distribution
from probpipe.core._random_functions import RandomFunction
from probpipe.core._random_measures import NumericRandomMeasure
from probpipe.core.constraints import Constraint

from sabi.density_decomposition import DensityDecomposition
from sabi.emulators.base import Emulator


class SurrogateDistribution(NumericRandomMeasure):
    """Abstract base for sabi's surrogate posteriors.

    Two concrete subtypes:

    - :class:`EmulatedDistribution` — emulator-backed; pushes the
      emulator's predictive through a :class:`DensityDecomposition`.
    - :class:`WeightedEmpiricalRandomMeasure` — degenerate (Dirac) at
      a weighted empirical of design points; no emulator, no decomposition.

    Code that needs to operate on either subtype types against this
    base. Code that needs an emulator narrows to
    `EmulatedDistribution` via ``isinstance`` and raises if the
    runtime type is wrong.
    """

    @property
    @abstractmethod
    def inner_support(self) -> Constraint:
        """Support shared by every inner `Distribution[Array]`'s samples."""

    @property
    @abstractmethod
    def inner_event_shape(self) -> tuple[int, ...]:
        """`event_shape` shared by the inner distributions."""


class EmulatedDistribution(SurrogateDistribution):
    """Surrogate posterior obtained by composing an emulator's predictive
    distribution with a :class:`DensityDecomposition` via pushforward.

    For each emulator function realization, the decomposition defines a
    deterministic posterior; the random measure is the distribution
    over these posteriors as the emulator's random function varies.

    Args:
        emulator: a fittable :class:`Emulator` (an
            ``ArrayRandomFunction`` over the parameter space). Provides
            the predictive distribution at query points via
            ``__call__(X, joint_inputs, joint_outputs)``.
        decomposition: the :class:`DensityDecomposition` that composes
            the emulator's outputs with link / shift into an unnormalized
            log-posterior.
        support: ``Constraint`` over the parameter space.
        input_shape: shape of one parameter-space point. Must equal
            ``emulator.input_shape``.
        name: optional ProbPipe distribution name.
    """

    def __init__(
        self,
        emulator: Emulator,
        decomposition: DensityDecomposition,
        *,
        support: Constraint,
        input_shape: tuple[int, ...],
        name: str | None = None,
    ):
        if emulator is None:
            raise ValueError(
                "EmulatedDistribution requires a non-None `emulator`. "
                "Use `WeightedEmpiricalRandomMeasure` for the no-emulator "
                "baseline."
            )
        if decomposition is None:
            raise ValueError(
                "EmulatedDistribution requires a non-None `decomposition`."
            )
        if support is None:
            raise ValueError("EmulatedDistribution requires a non-None `support`.")
        if tuple(input_shape) != tuple(emulator.input_shape):
            raise ValueError(
                f"input_shape={tuple(input_shape)} must match "
                f"emulator.input_shape={tuple(emulator.input_shape)}."
            )
        self._emulator = emulator
        self._decomposition = decomposition
        self._support = support
        self._input_shape = tuple(input_shape)
        super().__init__(name=name or type(self).__name__)

    # NumericRandomMeasure abstract properties

    @property
    def inner_support(self) -> Constraint:
        return self._support

    @property
    def inner_event_shape(self) -> tuple[int, ...]:
        return self._input_shape

    @property
    def emulator(self) -> Emulator:
        return self._emulator

    @property
    def decomposition(self) -> DensityDecomposition:
        return self._decomposition

    # Protocol implementation -------------------------------------------------

    def _random_unnormalized_log_prob(self) -> RandomFunction:
        return _EmulatedDistributionPushforward(self)


class _EmulatedDistributionPushforward(RandomFunction):
    """The random unnormalized log-density of an `EmulatedDistribution`.

    ``__call__(X)`` evaluates the emulator's predictive at ``X`` (a
    ``Distribution``) and pushes it through the surrogate's
    :class:`DensityDecomposition` via ``decomposition.pushforward(X, ...)``.
    Joint flags pass through to the emulator (so callers can request
    joint over inputs / outputs when the emulator supports it).
    """

    _sampling_cost: ClassVar[str] = "low"
    _preferred_orchestration: ClassVar[str | None] = None

    def __init__(self, surrogate_distribution: EmulatedDistribution):
        self._surrogate_distribution = surrogate_distribution
        super().__init__(
            name=f"{surrogate_distribution.name}_random_unnormalized_log_prob"
        )

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._surrogate_distribution.inner_event_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return ()

    def __call__(
        self,
        X: Array,
        *,
        joint_inputs: bool = False,
        joint_outputs: bool = False,
    ) -> Distribution:
        surrogate_distribution = self._surrogate_distribution
        # The emulator validates X against its own input_shape contract.
        input_dist = surrogate_distribution.emulator(
            X, joint_inputs=joint_inputs, joint_outputs=joint_outputs
        )
        return surrogate_distribution.decomposition.pushforward(X, input_dist)
