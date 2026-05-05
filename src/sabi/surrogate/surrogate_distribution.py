"""`SurrogateDistribution` — random measure induced by an emulator of the
target map composed with a `LogDensityForm`.

A `SurrogateDistribution` is a ProbPipe `NumericRandomMeasure[Array]`:
every emulator function realization defines a deterministic posterior,
and the random measure is the distribution over these. Decoupled from
`Problem`: takes math primitives directly (`support`, `prior`,
`log_density_form`, `input_shape`); the loop pulls those from the
`Problem` per round.

`emulator=None` denotes a Dirac surrogate posterior, concretely realized
by `WeightedEmpiricalRandomMeasure`. The base class raises
`NotImplementedError` from emulator-touching protocol methods on a
None emulator; degenerate subclasses opt in by overriding.

Protocol opt-ins:

- `SupportsRandomUnnormalizedLogProb`: returns a thin `RandomFunction`
  that pushes the emulator's predictive at `X` through the form via
  `pushforward_marginal`. Requires a non-None `emulator`.
- `SupportsSampling`, `SupportsMean`, `SupportsRandomLogProb`: not
  implemented on the base; degenerate subclasses may implement them.

See ``docs/design.md`` §4.5 and ``docs/notation.md`` for the full
"emulator" vs. "surrogate" naming convention and the architectural
context.
"""

from __future__ import annotations

from typing import ClassVar

from jax import Array
from probpipe.core._distribution_base import Distribution
from probpipe.core._random_functions import RandomFunction
from probpipe.core._random_measures import NumericRandomMeasure
from probpipe.core.constraints import Constraint

from sabi.emulators.base import Emulator
from sabi.surrogate._pushforward import pushforward_marginal
from sabi.problems.forms import LogDensityForm


class SurrogateDistribution(NumericRandomMeasure):
    """Random measure induced by an emulator of the target map composed
    with a `LogDensityForm`.

    Args:
        emulator: a fittable `Emulator` (an `ArrayRandomFunction` over
            the parameter space). Provides the predictive distribution at
            query points via `__call__(X, joint_inputs, joint_outputs)`.
            ``None`` denotes a degenerate surrogate posterior — see the
            module docstring; subclasses must override the relevant
            protocol methods for this to work.
        log_density_form: composes the emulator's outputs with the prior
            into an unnormalized log-posterior. ``None`` is allowed for
            degenerate subclasses where the form is consumed at
            construction time (e.g. `WeightedEmpiricalRandomMeasure`).
        support: `Constraint` over the parameter space.
        input_shape: shape of one parameter-space point. Must equal
            `emulator.input_shape` when ``emulator is not None``.
        prior: optional `Distribution`; required by forms that access it
            (`LogLikPlusPrior`, `ForwardModel`).
        name: optional ProbPipe distribution name.
    """

    def __init__(
        self,
        emulator: Emulator | None,
        log_density_form: LogDensityForm | None,
        *,
        support: Constraint,
        input_shape: tuple[int, ...],
        prior: Distribution | None = None,
        name: str | None = None,
    ):
        if support is None:
            raise ValueError("SurrogateDistribution requires a non-None `support`.")
        if emulator is not None and tuple(input_shape) != tuple(emulator.input_shape):
            raise ValueError(
                f"input_shape={tuple(input_shape)} must match "
                f"emulator.input_shape={tuple(emulator.input_shape)}."
            )
        self._emulator = emulator
        self._log_density_form = log_density_form
        self._support = support
        self._input_shape = tuple(input_shape)
        self._prior = prior
        super().__init__(name=name or type(self).__name__)

    # NumericRandomMeasure abstract properties

    @property
    def inner_support(self) -> Constraint:
        return self._support

    @property
    def inner_event_shape(self) -> tuple[int, ...]:
        return self._input_shape

    @property
    def emulator(self) -> Emulator | None:
        return self._emulator

    @property
    def log_density_form(self) -> LogDensityForm | None:
        return self._log_density_form

    @property
    def prior(self) -> Distribution | None:
        return self._prior

    # Helpers -----------------------------------------------------------------

    def require_emulator(self, caller: str) -> Emulator:
        """Return `self.emulator`, raising `ValueError` if it is `None`.

        Used by code paths (e.g. emulator-touching acquisitions, fantasy
        imputers) that cannot operate on the degenerate / no-emulator
        baseline. Centralizes the "...requires a non-degenerate emulator"
        message so consumers don't each hand-roll their own.

        Args:
            caller: short name of the caller (typically a class name like
                ``"ExpectedImprovement"``); included in the error message
                so users can see who rejected the degenerate surrogate.
        """
        if self._emulator is None:
            raise ValueError(
                f"{caller} requires a non-degenerate emulator; got "
                f"`surrogate_distribution.emulator=None` (the no-emulator "
                f"baseline, e.g. WeightedEmpiricalRandomMeasure). Switch "
                f"to a real emulator or use a sampling-style acquisition "
                f"like PriorSampling."
            )
        return self._emulator

    # Protocol implementation -------------------------------------------------

    def _random_unnormalized_log_prob(self) -> RandomFunction:
        if self._emulator is None:
            raise NotImplementedError(
                f"{type(self).__name__}._random_unnormalized_log_prob: "
                "emulator is None (degenerate SurrogateDistribution); "
                "subclass must override this method."
            )
        return _SurrogateDistributionPushforward(self)


class _SurrogateDistributionPushforward(RandomFunction):
    """The random unnormalized log-density of a `SurrogateDistribution`.

    `__call__(X)` evaluates the emulator's predictive at `X` (a
    `Distribution`) and pushes it through the SP's `log_density_form` via
    `pushforward_marginal`. Joint flags pass through to the emulator
    (so callers can request joint over inputs / outputs when the
    emulator supports it).
    """

    _sampling_cost: ClassVar[str] = "low"
    _preferred_orchestration: ClassVar[str | None] = None

    def __init__(self, sp: SurrogateDistribution):
        self._sp = sp
        super().__init__(name=f"{sp.name}_random_unnormalized_log_prob")

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._sp.inner_event_shape

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
        sp = self._sp
        # The emulator validates X against its own input_shape contract.
        input_dist = sp.emulator(
            X, joint_inputs=joint_inputs, joint_outputs=joint_outputs
        )
        return pushforward_marginal(
            input_dist,
            sp.log_density_form,
            X=X,
            prior=sp.prior,
        )
