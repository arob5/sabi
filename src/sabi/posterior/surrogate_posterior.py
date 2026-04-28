"""`SurrogatePosterior` — random measure induced by a surrogate of the
target map composed with a `LogDensityForm`.

A `SurrogatePosterior` is a ProbPipe `NumericRandomMeasure[Array]`: every
surrogate function realization defines a deterministic posterior, and the
random measure is the distribution over these as the surrogate's random
function varies.

Decoupled from `Problem`: takes math primitives directly (`support`,
`prior`, `log_density_form`, `input_shape`). The algorithm loop pulls
those from a `Problem` when constructing the SP each round.

For the no-surrogate / Dirac baseline, see
`sabi.posterior.weighted_empirical.WeightedEmpiricalRandomMeasure` —
not a subclass of this class, but plays the same algorithmic role in
the loop (the round's posterior estimate / random measure).

Protocol opt-ins (v1.2):

- `SupportsRandomUnnormalizedLogProb`: returns a thin `RandomFunction`
  whose `__call__(X)` gets the surrogate's predictive at `X` and pushes
  it through the form via `pushforward_marginal` (closed-form for
  Gaussian-affine cases, MC empirical via ProbPipe broadcasting otherwise).
- `SupportsSampling`: NOT implemented in v1.2 (sabi's `Surrogate` doesn't
  yet expose function-trajectory sampling — v1.6).
- `SupportsMean` (the unbiased "expected posterior"): NOT implemented
  (no closed-form expected posterior; MC backend is a v2 item).
- `SupportsRandomLogProb`: NOT implemented (normalization intractable).
"""

from __future__ import annotations

from typing import ClassVar

from jax import Array
from probpipe.core._distribution_base import Distribution
from probpipe.core._random_functions import RandomFunction
from probpipe.core._random_measures import NumericRandomMeasure
from probpipe.core.constraints import Constraint

from sabi.posterior._pushforward import pushforward_marginal
from sabi.problems.forms import LogDensityForm
from sabi.surrogates.base import Surrogate


class SurrogatePosterior(NumericRandomMeasure):
    """Random measure induced by a surrogate of the target map composed
    with a `LogDensityForm`.

    Args:
        surrogate: a fittable `Surrogate` (an `ArrayRandomFunction` over
            the parameter space). Provides the predictive distribution at
            query points via `__call__(X, joint_inputs, joint_outputs)`.
        log_density_form: composes the surrogate's outputs with the prior
            into an unnormalized log-posterior.
        support: `Constraint` over the parameter space.
        input_shape: shape of one parameter-space point. Must equal
            `surrogate.input_shape`.
        prior: optional `Distribution`; required by forms that access it
            (`LogLikPlusPrior`, `ForwardModel`).
        name: optional ProbPipe distribution name.
    """

    def __init__(
        self,
        surrogate: Surrogate,
        log_density_form: LogDensityForm,
        *,
        support: Constraint,
        input_shape: tuple[int, ...],
        prior: Distribution | None = None,
        name: str | None = None,
    ):
        if support is None:
            raise ValueError("SurrogatePosterior requires a non-None `support`.")
        if tuple(input_shape) != tuple(surrogate.input_shape):
            raise ValueError(
                f"input_shape={tuple(input_shape)} must match "
                f"surrogate.input_shape={tuple(surrogate.input_shape)}."
            )
        self._surrogate = surrogate
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
    def surrogate(self) -> Surrogate:
        return self._surrogate

    @property
    def log_density_form(self) -> LogDensityForm:
        return self._log_density_form

    @property
    def prior(self) -> Distribution | None:
        return self._prior

    # Protocol implementation -------------------------------------------------

    def _random_unnormalized_log_prob(self) -> RandomFunction:
        return _SurrogatePosteriorPushforward(self)


class _SurrogatePosteriorPushforward(RandomFunction):
    """The random unnormalized log-density of a `SurrogatePosterior`.

    `__call__(X)` evaluates the surrogate's predictive at `X` (a
    `Distribution`) and pushes it through the SP's `log_density_form` via
    `pushforward_marginal`. Joint flags pass through to the surrogate
    (so callers can request joint over inputs / outputs when the
    surrogate supports it).
    """

    _sampling_cost: ClassVar[str] = "low"
    _preferred_orchestration: ClassVar[str | None] = None

    def __init__(self, sp: SurrogatePosterior):
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
        # The surrogate validates X against its own input_shape contract.
        input_dist = sp.surrogate(
            X, joint_inputs=joint_inputs, joint_outputs=joint_outputs
        )
        return pushforward_marginal(
            input_dist,
            sp.log_density_form,
            X=X,
            prior=sp.prior,
        )
