"""`SurrogatePosterior` — random measure induced by a surrogate of the
target map composed with a log-density form.

A `SurrogatePosterior` is a ProbPipe `NumericRandomMeasure[Array]`: a
distribution over `Distribution[Array]`s on the parameter space.

The base class is **decoupled from `Problem`**: it carries math primitives
directly (`support`, `prior`, `log_density_form`, `input_shape`). The
algorithm loop is responsible for extracting these from a `Problem` and
choosing whichever (possibly tempered) form is current for the round; the
SP itself is timeline-agnostic.

Two concrete subclasses ship in v1.2:

- `WeightedEmpiricalSurrogatePosterior` — Dirac random measure at the
  weighted empirical of design points. No surrogate uncertainty; serves
  as a no-GP baseline.
- `GPPushforwardSurrogatePosterior` — proper random measure: every
  surrogate function realization defines a deterministic posterior, and
  the random measure is the distribution over these as the surrogate's
  random function varies.

Each subclass opts into individual `Supports*` protocols by implementing
the corresponding methods. See the per-class docstrings for the
opt-in matrix.
"""

from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import ClassVar

import jax.numpy as jnp
from jax import Array
from probpipe import log_prob
from probpipe._weights import Weights
from probpipe.core._distribution_array import DistributionArray
from probpipe.core._distribution_base import Distribution
from probpipe.core._empirical import NumericEmpiricalDistribution
from probpipe.core._random_measures import NumericRandomMeasure
from probpipe.core.constraints import Constraint

from sabi.posterior._dirac import _DiracArrayRandomFunction
from sabi.posterior._pushforward import _PushforwardLogDensityRandomFunction
from sabi.problems.forms import LogDensityForm
from sabi.surrogates.base import Surrogate


class SurrogatePosterior(NumericRandomMeasure, ABC):
    """Abstract base. A `SurrogatePosterior` is the random measure induced
    by a surrogate of the target map composed with a `LogDensityForm`.

    Constructor takes math primitives directly:

    Args:
        support: `Constraint` over the inner samples (parameter space).
            Required (raises if `None`).
        input_shape: shape of one parameter-space point.
        log_density_form: composes the surrogate's outputs with the prior
            into an unnormalized log-posterior.
        prior: optional `Distribution`; required by forms that access it
            (`LogLikPlusPrior`, `ForwardModel`).
        name: optional ProbPipe distribution name.
    """

    def __init__(
        self,
        *,
        support: Constraint,
        input_shape: tuple[int, ...],
        log_density_form: LogDensityForm,
        prior: Distribution | None = None,
        name: str | None = None,
    ):
        if support is None:
            raise ValueError("SurrogatePosterior requires a non-None `support`.")
        self._support = support
        self._input_shape = tuple(input_shape)
        self._log_density_form = log_density_form
        self._prior = prior
        super().__init__(name=name or type(self).__name__)

    # NumericRandomMeasure abstract properties

    @property
    def inner_support(self) -> Constraint:
        return self._support

    @property
    def inner_event_shape(self) -> tuple[int, ...]:
        return self._input_shape

    # The rest of the SP's identity

    @property
    def log_density_form(self) -> LogDensityForm:
        return self._log_density_form

    @property
    def prior(self) -> Distribution | None:
        return self._prior


class WeightedEmpiricalSurrogatePosterior(SurrogatePosterior):
    """Dirac random measure at the weighted empirical of design points.

    Stores `(X, Y_log_density)` where `Y_log_density[i]` is the
    deterministic unnormalized log-posterior at `X[i]` (already composed
    via the SP's `log_density_form`). The induced inner posterior is a
    `NumericEmpiricalDistribution(samples=X, log_weights=Y_log_density)`.
    The random measure has zero variance — every "draw" is the same
    empirical.

    Protocol opt-ins:
        - `SupportsMean`: returns the inner empirical (the Dirac point).
        - `SupportsSampling`: returns the inner empirical for
          `sample_shape == ()`; for non-trivial `sample_shape`, returns a
          `DistributionArray` of repeats (Dirac means every cell is the
          same distribution).
        - `SupportsRandomLogProb` / `SupportsRandomUnnormalizedLogProb`:
          a degenerate Dirac random function (sabi-local
          `_DiracArrayRandomFunction`) whose marginal at each `x` is a
          `_DiracDistribution` at the empirical's log-density value.

    For Dirac surrogate posteriors, `mean(sp)` and any sensible
    deterministic estimator (`expected_target(sp)`, etc.) coincide and
    return the inner empirical.
    """

    _sampling_cost: ClassVar[str] = "low"
    _preferred_orchestration: ClassVar[str | None] = None

    def __init__(
        self,
        X: Array,
        Y: Array,
        *,
        support: Constraint,
        input_shape: tuple[int, ...],
        log_density_form: LogDensityForm,
        prior: Distribution | None = None,
        name: str | None = None,
    ):
        X = jnp.asarray(X)
        Y = jnp.asarray(Y)
        if X.shape[1:] != tuple(input_shape):
            raise ValueError(
                f"X.shape[1:]={X.shape[1:]} must match "
                f"input_shape={tuple(input_shape)}."
            )
        if Y.shape != (X.shape[0],):
            raise ValueError(
                f"Y must have shape ({X.shape[0]},), got {Y.shape}."
            )
        self._X = X
        self._Y = Y
        super().__init__(
            support=support,
            input_shape=input_shape,
            log_density_form=log_density_form,
            prior=prior,
            name=name,
        )

    @property
    def X(self) -> Array:
        return self._X

    @property
    def Y(self) -> Array:
        return self._Y

    @cached_property
    def inner_distribution(self) -> NumericEmpiricalDistribution:
        return NumericEmpiricalDistribution(
            samples=self._X,
            weights=Weights(n=self._X.shape[0], log_weights=self._Y),
            name=f"{self.name}_empirical",
        )

    # Protocol implementations

    def _mean(self) -> Distribution:
        return self.inner_distribution

    def _sample(self, key, sample_shape: tuple[int, ...] = ()) -> Distribution:
        if sample_shape == ():
            return self.inner_distribution
        n = 1
        for s in sample_shape:
            n *= s
        components = tuple(self.inner_distribution for _ in range(n))
        return DistributionArray(components, batch_shape=sample_shape)

    def _random_log_prob(self):
        emp = self.inner_distribution
        return _DiracArrayRandomFunction(
            evaluator=lambda x: jnp.asarray(log_prob(emp, x)),
            input_shape=self._input_shape,
            output_shape=(),
            name=f"{self.name}_random_log_prob",
        )

    def _random_unnormalized_log_prob(self):
        # `NumericEmpiricalDistribution` is normalized, so its
        # unnormalized log-prob coincides with its log-prob. (ProbPipe's
        # empirical doesn't expose `_unnormalized_log_prob` directly, so
        # we delegate via `log_prob` here.)
        emp = self.inner_distribution
        return _DiracArrayRandomFunction(
            evaluator=lambda x: jnp.asarray(log_prob(emp, x)),
            input_shape=self._input_shape,
            output_shape=(),
            name=f"{self.name}_random_unnormalized_log_prob",
        )


class GPPushforwardSurrogatePosterior(SurrogatePosterior):
    """Random measure induced by a stochastic surrogate of the target map
    pushed through a `LogDensityForm`.

    Each surrogate function realization defines a deterministic
    log-posterior; the random measure is the distribution over these as
    the surrogate's random function varies.

    Protocol opt-ins (v1.2):
        - `SupportsRandomUnnormalizedLogProb`: implemented for `Identity`
          and `LogLikPlusPrior` forms (closed-form pushforward of the
          surrogate's pointwise marginals through an affine transform).
          `ForwardModel` raises until the partial-pushforward primitive
          lands.

    NOT implemented in v1.2:
        - `SupportsSampling`: requires the surrogate to expose a
          function-trajectory sampler, which the v1.2 `Surrogate`
          interface doesn't have. v1.6 (real GP backend) reshapes this.
        - `SupportsMean` (the unbiased "expected posterior"): no
          closed-form path and no MC backend in v1.2. `mean(gp_sp)`
          raises clearly via the protocol-not-implemented path.
        - `SupportsRandomLogProb`: normalization of the inner posterior
          is intractable; only the unnormalized variant is exposed.

    For deterministic posterior approximations, see
    `sabi.posterior.estimators` — `expected_target(sp)` returns the
    biased plug-in posterior (predictive mean of the surrogate plugged
    into the log-density form), which is what the v1.2 loop uses.
    """

    def __init__(
        self,
        surrogate: Surrogate,
        *,
        support: Constraint,
        input_shape: tuple[int, ...],
        log_density_form: LogDensityForm,
        prior: Distribution | None = None,
        name: str | None = None,
    ):
        self._surrogate = surrogate
        super().__init__(
            support=support,
            input_shape=input_shape,
            log_density_form=log_density_form,
            prior=prior,
            name=name,
        )

    @property
    def surrogate(self) -> Surrogate:
        return self._surrogate

    def _random_unnormalized_log_prob(self):
        return _PushforwardLogDensityRandomFunction(
            surrogate=self._surrogate,
            log_density_form=self._log_density_form,
            prior=self._prior,
            input_shape=self._input_shape,
            name=f"{self.name}_random_unnormalized_log_prob",
        )
