"""`WeightedEmpiricalRandomMeasure` — Dirac at a weighted empirical of design points.

The simplest non-trivial random measure: zero variance over inner-distribution
draws, with the inner distribution being a `NumericEmpiricalDistribution`
weighted by `softmax(log_weights)`.

Sibling of `EmulatedDistribution` under `SurrogateDistribution` (per
``docs/design.md §4.5``): both extend the abstract base
`SurrogateDistribution`, neither inherits from the other. The design
points carry the posterior structure directly — there is no underlying
emulator and no carried decomposition (the decomposition's role ends at
construction, when `log_weights` is computed from `(X, Y)`).

In sabi this serves as the no-emulator baseline for the loop. The loop's
factory applies a :class:`sabi.density_decomposition.DensityDecomposition`
to `(X, Y)` to compute `log_weights`, then hands the resulting
`(X, log_weights)` to this class.

Tracked for promotion to ProbPipe — see `docs/probpipe_issues.md`:
"`WeightedEmpiricalRandomMeasure` as a ProbPipe primitive".
"""

from __future__ import annotations

from functools import cached_property
from typing import ClassVar

import jax.numpy as jnp
from jax import Array
from probpipe import log_prob
from probpipe._weights import Weights
from probpipe.core._distribution_array import DistributionArray
from probpipe.core._distribution_base import Distribution
from probpipe.core._empirical import NumericEmpiricalDistribution
from probpipe.core.constraints import Constraint

from sabi.surrogate._dirac import _DiracArrayRandomFunction
from sabi.surrogate.surrogate_distribution import SurrogateDistribution


class WeightedEmpiricalRandomMeasure(SurrogateDistribution):
    """Dirac surrogate posterior at a weighted empirical of design points.

    A draw from this random measure is always the same
    `NumericEmpiricalDistribution` over `(X, log_weights)`. Implements
    the full `NumericRandomMeasure` protocol surface (`SupportsMean`,
    `SupportsSampling`, `SupportsRandomLogProb`,
    `SupportsRandomUnnormalizedLogProb`) via the underlying empirical
    and Dirac shims. Sibling of `EmulatedDistribution`; carries no
    emulator and no form.

    Args:
        X: design points, shape `(n,) + inner_event_shape`.
        log_weights: shape `(n,)` — unnormalized log-weights at each
            design point. Typically the deterministic log-posterior at
            `X[i]` under the loop's current `DensityDecomposition`.
        support: `Constraint` over the inner samples (parameter space).
        inner_event_shape: shape of one inner sample (one
            parameter-space point).
        name: optional ProbPipe distribution name.
    """

    _sampling_cost: ClassVar[str] = "low"
    _preferred_orchestration: ClassVar[str | None] = None

    def __init__(
        self,
        X: Array,
        log_weights: Array,
        *,
        support: Constraint,
        inner_event_shape: tuple[int, ...],
        name: str | None = None,
    ):
        X = jnp.asarray(X)
        log_weights = jnp.asarray(log_weights)
        if X.shape[1:] != tuple(inner_event_shape):
            raise ValueError(
                f"X.shape[1:]={X.shape[1:]} must match "
                f"inner_event_shape={tuple(inner_event_shape)}."
            )
        if log_weights.shape != (X.shape[0],):
            raise ValueError(
                f"log_weights must have shape ({X.shape[0]},), "
                f"got {tuple(log_weights.shape)}."
            )
        if support is None:
            raise ValueError(
                "WeightedEmpiricalRandomMeasure requires a non-None `support`."
            )
        self._X = X
        self._log_weights = log_weights
        self._support = support
        self._inner_event_shape = tuple(inner_event_shape)
        super().__init__(name=name or type(self).__name__)

    # NumericRandomMeasure abstract properties

    @property
    def inner_support(self) -> Constraint:
        return self._support

    @property
    def inner_event_shape(self) -> tuple[int, ...]:
        return self._inner_event_shape

    @property
    def X(self) -> Array:
        return self._X

    @property
    def log_weights(self) -> Array:
        return self._log_weights

    @cached_property
    def inner_distribution(self) -> NumericEmpiricalDistribution:
        return NumericEmpiricalDistribution(
            samples=self._X,
            weights=Weights(n=self._X.shape[0], log_weights=self._log_weights),
            name=f"{self.name}_empirical",
        )

    # Protocol implementations -----------------------------------------------

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
            input_shape=self._inner_event_shape,
            output_shape=(),
            name=f"{self.name}_random_log_prob",
        )

    def _random_unnormalized_log_prob(self):
        # NumericEmpiricalDistribution is normalized, so its
        # unnormalized log-prob coincides with its log-prob.
        emp = self.inner_distribution
        return _DiracArrayRandomFunction(
            evaluator=lambda x: jnp.asarray(log_prob(emp, x)),
            input_shape=self._inner_event_shape,
            output_shape=(),
            name=f"{self.name}_random_unnormalized_log_prob",
        )
