"""Pushforward of a stochastic surrogate's pointwise marginals through a
`LogDensityForm`.

For a surrogate of the target map `y = f(x)` with Gaussian predictive
marginals (mean, variance), the random unnormalized log-density at `x` is

    log p̃(x; f) = log_density_form(x, f(x), prior=prior)

a random scalar in `f`. For closed-form forms (`Identity`, `LogLikPlusPrior`)
this is an affine transform of `f(x)`, so the marginal at each `x` is a
`Normal` whose mean is the surrogate's predictive mean (plus the form's
shift) and whose variance is the surrogate's predictive variance.

`ForwardModel` is a non-linear pushforward through `log_lik_from_outputs`
and is not closed-form in general; raise until partial-pushforward
infrastructure lands. See `docs/probpipe_issues.md`: "Pushforward with
partial information".
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
from jax import Array
from probpipe.core._distribution_base import Distribution
from probpipe.core._random_functions import RandomFunction
from probpipe.distributions.continuous import Normal

from sabi.problems.forms import (
    ForwardModel,
    Identity,
    LogDensityForm,
    LogLikPlusPrior,
    _joint_log_prior,
)
from sabi.surrogates.base import Surrogate


class _PushforwardLogDensityRandomFunction(RandomFunction):
    """Marginal random log-density at each input.

    `__call__(x)` returns a `Distribution[Array]` for the marginal at the
    single point `x` (shape `input_shape`). Subclasses `RandomFunction`
    directly (not `ArrayRandomFunction`) because we expose only the
    point-marginal interface — the v1.2 surrogate doesn't yet expose joint
    multi-point predictives that `ArrayRandomFunction`'s shape contract
    would require.
    """

    _sampling_cost: ClassVar[str] = "low"
    _preferred_orchestration: ClassVar[str | None] = None

    def __init__(
        self,
        surrogate: Surrogate,
        log_density_form: LogDensityForm,
        prior: Distribution | None,
        *,
        input_shape: tuple[int, ...],
        name: str | None = None,
    ):
        self._surrogate = surrogate
        self._form = log_density_form
        self._prior = prior
        self._input_shape = tuple(input_shape)
        super().__init__(name=name or "pushforward_log_density")

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return ()

    def __call__(self, x: Array) -> Distribution:
        x = jnp.asarray(x)
        if x.shape != self._input_shape:
            raise ValueError(
                f"_PushforwardLogDensityRandomFunction expects a single point "
                f"of shape {self._input_shape}, got {x.shape}."
            )
        # surrogate.predict expects a leading batch axis
        pred = self._surrogate.predict(x[None])
        mu = pred.mean[0]
        scale = jnp.sqrt(pred.variance[0])

        if isinstance(self._form, Identity):
            return Normal(loc=mu, scale=scale, name="lp_marginal")
        if isinstance(self._form, LogLikPlusPrior):
            if self._prior is None:
                raise ValueError(
                    "LogLikPlusPrior pushforward requires a non-None prior."
                )
            shift = _joint_log_prior(self._prior, x)
            return Normal(loc=mu + shift, scale=scale, name="lp_marginal")
        if isinstance(self._form, ForwardModel):
            raise NotImplementedError(
                "Random unnormalized log-prob for ForwardModel is not "
                "implemented in v1.2 — needs pushforward through the "
                "likelihood. See docs/probpipe_issues.md: 'Pushforward "
                "with partial information'."
            )
        raise TypeError(
            f"Unknown LogDensityForm subclass: {type(self._form).__name__}."
        )
