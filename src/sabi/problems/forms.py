"""Log-density forms — sabi's local **pushforward operators**.

Each `LogDensityForm` composes a target evaluation `y = f(x)` with a prior
log-density (and any extra observation-model machinery) to produce an
unnormalized log-posterior. This is exactly the pushforward of `f` through
`log_prior(x)` — sabi treats `LogDensityForm` as its local pushforward
implementation until ProbPipe's pushforward op lands (see
`docs/probpipe_issues.md`).

Each form is called on a **single** point `(x, y)` plus an optional `prior`
argument and returns a scalar log-density; vectorize externally (e.g.
`jax.vmap`) when working with batches. Shape / symbol conventions: see
`docs/notation.md`.

Forms take `prior` directly rather than a `Problem` so they can be reused
in algorithm-agnostic contexts (e.g. by `SurrogatePosterior`, which is
decoupled from `Problem` per the v1.2 design). The `Problem.log_posterior`
method passes its own `prior` field through.

`LogLikPlusPrior` and `ForwardModel` access the prior via
`probpipe.log_prob(prior, x)`. ProbPipe ops return `NumericRecord`
containers; we extract the scalar with `jnp.asarray(...)` to interoperate
with sabi's float arithmetic (see `NumericRecord operator overloading`
entry in `docs/probpipe_issues.md`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from probpipe import log_prob
from probpipe.core._distribution_base import Distribution


class LogDensityForm(ABC):
    """Deterministic assembler `(x, y) → unnormalized log-posterior(x)`."""

    @abstractmethod
    def __call__(
        self,
        x: Array,
        y: Array,
        *,
        prior: Distribution | None = None,
    ) -> Array: ...


@dataclass(frozen=True)
class Identity(LogDensityForm):
    """`y` is already the unnormalized log-posterior. Emulator learns log-post."""

    def __call__(
        self,
        x: Array,
        y: Array,
        *,
        prior: Distribution | None = None,
    ) -> Array:
        return y


def _joint_log_prior(prior: Distribution, x: Array) -> Array:
    """Joint log-density of `prior` at `x`, summed across all dims.

    Sabi's forms treat `prior` as a joint distribution over parameter space
    and need a scalar log-density at each `x`. ProbPipe distributions like
    `Uniform` are element-wise (returning a per-dim log-density when
    `event_shape == ()` and the parameters are batched), so we sum across
    all returned dims to get the joint log-density. For a properly
    multivariate prior whose `log_prob` already returns a scalar, the sum
    is a no-op.

    This is a v1.2 pragma — when ProbPipe ships joint multivariate priors
    natively (via `Independent`-style wrappers or composed
    `MultivariateNormal`), the form can drop the sum and require scalar
    `log_prob` output explicitly.
    """
    return jnp.sum(jnp.asarray(log_prob(prior, x)))


@dataclass(frozen=True)
class LogLikPlusPrior(LogDensityForm):
    """`y` is the log-likelihood; emulator learns log-lik only, prior added here."""

    def __call__(
        self,
        x: Array,
        y: Array,
        *,
        prior: Distribution | None = None,
    ) -> Array:
        if prior is None:
            raise ValueError(
                f"{type(self).__name__} requires a non-None prior."
            )
        return y + _joint_log_prior(prior, x)


@dataclass(frozen=True)
class ForwardModel(LogDensityForm):
    """Emulator learns a forward model `y = g(x)`; observation model + prior are
    applied here to form the log-posterior."""

    log_lik_from_outputs: Callable[[Array, Array], Array]  # (x, y) -> log-likelihood

    def __call__(
        self,
        x: Array,
        y: Array,
        *,
        prior: Distribution | None = None,
    ) -> Array:
        if prior is None:
            raise ValueError(
                f"{type(self).__name__} requires a non-None prior."
            )
        return self.log_lik_from_outputs(x, y) + _joint_log_prior(prior, x)
