"""Log-density forms — sabi's local **pushforward operators**.

Each `LogDensityForm` composes a target evaluation `y = f(x)` with the
problem's prior log-density (and any extra observation-model machinery)
to produce an unnormalized log-posterior. This is exactly the pushforward
of `f` through `log_prior(x)` — sabi treats `LogDensityForm` as its
local pushforward implementation until ProbPipe's pushforward op lands
(see `docs/probpipe_issues.md` and `docs/v1_plan.md` v1.1).

Each form is called on a **single** point `(x, y)` and returns a scalar
log-density; vectorize externally (e.g. `jax.vmap`) when working with
batches. Shape / symbol conventions: see `docs/notation.md`.

`LogLikPlusPrior` and `ForwardModel` access the prior via
`probpipe.core.ops.log_prob(problem.prior, x)`. ProbPipe ops return
`NumericRecord` containers; we extract the scalar with `jnp.asarray(...)`
to interoperate with sabi's float arithmetic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import probpipe.core.ops as pp_ops
from jax import Array

if TYPE_CHECKING:
    from sabi.problems.base import Problem


class LogDensityForm(ABC):
    """Deterministic assembler `(x, y) → unnormalized log-posterior(x)`."""

    @abstractmethod
    def __call__(self, x: Array, y: Array, problem: Problem) -> Array: ...


@dataclass(frozen=True)
class Identity(LogDensityForm):
    """`y` is already the unnormalized log-posterior. Emulator learns log-post."""

    def __call__(self, x: Array, y: Array, problem: Problem) -> Array:
        return y


@dataclass(frozen=True)
class LogLikPlusPrior(LogDensityForm):
    """`y` is the log-likelihood; emulator learns log-lik only, prior added here."""

    def __call__(self, x: Array, y: Array, problem: Problem) -> Array:
        if problem.prior is None:
            raise ValueError(
                f"{type(self).__name__} requires problem.prior to be set."
            )
        return y + jnp.asarray(pp_ops.log_prob(problem.prior, x))


@dataclass(frozen=True)
class ForwardModel(LogDensityForm):
    """Emulator learns a forward model `y = g(x)`; observation model + prior are
    applied here to form the log-posterior."""

    log_lik_from_outputs: Callable[[Array, Array], Array]  # (x, y) -> log-likelihood

    def __call__(self, x: Array, y: Array, problem: Problem) -> Array:
        if problem.prior is None:
            raise ValueError(
                f"{type(self).__name__} requires problem.prior to be set."
            )
        return self.log_lik_from_outputs(x, y) + jnp.asarray(
            pp_ops.log_prob(problem.prior, x)
        )
