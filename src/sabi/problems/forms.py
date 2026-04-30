r"""Log-density forms — sabi's local **pushforward operators**.

A `LogDensityForm` composes target evaluations ``y = f(x)`` with a prior
log-density (and any extra observation-model machinery) to produce an
unnormalized log-posterior. This is the pushforward of ``f`` through
``log_prior(x)`` — sabi treats `LogDensityForm` as its local pushforward
implementation until ProbPipe's pushforward op lands (see
``docs/probpipe_issues.md``).

Shape contract (mirrors ProbPipe's batch / event semantics)
-----------------------------------------------------------

The public ``__call__(X, Y, *, prior)`` is **batched**:

- ``X.shape == (n,) + input_shape``  (a leading batch dim of size n
  prepended to the event shape).
- ``Y.shape == (n,) + output_shape``.
- Returns shape ``(n,)`` — one scalar log-density per row of X.

Subclasses implement the per-point hook ``_call_single(x, y, *, prior)``:

- ``x.shape == input_shape``, ``y.shape == output_shape``.
- Returns scalar.

Default ``__call__`` does ``jax.vmap(self._call_single, in_axes=(0, 0,
None))(X, Y)`` — vmaps over the leading axis of X and Y; broadcasts the
``prior`` Distribution. Subclasses can override ``__call__`` directly
when a vectorized batched implementation is more efficient than
vmap-of-single-point (e.g., `Identity` is batched-trivial since
``Y_batched`` is already the answer).

Prior shape contract
--------------------

The ``prior`` argument is a multivariate-event Distribution over the
parameter space: ``prior.event_shape == input_shape``,
``prior.batch_shape == ()``. Then ``log_prob(prior, x)`` for ``x`` of
shape ``input_shape`` returns a *scalar* — exactly what the form
needs.

For per-dim distributions (e.g., a per-dim ``Uniform(low_array,
high_array)`` with ``batch_shape == (d,), event_shape == ()``), wrap
with `sabi._probpipe_compat.independent_uniform` (or the more general
`_IndependentArrayDistribution`) to re-interpret the batch dims as
event dims. Forms call `log_prob(prior, x)` directly and assume the
scalar result.

Naming
------

``_call_single`` is the **private hook** subclasses implement
(underscore by convention; not part of the public API). External
callers use the batched ``__call__``. Forms that are pointwise scalar
operations on ``y`` (like `Identity`) override ``__call__`` directly
to skip the vmap overhead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from probpipe import log_prob
from probpipe.core._distribution_base import Distribution


class LogDensityForm(ABC):
    """Deterministic assembler `(x, y) -> unnormalized log-posterior(x)`.

    Subclasses implement ``_call_single(x, y, *, prior) -> scalar``;
    the default ``__call__(X, Y, *, prior)`` ``jax.vmap``s it over the
    leading batch axis of X and Y. Override ``__call__`` directly for
    a batched implementation more efficient than vmap.
    """

    @abstractmethod
    def _call_single(
        self,
        x: Array,
        y: Array,
        *,
        prior: Distribution | None = None,
    ) -> Array:
        """Single-point hook: ``x.shape == input_shape``,
        ``y.shape == output_shape``; returns scalar.
        """

    def __call__(
        self,
        X: Array,
        Y: Array,
        *,
        prior: Distribution | None = None,
    ) -> Array:
        """Batched: ``X.shape == (n,) + input_shape``,
        ``Y.shape == (n,) + output_shape``; returns shape ``(n,)``.

        Default: ``jax.vmap`` of ``_call_single`` over the leading
        axis. Override for a vectorized implementation when more
        efficient.
        """
        return jax.vmap(
            lambda xi, yi: self._call_single(xi, yi, prior=prior),
            in_axes=(0, 0),
        )(X, Y)


@dataclass(frozen=True)
class Identity(LogDensityForm):
    """`y` is already the unnormalized log-posterior. Emulator learns log-post."""

    def _call_single(
        self,
        x: Array,
        y: Array,
        *,
        prior: Distribution | None = None,
    ) -> Array:
        return y

    def __call__(
        self,
        X: Array,
        Y: Array,
        *,
        prior: Distribution | None = None,
    ) -> Array:
        # Trivially batched: Y already is the answer at every row.
        return Y


@dataclass(frozen=True)
class LogLikPlusPrior(LogDensityForm):
    """`y` is the log-likelihood; emulator learns log-lik only, prior added here.

    Requires a multivariate-event ``prior`` whose ``log_prob(prior, x)``
    for ``x`` of shape ``input_shape`` returns a scalar. See the
    module-level shape contract.
    """

    def _call_single(
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
        return y + jnp.asarray(log_prob(prior, x))


@dataclass(frozen=True)
class ForwardModel(LogDensityForm):
    """Emulator learns a forward model ``y = g(x)``; observation model and
    prior are applied here to form the log-posterior.

    The ``log_lik_from_outputs(x, y)`` callable is single-point: takes
    ``(x.shape == input_shape, y.shape == output_shape)`` and returns
    a scalar log-likelihood. Requires a multivariate-event ``prior``.
    """

    log_lik_from_outputs: Callable[[Array, Array], Array]  # (x, y) -> scalar

    def _call_single(
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
        return self.log_lik_from_outputs(x, y) + jnp.asarray(log_prob(prior, x))
