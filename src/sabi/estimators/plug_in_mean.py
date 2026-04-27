"""`PlugInMean` — concrete plug-in-mean posterior.

Constructing `PlugInMean(surrogate_posterior)` returns a ProbPipe
`NumericRecordDistribution` (or a subclass thereof) whose protocols depend
on the type of `surrogate_posterior`:

- `WeightedEmpiricalSurrogatePosterior` (Dirac random measure) →
  the underlying `NumericEmpiricalDistribution`. Supports sampling and
  exact moments via ProbPipe's empirical machinery — no GP needed.
- `GPPushforwardSurrogatePosterior` → a `_GPPlugInMean` instance whose
  `_sample` is importance resampling against the surrogate-mean
  log-density and whose `_unnormalized_log_prob` evaluates
  `current_form(x, surrogate.predict(x).mean, problem)`.

The `_GPPlugInMean` IS implementation is the same backend v2 will replace
with MCMC / SMC / VI dispatch. v1.1 keeps the dispatch via isinstance; v1.2
formalizes via `RandomMeasure` op dispatch.
"""

from __future__ import annotations

from typing import ClassVar

import jax
import jax.numpy as jnp
from jax import Array
from probpipe.core._numeric_record_distribution import NumericRecordDistribution

from sabi.estimators.surrogate_posterior import (
    GPPushforwardSurrogatePosterior,
    SurrogatePosterior,
    WeightedEmpiricalSurrogatePosterior,
)
from sabi.initial_designs.base import sample_initial


class PlugInMean:
    """Factory for the plug-in-mean concrete posterior.

    `PlugInMean(surrogate_posterior, ...)` returns a `Distribution[Array]`
    (a `NumericRecordDistribution` subclass) — see module docstring for the
    dispatch table. Not itself a Distribution; instances of `PlugInMean` are
    never created.
    """

    def __new__(
        cls,
        surrogate_posterior: SurrogatePosterior,
        *,
        n_proposals: int = 4096,
        name: str | None = None,
    ) -> NumericRecordDistribution:
        if isinstance(surrogate_posterior, WeightedEmpiricalSurrogatePosterior):
            return surrogate_posterior.empirical_distribution
        if isinstance(surrogate_posterior, GPPushforwardSurrogatePosterior):
            return _GPPlugInMean(
                surrogate_posterior, n_proposals=n_proposals, name=name
            )
        raise TypeError(
            f"PlugInMean: unsupported surrogate posterior type "
            f"{type(surrogate_posterior).__name__}"
        )


class _GPPlugInMean(NumericRecordDistribution):
    """Plug-in-mean posterior over the GP-pushforward of the surrogate.

    `_sample` is importance resampling against the surrogate-mean log-density;
    `_unnormalized_log_prob` evaluates the `current_form` at the surrogate's
    mean prediction. The IS proposal is `problem.prior` (the design
    distribution).
    """

    _sampling_cost: ClassVar[str] = "medium"
    _preferred_orchestration: ClassVar[str | None] = None

    def __init__(
        self,
        surrogate_posterior: GPPushforwardSurrogatePosterior,
        *,
        n_proposals: int = 4096,
        name: str | None = None,
    ):
        self._sp = surrogate_posterior
        self._n_proposals = n_proposals
        super().__init__(
            name=name or f"plug_in_mean_{surrogate_posterior.problem.name}"
        )

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._sp.problem.input_shape

    def _sample(self, key: Array, sample_shape: tuple[int, ...] = ()) -> Array:
        sp = self._sp
        n_total = 1
        for d in sample_shape:
            n_total *= d

        key_prop, key_resample = jax.random.split(key)
        proposals = sample_initial(sp.problem, key_prop, self._n_proposals)

        # Plug-in-mean log-density at each proposal: form(x, surrogate_mean(x), problem).
        pred = sp.surrogate.predict(proposals)
        log_w = jax.vmap(lambda x, y: sp.current_form(x, y, sp.problem))(
            proposals, pred.mean
        )
        log_w = log_w - jnp.max(log_w)
        w = jnp.exp(log_w)
        w = jnp.where(jnp.isfinite(w), w, 0.0)
        total = jnp.sum(w)
        w = jnp.where(total > 0, w / total, jnp.ones_like(w) / w.shape[0])

        idx = jax.random.choice(
            key_resample, proposals.shape[0], shape=(n_total,), p=w, replace=True
        )
        flat = proposals[idx]  # (n_total, *event_shape)
        if sample_shape == ():
            # Protocol: shape == event_shape for sample_shape == ().
            return flat[0]
        return flat.reshape(sample_shape + sp.problem.input_shape)

    def _unnormalized_log_prob(self, value: Array) -> Array:
        sp = self._sp
        pred = sp.surrogate.predict(value[None, ...])
        return sp.current_form(value, pred.mean[0], sp.problem)
