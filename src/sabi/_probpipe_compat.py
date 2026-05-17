"""Temporary compatibility shims around ProbPipe gaps.

Each utility here exists because the equivalent isn't yet available
upstream in ProbPipe. Each is marked with a TODO pointing at what
needs to land in ProbPipe before the shim can be retired. **Sabi code
should prefer the ProbPipe-native API the moment it ships.**
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from probpipe.core.constraints import Constraint, interval
from probpipe.distributions._tfp_base import TFPDistribution


# ---------------------------------------------------------------------------
# Independent: re-interpret batch dims as event dims
# ---------------------------------------------------------------------------
#
# TODO(probpipe-independent): retire this class once ProbPipe ships an
# `Independent`-style wrapper. The functionality here is a thin shim
# around `tfd.Independent(reinterpreted_batch_ndims=...)`.
#
# What the wrapper does:
#
#   Take a per-dim distribution (e.g.,
#   `Uniform(low=array_d, high=array_d)` with `batch_shape=(d,),
#   event_shape=()`) and re-interpret the trailing `n` batch dims as
#   event dims. Result: `event_shape=(d,) + base.event_shape`,
#   `batch_shape=base.batch_shape[:-n]`.
#
#   `log_prob(x)` for `x` of shape `(d,) + base.event_shape` returns
#   scalar — TFP sums the per-element log-probs across the
#   reinterpreted dims. This is what sabi's `DensityDecomposition`'s
#   `LogProb` shift requires from a multivariate-event modeling prior.
#
# Why not use `ProbPipe.distributions.ProductDistribution`:
#
#   `ProductDistribution` is a record-style joint over multiple named
#   components (each component sampled independently, named fields).
#   That's a different abstraction. We want to take *one* per-dim
#   distribution and re-interpret its batch dims as event dims, which
#   is what `tfd.Independent` does and what we wrap here.

class _IndependentArrayDistribution(TFPDistribution):
    """Re-interpret a per-dim distribution's batch dims as event dims.

    Args:
        base: a `TFPDistribution` whose `batch_shape` includes the
            dimensions to re-interpret as event dims.
        reinterpreted_batch_ndims: number of trailing batch dims to
            re-interpret as event dims. Default 1 (the common case for
            d-dim parameter spaces with per-dim priors).
        support: optional explicit `Constraint` for the resulting
            multivariate-event distribution. If `None`, sabi tries to
            derive it from `base` — currently supports `tfd.Uniform`
            with array `low`/`high` (returns
            `interval(low_array, high_array)`). For other base types,
            pass `support` explicitly.
        name: ProbPipe distribution name.
    """

    def __init__(
        self,
        base: TFPDistribution,
        *,
        reinterpreted_batch_ndims: int = 1,
        support: Constraint | None = None,
        name: str,
    ):
        self._base = base
        self._reinterpreted_batch_ndims = reinterpreted_batch_ndims
        self._tfp_dist = tfd.Independent(
            base._tfp_dist,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
        )
        if support is not None:
            self._support = support
        else:
            self._support = _derive_support(
                base._tfp_dist, reinterpreted_batch_ndims
            )
        super().__init__(name=name)

    @property
    def support(self) -> Constraint:
        return self._support


def _derive_support(
    base_tfp: tfd.Distribution,
    reinterpreted_batch_ndims: int,
) -> Constraint:
    """Best-effort support derivation for common per-dim distributions.

    For a `tfd.Uniform(low, high)` with array `low` and `high`, returns
    `interval(low, high)` over the full multivariate event. For other
    distributions, raises with a pointer to pass `support` explicitly.

    The derivation here sidesteps a ProbPipe bug where
    `Uniform.support` casts its (potentially-array) `low`/`high` to
    Python `float`, which crashes on arrays. Once that's fixed
    upstream, the `Uniform` branch here can be replaced with
    `_unwrap_probpipe_uniform_support(...)` or similar.
    """
    if isinstance(base_tfp, tfd.Uniform):
        # Reinterpret batch_shape[-n:] as event dims; the per-dim
        # interval on each becomes a multivariate box.
        low = jnp.asarray(base_tfp.low)
        high = jnp.asarray(base_tfp.high)
        return interval(low, high)
    raise NotImplementedError(
        f"_IndependentArrayDistribution: cannot auto-derive support for "
        f"base of type {type(base_tfp).__name__}. Pass `support=...` "
        "explicitly."
    )


def independent_uniform(
    low: jnp.ndarray,
    high: jnp.ndarray,
    *,
    name: str,
) -> _IndependentArrayDistribution:
    """Convenience: build an `Independent`-wrapped `Uniform` over a box.

    Equivalent to `Independent(Uniform(low, high))` with
    `event_shape=(d,)` and `batch_shape=()` for a `d`-dim box. The
    resulting distribution's `log_prob(x)` returns a scalar for `x` of
    shape `(d,)` — which is what sabi's `DensityDecomposition` expects
    of a multivariate-event prior wrapped in a `LogProb` shift.
    """
    from probpipe.distributions.continuous import Uniform

    base = Uniform(low=low, high=high, name=f"{name}_per_dim")
    return _IndependentArrayDistribution(base, name=name)
