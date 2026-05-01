"""Dimension-scaled-prior (DSP) primitives for the gpjax-backed GP emulator.

Implements the prior + parameterization recipe from Hvarfner et al.
(2024), "Vanilla Bayesian Optimization Performs Great in High
Dimensions" (https://arxiv.org/abs/2402.02229):

- ``dsp_lengthscale_prior(d) = LogNormal(loc = √2 + 0.5·log(d), scale = √3)``
  applied IID per ARD lengthscale.
- ``dsp_noise_prior() = LogNormal(loc = -4, scale = 1)`` on the
  **standard deviation** parameter (gpjax's ``obs_stddev``). Note the
  GPyTorch reference puts the same prior on the variance — we track
  that divergence here rather than silently absorb it.
- ``BoundedPositive(value, *, lower)``: ``paramax`` wrapper that
  parameterizes a strictly-positive scalar/array via
  ``lower + softplus(unconstrained)``. Used to enforce hard floors of
  ``2.5e-2`` on lengthscales and ``1e-4`` on noise stddev (the floors
  guard against fit_scipy collapsing onto a degenerate solution when
  the data-likelihood gradient overwhelms the prior).

These primitives are independent of any ``Emulator`` subclass so they
can be unit-tested in isolation.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as ndist
import paramax
from jax import Array

# --- Floors ------------------------------------------------------------------
# Hard lower bounds for the parameterization. Chosen to match Hvarfner et al.
# 2024 (lengthscale floor) and a conservative numerical safety margin (noise
# floor) — small enough not to bias well-posed fits, large enough to keep
# fit_scipy out of degenerate regions on near-noiseless data.
DSP_LENGTHSCALE_FLOOR: float = 2.5e-2
DSP_NOISE_FLOOR: float = 1e-4


# --- Priors ------------------------------------------------------------------


def dsp_lengthscale_prior(d: int) -> ndist.LogNormal:
    """LogNormal lengthscale prior, scaled with input dimension `d`.

    Args:
        d: input dimension. Must be a positive integer.

    Returns:
        ``LogNormal(loc = √2 + 0.5·log(d), scale = √3)``. Apply IID per
        ARD lengthscale; sum the per-dim log-densities for the joint.
    """
    if d < 1:
        raise ValueError(f"dsp_lengthscale_prior: d must be >= 1, got {d}.")
    loc = jnp.sqrt(jnp.asarray(2.0)) + 0.5 * jnp.log(jnp.asarray(float(d)))
    scale = jnp.sqrt(jnp.asarray(3.0))
    return ndist.LogNormal(loc=loc, scale=scale)


def dsp_noise_prior() -> ndist.LogNormal:
    """LogNormal noise-stddev prior: ``LogNormal(loc=-4, scale=1)``.

    Applied to ``posterior.likelihood.obs_stddev`` (gpjax exposes the
    standard deviation, not the variance). Tracked separately from the
    GPyTorch reference, which puts the same prior on the variance.
    """
    return ndist.LogNormal(loc=jnp.asarray(-4.0), scale=jnp.asarray(1.0))


# --- Bounded-positive parameter wrapper -------------------------------------


class BoundedPositive(paramax.AbstractUnwrappable):
    """Strictly-positive parameter with a hard floor.

    Stored as an unconstrained real array; on ``unwrap()`` produces
    ``lower + softplus(unconstrained)``, which is bounded below by
    ``lower`` for any finite unconstrained value.

    Mirrors the role of ``gpjax.parameters.PositiveReal`` (which uses
    plain softplus with a 0 floor) but lets us pin a non-zero floor
    used by the DSP recipe.

    Args:
        value: the desired constrained value at construction time.
            Must be > ``lower`` (a tiny safety margin is added if the
            user passes the floor exactly).
        lower: the hard floor. Stored as a static (non-trainable) field.
    """

    _unconstrained: Array
    lower: float = eqx.field(static=True)

    def __init__(self, value, *, lower: float):
        v = jnp.asarray(value, dtype=jnp.float64)
        # softplus⁻¹(y) = log(exp(y) - 1) = log(expm1(y)). Numerically stable
        # for moderate y; we clamp the residual above the floor to a tiny
        # positive value so log(expm1(.)) doesn't return -inf at the floor.
        delta = jnp.maximum(v - jnp.asarray(lower, dtype=jnp.float64), 1e-12)
        self._unconstrained = jnp.log(jnp.expm1(delta))
        self.lower = float(lower)

    def unwrap(self) -> Array:
        return self.lower + jax.nn.softplus(self._unconstrained)


# --- MAP objective ----------------------------------------------------------


def dsp_map_objective(posterior, data) -> Array:
    """MAP objective: ``conjugate_mll + Σ log_prior_lengthscale + log_prior_noise``.

    Designed to be called by ``gpx.fit_scipy`` via
    ``lambda p, d: -dsp_map_objective(p, d)``. ``fit_scipy`` calls
    ``paramax.unwrap(model)`` BEFORE evaluating the objective, so by
    the time we get ``posterior`` here the kernel lengthscales and
    likelihood ``obs_stddev`` are concrete arrays (post-floor +
    softplus), and the LogNormal log-prob is well-defined.

    Args:
        posterior: a ``gpjax.gps.ConjugatePosterior`` (already unwrapped
            by fit_scipy's machinery).
        data: a ``gpjax.Dataset`` with ``X``, ``y``, and ``n``.

    Returns:
        Scalar JAX array — the log MAP density (up to constants), with
        the conjugate marginal log-likelihood and the DSP priors on
        lengthscales (per-dim, summed) and noise stddev.
    """
    import gpjax as gpx

    mll = gpx.objectives.conjugate_mll(posterior, data)

    lengthscale = posterior.prior.kernel.lengthscale
    obs_stddev = posterior.likelihood.obs_stddev

    d = int(lengthscale.shape[-1]) if lengthscale.ndim > 0 else 1
    lp_ls = dsp_lengthscale_prior(d).log_prob(lengthscale).sum()
    lp_noise = dsp_noise_prior().log_prob(obs_stddev).sum()

    return mll + lp_ls + lp_noise
