r"""Neal's funnel posterior — a stress benchmark with no analytic posterior.

Standard form (Radford Neal, 2003):

.. math::

    v &\sim \mathcal{N}(0, \sigma_v^2), \\
    x_i \mid v &\sim \mathcal{N}(0, e^{v}), \quad i = 1, \dots, d.

The joint log-density is

.. math::

    \log p(v, x) =
        -\tfrac{1}{2} \frac{v^2}{\sigma_v^2}
        - \tfrac{d v}{2}
        - \tfrac{e^{-v}}{2} \sum_{i=1}^{d} x_i^2
        + C,

with constant :math:`C = -\tfrac{1}{2} \log(2\pi \sigma_v^2)
- \tfrac{d}{2} \log(2\pi)`.

The joint posterior over :math:`(v, x_1, \dots, x_d)` is highly
anisotropic — the "funnel" geometry: tight neck at low :math:`v`
(:math:`\mathrm{Var}(x_i \mid v=-9) = e^{-9} \approx 10^{-4}`) opening
to a fat bulb at high :math:`v` (:math:`\mathrm{Var}(x_i \mid v=+9)
= e^{9} \approx 8100`). MCMC diagnostics struggle without
geometry-aware sampling; we run NUTS and accept what comes out within
reason.

The reference distribution is generated via NUTS
(``condition_on(target)``) and cached on disk. First call to
``neals_funnel(...)`` with a new parameter combination triggers
regeneration (slow, minutes); subsequent calls load from the parquet
artifact (fast).

Shapes: ``input_shape=(d+1,)``, ``output_shape=()``. See
``docs/notation.md`` for sabi's shape conventions.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from sabi._probpipe_compat import independent_uniform
from sabi.problems.base import Problem
from sabi.problems.forms import Identity
from sabi.target_distribution import TargetDistribution
from sabi.reference.cache import load_or_generate_reference_samples


def neals_funnel(
    d: int = 2,
    sigma_v: float = 3.0,
    *,
    v_bound: float = 9.0,
    x_bound: float = 30.0,
    num_results: int = 2000,
    num_warmup: int = 2000,
    num_chains: int = 4,
    random_seed: int = 0,
    quality_thresholds: dict[str, float] | None = None,
) -> Problem:
    """Build the Neal's funnel benchmark.

    Args:
        d: number of x dimensions. Total parameter dimension is `d + 1`
            (one v, d x's).
        sigma_v: standard deviation of v's marginal.
        v_bound: design-distribution / support range for v ∈ [-v_bound, v_bound].
            Default 9.0 covers ±3 σ_v.
        x_bound: design-distribution / support range for each x_i ∈
            [-x_bound, x_bound]. Default 30 covers most of the bulk for
            moderate v; high-v tails are not fully covered (acceptable
            for a stretch benchmark).
        num_results, num_warmup, num_chains, random_seed: NUTS configuration
            for reference generation. Different values → different cached
            artifact.
        quality_thresholds: optional override of the
            (max_rhat, min_ess, max_divergence_rate) gates. The funnel is
            harder than the affine benchmarks; defaults can be loosened
            here if NUTS without geometry-aware reparameterization
            produces marginal diagnostics.
    """
    if d < 1:
        raise ValueError(f"d must be ≥ 1, got {d}.")
    if sigma_v <= 0:
        raise ValueError(f"sigma_v must be positive, got {sigma_v}.")

    total_dim = d + 1  # total parameter dimension: v + d x_i's

    # --- target log-density ------------------------------------------------
    # log p(v, x) = log p(v) + sum_i log p(x_i | v)
    #             = -0.5 v² / σ_v² - 0.5 log(2π σ_v²)
    #               + sum_i [ -0.5 x_i² exp(-v) - 0.5 v - 0.5 log(2π) ]
    log2pi = jnp.log(2.0 * jnp.pi)
    sigma_v_sq = sigma_v * sigma_v
    norm_const = -0.5 * (log2pi + jnp.log(sigma_v_sq))  # log p(v) normalizer

    def log_prob_single(theta: Array) -> Array:
        v = theta[0]
        x = theta[1:]
        log_p_v = norm_const - 0.5 * v * v / sigma_v_sq
        # log p(x_i | v) per dim, summed
        log_p_x_given_v = -0.5 * d * (log2pi + v) - 0.5 * jnp.exp(-v) * jnp.sum(x * x)
        return log_p_v + log_p_x_given_v

    # --- support + design distribution ------------------------------------
    lower = jnp.asarray([-v_bound] + [-x_bound] * d, dtype=jnp.float64)
    upper = jnp.asarray([v_bound] + [x_bound] * d, dtype=jnp.float64)
    # Multivariate-event prior over R^total_dim (event_shape == (total_dim,)). See
    # `sabi/_probpipe_compat.py` for the shim.
    prior = independent_uniform(
        low=lower, high=upper, name=f"neals_funnel_design_d{d}_sv{sigma_v}"
    )

    # --- reference distribution via NUTS (cached on disk) -----------------
    cache_key = (
        f"d{d}_sv{sigma_v}_vb{v_bound}_xb{x_bound}"
    )
    # Funnel-specific quality thresholds: vanilla NUTS without
    # reparameterization mixes poorly on the funnel geometry. The defaults
    # here loosen R-hat / ESS expectations relative to the strict tier-A
    # gates used by analytic-friendly benchmarks. Override via
    # `quality_thresholds=` if you want stricter guarantees (and a bigger
    # NUTS budget).
    funnel_thresholds = {
        "max_rhat": 1.15,
        # ESS threshold loosened to acknowledge that vanilla NUTS without
        # geometry-aware reparameterization mixes slowly on the funnel.
        # 30 ESS-per-dim is enough for sabi-side metric comparisons but
        # well below what one would expect on an isotropic target.
        "min_ess": 30.0,
        "max_divergence_rate": 0.10,
    }
    if quality_thresholds:
        funnel_thresholds.update(quality_thresholds)

    reference = load_or_generate_reference_samples(
        problem_name="neals_funnel",
        cache_key=cache_key,
        target_map=log_prob_single,
        log_density_form=Identity(),
        prior=prior,
        input_shape=(total_dim,),
        problem_params={
            "d": d,
            "sigma_v": sigma_v,
            "v_bound": v_bound,
            "x_bound": x_bound,
        },
        num_results=num_results,
        num_warmup=num_warmup,
        num_chains=num_chains,
        random_seed=random_seed,
        quality_thresholds=funnel_thresholds,
        name=f"neals_funnel_d{d}_reference",
    )

    target = TargetDistribution(
        target_single=log_prob_single,
        name=f"neals_funnel_d{d}_target",
        input_shape=(total_dim,),
        output_shape=(),
        log_density_form=Identity(),
        prior=prior,
    )
    return Problem(
        target_distribution=target,
        reference_distribution=reference,
        name="neals_funnel",
    )
