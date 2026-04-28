"""Neal's funnel posterior — a stress benchmark with no analytic posterior.

Standard form (Radford Neal, 2003):

    v ~ N(0, σ_v²)
    x_i | v ~ N(0, exp(v))   for i = 1, ..., d

The joint posterior over (v, x_1, ..., x_d) is highly anisotropic — the
"funnel" geometry: tight neck at low v (variance ≈ exp(-9) ≈ 0.0001 for
v = -9) opening to a fat bulb at high v (variance ≈ exp(9) ≈ 8100 for
v = +9). MCMC diagnostics struggle without proper geometry-aware
sampling; we run NUTS and accept what comes out within reason.

The reference distribution is generated via NUTS (`condition_on(target)`)
and cached on disk. First call to `neals_funnel(...)` with a new
parameter combination triggers regeneration (slow, ~minutes); subsequent
calls load from the parquet artifact (fast).

Shapes: `input_shape=(d+1,)`, `output_shape=()`.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from probpipe.core.constraints import interval
from probpipe.distributions.continuous import Uniform

from sabi.problems.base import Problem
from sabi.problems.forms import Identity
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

    p = d + 1  # total dimension: v + d x_i's

    # --- target log-density ------------------------------------------------
    # log p(v, x) = log p(v) + sum_i log p(x_i | v)
    #             = -0.5 v² / σ_v² - 0.5 log(2π σ_v²)
    #               + sum_i [ -0.5 x_i² exp(-v) - 0.5 v - 0.5 log(2π) ]
    log2pi = jnp.log(2.0 * jnp.pi)
    sigma_v_sq = sigma_v * sigma_v
    norm_const = -0.5 * (log2pi + jnp.log(sigma_v_sq))  # log p(v) normalizer

    def log_prob(theta: Array) -> Array:
        v = theta[0]
        x = theta[1:]
        log_p_v = norm_const - 0.5 * v * v / sigma_v_sq
        # log p(x_i | v) per dim, summed
        log_p_x_given_v = -0.5 * d * (log2pi + v) - 0.5 * jnp.exp(-v) * jnp.sum(x * x)
        return log_p_v + log_p_x_given_v

    # --- support + design distribution ------------------------------------
    lower = jnp.asarray([-v_bound] + [-x_bound] * d, dtype=jnp.float64)
    upper = jnp.asarray([v_bound] + [x_bound] * d, dtype=jnp.float64)
    support = interval(lower, upper)
    prior = Uniform(low=lower, high=upper, name=f"neals_funnel_design_d{d}_sv{sigma_v}")

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
        target_function=log_prob,
        log_density_form=Identity(),
        prior=prior,
        support=support,
        input_shape=(p,),
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

    return Problem(
        name="neals_funnel",
        input_shape=(p,),
        output_shape=(),
        target_function=log_prob,
        log_density_form=Identity(),
        prior=prior,
        support=support,
        reference_distribution=reference,
    )
