r"""Neal's funnel posterior — a stress benchmark with no analytic posterior.

Standard form (Radford Neal, 2003):

.. math::

    v &\sim \mathcal{N}(0, \sigma_v^2), \\
    x_i \mid v &\sim \mathcal{N}(0, e^{v}), \quad i = 1, \dots, d.

The reference distribution is generated via NUTS
(``condition_on(target)``) and cached on disk. First call to
``neals_funnel(...)`` with a new parameter combination triggers
regeneration (slow, minutes); subsequent calls load from the parquet
artifact (fast).

Shapes: ``input_shape=(d+1,)``. The factory returns a `Problem` with
an analytical ``_unnormalized_log_prob`` on the ``target_distribution``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from sabi._probpipe_compat import independent_uniform
from sabi.problems.base import Problem
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
        d: number of x dimensions. Total parameter dimension is `d + 1`.
        sigma_v: standard deviation of v's marginal.
        v_bound: support / NUTS-init range for v ∈ [-v_bound, v_bound].
        x_bound: support / NUTS-init range for each x_i.
        num_results, num_warmup, num_chains, random_seed: NUTS configuration.
        quality_thresholds: optional override of the
            (max_rhat, min_ess, max_divergence_rate) gates.
    """
    if d < 1:
        raise ValueError(f"d must be ≥ 1, got {d}.")
    if sigma_v <= 0:
        raise ValueError(f"sigma_v must be positive, got {sigma_v}.")

    total_dim = d + 1

    log2pi = jnp.log(2.0 * jnp.pi)
    sigma_v_sq = sigma_v * sigma_v
    norm_const = -0.5 * (log2pi + jnp.log(sigma_v_sq))

    def log_prob_single(theta: Array) -> Array:
        v = theta[0]
        x = theta[1:]
        log_p_v = norm_const - 0.5 * v * v / sigma_v_sq
        log_p_x_given_v = -0.5 * d * (log2pi + v) - 0.5 * jnp.exp(-v) * jnp.sum(x * x)
        return log_p_v + log_p_x_given_v

    lower = jnp.asarray([-v_bound] + [-x_bound] * d, dtype=jnp.float64)
    upper = jnp.asarray([v_bound] + [x_bound] * d, dtype=jnp.float64)
    support = independent_uniform(
        low=lower, high=upper, name=f"neals_funnel_support_d{d}_sv{sigma_v}"
    ).support

    cache_key = (
        f"d{d}_sv{sigma_v}_vb{v_bound}_xb{x_bound}"
    )
    funnel_thresholds = {
        "max_rhat": 1.15,
        "min_ess": 30.0,
        "max_divergence_rate": 0.10,
    }
    if quality_thresholds:
        funnel_thresholds.update(quality_thresholds)

    reference = load_or_generate_reference_samples(
        problem_name="neals_funnel",
        cache_key=cache_key,
        target_log_prob=log_prob_single,
        support=support,
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
        name=f"neals_funnel_d{d}_target",
        input_shape=(total_dim,),
        support=support,
        unnormalized_log_prob=log_prob_single,
    )
    return Problem(
        target_distribution=target,
        reference_distribution=reference,
        name="neals_funnel",
    )
