r"""Neal's funnel posterior — a stress benchmark with no analytic posterior.

Standard form (Radford Neal, 2003):

.. math::

    v &\sim \mathcal{N}(0, \sigma_v^2), \\
    x_i \mid v &\sim \mathcal{N}(0, e^{v}), \quad i = 1, \dots, d.

Reference samples are generated via NUTS and cached on disk.

Public surface:

- :func:`neals_funnel` — factory returning a ``Problem``.
- :class:`NealsFunnelTarget` — :class:`TargetDistribution` subclass.
- :class:`NealsFunnelLogProbDecomposition` —
  :class:`LogProbTermTarget` subclass that emulates the full log-density.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from probpipe.core.constraints import Constraint

from sabi._probpipe_compat import independent_uniform
from sabi.density_decomposition import LogProbTermTarget
from sabi.problems.base import Problem
from sabi.target_distribution import TargetDistribution
from sabi.reference.cache import load_or_generate_reference_samples


# ---------------------------------------------------------------------------
# Private density helper — single-event input.
# ---------------------------------------------------------------------------


def _funnel_log_density(theta: Array, *, d: int, sigma_v: float) -> Array:
    """Vectorized: ``theta.shape == batch_shape + (d+1,)`` → ``batch_shape``.

    Indexes the trailing event axis with ``theta[..., 0]`` /
    ``theta[..., 1:]`` so leading batch dims pass through.
    """
    log2pi = jnp.log(2.0 * jnp.pi)
    sigma_v_sq = sigma_v * sigma_v
    norm_const = -0.5 * (log2pi + jnp.log(sigma_v_sq))
    v = theta[..., 0]
    x = theta[..., 1:]
    log_p_v = norm_const - 0.5 * v * v / sigma_v_sq
    log_p_x_given_v = (
        -0.5 * d * (log2pi + v) - 0.5 * jnp.exp(-v) * jnp.sum(x * x, axis=-1)
    )
    return log_p_v + log_p_x_given_v


# ---------------------------------------------------------------------------
# Subclasses
# ---------------------------------------------------------------------------


class NealsFunnelTarget(TargetDistribution):
    """``TargetDistribution`` for Neal's funnel."""

    def __init__(
        self,
        *,
        name: str,
        input_shape: tuple[int, ...],
        support: Constraint,
        d: int,
        sigma_v: float,
    ):
        self._d, self._sigma_v = d, sigma_v
        super().__init__(name=name, input_shape=input_shape, support=support)

    def _unnormalized_log_prob(self, theta: Array) -> Array:
        return _funnel_log_density(theta, d=self._d, sigma_v=self._sigma_v)


class NealsFunnelLogProbDecomposition(LogProbTermTarget):
    """Decomposition for Neal's funnel — full log-density emulation."""

    def __init__(
        self,
        *,
        name: str,
        input_shape: tuple[int, ...],
        support: Constraint,
        d: int,
        sigma_v: float,
    ):
        self._d, self._sigma_v = d, sigma_v
        super().__init__(
            name=name, input_shape=input_shape, support=support, prior=None
        )

    def target_map(self, theta: Array) -> Array:
        return _funnel_log_density(theta, d=self._d, sigma_v=self._sigma_v)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


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
    """Build the Neal's funnel benchmark."""
    if d < 1:
        raise ValueError(f"d must be ≥ 1, got {d}.")
    if sigma_v <= 0:
        raise ValueError(f"sigma_v must be positive, got {sigma_v}.")

    total_dim = d + 1

    lower = jnp.asarray([-v_bound] + [-x_bound] * d, dtype=jnp.float64)
    upper = jnp.asarray([v_bound] + [x_bound] * d, dtype=jnp.float64)
    support = independent_uniform(
        low=lower, high=upper, name=f"neals_funnel_support_d{d}_sv{sigma_v}"
    ).support

    target = NealsFunnelTarget(
        name=f"neals_funnel_d{d}_target",
        input_shape=(total_dim,),
        support=support,
        d=d,
        sigma_v=sigma_v,
    )

    cache_key = f"d{d}_sv{sigma_v}_vb{v_bound}_xb{x_bound}"
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
        target=target,
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
        target_distribution=target,
        reference_distribution=reference,
        name="neals_funnel",
    )
