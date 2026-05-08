"""NUTS-based reference posterior generation via ProbPipe ``condition_on``.

Given a :class:`TargetDistribution` whose subclass implements an
analytical ``_unnormalized_log_prob``, calls
``condition_on(target, ...)``. ProbPipe's inference registry
auto-dispatches to ``tfp_nuts``.

Diagnostics (R-hat, ESS, divergence count) are computed via ArviZ on
the returned ``ApproximateDistribution.inference_data`` and embedded in
the saved metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array
from probpipe import condition_on

from sabi.target_distribution import TargetDistribution


@dataclass(frozen=True)
class MCMCDiagnostics:
    """ArviZ-derived MCMC quality metrics, embedded in artifact metadata."""

    rhat: dict[str, float]
    ess_bulk: dict[str, float]
    ess_tail: dict[str, float]
    num_divergences: int
    num_chains: int
    num_draws_per_chain: int

    def max_rhat(self) -> float:
        return max(self.rhat.values()) if self.rhat else float("nan")

    def min_ess(self) -> float:
        all_ess = list(self.ess_bulk.values()) + list(self.ess_tail.values())
        return min(all_ess) if all_ess else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rhat": dict(self.rhat),
            "ess_bulk": dict(self.ess_bulk),
            "ess_tail": dict(self.ess_tail),
            "num_divergences": int(self.num_divergences),
            "num_chains": int(self.num_chains),
            "num_draws_per_chain": int(self.num_draws_per_chain),
            "max_rhat": float(self.max_rhat()),
            "min_ess": float(self.min_ess()),
        }


def _compute_diagnostics(approx) -> MCMCDiagnostics:
    """Pull R-hat / ESS / divergence diagnostics from ArviZ."""
    import arviz as az

    idata = approx.auxiliary
    if idata is None:
        idata = az.from_dict(
            posterior={
                "params": np.stack([np.asarray(c) for c in approx.chains], axis=0)
            }
        )

    rhat_da = az.rhat(idata)
    ess_bulk_da = az.ess(idata, method="bulk")
    ess_tail_da = az.ess(idata, method="tail")

    rhat = _flatten_diag_dataarray(rhat_da)
    ess_bulk = _flatten_diag_dataarray(ess_bulk_da)
    ess_tail = _flatten_diag_dataarray(ess_tail_da)

    num_divergences = 0
    if hasattr(idata, "sample_stats") or (
        hasattr(idata, "children") and "sample_stats" in idata.children
    ):
        try:
            stats = idata["sample_stats"] if hasattr(idata, "children") else idata.sample_stats
            if "diverging" in stats:
                num_divergences = int(np.sum(np.asarray(stats["diverging"])))
        except (KeyError, AttributeError):
            pass

    return MCMCDiagnostics(
        rhat=rhat,
        ess_bulk=ess_bulk,
        ess_tail=ess_tail,
        num_divergences=num_divergences,
        num_chains=approx.num_chains,
        num_draws_per_chain=approx.num_draws,
    )


def _flatten_diag_dataarray(da) -> dict[str, float]:
    out: dict[str, float] = {}
    for var_name in da.data_vars:
        arr = np.asarray(da[var_name].values)
        flat = arr.ravel()
        for i, val in enumerate(flat):
            label = f"dim_{i}" if len(da.data_vars) == 1 else f"{var_name}_{i}"
            out[label] = float(val)
    return out


def generate_via_nuts(
    *,
    target: TargetDistribution,
    num_results: int = 1000,
    num_warmup: int = 500,
    num_chains: int = 4,
    random_seed: int = 0,
) -> tuple[Array, MCMCDiagnostics]:
    """Run NUTS via ``condition_on(target, ...)``.

    Returns ``(samples, diagnostics)`` — flat
    ``(num_chains * num_results, *target.input_shape)`` JAX array of
    post-warmup draws plus R-hat / ESS / divergence counts.
    """
    approx = condition_on(
        target,
        num_results=num_results,
        num_warmup=num_warmup,
        num_chains=num_chains,
        random_seed=random_seed,
    )
    diagnostics = _compute_diagnostics(approx)
    chains = jnp.stack([jnp.asarray(c) for c in approx.chains], axis=0)
    flat = chains.reshape((-1,) + tuple(target.input_shape))
    return flat, diagnostics
