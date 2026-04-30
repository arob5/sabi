"""NUTS-based reference posterior generation via ProbPipe `condition_on`.

Given a `TargetDistribution` (defined by a target function + form +
prior + support), call `condition_on(target_dist, ...)`. ProbPipe's
inference registry auto-dispatches to `tfp_nuts` (post-PR-#151, MCMC
methods accept `SupportsUnnormalizedLogProb`).

Diagnostics (R-hat, ESS, divergence count) are computed via ArviZ on
the returned `ApproximateDistribution.inference_data` and embedded in
the saved metadata. The regen script enforces minimum quality
thresholds before allowing the artifact to land.

Historical note: prior to the `TargetDistribution` introduction, this
module carried a local `_ProblemTargetDistribution` shim that wrapped
the problem's components into a Distribution. That shim is now
redundant — `TargetDistribution` is itself the right Distribution and
goes directly into `condition_on`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array
from probpipe import condition_on
from probpipe.core._distribution_base import Distribution

from sabi.problems.forms import LogDensityForm
from sabi.target_distribution import TargetDistribution


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCMCDiagnostics:
    """ArviZ-derived MCMC quality metrics, embedded in artifact metadata."""

    rhat: dict[str, float]            # per-dim R-hat
    ess_bulk: dict[str, float]        # per-dim bulk ESS
    ess_tail: dict[str, float]        # per-dim tail ESS
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
    """Pull R-hat / ESS / divergence diagnostics from ArviZ.

    `approx` is a ProbPipe `ApproximateDistribution` whose `auxiliary` is
    an ArviZ-compatible `InferenceData` / `DataTree`.
    """
    import arviz as az

    idata = approx.auxiliary
    if idata is None:
        # Fall back to constructing minimal InferenceData from chains.
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
    """Flatten an ArviZ R-hat / ESS xarray to a {label: value} dict.

    ArviZ returns a Dataset with one variable per posterior group; for
    sabi's flat-array posteriors there's typically one variable
    (`params`) carrying a per-dim array. We label entries `dim_0`,
    `dim_1`, … to mirror the parquet column names.
    """
    out: dict[str, float] = {}
    for var_name in da.data_vars:
        arr = np.asarray(da[var_name].values)
        flat = arr.ravel()
        for i, val in enumerate(flat):
            label = f"dim_{i}" if len(da.data_vars) == 1 else f"{var_name}_{i}"
            out[label] = float(val)
    return out


# ---------------------------------------------------------------------------
# NUTS-driven reference generation
# ---------------------------------------------------------------------------


def generate_via_nuts(
    *,
    target_function: Callable[[Array], Array],
    log_density_form: LogDensityForm,
    prior: Distribution,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...] = (),
    num_results: int = 1000,
    num_warmup: int = 500,
    num_chains: int = 4,
    random_seed: int = 0,
    name: str | None = None,
) -> tuple[Array, MCMCDiagnostics]:
    """Run NUTS via ProbPipe `condition_on` against the problem's
    unnormalized posterior.

    `target_function` is the **single-point** target callable (shape
    ``input_shape -> output_shape``) — that's what NUTS evaluates
    pointwise. The function builds a `TargetDistribution` from this
    (the constructor `jax.vmap`s the single-point callable for the
    batched view) and feeds it directly to `condition_on`. Default
    ``output_shape=()`` corresponds to the log-density-emulation case;
    set explicitly for forward-model targets.

    The support is derived from ``prior.support`` (sabi convention:
    ``prior`` is required and defines the support).

    Returns:
        (samples, diagnostics) where `samples` is a flat
        `(num_chains * num_results, *input_shape)` JAX array of
        post-warmup draws, and `diagnostics` carries R-hat / ESS /
        divergence counts.
    """
    target = TargetDistribution(
        target_single=target_function,
        name=name or "reference_target",
        input_shape=input_shape,
        output_shape=output_shape,
        log_density_form=log_density_form,
        prior=prior,
    )
    approx = condition_on(
        target,
        num_results=num_results,
        num_warmup=num_warmup,
        num_chains=num_chains,
        random_seed=random_seed,
    )
    diagnostics = _compute_diagnostics(approx)
    # Flatten chain structure: shape (num_chains, num_results, *input_shape)
    # → (num_chains * num_results, *input_shape).
    chains = jnp.stack([jnp.asarray(c) for c in approx.chains], axis=0)
    flat = chains.reshape((-1,) + tuple(input_shape))
    return flat, diagnostics
