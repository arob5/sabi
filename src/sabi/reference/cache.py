"""Disk-backed cache for reference posterior samples.

Public entry point: `load_or_generate_reference_samples(...)`.

Cache key is built from `(problem_param_tag, sampler_param_tag)`. The
filename encodes both for human-readable inspection — different problem
params or sampler configs land at different paths, so a Tier-A change
that invalidates the existing artifact is visible at a glance.

Behavior:

- If the artifact (`<key>.parquet` + `<key>.json`) exists and
  `regenerate=False`: load samples + return as
  `NumericEmpiricalDistribution`.
- Otherwise: run `generate_via_nuts(...)`, validate ArviZ diagnostics
  against thresholds, save artifact, return.

The artifact root defaults to `<repo_root>/reference_posteriors/`. Tier-A
artifacts are small (<1 MB) and committed to the repo. Tier-B (large /
expensive) artifacts will use git-lfs in a future phase.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from jax import Array
from probpipe._weights import Weights
from probpipe.core._distribution_base import Distribution
from probpipe.core._empirical import NumericEmpiricalDistribution

from sabi.problems.forms import LogDensityForm
from sabi.reference.io import (
    now_iso,
    read_metadata_json,
    read_samples_parquet,
    write_metadata_json,
    write_samples_parquet,
)
from sabi.reference.nuts import MCMCDiagnostics, generate_via_nuts


# Default artifact root: repo_root / reference_posteriors/
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "reference_posteriors"


# Quality thresholds enforced during regeneration. Tunable; current values
# are conservative defaults. Loosen via `quality_thresholds=...` if needed
# (e.g., for stretch benchmarks like Neal's funnel where some tail-ESS
# values can be marginal).
DEFAULT_MAX_RHAT = 1.05
DEFAULT_MIN_ESS = 400.0
DEFAULT_MAX_DIVERGENCE_RATE = 0.01  # fraction of post-warmup samples


def _artifact_path(
    cache_dir: Path,
    problem_name: str,
    cache_key: str,
    sampler_tag: str,
) -> tuple[Path, Path]:
    """Return (parquet_path, json_path) for a given key. Subdirectory per problem."""
    base = cache_dir / problem_name / f"{cache_key}_{sampler_tag}"
    return base.with_suffix(".parquet"), base.with_suffix(".json")


def _sampler_tag(num_results: int, num_warmup: int, num_chains: int, seed: int) -> str:
    return f"nuts_n{num_results}_w{num_warmup}_c{num_chains}_s{seed}"


def load_or_generate_reference_samples(
    *,
    problem_name: str,
    cache_key: str,
    target_map: Callable[[Array], Array],
    log_density_form: LogDensityForm,
    prior: Distribution,
    input_shape: tuple[int, ...],
    problem_params: dict[str, Any] | None = None,
    num_results: int = 1000,
    num_warmup: int = 500,
    num_chains: int = 4,
    random_seed: int = 0,
    cache_dir: Path | None = None,
    regenerate: bool = False,
    quality_thresholds: dict[str, float] | None = None,
    name: str | None = None,
) -> NumericEmpiricalDistribution:
    """Load reference samples from cache or generate them via NUTS.

    Args:
        problem_name: subdirectory under `cache_dir` (e.g., "neals_funnel").
        cache_key: human-readable identifier of the problem variant
            (e.g., "d2_sv3.0"). Different cache_keys → different artifacts.
        target_map, log_density_form, prior, input_shape:
            forwarded to `generate_via_nuts`. Support is derived from
            ``prior.support``.
        problem_params: optional dict embedded verbatim in metadata
            (e.g., `{"d": 2, "sigma_v": 3.0}`).
        num_results, num_warmup, num_chains, random_seed: NUTS
            configuration. Must match across saves and loads (sampler tag
            is hashed into the filename).
        cache_dir: artifact root; defaults to repo's `reference_posteriors/`.
        regenerate: if True, force NUTS regeneration even when a cached
            artifact exists.
        quality_thresholds: override `{max_rhat, min_ess, max_divergence_rate}`.

    Returns:
        `NumericEmpiricalDistribution` over the cached / generated samples.

    Raises:
        ValueError: if regeneration produced samples that fail the quality
            thresholds.
    """
    cache_dir = cache_dir or _DEFAULT_CACHE_DIR
    sampler_tag = _sampler_tag(num_results, num_warmup, num_chains, random_seed)
    parquet_path, json_path = _artifact_path(
        cache_dir, problem_name, cache_key, sampler_tag
    )

    if not regenerate and parquet_path.exists() and json_path.exists():
        samples = read_samples_parquet(parquet_path)
        return NumericEmpiricalDistribution(
            samples=samples,
            weights=Weights(n=samples.shape[0]),  # uniform weights
            name=name or f"{problem_name}_reference",
        )

    # Regenerate via NUTS.
    samples, diagnostics = generate_via_nuts(
        target_map=target_map,
        log_density_form=log_density_form,
        prior=prior,
        input_shape=input_shape,
        num_results=num_results,
        num_warmup=num_warmup,
        num_chains=num_chains,
        random_seed=random_seed,
        name=name,
    )

    thresholds = {
        "max_rhat": DEFAULT_MAX_RHAT,
        "min_ess": DEFAULT_MIN_ESS,
        "max_divergence_rate": DEFAULT_MAX_DIVERGENCE_RATE,
    }
    if quality_thresholds:
        thresholds.update(quality_thresholds)

    _validate_diagnostics(diagnostics, thresholds, problem_name=problem_name)

    metadata = {
        "problem_name": problem_name,
        "cache_key": cache_key,
        "problem_params": problem_params or {},
        "sampler": {
            "method": "tfp_nuts",
            "num_results": num_results,
            "num_warmup": num_warmup,
            "num_chains": num_chains,
            "random_seed": random_seed,
        },
        "diagnostics": diagnostics.to_dict(),
        "input_shape": list(input_shape),
        "n_samples": int(samples.shape[0]),
        "generated_at": now_iso(),
    }
    write_samples_parquet(samples, parquet_path)
    write_metadata_json(metadata, json_path)

    return NumericEmpiricalDistribution(
        samples=samples,
        weights=Weights(n=samples.shape[0]),
        name=name or f"{problem_name}_reference",
    )


def _validate_diagnostics(
    diagnostics: MCMCDiagnostics,
    thresholds: dict[str, float],
    *,
    problem_name: str,
) -> None:
    """Raise ValueError if NUTS quality is below the configured thresholds."""
    issues: list[str] = []
    max_rhat = diagnostics.max_rhat()
    min_ess = diagnostics.min_ess()
    n_total = diagnostics.num_chains * diagnostics.num_draws_per_chain
    div_rate = diagnostics.num_divergences / max(n_total, 1)

    if max_rhat > thresholds["max_rhat"]:
        issues.append(
            f"max R-hat = {max_rhat:.4f} > {thresholds['max_rhat']:.4f}"
        )
    if min_ess < thresholds["min_ess"]:
        issues.append(
            f"min ESS = {min_ess:.1f} < {thresholds['min_ess']:.1f}"
        )
    if div_rate > thresholds["max_divergence_rate"]:
        issues.append(
            f"divergence rate = {div_rate:.4f} > "
            f"{thresholds['max_divergence_rate']:.4f} "
            f"({diagnostics.num_divergences} divergences / {n_total} draws)"
        )

    if issues:
        raise ValueError(
            f"NUTS quality below thresholds for {problem_name!r}:\n  - "
            + "\n  - ".join(issues)
            + "\nLoosen quality_thresholds= or rerun with more warmup / chains."
        )
