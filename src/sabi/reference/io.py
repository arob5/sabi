"""Parquet + JSON I/O for cached reference posteriors.

Artifact layout under `reference_posteriors/<problem>/`:

- `<problem>_<param-tag>_<sampler-tag>.parquet` — flat sample matrix
  with one row per draw; columns named `dim_0`, `dim_1`, ...
- `<problem>_<param-tag>_<sampler-tag>.json` — metadata: problem
  parameters, sampler configuration, ArviZ diagnostics (R-hat per dim,
  min ESS, divergences), generation timestamp, ProbPipe / sabi commit
  SHAs.

The flat sample matrix loses per-chain structure on disk. Per-chain
diagnostics are computed during regeneration (when the chain structure
is still available from `ApproximateDistribution.chains`) and embedded
in the metadata JSON; sabi consumers see the loaded reference as a
`NumericEmpiricalDistribution`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from jax import Array


def write_samples_parquet(samples: Array, path: Path) -> None:
    """Write a flat `(n_samples, dim)` array to a Parquet file."""
    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError(
            f"Expected (n_samples, dim) array; got shape {samples.shape}."
        )
    n, d = samples.shape
    table = pa.table({f"dim_{i}": samples[:, i] for i in range(d)})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def read_samples_parquet(path: Path) -> Array:
    """Read a Parquet file written by :func:`write_samples_parquet`."""
    table = pq.read_table(path)
    cols = [c for c in table.column_names if c.startswith("dim_")]
    cols.sort(key=lambda s: int(s.split("_", 1)[1]))
    arr = np.stack([table.column(c).to_numpy() for c in cols], axis=-1)
    return jnp.asarray(arr)


def write_metadata_json(metadata: dict[str, Any], path: Path) -> None:
    """Write metadata dict to a JSON file (UTF-8, 2-space indent)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=_json_default, sort_keys=True)
        f.write("\n")


def read_metadata_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def now_iso() -> str:
    """UTC timestamp in ISO 8601 (suitable for metadata.generated_at)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(o: Any) -> Any:
    """JSON-encode numpy / jax arrays + scalars."""
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"Cannot JSON-encode {type(o).__name__}")
