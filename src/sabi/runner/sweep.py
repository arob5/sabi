"""Sweep enumeration and result aggregation.

Sweep enumeration
-----------------
:func:`enumerate_sweep` expands a list of Hydra-style override strings into
the full Cartesian product of :class:`~sabi.runner.backends.RunSpec` objects.
Multi-value overrides use comma-separated notation (``key=v1,v2,...``),
mirroring Hydra's multirun syntax.

Result aggregation
------------------
:func:`aggregate_sweep` walks a completed sweep directory, reads each run's
``summary.json`` and ``metrics.jsonl``, and returns a :class:`pyarrow.Table`
(also written to ``sweep_results.parquet`` in the sweep directory).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sabi.runner.backends import RunSpec


def enumerate_sweep(
    overrides: list[str],
    sweep_dir: Path,
    *,
    n_seeds: int | None = None,
) -> list[RunSpec]:
    """Parse override strings and enumerate the Cartesian product as RunSpecs.

    Each override of the form ``key=v1,v2,...`` is expanded; single-value
    overrides (``key=v``) are held constant across all specs.

    Parameters
    ----------
    overrides:
        Hydra-style override strings.  Multi-value overrides use comma
        separation, e.g. ``["problem=gaussian_2d,banana", "seed=0,1"]``.
    sweep_dir:
        Root directory for this sweep.  Each :class:`RunSpec` receives a
        unique subdirectory derived from the override combination.
    n_seeds:
        When provided, inject ``seed=0,1,...,n_seeds-1`` automatically.
        Raises :class:`ValueError` if ``seed=`` already appears in
        *overrides* to prevent ambiguity.

    Returns
    -------
    list[RunSpec]
        One spec per Cartesian combination, in lexicographic override order.
    """
    overrides = list(overrides)
    if n_seeds is not None:
        if n_seeds < 1:
            raise ValueError(f"n_seeds must be >= 1, got {n_seeds}.")
        if any(o.startswith("seed=") for o in overrides):
            raise ValueError(
                "Cannot combine --n-seeds with an explicit seed= override."
            )
        overrides.append("seed=" + ",".join(str(i) for i in range(n_seeds)))

    parsed: list[tuple[str, list[str]]] = []
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Override {override!r} has no '=' separator.  "
                "Expected the form key=value or key=v1,v2,..."
            )
        key, vals_str = override.split("=", 1)
        parsed.append((key, vals_str.split(",")))

    keys = [k for k, _ in parsed]
    value_lists = [vs for _, vs in parsed]

    specs: list[RunSpec] = []
    for combo in itertools.product(*value_lists):
        combo_overrides = tuple(f"{k}={v}" for k, v in zip(keys, combo))
        run_name = "_".join(f"{k}-{v}" for k, v in zip(keys, combo))
        specs.append(RunSpec(overrides=combo_overrides, output_dir=sweep_dir / run_name))

    return specs


def _load_manifest(sweep_dir: Path) -> dict[str, dict[str, str | int | float]]:
    """Parse ``manifest.tsv`` → mapping from output_dir string to override dict.

    Override values are coerced to ``int`` or ``float`` where possible so
    that numeric sweep dimensions (``seed``, ``algorithm.n_rounds``, etc.)
    arrive as the right type in the aggregated table.
    """
    manifest_path = sweep_dir / "manifest.tsv"
    if not manifest_path.exists():
        return {}
    result: dict[str, dict[str, str | int | float]] = {}
    for line in manifest_path.read_text().splitlines()[1:]:  # skip header
        parts = line.split("\t", 2)
        if len(parts) < 3:
            continue
        _, out_dir, overrides_str = parts
        overrides: dict[str, str | int | float] = {}
        for token in overrides_str.split():
            if "=" in token:
                k, v = token.split("=", 1)
                overrides[k] = _coerce(v)
        result[out_dir.strip()] = overrides
    return result


def _coerce(v: str) -> str | int | float:
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def aggregate_sweep(sweep_dir: Path) -> pa.Table:
    """Collect per-run outputs from a completed sweep into a tidy table.

    For each subdirectory of *sweep_dir* that contains a ``summary.json``
    file, reads the manifest (primary), ``summary.json``, and the last row of
    ``metrics.jsonl`` (if present) and merges everything into one row.

    Override columns come from ``manifest.tsv`` when available, covering all
    submitted keys including nested ones (e.g. ``algorithm.n_rounds``).  This
    is more complete than reading ``config.yaml``, which only surfaces fields
    that happen to appear at the Hydra top level.

    Writes the result to ``sweep_dir/sweep_results.parquet`` and returns the
    :class:`pyarrow.Table` (rows = runs, columns = override keys + metrics).

    Parameters
    ----------
    sweep_dir:
        Root directory of a sweep previously dispatched by
        :class:`~sabi.runner.backends.RunBackend`.

    Raises
    ------
    ValueError
        If no completed runs (subdirectories with ``summary.json``) are found.
    """
    manifest_overrides = _load_manifest(sweep_dir)
    rows: list[dict] = []

    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue

        summary = json.loads(summary_path.read_text())
        row: dict = {"run_dir": str(run_dir)}

        # Override values from manifest (covers all submitted keys).
        row.update(manifest_overrides.get(str(run_dir), {}))

        row["n_evals_final"] = summary.get("n_evals_final")
        for k, v in summary.get("final_metrics", {}).items():
            row[f"final_{k}"] = v

        # Last-round metrics from metrics.jsonl.
        metrics_path = run_dir / "metrics.jsonl"
        if metrics_path.exists():
            lines = [ln for ln in metrics_path.read_text().splitlines() if ln.strip()]
            if lines:
                for k, v in json.loads(lines[-1]).items():
                    row[f"last_round_{k}"] = v

        rows.append(row)

    if not rows:
        raise ValueError(f"No completed runs found under {sweep_dir}.")

    table = pa.Table.from_pylist(rows)
    out_path = sweep_dir / "sweep_results.parquet"
    pq.write_table(table, out_path)
    print(f"Wrote {len(rows)} rows → {out_path}")
    return table
