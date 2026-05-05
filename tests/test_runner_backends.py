"""Tests for the sweep orchestration layer: backends, enumeration, and aggregation.

Fast tests (no subprocess / no JAX warm-up)
--------------------------------------------
- ``test_enumerate_sweep_*``   — pure Cartesian-product logic
- ``test_n_seeds_expansion_*`` — seed shorthand expansion
- ``test_scc_backend_*``       — script/manifest generation (no qsub)
- ``test_aggregate_sweep_*``   — reads hand-crafted JSON fixtures

Integration tests (gated behind SABI_BACKENDS_INTEGRATION=1)
-------------------------------------------------------------
- ``test_local_sequential_trivial_sweep``
- ``test_local_parallel_matches_sequential``

These integration tests invoke ``python -m sabi.runner`` as subprocesses,
so they pay JAX startup overhead (~10–30 s on a cold machine).  Gate them
in CI with:

    SABI_BACKENDS_INTEGRATION=1 scripts/python -m pytest tests/test_runner_backends.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sabi.runner.backends import (
    LocalParallelBackend,
    LocalSequentialBackend,
    RunSpec,
    SCCArrayBackend,
)
from sabi.runner.sweep import aggregate_sweep, enumerate_sweep

_INTEGRATION = pytest.mark.skipif(
    not os.environ.get("SABI_BACKENDS_INTEGRATION"),
    reason="set SABI_BACKENDS_INTEGRATION=1 to run backend integration tests",
)

# ---------------------------------------------------------------------------
# enumerate_sweep
# ---------------------------------------------------------------------------


def test_enumerate_sweep_cartesian_product(tmp_path):
    specs = enumerate_sweep(["problem=a,b", "seed=0,1"], tmp_path)
    assert len(specs) == 4
    combos = {s.overrides for s in specs}
    assert ("problem=a", "seed=0") in combos
    assert ("problem=a", "seed=1") in combos
    assert ("problem=b", "seed=0") in combos
    assert ("problem=b", "seed=1") in combos


def test_enumerate_sweep_single_value_is_constant(tmp_path):
    specs = enumerate_sweep(["problem=gaussian_2d", "seed=0,1"], tmp_path)
    assert len(specs) == 2
    assert all("problem=gaussian_2d" in s.overrides for s in specs)


def test_enumerate_sweep_output_dirs_are_unique(tmp_path):
    specs = enumerate_sweep(["problem=a,b", "seed=0,1"], tmp_path)
    dirs = [s.output_dir for s in specs]
    assert len(dirs) == len(set(dirs))


def test_enumerate_sweep_output_dirs_under_sweep_dir(tmp_path):
    specs = enumerate_sweep(["problem=a,b"], tmp_path)
    for spec in specs:
        assert spec.output_dir.parent == tmp_path


def test_enumerate_sweep_missing_equals_raises(tmp_path):
    with pytest.raises(ValueError, match="no '=' separator"):
        enumerate_sweep(["problem"], tmp_path)


def test_enumerate_sweep_empty_overrides(tmp_path):
    # No overrides → one spec with empty overrides (the "null sweep").
    specs = enumerate_sweep([], tmp_path)
    assert len(specs) == 1
    assert specs[0].overrides == ()


# ---------------------------------------------------------------------------
# --n-seeds expansion
# ---------------------------------------------------------------------------


def test_n_seeds_expansion_count(tmp_path):
    specs = enumerate_sweep(["problem=a,b"], tmp_path, n_seeds=50)
    assert len(specs) == 100  # 2 problems × 50 seeds


def test_n_seeds_expansion_seed_range(tmp_path):
    specs = enumerate_sweep(["problem=a"], tmp_path, n_seeds=5)
    seeds = {int(o.split("=")[1]) for s in specs for o in s.overrides if o.startswith("seed=")}
    assert seeds == {0, 1, 2, 3, 4}


def test_n_seeds_conflicts_with_explicit_seed(tmp_path):
    with pytest.raises(ValueError, match="Cannot combine"):
        enumerate_sweep(["seed=0,1"], tmp_path, n_seeds=3)


def test_n_seeds_zero_raises(tmp_path):
    with pytest.raises(ValueError, match="n_seeds must be >= 1"):
        enumerate_sweep([], tmp_path, n_seeds=0)


# ---------------------------------------------------------------------------
# SCCArrayBackend — script/manifest generation (no qsub required)
# ---------------------------------------------------------------------------


def _make_specs(sweep_dir: Path, n: int = 3) -> list[RunSpec]:
    return [
        RunSpec(
            overrides=(f"problem=gaussian_2d", f"seed={i}"),
            output_dir=sweep_dir / f"run_{i}",
        )
        for i in range(n)
    ]


def test_scc_backend_creates_manifest(tmp_path):
    specs = _make_specs(tmp_path)
    SCCArrayBackend().dispatch(specs)
    manifest = tmp_path / "manifest.tsv"
    assert manifest.exists()
    lines = manifest.read_text().splitlines()
    assert lines[0] == "task_id\toutput_dir\toverrides"
    assert len(lines) == 1 + len(specs)  # header + data rows


def test_scc_backend_creates_qsub_script(tmp_path):
    specs = _make_specs(tmp_path, n=3)
    SCCArrayBackend().dispatch(specs)
    script = (tmp_path / "qsub_array.sh").read_text()
    assert "#$ -t 1-3" in script
    assert "python -m sabi.runner" in script


def test_scc_backend_script_array_size_matches_specs(tmp_path):
    n = 7
    specs = _make_specs(tmp_path, n=n)
    SCCArrayBackend().dispatch(specs)
    script = (tmp_path / "qsub_array.sh").read_text()
    assert f"#$ -t 1-{n}" in script


def test_scc_backend_manifest_task_ids_sequential(tmp_path):
    specs = _make_specs(tmp_path, n=4)
    SCCArrayBackend().dispatch(specs)
    lines = (tmp_path / "manifest.tsv").read_text().splitlines()[1:]  # skip header
    task_ids = [int(line.split("\t")[0]) for line in lines]
    assert task_ids == list(range(1, 5))


def test_scc_backend_custom_knobs_reflected_in_script(tmp_path):
    specs = _make_specs(tmp_path, n=2)
    backend = SCCArrayBackend(
        cores=8,
        mem_gb=32,
        walltime="04:00:00",
        queue="gpu_scc",
        modules=("python3/3.12.3", "cuda/12.2"),
    )
    backend.dispatch(specs)
    script = (tmp_path / "qsub_array.sh").read_text()
    assert "#$ -pe omp 8" in script
    assert "l mem_per_core=4G" in script
    assert "04:00:00" in script
    assert "#$ -q gpu_scc" in script
    assert "module load python3/3.12.3" in script
    assert "module load cuda/12.2" in script


def test_scc_backend_logs_dir_created(tmp_path):
    specs = _make_specs(tmp_path, n=2)
    SCCArrayBackend().dispatch(specs)
    assert (tmp_path / "logs").is_dir()


def test_scc_backend_empty_specs_is_noop(tmp_path):
    SCCArrayBackend().dispatch([])
    assert not (tmp_path / "manifest.tsv").exists()


# ---------------------------------------------------------------------------
# LocalParallelBackend — failure handling
# ---------------------------------------------------------------------------


@_INTEGRATION
def test_local_parallel_all_failures_surfaced(tmp_path):
    """All subprocess failures must be reported, not just the first."""
    # Specs with an unknown problem name exit non-zero immediately after
    # Hydra resolves config (before any JAX compute).
    bad_specs = [
        RunSpec(
            overrides=("problem=__nonexistent__",),
            output_dir=tmp_path / f"bad_{i}",
        )
        for i in range(3)
    ]
    with pytest.raises(RuntimeError) as exc_info:
        LocalParallelBackend(n_workers=3).dispatch(bad_specs)

    msg = str(exc_info.value)
    assert "3 of 3 runs failed" in msg


# ---------------------------------------------------------------------------
# aggregate_sweep — reads hand-crafted JSON fixtures
# ---------------------------------------------------------------------------


def _write_fake_run(run_dir: Path, seed: int, problem: str = "gaussian") -> None:
    run_dir.mkdir(parents=True)
    summary = {
        "problem": problem,
        "n_evals_final": 20,
        "final_metrics": {"mmd2": 0.01 + seed * 0.001},
        "tempering_states": [],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    metrics_rows = [
        json.dumps({"round": r, "mmd2": 0.05 - r * 0.005}) for r in range(3)
    ]
    (run_dir / "metrics.jsonl").write_text("\n".join(metrics_rows))
    # Minimal config.yaml so aggregate_sweep can read seed/problem/etc.
    config_yaml = f"seed: {seed}\nproblem:\n  name: {problem}\nemulator:\n  name: gp\nacquisition:\n  name: ei\n"
    (run_dir / "config.yaml").write_text(config_yaml)


def test_aggregate_sweep_row_count(tmp_path):
    for i in range(4):
        _write_fake_run(tmp_path / f"run_{i}", seed=i)
    table = aggregate_sweep(tmp_path)
    assert len(table) == 4


def test_aggregate_sweep_writes_parquet(tmp_path):
    _write_fake_run(tmp_path / "run_0", seed=0)
    aggregate_sweep(tmp_path)
    assert (tmp_path / "sweep_results.parquet").exists()


def test_aggregate_sweep_columns_include_seed_and_metrics(tmp_path):
    _write_fake_run(tmp_path / "run_0", seed=42)
    table = aggregate_sweep(tmp_path)
    assert "seed" in table.schema.names
    assert "final_mmd2" in table.schema.names


def test_aggregate_sweep_seed_values_round_trip(tmp_path):
    for seed in (7, 13):
        _write_fake_run(tmp_path / f"run_{seed}", seed=seed)
    table = aggregate_sweep(tmp_path)
    seeds = sorted(table.column("seed").to_pylist())
    assert seeds == [7, 13]


def test_aggregate_sweep_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="No completed runs"):
        aggregate_sweep(tmp_path)


def test_aggregate_sweep_skips_dirs_without_summary(tmp_path):
    _write_fake_run(tmp_path / "run_0", seed=0)
    (tmp_path / "not_a_run").mkdir()  # no summary.json
    table = aggregate_sweep(tmp_path)
    assert len(table) == 1


# ---------------------------------------------------------------------------
# Integration tests — actual subprocess execution (slow, opt-in)
# ---------------------------------------------------------------------------

# Tiny config: 4 initial points, 1 round, prior-sampling acquisition, no metrics.
# Keeps JAX work minimal while exercising the full subprocess path.
_TINY_OVERRIDES_TEMPLATE = (
    "algorithm.n_initial=4",
    "algorithm.n_rounds=1",
    "acquisition=prior_sampling",
    "metrics=[]",
)


@_INTEGRATION
def test_local_sequential_trivial_sweep(tmp_path):
    specs = [
        RunSpec(
            overrides=(*_TINY_OVERRIDES_TEMPLATE, f"seed={i}"),
            output_dir=tmp_path / f"run_{i}",
        )
        for i in range(2)
    ]
    LocalSequentialBackend().dispatch(specs)
    for spec in specs:
        assert (spec.output_dir / "summary.json").exists()
        assert (spec.output_dir / "metrics.jsonl").exists()


@_INTEGRATION
def test_local_parallel_trivial_sweep(tmp_path):
    specs = [
        RunSpec(
            overrides=(*_TINY_OVERRIDES_TEMPLATE, f"seed={i}"),
            output_dir=tmp_path / f"run_{i}",
        )
        for i in range(2)
    ]
    LocalParallelBackend(n_workers=2).dispatch(specs)
    for spec in specs:
        assert (spec.output_dir / "summary.json").exists()
        assert (spec.output_dir / "metrics.jsonl").exists()


@_INTEGRATION
def test_sequential_and_parallel_produce_same_outputs(tmp_path):
    """Same seed → same summary regardless of dispatch backend."""
    seed = 99
    overrides = (*_TINY_OVERRIDES_TEMPLATE, f"seed={seed}")

    seq_dir = tmp_path / "sequential"
    par_dir = tmp_path / "parallel"

    LocalSequentialBackend().dispatch(
        [RunSpec(overrides=overrides, output_dir=seq_dir)]
    )
    LocalParallelBackend(n_workers=1).dispatch(
        [RunSpec(overrides=overrides, output_dir=par_dir)]
    )

    seq_summary = json.loads((seq_dir / "summary.json").read_text())
    par_summary = json.loads((par_dir / "summary.json").read_text())
    assert seq_summary["n_evals_final"] == par_summary["n_evals_final"]
    assert seq_summary["problem"] == par_summary["problem"]
