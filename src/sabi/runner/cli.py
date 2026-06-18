"""CLI entry points for sabi sweep orchestration.

``sabi-submit``
    Enumerate a Cartesian product of config overrides and dispatch them to
    an execution backend (local sequential, local parallel, or SCC array).

``sabi-aggregate``
    Collect per-run outputs from a completed sweep into a single Parquet table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sabi.runner.backends import (
    LocalParallelBackend,
    LocalSequentialBackend,
    RunBackend,
    SCCArrayBackend,
)
from sabi.runner.sweep import aggregate_sweep, enumerate_sweep


def sabi_submit() -> None:
    """Entry point for the ``sabi-submit`` command."""
    parser = argparse.ArgumentParser(
        prog="sabi-submit",
        description=(
            "Enumerate a sweep and dispatch runs to a backend.\n\n"
            "Overrides use Hydra multirun notation: comma-separated values\n"
            "expand into the Cartesian product, e.g.:\n\n"
            "  sabi-submit --backend local_parallel --output-dir ./sweep \\\n"
            "      problem=gaussian_2d,banana acquisition=ei seed=0,1,2"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        default="local_sequential",
        choices=["local_sequential", "local_parallel", "scc_array"],
        help="Execution backend (default: local_sequential).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        dest="output_dir",
        metavar="DIR",
        help="Root directory for sweep outputs.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        dest="n_seeds",
        metavar="N",
        help=(
            "Number of random seeds (expands to seed=0,1,...,N-1). "
            "Cannot be combined with an explicit seed= override."
        ),
    )
    # local_parallel knob
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        dest="n_workers",
        metavar="N",
        help="Parallel worker processes (local_parallel only, default: 4).",
    )
    # SCC knobs
    parser.add_argument(
        "--scc-cores",
        type=int,
        default=4,
        dest="scc_cores",
        metavar="N",
        help="CPU slots per array task (default: 4).",
    )
    parser.add_argument(
        "--scc-mem-gb",
        type=int,
        default=8,
        dest="scc_mem_gb",
        metavar="GB",
        help="Total memory (GB) per array task (default: 8).",
    )
    parser.add_argument(
        "--scc-walltime",
        default="12:00:00",
        dest="scc_walltime",
        metavar="HH:MM:SS",
        help="Wall-clock time limit (default: 12:00:00).",
    )
    parser.add_argument(
        "--scc-queue",
        default="shared",
        dest="scc_queue",
        metavar="QUEUE",
        help="SGE queue name (default: shared).",
    )
    parser.add_argument(
        "--scc-batch-size",
        type=int,
        default=1,
        dest="scc_batch_size",
        metavar="N",
        help="Runs per array task (default: 1).",
    )
    parser.add_argument(
        "--scc-module",
        action="append",
        default=[],
        dest="scc_modules",
        metavar="MODULE",
        help="Module to load on SCC nodes (repeatable), e.g. python3/3.12.3.",
    )
    parser.add_argument(
        "--scc-python",
        default=sys.executable,
        dest="scc_python",
        metavar="PATH",
        help=(
            "Python executable path written into the qsub script "
            "(default: current interpreter)."
        ),
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        default=False,
        help="SCC only: call qsub automatically after writing the script.",
    )
    parser.add_argument(
        "--group-by-experiment",
        action="store_true",
        default=False,
        dest="group_by_experiment",
        help=(
            "SCC only: group specs sharing the same non-seed overrides into "
            "one array task, so all replicates of an experiment run on the "
            "same node.  Writes groups.tsv alongside the manifest."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help=(
            "Print what would be dispatched (spec count, override combinations, "
            "estimated array size) without running anything."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE[,VALUE...]",
        help="Hydra-style override strings.",
    )

    args = parser.parse_args()
    backend = _build_backend(args)

    try:
        specs = enumerate_sweep(args.overrides, args.output_dir, n_seeds=args.n_seeds)
    except ValueError as exc:
        print(f"sabi-submit: {exc}", file=sys.stderr)
        sys.exit(1)

    if not specs:
        print("sabi-submit: no run specs generated — check your overrides.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        _print_dry_run(specs, args)
        return

    print(f"Dispatching {len(specs)} runs via {args.backend}...")
    backend.dispatch(specs)
    print("Done.")


def _print_dry_run(specs: list, args: argparse.Namespace) -> None:
    from sabi.runner.backends import _group_by_experiment

    print(f"DRY RUN — {len(specs)} spec(s), backend={args.backend}")
    print(f"Output dir: {args.output_dir}")
    if args.backend == "scc_array":
        if args.group_by_experiment:
            _, groups = _group_by_experiment(specs)
            n_tasks = len(groups)
            print(f"group_by_experiment=True → {n_tasks} array task(s)")
        else:
            import math
            n_tasks = math.ceil(len(specs) / args.scc_batch_size)
            print(f"batch_size={args.scc_batch_size} → {n_tasks} array task(s)")
    print("Specs:")
    for spec in specs:
        print(f"  {' '.join(spec.overrides)}  →  {spec.output_dir}")


def _build_backend(args: argparse.Namespace) -> RunBackend:
    if args.backend == "local_sequential":
        return LocalSequentialBackend()
    if args.backend == "local_parallel":
        return LocalParallelBackend(n_workers=args.n_workers)
    if args.backend == "scc_array":
        return SCCArrayBackend(
            cores=args.scc_cores,
            mem_gb=args.scc_mem_gb,
            walltime=args.scc_walltime,
            queue=args.scc_queue,
            batch_size=args.scc_batch_size,
            modules=tuple(args.scc_modules),
            python_exe=args.scc_python,
            submit=args.submit,
            group_by_experiment=args.group_by_experiment,
        )
    raise ValueError(f"Unknown backend: {args.backend!r}")  # unreachable


def sabi_aggregate() -> None:
    """Entry point for the ``sabi-aggregate`` command."""
    parser = argparse.ArgumentParser(
        prog="sabi-aggregate",
        description="Collect sweep run outputs into a single Parquet table.",
    )
    parser.add_argument(
        "sweep_dir",
        type=Path,
        metavar="DIR",
        help="Sweep root directory produced by sabi-submit.",
    )
    args = parser.parse_args()

    try:
        table = aggregate_sweep(args.sweep_dir)
    except ValueError as exc:
        print(f"sabi-aggregate: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Columns : {table.schema.names}")
    print(f"Rows    : {len(table)}")
