"""Execution backends for sabi sweeps.

Each backend implements ``dispatch(specs)`` — the contract is:

- :class:`LocalSequentialBackend` and :class:`LocalParallelBackend` block
  until all runs complete.
- :class:`SCCArrayBackend` writes ``manifest.tsv`` + ``qsub_array.sh`` to the
  sweep directory and returns immediately; the user submits the script with
  ``qsub qsub_array.sh`` (or passes ``--submit`` to ``sabi-submit``).

All backends execute runs via ``python -m sabi.runner <overrides>``, so
local ↔ cluster parity is exact.
"""

from __future__ import annotations

import math
import subprocess
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RunSpec:
    """Specification for a single sabi run.

    Parameters
    ----------
    overrides:
        Hydra override strings, e.g. ``("problem=gaussian_2d", "seed=0")``.
    output_dir:
        Directory where this run's outputs (``metrics.jsonl``,
        ``summary.json``, ``config.yaml``) will be written.
    """

    overrides: tuple[str, ...]
    output_dir: Path


def _run_subprocess(spec: RunSpec) -> None:
    """Execute one run by invoking ``sabi.runner.main`` as a subprocess.

    The subprocess inherits the current environment (including ``PYTHONPATH``),
    so whichever ``python`` is active in the caller also resolves ``sabi``.
    """
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "sabi.runner",
        f"hydra.run.dir={spec.output_dir}",
        *spec.overrides,
    ]
    subprocess.run(cmd, check=True)


class RunBackend(ABC):
    """Abstract base for sabi execution backends."""

    @abstractmethod
    def dispatch(self, specs: list[RunSpec]) -> None:
        """Execute or schedule all runs in *specs*.

        Parameters
        ----------
        specs:
            Non-empty list of :class:`RunSpec` objects to dispatch.
        """


class LocalSequentialBackend(RunBackend):
    """Execute runs one at a time in the calling process."""

    def dispatch(self, specs: list[RunSpec]) -> None:
        for spec in specs:
            _run_subprocess(spec)


@dataclass(frozen=True)
class LocalParallelBackend(RunBackend):
    """Execute runs in parallel using a local process pool.

    Parameters
    ----------
    n_workers:
        Maximum number of concurrent subprocesses.
    """

    n_workers: int = 4

    def dispatch(self, specs: list[RunSpec]) -> None:
        failures: list[tuple[RunSpec, BaseException]] = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
            futures = {pool.submit(_run_subprocess, s): s for s in specs}
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc is not None:
                    failures.append((futures[fut], exc))
        if failures:
            lines = [
                f"  {spec.output_dir}: {type(exc).__name__}: {exc}"
                for spec, exc in failures
            ]
            raise RuntimeError(
                f"{len(failures)} of {len(specs)} runs failed:\n" + "\n".join(lines)
            )


@dataclass(frozen=True)
class SCCArrayBackend(RunBackend):
    """Generate a qsub job-array script for the BU SCC (SGE scheduler).

    :meth:`dispatch` writes two files under the sweep directory:

    ``manifest.tsv``
        One row per run: ``task_id``, ``output_dir``, ``overrides``.
    ``qsub_array.sh``
        An SGE array job script.  Each task reads its row from the manifest
        and calls ``python -m sabi.runner`` with the stored overrides.

    The script is *not* submitted automatically — run ``qsub qsub_array.sh``
    in the sweep directory, or pass ``--submit`` to ``sabi-submit``.

    Parameters
    ----------
    cores:
        Cores (slots) requested per array task.
    mem_gb:
        Total memory (GB) requested per array task.
    walltime:
        Wall-clock time limit in ``HH:MM:SS`` format.
    queue:
        SGE queue / parallel environment name (e.g. ``"shared"``,
        ``"gpu_scc"``).
    batch_size:
        Number of consecutive manifest rows executed inside a single array
        task.  ``1`` (default) gives the finest-grained retry granularity.
    modules:
        ``module load`` arguments to prepend to the script, e.g.
        ``("python3/3.12.3", "cuda/12.2")``.
    python_exe:
        Python executable path to use inside the qsub script.  Defaults to
        the interpreter running ``sabi-submit``, which is correct when the
        cluster nodes share the same file system mount.
    submit:
        When ``True``, call ``qsub qsub_array.sh`` automatically after
        writing the script.  Requires ``qsub`` to be on ``PATH``.
    group_by_experiment:
        When ``True``, group specs that share the same non-``seed`` overrides
        into a single array task, so that all replicates of one experiment run
        sequentially on the same node.  A companion ``groups.tsv`` is written
        alongside the manifest so the script knows each task's row range.
        ``batch_size`` is ignored when this flag is set.
    """

    cores: int = 4
    mem_gb: int = 8
    walltime: str = "12:00:00"
    queue: str = "shared"
    batch_size: int = 1
    modules: tuple[str, ...] = field(default_factory=tuple)
    python_exe: str = sys.executable
    submit: bool = False
    group_by_experiment: bool = False

    def dispatch(self, specs: list[RunSpec]) -> None:
        if not specs:
            return
        sweep_dir = specs[0].output_dir.parent
        sweep_dir.mkdir(parents=True, exist_ok=True)
        (sweep_dir / "logs").mkdir(exist_ok=True)

        if self.group_by_experiment:
            ordered, groups = _group_by_experiment(specs)
            self._write_manifest(ordered, sweep_dir)
            self._write_groups_file(groups, sweep_dir)
            self._write_qsub_script_grouped(groups, sweep_dir)
        else:
            self._write_manifest(specs, sweep_dir)
            self._write_qsub_script(specs, sweep_dir)

        if self.submit:
            script_path = sweep_dir / "qsub_array.sh"
            subprocess.run(["qsub", str(script_path)], check=True, cwd=sweep_dir)

    def _write_manifest(self, specs: list[RunSpec], sweep_dir: Path) -> None:
        manifest_path = sweep_dir / "manifest.tsv"
        with manifest_path.open("w") as f:
            f.write("task_id\toutput_dir\toverrides\n")
            for i, spec in enumerate(specs, start=1):
                overrides_str = " ".join(spec.overrides)
                f.write(f"{i}\t{spec.output_dir}\t{overrides_str}\n")

    def _write_qsub_script(self, specs: list[RunSpec], sweep_dir: Path) -> None:
        n_specs = len(specs)
        n_array_tasks = math.ceil(n_specs / self.batch_size)
        module_lines = (
            "\n".join(f"module load {m}" for m in self.modules) if self.modules else ""
        )
        mem_per_core = max(1, self.mem_gb // self.cores)

        # Each array task runs batch_size consecutive manifest rows.
        # Manifest rows are 1-indexed; file line = row + 1 (header offset).
        # The last task may run fewer than batch_size rows when n_specs is not
        # a multiple of batch_size.
        script = f"""\
#!/bin/bash
#$ -N sabi_sweep
#$ -t 1-{n_array_tasks}
#$ -pe omp {self.cores}
#$ -l h_rt={self.walltime}
#$ -l mem_per_core={mem_per_core}G
#$ -q {self.queue}
#$ -j y
#$ -o {sweep_dir}/logs/
#$ -cwd

{module_lines}

MANIFEST={sweep_dir}/manifest.tsv
BATCH_SIZE={self.batch_size}
N_SPECS={n_specs}

START=$(( (SGE_TASK_ID - 1) * BATCH_SIZE + 1 ))
END=$(( SGE_TASK_ID * BATCH_SIZE ))
if [ $END -gt $N_SPECS ]; then END=$N_SPECS; fi

for ROW in $(seq $START $END); do
    LINE=$(awk -v r="$ROW" 'NR == r + 1' "$MANIFEST")
    OUTPUT_DIR=$(printf '%s' "$LINE" | cut -f2)
    OVERRIDES=$(printf '%s' "$LINE" | cut -f3)
    mkdir -p "$OUTPUT_DIR"
    {self.python_exe} -m sabi.runner hydra.run.dir="$OUTPUT_DIR" $OVERRIDES
done
"""
        (sweep_dir / "qsub_array.sh").write_text(script)

    def _write_groups_file(
        self, groups: list[list[RunSpec]], sweep_dir: Path
    ) -> None:
        """Write groups.tsv: task_id → (start_row, end_row) in the manifest."""
        groups_path = sweep_dir / "groups.tsv"
        row = 1
        with groups_path.open("w") as f:
            f.write("task_id\tstart_row\tend_row\n")
            for task_id, group in enumerate(groups, start=1):
                f.write(f"{task_id}\t{row}\t{row + len(group) - 1}\n")
                row += len(group)

    def _write_qsub_script_grouped(
        self, groups: list[list[RunSpec]], sweep_dir: Path
    ) -> None:
        """Write qsub_array.sh for group_by_experiment mode.

        Each array task reads its row range from groups.tsv and runs
        all specs in that range sequentially.
        """
        n_array_tasks = len(groups)
        module_lines = (
            "\n".join(f"module load {m}" for m in self.modules) if self.modules else ""
        )
        mem_per_core = max(1, self.mem_gb // self.cores)

        script = f"""\
#!/bin/bash
#$ -N sabi_sweep
#$ -t 1-{n_array_tasks}
#$ -pe omp {self.cores}
#$ -l h_rt={self.walltime}
#$ -l mem_per_core={mem_per_core}G
#$ -q {self.queue}
#$ -j y
#$ -o {sweep_dir}/logs/
#$ -cwd

{module_lines}

MANIFEST={sweep_dir}/manifest.tsv
GROUPS={sweep_dir}/groups.tsv

GROUP_LINE=$(awk -v t="$SGE_TASK_ID" 'NR == t + 1' "$GROUPS")
START=$(printf '%s' "$GROUP_LINE" | cut -f2)
END=$(printf '%s' "$GROUP_LINE" | cut -f3)

for ROW in $(seq "$START" "$END"); do
    LINE=$(awk -v r="$ROW" 'NR == r + 1' "$MANIFEST")
    OUTPUT_DIR=$(printf '%s' "$LINE" | cut -f2)
    OVERRIDES=$(printf '%s' "$LINE" | cut -f3)
    mkdir -p "$OUTPUT_DIR"
    {self.python_exe} -m sabi.runner hydra.run.dir="$OUTPUT_DIR" $OVERRIDES
done
"""
        (sweep_dir / "qsub_array.sh").write_text(script)


def _group_by_experiment(
    specs: list[RunSpec],
) -> tuple[list[RunSpec], list[list[RunSpec]]]:
    """Group specs sharing the same non-seed overrides.

    Returns a tuple of:
    - *ordered*: specs sorted so all seeds of each experiment are consecutive.
    - *groups*: the same specs partitioned into per-experiment lists,
      preserving the order in which experiments first appear in *specs*.
    """
    group_map: dict[tuple[str, ...], list[RunSpec]] = defaultdict(list)
    for spec in specs:
        key = tuple(o for o in spec.overrides if not o.startswith("seed="))
        group_map[key].append(spec)
    groups = list(group_map.values())
    ordered = [s for group in groups for s in group]
    return ordered, groups
