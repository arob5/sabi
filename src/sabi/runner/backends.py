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

import subprocess
import sys
from abc import ABC, abstractmethod
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
    """

    cores: int = 4
    mem_gb: int = 8
    walltime: str = "12:00:00"
    queue: str = "shared"
    batch_size: int = 1
    modules: tuple[str, ...] = field(default_factory=tuple)
    python_exe: str = sys.executable

    def dispatch(self, specs: list[RunSpec]) -> None:
        if not specs:
            return
        sweep_dir = specs[0].output_dir.parent
        sweep_dir.mkdir(parents=True, exist_ok=True)
        (sweep_dir / "logs").mkdir(exist_ok=True)
        self._write_manifest(specs, sweep_dir)
        self._write_qsub_script(specs, sweep_dir)

    def _write_manifest(self, specs: list[RunSpec], sweep_dir: Path) -> None:
        manifest_path = sweep_dir / "manifest.tsv"
        with manifest_path.open("w") as f:
            f.write("task_id\toutput_dir\toverrides\n")
            for i, spec in enumerate(specs, start=1):
                overrides_str = " ".join(spec.overrides)
                f.write(f"{i}\t{spec.output_dir}\t{overrides_str}\n")

    def _write_qsub_script(self, specs: list[RunSpec], sweep_dir: Path) -> None:
        n_tasks = len(specs)
        module_lines = (
            "\n".join(f"module load {m}" for m in self.modules) if self.modules else ""
        )
        mem_per_core = max(1, self.mem_gb // self.cores)

        script = f"""\
#!/bin/bash
#$ -N sabi_sweep
#$ -t 1-{n_tasks}
#$ -pe omp {self.cores}
#$ -l h_rt={self.walltime}
#$ -l mem_per_core={mem_per_core}G
#$ -q {self.queue}
#$ -j y
#$ -o {sweep_dir}/logs/
#$ -cwd

{module_lines}

MANIFEST={sweep_dir}/manifest.tsv

# Manifest line for this task: header is line 1, so task N is line N+1.
LINE=$(awk -v t=$SGE_TASK_ID 'NR == t + 1' "$MANIFEST")
OUTPUT_DIR=$(printf '%s' "$LINE" | cut -f2)
OVERRIDES=$(printf '%s' "$LINE" | cut -f3)

mkdir -p "$OUTPUT_DIR"
{self.python_exe} -m sabi.runner hydra.run.dir="$OUTPUT_DIR" $OVERRIDES
"""
        (sweep_dir / "qsub_array.sh").write_text(script)
