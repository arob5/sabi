"""Hydra entry point for sabi v0.

Resolves the config tree under `configs/`, builds the problem + algorithm,
runs the loop, and writes per-round metrics + final summary as JSONL + JSON.
No W&B / Parquet yet — minimal footprint for v0.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import jax
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from sabi.algorithms.loop import run
from sabi.runner.build import build_algorithm, build_problem


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    return x


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # v0: x64 on by default — MMD and analytic refs need the precision.
    jax.config.update("jax_enable_x64", True)

    # Hydra 1.2+ does not chdir into the run dir by default; query it explicitly.
    out_dir = Path(HydraConfig.get().runtime.output_dir)

    problem = build_problem(cfg.problem)
    algorithm = build_algorithm(cfg, problem=problem)

    key = jax.random.key(int(cfg.get("seed", 0)))
    result = run(problem, algorithm, key)

    # Write per-round metrics as JSONL.
    with (out_dir / "metrics.jsonl").open("w") as f:
        for row in result.per_round_metrics:
            f.write(json.dumps({k: _jsonable(v) for k, v in row.items()}) + "\n")

    # Summary.
    summary = {
        "problem": problem.name,
        "n_evals_final": int(result.X.shape[0]),
        "final_metrics": {k: _jsonable(v) for k, v in result.final_metrics.items()},
        "tempering_states": _jsonable(result.tempering_states),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Echo resolved config.
    (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
