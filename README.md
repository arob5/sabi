# sabi

A test framework for algorithms that perform **s**equential **a**daptive **B**ayesian **i**nference — adaptive loops that build a surrogate of an expensive (possibly noisy) posterior density from a sparse set of evaluations, analogous to Bayesian optimization but targeting a full distribution rather than an optimum.

Status: design phase. See [`docs/design.md`](docs/design.md).

## Scope

- Pluggable benchmarks, surrogates, acquisitions, posterior estimators, and metrics
- YAML-driven experiments (Hydra) with ablations, replicates, and reproducible seeding
- JAX-native; optional integration with [ProbPipe](https://github.com/TARPS-group/prob-pipe) for probabilistic abstractions

## Getting started

sabi uses [`uv`](https://docs.astral.sh/uv/) for dependency management and reproducible installs.

```bash
# Install dependencies (including dev + probpipe extras, per uv.lock)
uv sync --extra dev --extra probpipe

# Run tests
uv run pytest

# Run a Hydra experiment (once the runner exists)
uv run python -m sabi.runner
```

The lockfile (`uv.lock`) pins exact resolved versions across platforms. Regenerate with `uv lock` after dependency changes; commit the result.

### ProbPipe

`probpipe` is sourced from a local sibling clone at `../prob-pipe` (see `[tool.uv.sources]` in `pyproject.toml`). Clone ProbPipe next to sabi before running `uv sync --extra probpipe`:

```bash
git clone git@github.com:TARPS-group/prob-pipe.git ../prob-pipe
```
