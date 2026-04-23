# sabi

A test framework for algorithms that perform **s**equential **a**daptive **B**ayesian **i**nference — adaptive loops that build a surrogate of an expensive (possibly noisy) posterior density from a sparse set of evaluations, analogous to Bayesian optimization but targeting a full distribution rather than an optimum.

Status: design phase. See [`docs/design.md`](docs/design.md).

## Scope

- Pluggable benchmarks, surrogates, acquisitions, posterior estimators, and metrics
- YAML-driven experiments (Hydra) with ablations, replicates, and reproducible seeding
- JAX-native; optional integration with [ProbPipe](https://github.com/TARPS-group/prob-pipe) for probabilistic abstractions
