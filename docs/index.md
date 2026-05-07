# sabi

Sabi is a test framework for sequential, adaptive Bayesian inference algorithms.
It composes pluggable benchmarks, surrogates, acquisitions, posterior estimators,
and metrics behind a Hydra-driven runner — JAX-native, with optional integration
into [ProbPipe](https://github.com/arob5/prob-pipe).

## Where to start

- **{doc}`Getting Started <getting_started>`** — a 2D run end to end with
  visualizations of the target, acquired points, surrogate posterior, and a
  per-round metric trajectory.
- **Concepts** — the design notes that have shaped the library:
  {doc}`design <design>`, {doc}`notation <notation>`,
  {doc}`emulators <emulators>`, {doc}`tempering <tempering>`,
  {doc}`link functions <link_functions>`,
  {doc}`scheduled metrics <scheduled_metrics>`.
- **API reference** — auto-generated from the `sabi.*` source tree; rendered
  docstrings include the math.
- **{doc}`Contributing <contributing>`** — code/docs conventions for PRs.

## Project status

Sabi is in active design and prototyping. The roadmap and current scope live in
{doc}`v1_plan <v1_plan>`. Many features are gated on parallel work in
[ProbPipe](https://github.com/arob5/prob-pipe); cross-cutting issues are tracked
in {doc}`probpipe_issues <probpipe_issues>`.

```{toctree}
:hidden:
:caption: Tutorials

getting_started
```

```{toctree}
:hidden:
:caption: Concepts

design
notation
emulators
tempering
link_functions
scheduled_metrics
v1_plan
probpipe_random_measure_proposal
probpipe_issues
```

```{toctree}
:hidden:
:caption: Reference
:glob:

api/*
contributing
```
