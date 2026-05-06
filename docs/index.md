# sabi

Sabi is a test framework for sequential, adaptive Bayesian inference algorithms.
It composes pluggable benchmarks, surrogates, acquisitions, posterior estimators,
and metrics behind a Hydra-driven runner — JAX-native, with optional integration
into [ProbPipe](https://github.com/arob5/prob-pipe).

## Where to start

The **spine** below is the curated read path for a new user. Read it in
order; each page sets up the next.

1. **{doc}`Getting Started <getting_started>`** — a 2D run end to end
   with the four-panel result.
2. **{doc}`Sabi at a glance <overview>`** — the five abstractions, the
   emulator-vs-surrogate distinction, what `Algorithm` composes.
3. **{doc}`Notation <notation>`** — shape conventions, `x` vs `θ`, the
   public-batched / private-single-point split.
4. **{doc}`What happens in run() <run_walkthrough>`** — every helper
   inside the loop, with file:line links into `loop.py`.

Once you've read the spine:

- **Concepts** — the long-form design notes:
  {doc}`design <design>`, {doc}`emulators <emulators>`,
  {doc}`tempering <tempering>`, {doc}`scheduled metrics <scheduled_metrics>`.
- **API reference** — auto-generated from the `sabi.*` source tree;
  rendered docstrings include the math.
- **{doc}`Contributing <contributing>`** — code/docs conventions for PRs.

## Project status

Sabi is in active design and prototyping. The roadmap and current scope live in
{doc}`v1_plan <v1_plan>`. Many features are gated on parallel work in
[ProbPipe](https://github.com/arob5/prob-pipe); cross-cutting issues are tracked
in {doc}`probpipe_issues <probpipe_issues>`.

```{toctree}
:hidden:
:caption: Tutorials (the spine)

getting_started
overview
notation
run_walkthrough
```

```{toctree}
:hidden:
:caption: Concepts

design
emulators
tempering
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
