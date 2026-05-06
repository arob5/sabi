# sabi

SABI is a framework for quickly developing and testing algorithms for
Sequential Adaptive Bayesian Inference. A primary motivation for such
algorithms is performing Bayesian inference when the posterior density
is a computationally-expensive, black-box function. SABI provides
pluggable benchmarks, surrogate models, acquisition functions,
optimization routines, and other algorithmic components. SABI builds on
top of:

- [Hydra](https://hydra.cc/), for convenient algorithm comparison,
  replication, and ablation studies.
- [ProbPipe](https://github.com/arob5/prob-pipe), for probabilistic
  abstractions.

```{admonition} Connection to Bayesian Optimization
:class: note

SABI mirrors existing frameworks for black-box function optimization.
However, it aims to solve a different problem: rather than optimizing a
function, the goal is to approximate a probability distribution.
```

## Where to start

New users are encouraged to start by reading through the following
high-level tutorials in order:

1. **{doc}`Getting Started <getting_started>`** — an end-to-end run
   illustrated on a toy example.
2. **{doc}`SABI at a glance <overview>`** — understanding the main
   abstractions.
3. **{doc}`Notation <notation>`** — standardized SABI conventions.
4. **{doc}`Anatomy of a run <run_walkthrough>`** — walking through the
   code logic.

Follow-up reading:

- **Tutorial series** — in-depth walkthroughs of more advanced topics
  *(planned; not yet written)*.
- **{doc}`API reference <api/sabi/sabi>`**.
- **{doc}`Contributing <contributing>`**.

## Project status

SABI is under active development and does not yet have a stable API. It
is being developed in parallel with
[ProbPipe](https://github.com/arob5/prob-pipe).

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
