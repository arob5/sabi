# sabi — Design Document

**Status:** draft v0.1 — pre-implementation  
**Last updated:** 2026-04-23

## 1. Purpose

A test framework for algorithms performing **sequential surrogate-based Bayesian inference**: adaptive loops that approximate an expensive (possibly noisy) posterior density by fitting a stochastic surrogate over a sparse, adaptively chosen set of evaluations. The loop resembles Bayesian optimization, but the goal is to approximate a full distribution (and expectations w.r.t. it) rather than locate an optimum.

The framework is for **comparing algorithm variants** — different surrogates, acquisitions, posterior estimators, initial designs — on a set of benchmarks with reproducible, ablatable experiments.

## 2. Non-goals (v1)

- Production inference library. sabi is a testbed.
- Stochastic/noisy targets. Deferred (see §12).
- Framework-agnostic tensors. **JAX only** in v1.
- Built-in BOTorch/PyTorch backends. A surrogate interface leaves this open as a future plug-in.
- W&B logging. Local JSONL + Parquet in v1; W&B becomes a second logger backend later.

## 3. Relationship to ProbPipe

sabi is a separate repo with no hard dependency on ProbPipe in v1. ProbPipe is an **optional** source of probabilistic primitives via `pip install sabi[probpipe]`.

Primitives we expect to leverage as they mature:

| ProbPipe abstraction | sabi use |
|---|---|
| `Distribution[T]` | Return type of `PosteriorEstimator` |
| `EmpiricalDistribution` | Particle-based deterministic posterior estimates |
| `RandomFunction[X, Y]` | Contract for `Surrogate` (emulator) |
| `ArrayRandomFunction` | Shape + batch-dim conventions for `Surrogate`; v1+ `Surrogate` is an `ArrayRandomFunction` subclass |
| `GaussianRandomFunction` | GP-backed surrogate |
| `Constraint` + `support` | Parameter-space support in `Problem` |
| `RandomMeasure` (Phase 5) | Base for `SurrogatePosterior` |
| `Pushforward` (planned) | Assembly of `SurrogatePosterior` from surrogate + `LogDensityForm` |
| `condition_on` | Future implementation path for `Surrogate.update` |
| `_mc_expectation` / `BootstrapDistribution` | Metric computation with MC-error tracking |

**Strategy:** ship minimal local implementations of missing abstractions (pushforward, random-measure) inside sabi; swap to ProbPipe implementations as they land. The framework's interfaces are deliberately close to the ProbPipe shapes so the swap is a rename, not a refactor.

## 4. Core abstractions

### 4.1 `Problem` — a benchmark

The expensive Bayesian inference target plus everything needed to construct, transform, and evaluate against it. Shape conventions follow ProbPipe `ArrayRandomFunction` (see `docs/notation.md`): a single target input has shape `input_shape`, a single target output has shape `output_shape`, and design sets `X`, `Y` prepend a batch dimension `n`.

```
Problem:
  name: str
  input_shape: tuple[int, ...]                 # shape of one x (e.g., (d,))
  output_shape: tuple[int, ...]                # shape of one y = f(x) (e.g., () scalar, (p,))
  support: Constraint                          # required (ProbPipe Constraint)
  prior: Distribution | None                   # optional but preferred
  sampling_bounds: tuple[Array, Array] | None  # fallback for initial design; each has shape input_shape
  target_function: Callable[[x], y]            # the expensive thing being emulated; f(x) = y
  log_density_form: LogDensityForm             # (x, y) → log_unnorm_posterior
  reference_posterior: ReferencePosterior | None
```

**Initial-design / acquisition-space resolution** (priority order):
1. `prior` given → sample from prior.
2. Else `sampling_bounds` given → uniform over bounds.
3. Else `support` is bounded → uniform over support.
4. Else → raise `NoSamplingBoundsError`.

### 4.2 `LogDensityForm`

Deterministic function `φ(x, y) → log_unnorm_posterior(x)` mapping a single target output `y = f(x)` at a single input `x` to an unnormalized log-posterior value. Built-in canonical cases:

- `Identity` — `φ(x, y) = y`. Emulator learns the full log-posterior (prior absorbed).
- `LogLikPlusPrior` — `φ(x, y) = y + log_prior(x)`. Emulator learns log-likelihood only.
- `ForwardModel` — `φ(x, y) = log_lik_from_outputs(data, y) + log_prior(x)`. Emulator learns a multi-output forward model; observation model is supplied separately.

### 4.3 `Surrogate` — stochastic emulator of `target_function`

Minimum interface:

```
Surrogate:
  fit(X: Array, Y: Array) -> Self      # X: (n,) + input_shape, Y: (n,) + output_shape
  predict(X: Array) -> Distribution    # marginal predictive at each row of X
  sample_function(key) -> Callable     # joint realization (for Thompson)
```

Shape- and semantics-compatible with ProbPipe's `ArrayRandomFunction` — `input_shape` / `output_shape` and batch dims come from there. In v1+, `Surrogate` is an `ArrayRandomFunction` subclass once that abstraction is available in ProbPipe; until then, sabi ships a minimal local interface with the same shape contract. `sample_function` must respect joint semantics; independent-output surrogates are a modeling choice, not an API accident.

**The `Surrogate` is tempering-agnostic.** It emulates `target_function` — a raw function of `x` — and never sees a tempering state. Tempering is applied downstream in `LogDensityForm` (via `Tempering`, §4.11) and, if an algorithm wants to fit on tempered values instead of raw ones, through a future `EmulatorTarget` adapter that transforms `(x, y, state, prior)` into training values before `fit`. Keeping tempering out of the surrogate lets the same emulator be reused across tempering states and lets forward-model emulation work unchanged under tempering.

**v0 concrete implementation:** thin tinygp wrapper with data-adaptive fixed hyperparameters. **v1+:** proper GP via ProbPipe's `GaussianRandomFunction` / a mature GP library (e.g., gpjax). No hand-rolled hyperparameter optimization code beyond v0.

### 4.4 `Acquisition`

```
Acquisition:
  select_batch(state: AcquisitionState, q: int, key) -> Array   # shape (q,) + input_shape
```

Covers both (a) deterministic strategies that optimize an acquisition function, and (b) stochastic strategies such as Thompson sampling from `SurrogatePosterior`. Batch `q = 1` is pure sequential.

Acquisitions decouple **scoring** (the function to maximize) from **optimization** (how to maximize it). In v1 the shared optimization machinery lives in `acquisitions/optim.py` (§5): candidate-based scoring, multi-start continuous optimization, greedy multi-point batching, and support-aware reparameterization. A library of small helpers (candidate draws, top-k, local refinement, in-batch diversification) is factored out so each `Acquisition` implements only its scoring function plus a declaration of which optimizer modes it supports.

### 4.5 `SurrogatePosterior` — a random measure

Composition `(surrogate, log_density_form, prior)` behaving as a random measure (target of ProbPipe's Phase 5 `RandomMeasure`). Key methods:

- `sample(key) -> Distribution` — draw one function from the emulator, pushforward through `log_density_form` + prior, return a concrete distribution (typically a black-box `log_prob`; sampling from it needs inner MCMC).
- `plug_in_mean_log_density(X)` — vectorized plug-in-mean log-density at a batch of inputs.

### 4.6 `PosteriorEstimator` — random-measure → deterministic

Maps a `SurrogatePosterior` to a concrete `Distribution`. An algorithm can register **multiple** estimators and all get evaluated each logging round.

A single estimator like `PlugInMean` admits many computational backends (importance sampling, MCMC, SMC). Backend choice is resolved by **ProbPipe-style dispatch** on `(estimator_type, surrogate_posterior_type, backend)` so an algorithm can switch MCMC → SMC as a one-line config change without touching loop code. v0 ships a single IS-backed `PlugInMean`; MCMC / SMC backends land in v2 together with ProbPipe sampler integration.

Initial set:
- `PlugInMean` — deterministic posterior from the mean surrogate; cheap, biased. Backends: IS (v0), MCMC (v2), SMC (v2).
- `ExpectedPosterior` — MC over surrogate function draws; unbiased, expensive (v2).
- `MAPApproximation` — mode of `PlugInMean` (v2).
- `GaussianMixtureVI` (post-v1; for VBMC-style algorithms).

### 4.7 Metrics

Two protocols:

```
PosteriorMetric: (samples: Array, problem: Problem) -> dict[str, float]
EmulatorMetric:  (surrogate, validation_set, log_density_form?) -> dict[str, float]
```

A metric returns a **dict of named scalars**, not a single float, so that related quantities stay bundled (e.g., MMD² and its square root, forward and reverse KL, per-marginal TV). The dict keys are merged into each round's metric row; namespacing is the metric's responsibility when collisions are possible.

- `PosteriorMetric` examples: MMD (keys `mmd2`, `mmd`), forward/reverse KL, TV on 1-D marginals, log-score of estimate on reference samples.
- `EmulatorMetric` examples: log-score of emulator at validation points, log-score of induced unnormalized log-density (calibration of the pushforward).

The metric list is a field on `Algorithm`; the loop evaluates every metric each round with no metric-specific code paths in the loop itself.

**v0 limitation (tracked for post-ProbPipe):** the loop currently passes metrics a sample batch from the estimator, which assumes a sample-based posterior representation. Once ProbPipe's `Distribution` is available (v2, when `PosteriorEstimator` backend dispatch lands), the loop should ask the estimator for a `Distribution` and each metric should declare which representation it consumes — samples, density, or both. VI / Laplace / mixture estimators, and metrics like log-score or TV on 1-D marginals, are the motivating cases.

### 4.8 `InitialDesign`

Sobol, LHS, prior samples, uniform-in-bounds. Tiny.

### 4.9 `Algorithm` — composition

A dataclass bundling the above. Swapping any field is an ablation.

```
Algorithm:
  initial_design: InitialDesign
  surrogate_factory: Callable[[], Surrogate]
  acquisition: Acquisition
  posterior_estimators: list[PosteriorEstimator]
  tempering: Tempering                 # default NoTempering()
  schedule: TemperingSchedule          # default UntemperedSchedule()
  metrics: tuple[PosteriorMetric, ...] # default ()
  n_initial: int
  n_rounds: int
  q: int
```

### 4.10 `Runner`

Hydra entry point. Responsibilities: seed splitting across replicates, config hashing, git SHA capture, environment capture, per-round logging, artifact layout.

### 4.11 `Tempering`

A `Tempering` transforms a `LogDensityForm` at a given **tempering state**. The state is an opaque PyTree — its type and contents are tempering-strategy-specific, because not every bridging scheme is parameterized by a scalar. For `LikelihoodTempering` the state is a scalar `β ∈ [0, 1]`; for `DataTempering` it's a subset identifier; for more exotic bridges it could be a tuple, a dict, or a Distribution.

```
Tempering:
  apply(form: LogDensityForm, state: PyTree) -> LogDensityForm
```

`apply` is implemented by **dispatch on `(type(tempering), type(form))`** so new tempering strategies drop in without editing every existing `LogDensityForm`. The dispatch registry is the only place that knows how a particular tempering composes with a particular form; the state is passed through unchanged and unpacked inside the registered implementation.

Built-in:
- `NoTempering` — returns `form` unchanged (state ignored). Default.
- `LikelihoodTempering` (state = `β ∈ [0, 1]`) — raises the likelihood to power β:
  - `(LikelihoodTempering, LogLikPlusPrior)` → `(x, y) ↦ β·y + log_prior(x)`
  - `(LikelihoodTempering, ForwardModel)` → `(x, y) ↦ β·log_lik_from_outputs(data, y) + log_prior(x)`
  - `(LikelihoodTempering, Identity)` → `(x, y) ↦ (1−β)·log_prior(x) + β·y` (requires `prior` on the `Problem`; errors if absent, since full log-posterior emulation with no accessible prior cannot be tempered without refitting).
- `DataTempering(partition)` (state = subset index / identifier) — restricts the likelihood to data subset `S_state`. Requires a form that exposes per-datapoint likelihood structure (a later `PartitionedLogLikForm`); added when the first data-tempering benchmark lands.

### 4.12 `TemperingSchedule`

Selects the sequence of tempering states over the loop.

```
TemperingSchedule:
  next(round_idx: int, loop_state: LoopState) -> (tempering_state: PyTree, final: bool)
```

`final=True` signals that this is the terminal distribution (β=1 or equivalent) and no further rounds should advance the schedule. The schedule and the matching `Tempering` must agree on the state type.

Built-in:
- `UntemperedSchedule` — always returns `(None, True)`. Default; paired with `NoTempering` this recovers the v0 untempered loop.
- `FixedSchedule(states: tuple)` — iterates a pre-computed sequence; last entry is marked `final=True`. Typical for `LikelihoodTempering` with a geometric or linear β schedule.
- `ESSAdaptiveSchedule(target_ess_ratio)` — chooses the next state so that the ESS of importance weights between consecutive distributions hits the target ratio. Standard in SMC samplers; only defined for tempering schemes whose state supports ESS computation (likelihood tempering, data tempering).

### 4.13 Tempering in the algorithm loop

`Algorithm` gains two fields with no-op defaults:

```
tempering: Tempering = NoTempering()
schedule: TemperingSchedule = UntemperedSchedule()
```

Per round:
1. `tempering_state, final = schedule.next(round_idx, loop_state)`
2. `current_form = tempering.apply(problem.log_density_form, tempering_state)` — the **intermediate target log-density form** for this round.
3. Build `SurrogatePosterior(surrogate, current_form, prior)` — this is what `Acquisition` and `PosteriorEstimator`s consume.
4. `x_batch = acquisition.select_batch(acquisition_state, q, key)` where the acquisition state carries `current_form` and `tempering_state`.
5. Evaluate `f` at `x_batch`; `surrogate.fit` / `.update` on raw `(x, f(x))` values — the tempering state is **not** threaded through the surrogate.
6. Metrics: evaluate registered `PosteriorMetric`s each round. Metrics compare against the terminal reference (state=final) by default; metrics can also log quantities against the current intermediate target if they choose.

Each round's log row includes the `tempering_state` verbatim (serialized via a tempering-specific `to_json` when the state isn't JSON-primitive) so ablations over schedules can be reproduced exactly.

**Opt-in emulator-target tempering.** Some algorithms want the GP to learn the tempered log-density directly, not the raw `f`. This is handled by a future `EmulatorTarget` adapter:

```
EmulatorTarget:
  transform(x, y, tempering_state, prior) -> y_train
```

Default `RawTarget` returns `y` unchanged (v0 behavior). A `TemperedLogDensityTarget` would return the tempered log-density value, and the algorithm would refit the surrogate on those values each round. This keeps the tempering state out of `Surrogate` and out of `LogDensityForm` dispatch — the adapter is the one place that combines the two.

## 5. Optimization module (`acquisitions/optim.py`)

Acquisition optimization on non-convex surfaces is a dedicated v1 component, shared across all acquisitions so individual acquisitions implement only their scoring function. Built on [`optimistix`](https://github.com/patrick-kidger/optimistix) for JAX-native BFGS/LM primitives, wrapped with:

- **Candidate-based scoring** — draw a candidate set, score each, return top-q.
- **Continuous multi-start** — Sobol + acquisition-value-ranked starts, local refinement from top-k.
- **Greedy multi-point batching** — for `q > 1`, iteratively select, hallucinate, re-score.
- **Support-aware re-parameterization** — via ProbPipe `Constraint` → unconstrained transform.
- **Shared helpers** — candidate sampling, top-k selection, in-batch diversification, local refinement.

Each `Acquisition` declares which optimizer modes it supports. Must ship with benchmark tests against a BOTorch reference on a fixed acquisition surface. No merge without those tests.

## 6. Reference posteriors

Two tiers:

**Tier A (local, CI-runnable):** Analytic or very short NUTS.
- 2-D Gaussian (analytic marginals)
- Banana (analytic marginals)
- Neal's funnel
- (Add as needed)

**Tier B (cluster):** Realistic problems with long-running MCMC (blackjax). Reference computed once, stored as Parquet + JSON metadata, keyed by a content hash of `(problem_name, sampler, n_samples, sampler_seed)`. Regeneration script checked in. Reference artifacts live in `reference_posteriors/` with a versioned manifest.

## 7. Reproducibility

- Single top-level `seed`. Per-replicate keys via `jax.random.split`.
- Every run writes: resolved config, git SHA, `pip freeze`, Python+JAX versions, hostname, start/end timestamps.
- Config hash = SHA256 of resolved OmegaConf YAML. Used as run directory name (`outputs/<hash>/`).
- Metrics logged per-round as JSONL + end-of-run Parquet rollup.

## 8. Logging

`Logger` protocol with a v1 local backend (JSONL + Parquet). W&B is a second implementation of the same protocol — no W&B-shaped code paths in metrics or algorithm code.

## 9. Repo layout

```
sabi/
  src/sabi/
    problems/          # benchmarks (one subpackage each)
    surrogates/        # Surrogate interface + tinygp impl
    acquisitions/      # + optim.py
    estimators/        # PosteriorEstimator implementations
    metrics/           # PosteriorMetric + EmulatorMetric
    initial_designs/
    tempering/         # Tempering + TemperingSchedule + dispatch registry
    algorithms/        # composed dataclasses
    runner/            # Hydra entry, seeding, logging
    reference/         # reference-posterior tooling
  configs/             # Hydra tree: problem/, surrogate/, ...
  tests/
  reference_posteriors/ # versioned artifacts (git-lfs or DVC TBD)
  docs/
    design.md
    notation.md         # shape + symbol conventions (x, y, X, Y, n, d, p, q, f)
```

## 10. Dependencies

- **Required:** `jax`, `jaxlib`, `numpy`, `hydra-core`, `omegaconf`, `tinygp`, `blackjax`, `optimistix`, `pyarrow`.
- **Optional:** `probpipe`, `wandb`, `pyvbmc`.
- Reimplementing VBMC natively is a post-v1 goal; PyVBMC is the correctness oracle.

## 11. Roadmap

- **v0 (spike):** 2-D Gaussian + banana, toy tinygp-backed GP surrogate on log-posterior with fixed data-adaptive hyperparameters, random + EI acquisitions (candidate-set scoring), `PosteriorMetric` protocol with `ReferenceMMD`, Hydra configs, local logging. `Tempering` / `TemperingSchedule` shipped as no-op defaults (`NoTempering`, `UntemperedSchedule`); hook points exist, no tempering behavior exercised. **Explicitly not production-grade:** the GP, importance-resampling `PlugInMean`, and missing ProbPipe integration are all known v0 stubs.
- **v1:** `Problem` / reference posteriors re-expressed on ProbPipe `Distribution` + `Constraint` (no inline math for standard targets); `Surrogate` becomes an `ArrayRandomFunction` subclass; `SurrogatePosterior` becomes a sabi-local `RandomMeasure` (intended to graduate to ProbPipe); baseline weighted-empirical `SurrogatePosterior` ships first (for testing without a GP backend); GP surrogate backed by a proper GP library (gpjax or ProbPipe `GaussianRandomFunction`) with library-provided hyperparameter optimization (no hand-rolled BFGS); `SurrogatePosterior` / `PosteriorEstimator` abstractions formalized; Tier-A benchmarks with reference posteriors; optimization module (§5) with shared candidate / multi-start / greedy-batch helpers and BOTorch-comparison tests; `EmulatorMetric` protocol. **See [`v1_plan.md`](v1_plan.md) for the sub-phasing.**
- **v2:** `PosteriorEstimator` backend dispatch on `(estimator_type, surrogate_posterior_type, backend)` — swap IS / MCMC / SMC / VI as a config change, backed by ProbPipe sampler integration. Loop generalizes from sample-based metric evaluation to `Distribution`-valued estimates; `PosteriorMetric` declares which representation it consumes. `LikelihoodTempering` + `FixedSchedule` / `ESSAdaptiveSchedule`; `EmulatorTarget` adapter for optional tempered-target emulator fits. Native VBMC implementation (PyVBMC as oracle); more acquisitions (stochastic / Thompson); `ExpectedPosterior` estimator.
- **v3:** Batch `q > 1`, more benchmarks (Tier B), W&B backend. `DataTempering` and `PartitionedLogLikForm` land alongside the first benchmark that requires them.
- **v4+:** Noisy/stochastic target setting; non-GP surrogates (BNN, deep ensemble, ENN).

## 12. Deferred / explicit non-goals for v1

- Noisy / stochastic target functions (heteroscedastic GPs, noise-aware acquisitions).
- PyTorch / GPyTorch / BOTorch backends.
- Cluster orchestration (Prefect, Ray, etc.) — use existing cluster job scripts for now.
- W&B integration.

## 13. Open questions

- **Reference-posterior storage:** plain Parquet vs git-lfs vs DVC. Lean toward plain Parquet in-repo for Tier A (small), git-lfs for Tier B.
- **Inner-MCMC budget** for Thompson-style acquisitions: cheap short-NUTS per iteration, or warm-started sampler across iterations? Decide when we implement the first stochastic acquisition.
- **Multi-output surrogate kernels:** independent-output GPs for v1, or pull in LMC / linear-coregionalization? Start independent; add coupled as a v2 ablation.
- **`condition_on` as a `Surrogate.update` backend:** interface designed to allow this swap; decide timing based on ProbPipe `RandomFunction` evolution.
