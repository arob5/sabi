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
| `GaussianRandomFunction` | GP-backed surrogate |
| `Constraint` + `support` | Parameter-space support in `Problem` |
| `RandomMeasure` (Phase 5) | Base for `SurrogatePosterior` |
| `Pushforward` (planned) | Assembly of `SurrogatePosterior` from surrogate + `LogDensityForm` |
| `condition_on` | Future implementation path for `Surrogate.update` |
| `_mc_expectation` / `BootstrapDistribution` | Metric computation with MC-error tracking |

**Strategy:** ship minimal local implementations of missing abstractions (pushforward, random-measure) inside sabi; swap to ProbPipe implementations as they land. The framework's interfaces are deliberately close to the ProbPipe shapes so the swap is a rename, not a refactor.

## 4. Core abstractions

### 4.1 `Problem` — a benchmark

The expensive Bayesian inference target plus everything needed to construct, transform, and evaluate against it.

```
Problem:
  name: str
  support: Constraint                          # required (ProbPipe Constraint)
  prior: Distribution | None                   # optional but preferred
  sampling_bounds: tuple[Array, Array] | None  # fallback for initial design
  target_function: Callable[[Θ], R^k]          # the expensive thing being emulated
  log_density_form: LogDensityForm             # (θ, f(θ)) → log_unnorm_posterior
  reference_posterior: ReferencePosterior | None
```

**Initial-design / acquisition-space resolution** (priority order):
1. `prior` given → sample from prior.
2. Else `sampling_bounds` given → uniform over bounds.
3. Else `support` is bounded → uniform over support.
4. Else → raise `NoSamplingBoundsError`.

### 4.2 `LogDensityForm`

Deterministic function `φ(θ, y) → log_unnorm_posterior(θ)` mapping target output `y = f(θ)` to an unnormalized log-posterior value. Built-in canonical cases:

- `Identity` — `φ(θ, y) = y`. Emulator learns the full log-posterior (prior absorbed).
- `LogLikPlusPrior` — `φ(θ, y) = y + log_prior(θ)`. Emulator learns log-likelihood only.
- `ForwardModel` — `φ(θ, y) = log_lik_from_outputs(data, y) + log_prior(θ)`. Emulator learns a multi-output forward model; observation model is supplied separately.

### 4.3 `Surrogate` — stochastic emulator of `target_function`

Minimum interface:

```
Surrogate[Θ, Y]:
  fit(X: Array, Y: Array) -> Self          # or update(X_new, Y_new)
  predict(x: Θ) -> Distribution[Y]         # predictive distribution
  sample_function(key) -> Callable[[Θ], Y] # joint realization (for Thompson)
```

Shape-compatible with ProbPipe's `RandomFunction[Θ, Y]`. `sample_function` must respect joint semantics (`joint_inputs` / `joint_outputs` flags from ProbPipe); independent-output surrogates are a modeling choice, not an API accident.

**v1 concrete implementation:** GP via `tinygp`.

### 4.4 `Acquisition`

```
Acquisition:
  select_batch(surrogate, state, q: int, key) -> Array  # shape (q, d)
```

Covers both (a) deterministic strategies that optimize an acquisition function, and (b) stochastic strategies such as Thompson sampling from `SurrogatePosterior`. Batch `q = 1` is pure sequential.

### 4.5 `SurrogatePosterior` — a random measure

Composition `(surrogate, log_density_form, prior)` behaving as a random measure (target of ProbPipe's Phase 5 `RandomMeasure`). Key methods:

- `sample(key) -> Distribution[Θ]` — draw one function from the emulator, pushforward through `log_density_form` + prior, return a concrete distribution (typically a black-box `log_prob`; sampling from it needs inner MCMC).
- `mean_log_density(θ)` — plug-in mean predictor's induced log-density.

### 4.6 `PosteriorEstimator` — random-measure → deterministic

Maps a `SurrogatePosterior` to a concrete `Distribution[Θ]`. An algorithm can register **multiple** estimators and all get evaluated each logging round. Uses ProbPipe-style dispatch on `(estimator_type, surrogate_posterior_type, method)` so efficient specializations slot in without algorithm changes.

Initial set:
- `PlugInMean` — deterministic posterior from the mean surrogate; cheap, biased.
- `ExpectedPosterior` — MC over surrogate function draws; unbiased, expensive.
- `MAPApproximation` — mode of `PlugInMean`.
- `GaussianMixtureVI` (post-v1; for VBMC-style algorithms).

### 4.7 Metrics

Two protocols:

```
PosteriorMetric: (estimate: Distribution, reference: ReferencePosterior) -> float
EmulatorMetric:  (surrogate, validation_set, log_density_form?) -> float
```

- `PosteriorMetric` examples: MMD, forward/reverse KL (sample-based), TV on 1-D marginals, log-score of estimate on reference samples.
- `EmulatorMetric` examples: log-score of emulator at validation points, log-score of induced unnormalized log-density (calibration of the pushforward).

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
  n_initial: int
  n_rounds: int
  q: int
  metrics: MetricSpec
```

### 4.10 `Runner`

Hydra entry point. Responsibilities: seed splitting across replicates, config hashing, git SHA capture, environment capture, per-round logging, artifact layout.

## 5. Optimization module (`acquisitions/optim.py`)

Acquisition optimization on non-convex surfaces is a dedicated v1 component. Built on [`optimistix`](https://github.com/patrick-kidger/optimistix) for JAX-native BFGS/LM primitives, wrapped with:

- Multi-start initialization (Sobol + acquisition-value-ranked starts)
- Local refinement from top-k starts
- Support-aware re-parameterization (via ProbPipe `Constraint` → unconstrained transform)

Must ship with benchmark tests against a BOTorch reference on a fixed acquisition surface. No merge without those tests.

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
    algorithms/        # composed dataclasses
    runner/            # Hydra entry, seeding, logging
    reference/         # reference-posterior tooling
  configs/             # Hydra tree: problem/, surrogate/, ...
  tests/
  reference_posteriors/ # versioned artifacts (git-lfs or DVC TBD)
  docs/
```

## 10. Dependencies

- **Required:** `jax`, `jaxlib`, `numpy`, `hydra-core`, `omegaconf`, `tinygp`, `blackjax`, `optimistix`, `pyarrow`.
- **Optional:** `probpipe`, `wandb`, `pyvbmc`.
- Reimplementing VBMC natively is a post-v1 goal; PyVBMC is the correctness oracle.

## 11. Roadmap

- **v0 (spike):** 2-D Gaussian + banana, GP-on-log-posterior surrogate, one acquisition (random + EI), MMD metric, Hydra configs, local logging. No ProbPipe yet — lock the loop shape first.
- **v1:** `RandomFunction`-shaped `Surrogate`, `SurrogatePosterior` / `PosteriorEstimator` abstractions, Tier-A benchmarks with reference posteriors, optimization module with BOTorch-comparison tests, emulator metrics.
- **v2:** Native VBMC implementation (PyVBMC as oracle); more acquisitions (stochastic / Thompson); `ExpectedPosterior` estimator.
- **v3:** Batch `q > 1`, more benchmarks (Tier B), W&B backend.
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
