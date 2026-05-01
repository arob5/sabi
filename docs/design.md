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
| `RandomFunction[X, Y]` | Contract for `Emulator` |
| `ArrayRandomFunction` | Shape + batch-dim conventions for `Emulator`; v1+ `Emulator` is an `ArrayRandomFunction` subclass |
| `GaussianRandomFunction` | GP-backed emulator |
| `Constraint` + `support` | Parameter-space support in `Problem` |
| `RandomMeasure` / `NumericRandomMeasure` | Base for `SurrogatePosterior` (landed via PR #150) |
| `SupportsRandomLogProb` / `SupportsRandomUnnormalizedLogProb` | Optional protocols on `SurrogatePosterior` for the random log-density (landed via PR #150) |
| `Pushforward` (planned) | Assembly of `SurrogatePosterior` from surrogate + `LogDensityForm`; sabi ships a local pushforward in `LogDensityForm` until ProbPipe's primitive lands |
| `condition_on` | Future implementation path for `Emulator.update` |
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

### 4.3 `Emulator` — stochastic predictive model of `target_function`

`Emulator` IS-A ProbPipe `ArrayRandomFunction`. It inherits the full random-function shape contract (`input_shape`, `output_shape`, `batch_shape`, joint-input / joint-output flags) and the `__call__(X, joint_inputs, joint_outputs) -> Distribution` predictive interface. The only sabi-specific addition is an abstract `fit(X, Y) -> Self` that captures the algorithmic role (a fittable predictive process). Future migration: `condition_on(prior_rf, X=X_train, y=Y_train)` is the natural ProbPipe-native pattern; sabi's `fit` is a v1.x bridge that does the same conceptual work without requiring full ProbPipe conditioning machinery.

Concrete Gaussian emulators inherit from both `Emulator` and `GaussianRandomFunction` (diamond inheritance over `ArrayRandomFunction`, resolved by Python's C3 MRO). For example, `TinyGPEmulator(Emulator, GaussianRandomFunction)` implements `predict_mean(X)` and `predict_variance(X)`; `predict` / `__call__` come for free from the parent and assemble `Normal` (marginal) or `MultivariateNormal` (joint) at the right shape.

Naming: in sabi, "emulator" is reserved for the predictive model fit to evaluations of the target function. The broader word "surrogate" denotes any approximate quantity replacing its exact analog (hence `SurrogatePosterior` for the surrogate of the *true* posterior).

**The `Emulator` is tempering-agnostic.** It emulates `target_function` — a raw function of `x` — and never sees a tempering state. Tempering is applied downstream in `LogDensityForm` (via `Tempering`, §4.11) and, if an algorithm wants to fit on tempered values instead of raw ones, through a future `EmulatorTarget` adapter that transforms `(x, y, state, prior)` into training values before `fit`. Keeping tempering out of the emulator lets the same emulator be reused across tempering states and lets forward-model emulation work unchanged under tempering.

**v0/v1.2 concrete implementation:** `TinyGPEmulator(Emulator, GaussianRandomFunction)` — thin tinygp wrapper with data-adaptive fixed hyperparameters; `predict_mean` / `predict_variance` only (no `predict_covariance` until v1.5 emulator metrics). **v1.6:** proper GP via gpjax or a ProbPipe-native `GaussianRandomFunction` subclass; no hand-rolled hyperparameter optimization code in sabi.

### 4.4 `Acquisition`

```
Acquisition:
  select_batch(state: AcquisitionState, q: int, key) -> Array   # shape (q,) + input_shape
```

Covers both (a) deterministic strategies that optimize an acquisition function, and (b) stochastic strategies such as Thompson sampling from `SurrogatePosterior`. Batch `q = 1` is pure sequential.

Acquisitions decouple **scoring** (the function to maximize) from **optimization** (how to maximize it). The hierarchy:

- `Acquisition` (ABC) — the top-level `select_batch(state, q, key)` interface. Sampling-style acquisitions (`PriorSampling` today; `PosteriorThompsonSampling` / `MixtureSampling` v1.5+) implement this directly.
- `PointwiseScoredAcquisition(Acquisition)` — provides `score(x, state) -> scalar`; `select_batch` delegates to a `PointwiseOptimizer`. `ExpectedImprovement` inherits from this.
- `BatchScoredAcquisition(Acquisition)` (v1.5+) — provides `score_batch(X, state) -> scalar` over a joint q-batch; uses a parallel `BatchOptimizer` hierarchy. q-EI, max-min entropy, etc.

`PointwiseOptimizer` ships three implementations in `acquisitions/optim.py`:

- `CandidateSetOptimizer` — random candidates from `problem.prior` → top-q. Cheap; gradient-free; default.
- `ContinuousMultiStartOptimizer` — score-filter top-`n_starts` BFGS init points → optimistix BFGS in unconstrained reparameterization space → top-q. Sigmoid bijector for `interval(low, high)` supports; other supports raise.
- `GreedyMultiPointOptimizer` — for `q > 1`. Picks one point at a time via an inner optimizer; hallucinates a pending observation via a pluggable `FantasyImputer` (`KrigingBeliever`, `ConstantLiar`); refits the surrogate; iterates. Pluggable imputer makes the strategy interchangeable.

Acquisitions implement only their scoring function (and pick an optimizer per their needs). Helpers like reparameterization, top-k, candidate sampling are reused inside the optimizer module.

### 4.5 `SurrogatePosterior` — a `RandomMeasure`

`SurrogatePosterior` is a ProbPipe `NumericRandomMeasure[Array]`: a distribution over `Distribution[Array]`s on the parameter space. **Decoupled from `Problem`** — it carries math primitives directly (`support`, `prior`, `log_density_form`, `input_shape`). The algorithm loop pulls those primitives from a `Problem` when constructing the SP each round.

A `SurrogatePosterior` holds a `Surrogate` (a sabi-side `ArrayRandomFunction` subclass — see §4.3) and a `LogDensityForm`; its `_random_unnormalized_log_prob()` returns a `RandomFunction` whose `__call__(X)` evaluates the surrogate at `X` (yielding a `Distribution[Array]`) and pushes it through the form via the shared `pushforward_marginal` dispatch (closed-form for Gaussian × affine cases, MC empirical via ProbPipe `WorkflowFunction` broadcasting otherwise). The class itself is Gaussian-agnostic — only the dispatch knows about Gaussianness.

`WeightedEmpiricalRandomMeasure` is the **sibling** no-GP-baseline random measure (NOT a `SurrogatePosterior` subclass): a Dirac at a weighted empirical of design points. Useful for testing the loop without a fitted surrogate, and as a reference for any random-measure consumer. Tracked for potential graduation to ProbPipe.

Protocol opt-ins (v1.2):

- **`SurrogatePosterior`**: `SupportsRandomUnnormalizedLogProb`. Does NOT implement `SupportsSampling` (no function-trajectory sampler on the v1.x `Surrogate` interface yet — v1.6), `SupportsMean` (unbiased expected posterior needs an MC backend, deferred to v2), or `SupportsRandomLogProb` (normalization intractable).
- **`WeightedEmpiricalRandomMeasure`**: `SupportsMean` / `SupportsSampling` / `SupportsRandomLogProb` / `SupportsRandomUnnormalizedLogProb`, all via the underlying `NumericEmpiricalDistribution` and a Dirac random-function shim.

### 4.6 Deterministic posterior estimators

A `SurrogatePosterior` admits many deterministic posterior approximations. They are exposed as **free functions** with type-dispatch on `SurrogatePosterior` subtype, mirroring ProbPipe's op-dispatch style. Each returns a concrete `Distribution[Array]`. For `WeightedEmpiricalSurrogatePosterior` (Dirac), all estimators coincide and reduce to the underlying empirical.

Currently shipped:

- `mean(sp)` — the unbiased *expected posterior* `D̄(A) = ∫ D(A) dM(D)`, exposed via ProbPipe's `mean` op via `SupportsMean`. Implemented for the Dirac case (returns the inner empirical); for the GP path no general implementation in v1.2 — `mean(gp_sp)` raises until a v2 MC backend lands.
- `expected_target(sp)` — the *biased plug-in posterior*: plug the surrogate's predictive mean into the log-density form. Returns the inner empirical for Dirac SPs; for the GP path returns a `Distribution[Array]` whose `_unnormalized_log_prob(x)` evaluates `log_density_form(x, surrogate_mean(x), prior)` and whose `_sample` delegates to ProbPipe `condition_on(self)` (auto-dispatched MCMC, typically NUTS, post-PR-#151).

Future estimators (motivated by the partial-pushforward primitive — see `docs/probpipe_issues.md`):

- `expected_log_density(sp)` — pointwise mean of the random log-density. Coincides with `expected_target` for log-density emulators; differs for forward-model emulators.
- `expected_density(sp)` — pointwise mean of the random unnormalized density (`E_f[exp(log p̃(x; f))]`).
- `median_density(sp)` — pointwise median of the random unnormalized density.
- `MAPApproximation` — mode of `expected_target` (or another deterministic estimator).
- `GaussianMixtureVI` (post-v1; for VBMC-style algorithms).

### 4.7 Metrics

Two protocols:

```
PosteriorMetric: (samples: Array, problem: Problem) -> dict[str, float]
EmulatorMetric:  (emulator, validation_set, log_density_form?) -> dict[str, float]
```

A metric returns a **dict of named scalars**, not a single float, so that related quantities stay bundled (e.g., MMD² and its square root, forward and reverse KL, per-marginal TV). The dict keys are merged into each round's metric row; namespacing is the metric's responsibility when collisions are possible.

- `PosteriorMetric` examples: MMD (keys `mmd2`, `mmd`), forward/reverse KL, TV on 1-D marginals, log-score of estimate on reference samples.
- `EmulatorMetric` examples: log-score of emulator at validation points, log-score of induced unnormalized log-density (calibration of the pushforward).

The metric list is a field on `Algorithm`; the loop evaluates every metric each round with no metric-specific code paths in the loop itself.

**v0 limitation (tracked for post-ProbPipe):** the loop currently passes metrics a sample batch from the estimator, which assumes a sample-based posterior representation. Once ProbPipe's `Distribution` is available (v2, when `PosteriorEstimator` backend dispatch lands), the loop should ask the estimator for a `Distribution` and each metric should declare which representation it consumes — samples, density, or both. VI / Laplace / mixture estimators, and metrics like log-score or TV on 1-D marginals, are the motivating cases.

### 4.8 `BatchSampler`

Single abstraction for "draw `n` parameter-space points": Sobol, LHS,
prior samples, uniform-in-bounds. Lives in `sabi/sampling.py`. Used by

- the loop's initial-design step (`Algorithm.initial_sampler`),
- `PriorSampling.select_batch` (the prior-sampling acquisition),
- pointwise optimizers' candidate / seed sets
  (`CandidateSetOptimizer.candidate_sampler`,
  `ContinuousMultiStartOptimizer.seed_sampler`).

v1.4.1 ships `PriorSampler` (i.i.d. samples from `problem.prior`); Sobol /
LHS land alongside the first benchmark that needs them.

### 4.9 `Algorithm` — composition

A dataclass bundling the above. Swapping any field is an ablation.

```
Algorithm:
  initial_sampler: BatchSampler
  emulator_factory: Callable[[], Emulator]
  acquisition: Acquisition
  surrogate_posterior_factory: SurrogatePosteriorFactory   # default emulator_pushforward_factory
  estimator: Callable[[SurrogatePosterior], Distribution]  # default expected_target
  tempering_scheme: TemperingScheme    # default NoTempering()
  schedule: TemperingSchedule          # default UntemperedSchedule()
  acquisition_target: AcquisitionTarget # default CURRENT
  metrics: tuple[PosteriorMetric, ...] # default ()
  n_initial: int
  n_rounds: int
  q: int
```

### 4.10 `Runner`

Hydra entry point. Responsibilities: seed splitting across replicates, config hashing, git SHA capture, environment capture, per-round logging, artifact layout.

### 4.11 `TemperingScheme` and `IntermediateTarget`

A `TemperingScheme` is a family of intermediate target distributions
indexed by a state. The single method
`intermediate_target(base, state) -> IntermediateTarget` produces, for
each state, the per-state target distribution.

`IntermediateTarget` extends `TargetDistribution` with three pieces of
metadata:

- `state`: the tempering state that produced this intermediate.
- `output_transform(state, X, Y_raw) -> Y_train`: derives emulator
  training data from cached raw evaluations of the *base* target
  function — the loop uses this instead of evaluating the (possibly
  expensive) base target afresh.
- `base_target_function`: the un-tempered base target (for reference;
  the loop typically already has it via the base `TargetDistribution`).

Two orthogonal axes can be tempered:

- **Target axis**: the emulator's training target `f_state` varies
  with state. `output_transform` is non-trivial; the form is
  invariant.
- **Form axis**: the log-density form `phi_state` varies with state.
  `output_transform` is identity; the form is non-invariant.

Concrete schemes pick one axis at a time. The scheme advertises
which axis is invariant via
`is_invariant_target_function(state_a, state_b)` and
`is_invariant_form(state_a, state_b)`; the loop uses these to skip
redundant emulator refits / form rebuilds.

The state is an opaque PyTree — its type and contents are
strategy-specific. For likelihood tempering it's a scalar `β ∈ [0, 1]`;
for a future data tempering it's a subset identifier; for exotic
bridges it could be a tuple, a dict, or a Distribution. The
`TemperingSchedule` produces states; the scheme consumes them. The
two must agree on the state type — that's a user / config-level
convention (no static check).

Built-in schemes:

- `NoTempering` — identity on both axes. The intermediate is the
  base target distribution wrapped with `state=state`. Default.
- `LikelihoodTemperingViaForm` — form-axis tempering. Per-form-type
  dispatch:
  - `(LogLikPlusPrior, β)` → `(x, y) ↦ β·y + log_prior(x)`.
  - `(ForwardModel, β)` → `(x, y) ↦ β·log_lik_from_outputs(x, y) + log_prior(x)`.
  - `(Identity, β)` → `(x, y) ↦ (1−β)·log_prior(x) + β·y` (geometric
    bridge; requires `prior` for the bridge endpoints).
  Compatible with all three form types; emulator is invariant under
  state changes.
- `LikelihoodTemperingViaTarget` — target-axis tempering.
  `output_transform(β, X, Y_raw) = β·Y_raw`; form is unchanged.
  Restricted to `LogLikPlusPrior` base forms (the case where `f` is
  the log-likelihood directly).

A future `DataTempering` would land naturally as a third scheme with
state = subset identifier.

See [`tempering.md`](tempering.md) for the full case analysis with
worked examples.

### 4.12 `TemperingSchedule`

Selects the sequence of tempering states over the loop.

```
TemperingSchedule:
  next(round_idx: int, loop_state: LoopState) -> (tempering_state: PyTree, final: bool)
  terminal_state() -> tempering_state
```

`final=True` signals that this round is the terminal distribution
(β=1 or equivalent); the schedule should not advance past it.
`terminal_state()` returns the schedule's final state explicitly —
used by `AcquisitionTarget.TERMINAL` to build the look-ahead SP.

Built-in:
- `UntemperedSchedule` — always `(None, True)`; `terminal_state()` = `None`.
  Default; paired with `NoTempering` this recovers the untempered loop.
- `FixedSchedule(states: tuple)` — iterates a pre-computed sequence;
  last entry is marked `final=True`; `terminal_state()` = `states[-1]`.
  Typical for likelihood tempering with a geometric or linear β
  schedule.
- `ESSAdaptiveSchedule(target_ess_ratio)` — chooses the next state so
  the ESS of importance weights between consecutive distributions hits
  the target ratio. Standard in SMC samplers; only defined for
  tempering schemes whose state supports ESS computation. Tracked as
  [issue #5](https://github.com/arob5/sabi/issues/5).

### 4.13 `AcquisitionTarget`

Picks *which* tempering state the acquisition's `SurrogatePosterior`
is built at — independent of the round's "current" state.

- `CURRENT` (default): the round's current state. Acquisition
  optimizes against the current intermediate.
- `NEXT`: the next round's state (clamped to terminal at the last
  round). Standard SMC-flavor look-ahead.
- `TERMINAL`: the schedule's terminal state for every round.
  Acquisition optimizes toward the final target throughout.

For untempered loops, all three collapse. For tempered loops, the
loop builds a separate look-ahead `IntermediateTarget` at the
resolved state and may refit the emulator on its training data
before the acquisition runs (see §4.14).

Richer policies (ESS-adaptive look-ahead, custom callable that
depends on loop state) are tracked as
[issue #5](https://github.com/arob5/sabi/issues/5).

### 4.14 Tempering in the algorithm loop

`Algorithm` gains three fields with no-op defaults:

```
tempering_scheme: TemperingScheme = NoTempering()
schedule: TemperingSchedule = UntemperedSchedule()
acquisition_target: AcquisitionTarget = AcquisitionTarget.CURRENT
```

Per round (loop sketch):

1. `current_state, final = schedule.next(round_idx, None)`.
2. `target_state` resolved from `acquisition_target` — `CURRENT`
   gives `current_state`; `NEXT` gives `schedule.next(round_idx + 1)`;
   `TERMINAL` gives `schedule.terminal_state()`.
3. `current_intermediate = tempering_scheme.intermediate_target(target, current_state)`.
4. Acquisition's intermediate: reuse `current_intermediate` if the
   scheme reports both axes invariant under
   `(current_state, target_state)`; else build a fresh
   `target_intermediate` at `target_state`.
5. Acquisition's emulator: reuse the round's emulator if the scheme
   reports the target axis invariant; else refit on the look-ahead
   state's training data
   (`Y_train_acq = target_intermediate.output_transform(target_state, X, Y_raw)`).
   Cheap-update dispatch ([issue #4](https://github.com/arob5/sabi/issues/4))
   will replace the full refit.
6. `SurrogatePosterior` for acquisition = `(emulator_for_acq,
   target_intermediate.log_density_form, ...)`. Acquisition picks
   `x_new`, loop appends `y_new_raw = problem.target_function(x_new)`
   to `Y_raw`.
7. Round-end emulator + SP at the *current* state for metrics:
   `Y_train = current_intermediate.output_transform(current_state, X, Y_raw)`;
   refit emulator; build SP for metrics at `current_intermediate`.
8. Run metrics. Per-round metrics record both `tempering_state` and
   `target_tempering_state` for ablation reproducibility.

The base-class `Emulator` is tempering-agnostic — it just consumes
`(X, Y_train)`. The state-dependence enters through the scheme's
`output_transform`. Forward-model emulation with form-side tempering
works unchanged because the emulator never sees the state.

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

**Tier B (cluster):** Realistic problems with long-running MCMC. Same artifact infrastructure as Tier A (Parquet samples + JSON metadata under `reference_posteriors/<problem>/`); large Tier-B artifacts will use git-lfs when the first such benchmark lands. Sampler is ProbPipe's `condition_on` → `tfp_nuts` (with ArviZ diagnostics embedded in metadata); `scripts/regenerate_references` regenerates in place.

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
    emulators/         # Emulator interface + tinygp impl
    acquisitions/      # + optim.py
    posterior/         # SurrogatePosterior subclasses + deterministic estimators (expected_target, mean)
    metrics/           # PosteriorMetric + EmulatorMetric
    sampling.py        # BatchSampler + PriorSampler
    tempering/         # Tempering + TemperingSchedule + dispatch registry
    algorithms/        # composed dataclasses
    runner/            # Hydra entry, seeding, logging
    reference/         # reference-posterior tooling
  configs/             # Hydra tree: problem/, emulator/, ...
  tests/
  reference_posteriors/ # versioned artifacts (git-lfs or DVC TBD)
  docs/
    design.md
    notation.md         # shape + symbol conventions (x, y, X, Y, n, d, p, q, f)
```

## 10. Dependencies

- **Required:** `jax`, `jaxlib`, `numpy`, `hydra-core`, `omegaconf`, `tinygp`, `optimistix`, `pyarrow`.
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
- **`condition_on` as an `Emulator.update` backend:** interface designed to allow this swap; decide timing based on ProbPipe `RandomFunction` evolution.
