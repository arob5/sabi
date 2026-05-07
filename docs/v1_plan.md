# sabi — v1 implementation plan

**Status:** working plan; expected to evolve as ProbPipe progresses in parallel.
**Last updated:** 2026-04-26.

This document slices the v1 roadmap entry from `design.md §11` into ordered
sub-phases. Each sub-phase is a coherent chunk that can be reviewed and merged
on its own. The entries below capture scope, deliverables, exit criteria, and
the design questions deferred to that point.

The overarching v1 goal: replace v0 toy abstractions with ProbPipe-aligned
primitives. The framing principle is "distributions in, distributions out" —
sabi's APIs accept and return `Distribution` objects (or `RandomFunction`-shaped
objects); concrete representations (samples, densities, modes) are pulled
through optional protocols.

---

## v1.1 — ProbPipe `Distribution` + `Constraint` foundations + minimal pushforward

The first big lift. Switch `Problem` from callable-based prior/sampling to
`ProbPipe` primitives, formalize `LogDensityForm` as sabi's local
pushforward operator, ship a baseline `WeightedEmpiricalSurrogateDistribution`
that needs no GP at all, and make the loop pass `Distribution` objects to
metrics rather than raw sample arrays.

**Deliverables**
- `Problem.prior: Distribution | None` (replaces `prior_log_prob` + `prior_sample` callables). Where v0 benchmarks had `sampling_bounds=(low, high)` for uniform initial design, v1 expresses the same thing as `prior = Uniform(low, high, name=...)` — i.e. the prior field is the design distribution. (Renaming the field to something like `design_distribution` is a semantic bike-shed; v1.1 keeps `prior` to limit churn.)
- `Problem.support: Constraint | None` (replaces `sampling_bounds`). For v1.1 `support` is **metadata** — used by acquisitions / metrics that want to check feasibility, but sampling routes through `prior`. ProbPipe exposes TFP bijectors via `TransformedDistribution` (`probpipe/distributions/transformed.py`); we don't need them for v0 benchmarks (bounded boxes are sampled directly via `Uniform`), but they're available for v1.3+ benchmarks with non-trivial supports (positive parameters, simplex, etc.).
- `Problem.reference_distribution: Distribution | None` replacing `reference_samples` + `reference_log_prob`. For benchmarks with an analytic posterior we use the corresponding ProbPipe distribution directly (e.g. `gaussian2d`'s reference is a `MultivariateNormal`); for benchmarks without one we wrap pre-computed samples in `NumericEmpiricalDistribution`.
- Benchmarks rebuilt: `gaussian2d`'s posterior is a ProbPipe `MultivariateNormal`, and `target_map = lambda x: log_prob(posterior_dist, x)` per Option A. `banana` keeps its inline analytic log-density for v1.1 (a transformed-Normal expression of the banana posterior is a v1.5 exercise) and uses a `NumericEmpiricalDistribution` reference. **Watch:** `MultivariateNormal` currently casts to float32 internally, so the `target_map` may need an explicit cast back to float64 to interop with our x64-enabled tests; if that gets ugly, surface as a ProbPipe gap.
- `LogDensityForm` documented and labeled as **sabi's local pushforward operator**: it composes `f(x) = y` with `prior.log_prob(x)` to form an unnormalized log-posterior. Naming aligns with ProbPipe (`unnormalized_log_prob`), so the eventual swap to ProbPipe's pushforward is mechanical.
- `SurrogateDistribution` becomes abstract; subclasses declare which `Supports*` protocols they satisfy.
- New baseline: `WeightedEmpiricalSurrogateDistribution` — stores `(X, Y)` where `Y` is observed log-density at `X`. Exposes its inner posterior as a ProbPipe `NumericEmpiricalDistribution(samples=X, weights=Weights(log_weights=Y))` via a cached property. Sabi does **not** reimplement weighting / normalization / ESS — `Weights` in `probpipe/_weights.py` handles all of that. Inherits `SupportsSampling` and `SupportsMean` from the wrapped empirical.
- Existing GP-pushforward `SurrogateDistribution` (the v0 plug-in-mean object) refactored into a second concrete subclass `GPPushforwardSurrogateDistribution`. Implements `SupportsSampling` (via IS over the surrogate-mean log-density) only.
- **`PlugInMean` subclasses `NumericRecordDistribution`** (ProbPipe's base for all numeric distributions with TFP-style shape semantics) — not a generic `Distribution[Array]`. Constructor takes a `SurrogateDistribution`; the result is the concrete plug-in-mean posterior (a `Distribution[Array]` with `event_shape == problem.target_distribution.input_shape`). Internal type dispatch on the surrogate-posterior subtype (until v1.2 formalizes this as `RandomMeasure._mean()` op dispatch):
  - For `WeightedEmpiricalSurrogateDistribution` (Dirac random measure): `_sample` and `_unnormalized_log_prob` forward to the underlying `NumericEmpiricalDistribution`. `PlugInMean` and `mean(surrogate_distribution)` coincide here.
  - For `GPPushforwardSurrogateDistribution`: `_sample` is IS over the surrogate-mean log-density (the v0 backend); `_unnormalized_log_prob` evaluates `LogDensityForm(x, surrogate.predict(x).mean, problem)`. The IS backend is the same hook v2 backend dispatch will replace with MCMC / SMC / VI.
- Loop materialization: after constructing `posterior_estimate = PlugInMean(surrogate_distribution)`, the loop draws `samples = sample(posterior_estimate, key, n)` once per round and wraps in `NumericEmpiricalDistribution(samples)` to hand to metrics. Metrics see a `Distribution[Array]`, never a raw array.
- `PosteriorMetric` declares its required protocols via a class attribute (e.g., `requires: tuple[type, ...] = (SupportsSampling,)`). Loop introspects each metric and **raises** if the estimator distribution doesn't support a required protocol.
- `_evaluate_metrics` reshaped: takes the estimator-returned `Distribution[Array]`, dispatches per-metric via ProbPipe ops (`sample`, `log_prob`, `mean`, etc.). Removes the "draw samples once, pass as array" assumption from the loop.
- Tests:
  - End-to-end run with **only** `WeightedEmpiricalSurrogateDistribution` (no GP) — confirms the baseline path is self-contained.
  - `PlugInMean(weighted_empirical_baseline)` produces the same samples (up to PRNG) as direct sampling from the underlying empirical — Dirac mean property holds in code.
  - `PosteriorMetric.requires` honored: a metric requiring `SupportsLogProb` against an estimator that only supports `SupportsSampling` raises a clear error.
  - All v0 tests still pass after rename / reshape.

**Exit criteria**
- All 41+ tests green.
- `runner.main` smoke test: gaussian2d with `WeightedEmpiricalSurrogateDistribution` produces sensible MMD numbers (the baseline isn't expected to be *good*, just non-degenerate).
- No raw `Array` flowing from estimator → metric in the loop body; metrics receive `Distribution[Array]` only.

**Open design questions for v1.1**
1. Benchmark expression — answered: route `target_map` through ProbPipe's `Distribution.log_prob` / `unnormalized_log_prob` for benchmarks where a ProbPipe distribution describes the posterior natively. Inline math is kept only when no clean ProbPipe equivalent exists.
2. `sampling_bounds` removal — answered: drop. Use `Constraint` + TFP bijectors. Verify ProbPipe's TFP bijector exposure on implementation; if missing, surface as a ProbPipe gap.
3. `PosteriorMetric.requires` mismatch — answered: raise.
4. Should the v0 toy GP wrapper survive as a `Surrogate` example? It still works; it's just not the primary baseline anymore. Recommend keeping it (handy for testing acquisitions in v1.4) but not advertising it.

---

## v1.2 — `SurrogateDistribution` as a `NumericRandomMeasure` subclass — **landed**

### v1.2 refactor (post-MCMC-relaxation)

Following the pushforward-dispatch and class-hierarchy review, several of the v1.2 shapes have been refined:

- **`Surrogate(ArrayRandomFunction)`** — sabi's surrogate is now a real ProbPipe `ArrayRandomFunction`. `__call__(X, joint_inputs, joint_outputs) -> Distribution` is the predictive-distribution interface (free from the parent); `fit(X, Y) -> Self` is the only sabi-specific addition. `GPSurrogate(Surrogate, GaussianRandomFunction)` is the concrete Gaussian path — diamond inheritance over `ArrayRandomFunction`, resolved cleanly by C3 MRO. `SurrogatePrediction(mean, variance)` is removed; consumers use `mean` / `variance` ops on the returned `Normal` (or `MultivariateNormal` once joint modes land).
- **Class collapse + rename.** `SurrogateDistribution` (formerly `GPPushforwardSurrogateDistribution`) is now the only "surrogate posterior" — direct subclass of `NumericRandomMeasure`, holds `(surrogate, log_density_form, support, input_shape, prior)`. `WeightedEmpiricalRandomMeasure` (renamed from `WeightedEmpiricalSurrogateDistribution`) is a sibling class — a Dirac random measure that no longer carries `log_density_form`. The two are coordinate concepts in the loop's `surrogate_distribution_factory`, not parent/child classes.
- **`pushforward_marginal` dispatch.** A new free function in `sabi.surrogate._pushforward` does the work that was buried in `_PushforwardLogDensityRandomFunction.__call__`. Type-dispatched on `(input_dist, log_density_form)`:
  - `(Normal | MultivariateNormal, Identity | LogLikPlusPrior)` — closed-form affine pushforward (shift `loc`; same scale / scale_tril). Handles univariate marginals AND multivariate (joint over inputs / joint over outputs) uniformly.
  - `(samplable Distribution, anything)` — MC empirical via `@workflow_function`-wrapped helper. ProbPipe's broadcasting machinery samples from the input, runs the form pointwise (vmap when JAX-traceable), returns a `NumericEmpiricalDistribution`.
  - Otherwise — clear `NotImplementedError` naming the types and pointing at the partial-pushforward primitive in `docs/probpipe_issues.md`.
- **Acquisitions adopt the ProbPipe-native interface.** `ExpectedImprovement` now calls `state.surrogate(X)` to get a `Normal` and reads `mean(...)` / `variance(...)` via ProbPipe ops. No more sabi-specific `SurrogatePrediction` dataclass.

`RandomMeasure` lives in ProbPipe (PRs #150 + #151 — the latter relaxing
MCMC dispatch to `SupportsUnnormalizedLogProb`). Sabi's `SurrogateDistribution`
inherits from `NumericRandomMeasure` and is decoupled from `Problem` (takes
math primitives `support`, `prior`, `log_density_form`, `input_shape`
directly).

**Landed in this phase**
- `sabi.surrogate.SurrogateDistribution(NumericRandomMeasure)` — abstract base.
  Constructor takes math primitives, requires `support` to be set (raises
  otherwise), and exposes `inner_support` / `inner_event_shape` derived from
  those args. No protocol opt-ins on the base — subclasses choose.
- `sabi.surrogate.WeightedEmpiricalSurrogateDistribution` — Dirac random measure.
  Stores `(X, Y_log_density)`, exposes `inner_distribution: NumericEmpiricalDistribution`
  via cached property. Implements `SupportsMean` (returns inner empirical),
  `SupportsSampling` (returns the inner empirical for `sample_shape == ()`,
  `DistributionArray` of repeats otherwise), and
  `SupportsRandomLogProb` / `SupportsRandomUnnormalizedLogProb` via a Dirac
  random function shim.
- `sabi.surrogate.GPPushforwardSurrogateDistribution` — proper random measure.
  Implements `SupportsRandomUnnormalizedLogProb` for the closed-form forms
  (`Identity`, `LogLikPlusPrior`); `ForwardModel` raises until partial-pushforward
  primitive lands. Does NOT implement `SupportsSampling` (surrogate doesn't
  yet expose function-trajectory sampling — v1.6), `SupportsMean` (no closed-form
  expected posterior), or `SupportsRandomLogProb` (normalization intractable).
- Sabi-local Dirac shims (`_DiracDistribution`, `_DiracArrayRandomFunction`)
  in `sabi.surrogate._dirac` — graduate to ProbPipe when a general `Dirac`
  abstraction lands.
- `sabi.surrogate._pushforward._PushforwardLogDensityRandomFunction` — the
  marginal random log-density for the GP-pushforward path. Affine pushforward
  for closed-form forms; raises for `ForwardModel`.
- Free-function deterministic estimators in `sabi.surrogate.estimators`:
  - `expected_target(sp)` — biased plug-in (renamed from "plug-in mean" since
    it's the expectation of the target map under the surrogate). For Dirac SPs,
    coincides with `mean(sp)`; for the GP path, returns an
    `_ExpectedTargetDistribution` that satisfies `SupportsUnnormalizedLogProb`
    + `SupportsSampling` (the latter via `condition_on(self)` → NUTS).
- `LogDensityForm.__call__(x, y, *, prior=None)` — decoupled from `Problem`,
  takes `prior` directly. Sabi's `LogLikPlusPrior` / `ForwardModel` sum the
  prior's `log_prob` across all returned dims (a v1.2 pragma to handle
  ProbPipe's element-wise `Uniform` cleanly; revisit when ProbPipe ships
  joint multivariate distributions).
- Loop adapter (`_build_surrogate_distribution`) extracts math primitives from
  `Problem` and passes them to the factory; the factory + SP know nothing
  about `Problem`.
- Old `sabi.estimators` package removed.
- ProbPipe op imports switched from `import probpipe.core.ops as pp_ops` to
  top-level `from probpipe import condition_on, sample, log_prob, ...` per
  user preference.
- 20 new tests in `tests/test_surrogate_distribution.py`; 66 tests pass total.

**Deferred to later phases**
- `mean(gp_sp)` returning the unbiased expected posterior — v2 with MC backend
  + ProbPipe `PosteriorEstimator` backend dispatch.
- `SupportsSampling` on `GPPushforwardSurrogateDistribution` — needs a
  function-trajectory sampler on the surrogate (v1.6 swap to a real GP backend).
- `_random_unnormalized_log_prob` for `ForwardModel` — needs partial-pushforward
  primitive in ProbPipe (tracked in `docs/probpipe_issues.md`).
- Future estimators (`expected_log_density`, `expected_density`,
  `median_density`) — opt-in free functions, motivated by the partial-pushforward
  primitive.

---

## v1.3 — Reference posterior infrastructure — **landed**

NUTS-based reference computation for benchmarks without analytic
samples; on-disk caching of expensive references via Parquet artifacts.

**Landed in this phase**
- `sabi.reference` package: `io.py` (Parquet + JSON metadata read/write),
  `nuts.py` (`_ProblemTargetDistribution` + `generate_via_nuts` driving
  ProbPipe `condition_on` → NUTS, with ArviZ-derived diagnostics),
  `cache.py` (`load_or_generate_reference_samples` with cache hit / regen /
  quality-threshold gating).
- Cache layout: `reference_posteriors/<problem>/<key>_<sampler-tag>.{parquet,json}`.
  Filename encodes problem variant + NUTS config; different params → different
  artifact. Tier-A artifacts (small) committed to the repo.
- ArviZ diagnostics (R-hat, ESS bulk/tail, divergences) computed during
  regeneration and embedded in metadata JSON. `cache._validate_diagnostics`
  refuses to save if max R-hat / min ESS / divergence rate fail thresholds.
- Neal's funnel benchmark (`sabi.problems.neals_funnel.neals_funnel`):
  `input_shape=(d+1,)`, joint log-density of `v ~ N(0, σ_v²)` and
  `x_i | v ~ N(0, exp(v))`. Reference is loaded from cache; regenerates
  via NUTS on first call with new params. Funnel-specific quality thresholds
  loosen R-hat and ESS gates to acknowledge vanilla-NUTS difficulty without
  reparameterization (a v1.5+ stretch concern).
- `scripts/regenerate_references` — Python CLI for forced regeneration.
  `./scripts/regenerate_references --problem neals_funnel` refreshes the
  artifact in place.
- Runner / build wiring: `configs/problem/neals_funnel.yaml` + `build.py`
  dispatch on `name == "neals_funnel"`. `build_algorithm(cfg, problem=...)`
  signature now threads the problem's `input_shape` into the emulator
  factory so `TinyGPEmulator` is constructed with the right shape contract.
- `blackjax` removed from `pyproject.toml` dependencies. We use ProbPipe's
  `tfp_nuts` exclusively for both reference-posterior generation and the
  v1.2 `expected_target` sampling backend.
- Tests: `tests/test_reference.py` (Parquet roundtrip, cache hit/miss,
  cache-key disambiguation, quality threshold raising) +
  `tests/test_neals_funnel.py` (shape/type, log-density spot check, v
  marginal moments, funnel-geometry signature). 80 tests total pass.

**Deferred to later phases**
- Geometry-aware Neal's-funnel reference (non-centered reparameterization
  for substantially better ESS) — v1.5 stretch concern alongside emulator
  metrics on the funnel.
- `predict_covariance` on `TinyGPEmulator` (joint-mode pushforward) — v1.5+
  when emulator metrics need it.
- Tier-B reference posteriors (large, expensive) — would use git-lfs;
  schedule when first benchmark requires them.

---

## v1.4 — Acquisition optimization module (§5) — **landed**

Pluggable optimizer hierarchy for pointwise-scored acquisitions, plus
greedy multi-point batching with pluggable fantasy strategies.

**Landed in this phase**
- New `Acquisition` taxonomy in `sabi/acquisitions/base.py`:
  `PointwiseScoredAcquisition` provides `score(x, state) → scalar` and
  delegates `select_batch` to a configurable `PointwiseOptimizer`.
  Sampling-style and batch-scored acquisitions are noted as v1.5+.
- `sabi/acquisitions/optim.py`:
  - `CandidateSetOptimizer` — random candidates → top-q (the v1.x EI
    behavior; default for backwards compatibility).
  - `ContinuousMultiStartOptimizer` — random candidates → score-filter to
    top-`n_starts` → optimistix BFGS in unconstrained reparameterization
    space → return top-q. Sigmoid bijector for `interval(low, high)`
    supports; non-interval supports raise with a pointer to the
    `docs/probpipe_issues.md` Constraint→bijector entry.
  - `GreedyMultiPointOptimizer` — for `q > 1`. Wraps an inner optimizer
    plus a pluggable `FantasyImputer`.
  - Multi-start BFGS uses a Python loop with a `# TODO(vmap-multistart)`
    marker; `optimistix` doesn't ship a multi-start helper, so vmap is
    a future optimization.
- `sabi/acquisitions/fantasize.py`:
  - `FantasyImputer.impute(x_pending, state) -> Array`.
  - `KrigingBeliever` (default) uses surrogate predictive mean.
  - `ConstantLiar(value="min" | "max" | "mean" | float)`.
- `ExpectedImprovement` refactored to `PointwiseScoredAcquisition` with
  scalar `score(x, state)`. v1.x `n_candidates` field migrated to the
  optimizer; default `optimizer = CandidateSetOptimizer()` keeps the v1.x
  numeric behavior unchanged.
- `Random` renamed to `PriorSampling` (with `Random` alias retained for
  config / import-path compatibility).
- Hydra config: `configs/acquisition/ei.yaml` gains an `optimizer:`
  subsection; new `configs/acquisition/ei_continuous.yaml` selects the
  continuous multi-start backend. `runner/build.py._build_optimizer`
  dispatches.
- Tests (94 pass total):
  - `tests/test_fantasize.py` — `KrigingBeliever` returns surrogate mean,
    `ConstantLiar` produces correct constants for "min" / "max" / "mean"
    / explicit float; invalid string raises.
  - `tests/test_optim.py` — interval-sigmoid bijector roundtrips and
    boundary-clipping; `_make_bijector` raises on non-interval supports;
    `CandidateSetOptimizer` returns shape-correct top-q;
    `ContinuousMultiStartOptimizer` finds the known argmax of a synthetic
    concave score, clamps to support when target lies outside the box,
    and matches-or-beats `CandidateSetOptimizer` on score; greedy
    multi-point returns distinct picks under both `KrigingBeliever` and
    `ConstantLiar(value="min")`.
  - `tests/test_loop.py` — `test_loop_continuous_ei_beats_candidate_set_ei_on_gaussian2d`
    verifies the v1.4 exit criterion: continuous EI yields ≤ candidate-set
    EI MMD² on gaussian2d at matched budget.

**Deferred to later phases**
- BOTorch-comparison test on a fixed acquisition surface. The synthetic
  argmax tests in `test_optim.py` cover correctness; BOTorch becomes
  useful for benchmarking once we have a third-party baseline mindset.
  Note in code comment: revisit when BOTorch comparison is wanted.
- Sampling-style acquisitions: `PosteriorThompsonSampling`,
  `MixtureSampling`, `BatchSelector` family — v1.5.
- Batch-scored acquisitions (q-EI, max-min entropy) and the parallel
  `BatchOptimizer` hierarchy — v1.5.
- vmap-of-`optimistix.minimise` for multi-start parallelism — once
  validated against our solver settings.
- Optimizer-aware Hydra config for the greedy nested form (`inner` +
  `imputer` selection) — minimal in v1.4; deepens if v1.5 needs richer
  greedy configs.

---

## v1.5 — `EmulatorMetric` + Neal's funnel as a workout

Round out the metric surface and add a benchmark that stresses the
abstractions.

**Deliverables**
- `EmulatorMetric` protocol (signature in design doc §4.7), with `dict[str, float]` returns aligned with `PosteriorMetric`.
- Built-in `EmulatorMetric`s: log-score at validation points, log-score of induced unnormalized log-density (calibration of the pushforward).
- Loop support for emulator metrics in addition to posterior metrics.
- Neal's funnel as a Problem exercise (depends on v1.3 reference posterior).

**Exit criteria**
- Both metric protocols evaluable per round.
- Neal's funnel runs end-to-end (probably poorly with the toy GP / weighted-empirical baselines — that's fine, it's a stretch benchmark).

---

## v1.6 — Real GP backend (gpjax or ProbPipe `GaussianRandomFunction`)

Defer until the abstractions are stable. Drop the v0 toy GP in favour of a
library-backed GP with proper hyperparameter optimization.

**Deliverables**
- `Surrogate` GP backend swapped for gpjax (or ProbPipe `GaussianRandomFunction` if it lands first).
- Hyperparameter optimization driven by the library, not hand-rolled.
- v0 fixed-lengthscale heuristic and any leftover optimistix scaffolding deleted.
- `Surrogate` declared as an `ArrayRandomFunction` subclass formally (probably anticipated in v1.1 but consummated here).

**Exit criteria**
- Generalization metric on the smooth-2D test improves materially vs v0 toy GP.
- Loop run-time per round same order of magnitude as v0 (no regression from library overhead).

---

## After v1

**v2** — tempering becomes real (design doc §4.11–§4.13): `LikelihoodTempering` with `FixedSchedule` / `ESSAdaptiveSchedule`, `EmulatorTarget` adapter for tempered-target emulator fits. `PosteriorEstimator` backend dispatch on `(estimator_type, surrogate_distribution_type, backend)` for IS / MCMC / SMC / VI; native VBMC; stochastic acquisitions (Thompson); `ExpectedPosterior` estimator's full implementation. The "distributions in, distributions out" generalization for metrics consummated by v2.

**v3** — `q > 1` batches, Tier-B benchmarks, W&B logger backend, `DataTempering` + `PartitionedLogLikForm` when the first data-tempering benchmark lands.

**v4+** — noisy / stochastic targets, non-GP surrogates (BNN, deep ensemble, ENN).
