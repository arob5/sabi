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
pushforward operator, ship a baseline `WeightedEmpiricalSurrogatePosterior`
that needs no GP at all, and make the loop pass `Distribution` objects to
metrics rather than raw sample arrays.

**Deliverables**
- `Problem.prior: Distribution | None` (replaces `prior_log_prob` + `prior_sample` callables). Where v0 benchmarks had `sampling_bounds=(low, high)` for uniform initial design, v1 expresses the same thing as `prior = Uniform(low, high, name=...)` — i.e. the prior field is the design distribution. (Renaming the field to something like `design_distribution` is a semantic bike-shed; v1.1 keeps `prior` to limit churn.)
- `Problem.support: Constraint | None` (replaces `sampling_bounds`). For v1.1 `support` is **metadata** — used by acquisitions / metrics that want to check feasibility, but sampling routes through `prior`. ProbPipe exposes TFP bijectors via `TransformedDistribution` (`probpipe/distributions/transformed.py`); we don't need them for v0 benchmarks (bounded boxes are sampled directly via `Uniform`), but they're available for v1.3+ benchmarks with non-trivial supports (positive parameters, simplex, etc.).
- `Problem.reference_distribution: Distribution | None` replacing `reference_samples` + `reference_log_prob`. For benchmarks with an analytic posterior we use the corresponding ProbPipe distribution directly (e.g. `gaussian2d`'s reference is a `MultivariateNormal`); for benchmarks without one we wrap pre-computed samples in `NumericEmpiricalDistribution`.
- Benchmarks rebuilt: `gaussian2d`'s posterior is a ProbPipe `MultivariateNormal`, and `target_function = lambda x: log_prob(posterior_dist, x)` per Option A. `banana` keeps its inline analytic log-density for v1.1 (a transformed-Normal expression of the banana posterior is a v1.5 exercise) and uses a `NumericEmpiricalDistribution` reference. **Watch:** `MultivariateNormal` currently casts to float32 internally, so the `target_function` may need an explicit cast back to float64 to interop with our x64-enabled tests; if that gets ugly, surface as a ProbPipe gap.
- `LogDensityForm` documented and labeled as **sabi's local pushforward operator**: it composes `f(x) = y` with `prior.log_prob(x)` to form an unnormalized log-posterior. Naming aligns with ProbPipe (`unnormalized_log_prob`), so the eventual swap to ProbPipe's pushforward is mechanical.
- `SurrogatePosterior` becomes abstract; subclasses declare which `Supports*` protocols they satisfy.
- New baseline: `WeightedEmpiricalSurrogatePosterior` — stores `(X, Y)` where `Y` is observed log-density at `X`. Exposes its inner posterior as a ProbPipe `NumericEmpiricalDistribution(samples=X, weights=Weights(log_weights=Y))` via a cached property. Sabi does **not** reimplement weighting / normalization / ESS — `Weights` in `probpipe/_weights.py` handles all of that. Inherits `SupportsSampling` and `SupportsMean` from the wrapped empirical.
- Existing GP-pushforward `SurrogatePosterior` (the v0 plug-in-mean object) refactored into a second concrete subclass `GPPushforwardSurrogatePosterior`. Implements `SupportsSampling` (via IS over the surrogate-mean log-density) only.
- **`PlugInMean` subclasses `NumericRecordDistribution`** (ProbPipe's base for all numeric distributions with TFP-style shape semantics) — not a generic `Distribution[Array]`. Constructor takes a `SurrogatePosterior`; the result is the concrete plug-in-mean posterior (a `Distribution[Array]` with `event_shape == problem.input_shape`). Internal type dispatch on the surrogate-posterior subtype (until v1.2 formalizes this as `RandomMeasure._mean()` op dispatch):
  - For `WeightedEmpiricalSurrogatePosterior` (Dirac random measure): `_sample` and `_unnormalized_log_prob` forward to the underlying `NumericEmpiricalDistribution`. `PlugInMean` and `mean(surrogate_posterior)` coincide here.
  - For `GPPushforwardSurrogatePosterior`: `_sample` is IS over the surrogate-mean log-density (the v0 backend); `_unnormalized_log_prob` evaluates `LogDensityForm(x, surrogate.predict(x).mean, problem)`. The IS backend is the same hook v2 backend dispatch will replace with MCMC / SMC / VI.
- Loop materialization: after constructing `posterior_estimate = PlugInMean(surrogate_posterior)`, the loop draws `samples = sample(posterior_estimate, key, n)` once per round and wraps in `NumericEmpiricalDistribution(samples)` to hand to metrics. Metrics see a `Distribution[Array]`, never a raw array.
- `PosteriorMetric` declares its required protocols via a class attribute (e.g., `requires: tuple[type, ...] = (SupportsSampling,)`). Loop introspects each metric and **raises** if the estimator distribution doesn't support a required protocol.
- `_evaluate_metrics` reshaped: takes the estimator-returned `Distribution[Array]`, dispatches per-metric via ProbPipe ops (`sample`, `log_prob`, `mean`, etc.). Removes the "draw samples once, pass as array" assumption from the loop.
- Tests:
  - End-to-end run with **only** `WeightedEmpiricalSurrogatePosterior` (no GP) — confirms the baseline path is self-contained.
  - `PlugInMean(weighted_empirical_baseline)` produces the same samples (up to PRNG) as direct sampling from the underlying empirical — Dirac mean property holds in code.
  - `PosteriorMetric.requires` honored: a metric requiring `SupportsLogProb` against an estimator that only supports `SupportsSampling` raises a clear error.
  - All v0 tests still pass after rename / reshape.

**Exit criteria**
- All 41+ tests green.
- `runner.main` smoke test: gaussian2d with `WeightedEmpiricalSurrogatePosterior` produces sensible MMD numbers (the baseline isn't expected to be *good*, just non-degenerate).
- No raw `Array` flowing from estimator → metric in the loop body; metrics receive `Distribution[Array]` only.

**Open design questions for v1.1**
1. Benchmark expression — answered: route `target_function` through ProbPipe's `Distribution.log_prob` / `unnormalized_log_prob` for benchmarks where a ProbPipe distribution describes the posterior natively. Inline math is kept only when no clean ProbPipe equivalent exists.
2. `sampling_bounds` removal — answered: drop. Use `Constraint` + TFP bijectors. Verify ProbPipe's TFP bijector exposure on implementation; if missing, surface as a ProbPipe gap.
3. `PosteriorMetric.requires` mismatch — answered: raise.
4. Should the v0 toy GP wrapper survive as a `Surrogate` example? It still works; it's just not the primary baseline anymore. Recommend keeping it (handy for testing acquisitions in v1.4) but not advertising it.

---

## v1.2 — `SurrogatePosterior` as a `NumericRandomMeasure` subclass — **landed**

`RandomMeasure` lives in ProbPipe (PRs #150 + #151 — the latter relaxing
MCMC dispatch to `SupportsUnnormalizedLogProb`). Sabi's `SurrogatePosterior`
inherits from `NumericRandomMeasure` and is decoupled from `Problem` (takes
math primitives `support`, `prior`, `log_density_form`, `input_shape`
directly).

**Landed in this phase**
- `sabi.posterior.SurrogatePosterior(NumericRandomMeasure)` — abstract base.
  Constructor takes math primitives, requires `support` to be set (raises
  otherwise), and exposes `inner_support` / `inner_event_shape` derived from
  those args. No protocol opt-ins on the base — subclasses choose.
- `sabi.posterior.WeightedEmpiricalSurrogatePosterior` — Dirac random measure.
  Stores `(X, Y_log_density)`, exposes `inner_distribution: NumericEmpiricalDistribution`
  via cached property. Implements `SupportsMean` (returns inner empirical),
  `SupportsSampling` (returns the inner empirical for `sample_shape == ()`,
  `DistributionArray` of repeats otherwise), and
  `SupportsRandomLogProb` / `SupportsRandomUnnormalizedLogProb` via a Dirac
  random function shim.
- `sabi.posterior.GPPushforwardSurrogatePosterior` — proper random measure.
  Implements `SupportsRandomUnnormalizedLogProb` for the closed-form forms
  (`Identity`, `LogLikPlusPrior`); `ForwardModel` raises until partial-pushforward
  primitive lands. Does NOT implement `SupportsSampling` (surrogate doesn't
  yet expose function-trajectory sampling — v1.6), `SupportsMean` (no closed-form
  expected posterior), or `SupportsRandomLogProb` (normalization intractable).
- Sabi-local Dirac shims (`_DiracDistribution`, `_DiracArrayRandomFunction`)
  in `sabi.posterior._dirac` — graduate to ProbPipe when a general `Dirac`
  abstraction lands.
- `sabi.posterior._pushforward._PushforwardLogDensityRandomFunction` — the
  marginal random log-density for the GP-pushforward path. Affine pushforward
  for closed-form forms; raises for `ForwardModel`.
- Free-function deterministic estimators in `sabi.posterior.estimators`:
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
- Loop adapter (`_build_surrogate_posterior`) extracts math primitives from
  `Problem` and passes them to the factory; the factory + SP know nothing
  about `Problem`.
- Old `sabi.estimators` package removed.
- ProbPipe op imports switched from `import probpipe.core.ops as pp_ops` to
  top-level `from probpipe import condition_on, sample, log_prob, ...` per
  user preference.
- 20 new tests in `tests/test_surrogate_posterior.py`; 66 tests pass total.

**Deferred to later phases**
- `mean(gp_sp)` returning the unbiased expected posterior — v2 with MC backend
  + ProbPipe `PosteriorEstimator` backend dispatch.
- `SupportsSampling` on `GPPushforwardSurrogatePosterior` — needs a
  function-trajectory sampler on the surrogate (v1.6 swap to a real GP backend).
- `_random_unnormalized_log_prob` for `ForwardModel` — needs partial-pushforward
  primitive in ProbPipe (tracked in `docs/probpipe_issues.md`).
- Future estimators (`expected_log_density`, `expected_density`,
  `median_density`) — opt-in free functions, motivated by the partial-pushforward
  primitive.

---

## v1.3 — Reference posterior infrastructure

Smaller phase. Move analytic reference samples out of inline construction
into versioned artifacts; add NUTS-based reference computation for problems
that don't have analytic samples.

**Deliverables**
- Fix `blackjax` import in the shared venv (currently broken via fastprogress → IPython transitive).
- `reference_posteriors/` directory layout with content-hashed JSON manifest + Parquet artifacts.
- `Problem.reference_distribution` becomes a load-on-demand artifact (not computed at construction time).
- Regeneration script that runs blackjax NUTS for a `Problem` whose posterior isn't analytic.
- Add Neal's funnel as a Tier-A benchmark (no analytic samples, exercises the NUTS path).

**Exit criteria**
- Tier-A benchmarks (gaussian2d, banana, Neal's funnel) all load reference distributions from artifacts.
- Regeneration is a single CLI command.

---

## v1.4 — Acquisition optimization module (`§5`)

Build the shared optimizer that all acquisitions plug into.

**Deliverables**
- `acquisitions/optim.py` with multi-start, greedy multi-point batching (for `q > 1`), support-aware reparameterization (uses the `Constraint` from v1.1; bijector layer added if ProbPipe still doesn't have one).
- Shared helpers: candidate sampling, top-k, local refinement, in-batch diversification.
- Refactor `Random` and `EI` to declare which optimizer modes they support and use the shared machinery.
- At least one truly continuously-optimized acquisition (e.g., `EI` with multi-start BFGS rather than candidate-set scoring).
- BOTorch-comparison test on a fixed acquisition surface — design doc §5 says "no merge without those tests".

**Exit criteria**
- Continuous-optimization EI on gaussian2d produces visibly better MMD than candidate-set EI at matched budgets.
- BOTorch comparison test passes (within an agreed tolerance).

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

**v2** — tempering becomes real (design doc §4.11–§4.13): `LikelihoodTempering` with `FixedSchedule` / `ESSAdaptiveSchedule`, `EmulatorTarget` adapter for tempered-target emulator fits. `PosteriorEstimator` backend dispatch on `(estimator_type, surrogate_posterior_type, backend)` for IS / MCMC / SMC / VI; native VBMC; stochastic acquisitions (Thompson); `ExpectedPosterior` estimator's full implementation. The "distributions in, distributions out" generalization for metrics consummated by v2.

**v3** — `q > 1` batches, Tier-B benchmarks, W&B logger backend, `DataTempering` + `PartitionedLogLikForm` when the first data-tempering benchmark lands.

**v4+** — noisy / stochastic targets, non-GP surrogates (BNN, deep ensemble, ENN).
