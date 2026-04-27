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

## v1.2 — `RandomMeasure` abstraction; `SurrogatePosterior` becomes a `RandomMeasure` subclass

Bigger conceptual jump than v1.1. Implement the `RandomMeasure` base class as
a sabi-local primitive that follows ProbPipe conventions, intended to graduate
to ProbPipe once they add it. The user has noted this is significant enough
to be its own phase.

**Deliverables**
- `sabi.random_measure.RandomMeasure` base class. Subclass of ProbPipe `Distribution[Distribution[T]]` (a distribution whose values are themselves distributions). Declares the support of the inner distributions via a `Constraint`.
- Optional methods (mirrored on ProbPipe protocol patterns):
  - `_random_log_prob(...)` — returns a `RandomFunction[Array, Array]` whose evaluation at a point `x` yields the marginal `Distribution` of `log p(x)` under random draws from the random measure.
  - `_random_unnormalized_log_prob(...)` — same but without the (potentially intractable) normalizer. Distinguished per ProbPipe conventions.
  - `_mean()` — the *mean random measure* in the standard sense, i.e. the distribution `\bar p(x) = E[p(x)]`. In sabi's surrogate-modeling vocabulary this is the "expected posterior".
- `SurrogatePosterior` now inherits from `RandomMeasure`. The existing GP-pushforward variant declares `SupportsRandomLogProb` (or whatever name we agree on) — its random log-density is the pushforward of the `Surrogate`'s `RandomFunction` through `LogDensityForm`. The weighted-empirical baseline is a Dirac random measure with `SupportsMean` but not `SupportsRandomLogProb`.
- The `WeightedEmpiricalSurrogatePosterior` from v1.1 stays the same in behaviour but is re-typed as a deterministic `RandomMeasure`.
- New estimator: `ExpectedPosterior` uses `RandomMeasure._mean()` directly when available, falls back to MC over function draws otherwise.
- Tests for `RandomMeasure`: random log-density returns a `Distribution`, the `_mean()` of the GP-pushforward random measure matches the v0 plug-in-mean pushforward, the Dirac random measure's mean equals the underlying empirical distribution.

**Exit criteria**
- `RandomMeasure` is a clean base class that stays syntactically identical (or near-identical) to what would land in ProbPipe. Code review with the ProbPipe author (you) before declaring this phase done.
- All v1.1 tests still pass; new tests cover `RandomMeasure` semantics.

**Open design questions for v1.2**
1. Method naming for "random log density" vs "random unnormalized log density": follow ProbPipe's `_log_prob` / `_unnormalized_log_prob` pattern → `_random_log_prob` / `_random_unnormalized_log_prob`. Confirm.
2. Should `RandomMeasure` declare `output_type` (the inner `Distribution[T]` type) explicitly, or is it inferred from the random log-density's return type? Probably explicit, aligned with how `ArrayRandomFunction` declares `input_shape` / `output_shape`.
3. The `SupportsRandomLogProb` protocol: does it live in sabi during v1.2 and migrate to ProbPipe later, or do we propose it directly to ProbPipe and pull it from there as soon as it lands?

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
