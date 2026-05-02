# ProbPipe issues encountered while building sabi

A running ledger of ProbPipe limitations, gaps, and rough edges discovered as
sabi work proceeds in parallel with ProbPipe development. Some of these we
work around in sabi; others want a fix in ProbPipe; many are notes for
informing ProbPipe's design as features mature.

**Branch pinning.** While ProbPipe PRs are in flight, sabi shadows the venv's
editable `probpipe` install with a pinned worktree path stored in
`.probpipe-pin` at the repo root. The `scripts/python` shim consumes the pin
and prepends it to `PYTHONPATH`. Once all in-flight PRs land in `main`,
empty or delete `.probpipe-pin` and the shim falls back to the editable
install. The pin currently points at the `claude/awesome-payne-6faf6e`
worktree (PR #146, stacked on PR #145).

Each entry: title, status, sabi context, what we observed, what we'd want.
Append-only; mark resolutions in place rather than deleting.

Conventions for the **status** field:
- `open` — present today, not yet addressed.
- `in-flight` — fix in progress (link the PR or session).
- `resolved` — fixed in ProbPipe; entry retained for historical context.
- `wontfix` — confirmed out of scope; sabi works around.

---

## ProbPipe import fails on Python 3.12 (PEP 695 + `get_type_hints`)

**Status:** resolved ([PR #145](https://github.com/TARPS-group/prob-pipe/pull/145) merged 2026-04-27).

**Sabi context.** Blocked v1.1 at the very first step — `import probpipe` raises before any sabi code runs.

**What we observed.** `probpipe/core/transition.py` declares `def iterate[T, S](...)` with PEP 695 generic syntax. The `@workflow_function` decorator at module import time eventually calls `get_type_hints(func)` (`probpipe/core/node.py:369`) without supplying a `localns`. Python 3.12's `get_type_hints` doesn't pull PEP 695 type parameters into its eval namespace, so `Distribution[T]` annotations fail to resolve with `NameError: name 'T' is not defined`.

**Suggested fix.** Pass the function's `__type_params__` into `get_type_hints` via `localns`. Also grep for other bare `get_type_hints(...)` call sites that may need the same treatment.

**Why it matters for sabi.** Total blocker until fixed; no `import probpipe` ⇒ no v1.1. Sabi pins at the PR branch worktree until merged.

---

## TFP-backed distributions hard-cast parameters to float32

**Status:** resolved ([PR #146](https://github.com/TARPS-group/prob-pipe/pull/146) merged 2026-04-27).

**Sabi context.** `gaussian2d`'s posterior is a natural fit for `probpipe.distributions.multivariate.MultivariateNormal`, and per the v1 plan we route `target_map = lambda x: log_prob(posterior, x)`. Sabi tests run with `jax_enable_x64`.

**What we observed.** Every TFP-backed `__init__` in `probpipe/distributions/multivariate.py` and `probpipe/distributions/continuous.py` hard-codes `jnp.asarray(..., dtype=jnp.float32)`. Under x64, calling `log_prob(mvn, x)` with a float64 `x` raises a hard `TypeError` (not just a precision warning) from TFP's bijector internals. `Uniform.sample` also returns float32 regardless of x64.

**What's in the PR.** `probpipe/_dtype.py` adds private `_default_float_dtype()` and `_promote_floats()` helpers; constructors drop the explicit `float32` and use these helpers to coerce inputs to a common float dtype via `jnp.result_type` (with int → default-float promotion). The PR description reports 2102 (x64) / 2121 (x32) tests passing plus 19 new dtype regression tests.

**Why it matters for sabi.** Hard blocker for x64-mode use of TFP-backed distributions. Sabi pins at the PR branch worktree until merged.

---

## `NumericRecord` doesn't act as a JAX array under Python operators

**Status:** open.

**Sabi context.** ProbPipe ops (`log_prob`, `mean`, `sample`, etc.) return `NumericRecord` containers rather than bare arrays. Sabi's `LogDensityForm.LogLikPlusPrior` does `y + log_prob(prior, x)`; `target_map` for `gaussian2d` returns `log_prob(mvn, x)` and downstream code expects a plain scalar.

**What we observed.** `NumericRecord` implements `__array__` and `__jax_array__`, so JAX-namespace functions auto-unwrap it transparently:

```python
jnp.add(record, 1.0)      # works
jnp.exp(record)           # works
jnp.sum(jnp.asarray([record, record]))  # works
```

But Python's built-in arithmetic operators do not — `NumericRecord` doesn't implement `__add__` / `__radd__` / `__mul__` / etc., so:

```python
record + 1.0
# TypeError: unsupported operand type(s) for +: 'NumericRecord' and 'float'
```

This forces every sabi call site to wrap with `jnp.asarray(record)` (or use `jnp.add` instead of `+`) before doing arithmetic, which leaks the wrapper layer into otherwise-clean math.

**Why it matters for sabi.** The cleanest implementation of `LogLikPlusPrior` would be `return y + pp_ops.log_prob(problem.prior, x)`. Today we have to write `return y + jnp.asarray(pp_ops.log_prob(problem.prior, x))`. Same issue surfaces in any user code that combines op outputs with raw arrays via Python operators.

**What we'd want.** `NumericRecord` should act as a JAX array under Python arithmetic when it carries a single numeric leaf — implement `__add__` / `__radd__` / `__sub__` / `__mul__` / `__truediv__` / `__neg__` / `__pow__` / `__matmul__` (and right-hand variants) by delegating to the underlying array. Multi-field records can either delegate field-wise or raise. The single-leaf case is the common one and the one biting sabi today.

**Workaround.** Wrap with `jnp.asarray(record)` at every arithmetic boundary, or use `jnp.add` / `jnp.multiply` / etc. Sabi's `LogDensityForm` subclasses do the former.

---

## No `Constraint` → TFP bijector registry

**Status:** open. Sabi v1.4 hits this directly in
`ContinuousMultiStartOptimizer`'s reparameterization step.

**Sabi context.** sabi v1.4 adds `ContinuousMultiStartOptimizer`, which runs
BFGS in unconstrained ℝ^d coordinates and forward-maps back to the support
via a TFP bijector. The bijector itself comes from TFP — what's missing in
ProbPipe is the dispatch from a `Constraint` (the support) to the canonical
bijector for that constraint. v1.3+ benchmarks may also pick up non-box
supports (positive parameters, simplex) which the same mapping would unlock.

**What we observed.** ProbPipe ships `TransformedDistribution(base, bijector)`
and a small `_BIJECTOR_SUPPORT_MAP` that goes the other direction (bijector
class → output `Constraint`). There's no inverse mapping — given a
`Constraint`, callers have to manually pick the bijector. Sabi v1.4
hard-codes the dispatch in `sabi/acquisitions/optim.py::_make_bijector`:
`_Interval(low, high) → tfb.Sigmoid(low, high)`, with `NotImplementedError`
for everything else. That dispatch is the gap — there's no good place for
it inside ProbPipe today, so each consumer reinvents it.

**Why it matters for sabi.** v1.4's benchmarks (`gaussian2d`, `banana`,
`neals_funnel`) all have bounded-box supports, so today's `_make_bijector`
covers them. The moment we add a benchmark with `positive` /
`simplex` / `unit_interval` / etc., we have to extend sabi's local table.
A ProbPipe-side `bijector_for(constraint) -> Bijector` op (or a
`Constraint.default_bijector()` method) would absorb this growth.

**What we'd want.** A `bijector_for(constraint) -> tfb.Bijector` op (or
equivalent `Constraint.default_bijector()` method) that returns a canonical
bijector mapping ℝ^d → support, mirroring the existing reverse registry.
Initial coverage: `_Interval` → `Sigmoid(low, high)`, `_Positive` →
`Exp` (or `Softplus`), `_Simplex` → `SoftmaxCentered`, `_Real` → `Identity`.
Custom constraints register via the same mechanism.

**Workaround for v1.4.** `sabi/acquisitions/optim.py::_make_bijector`
holds the dispatch locally; non-interval constraints raise with a pointer
to this entry.

---

## `RandomMeasure` not yet in ProbPipe

**Status:** resolved ([PR #150](https://github.com/TARPS-group/prob-pipe/pull/150) merged 2026-04-27).

**Sabi context.** sabi's `SurrogateDistribution` is conceptually a distribution over distributions — needs `RandomMeasure`-shaped abstractions.

**What landed.** `RandomMeasure[T](Distribution[Distribution[T]])` and `NumericRandomMeasure(RandomMeasure[Array])` in `probpipe/core/_random_measures.py`, plus `SupportsRandomLogProb` / `SupportsRandomUnnormalizedLogProb` protocols and matching ops. v1.2 of sabi builds `SurrogateDistribution` directly on these.

---

## MCMC methods require `SupportsLogProb` but only need `SupportsUnnormalizedLogProb`

**Status:** open.

**Sabi context.** v1.2's `expected_target(sp)` returns a `Distribution[Array]` whose density is `log_density_form(x, surrogate.predictive_mean(x), problem)` — an *unnormalized* log-posterior (the surrogate gives no normalizer). Sabi wants ProbPipe's `condition_on(et_dist)` to auto-dispatch to NUTS for sampling.

**What we observed.** `tfp_nuts.check()` and `tfp_hmc.check()` (`probpipe/inference/_tfp_mcmc.py:306`) and `tfp_rwmh.check()` (`probpipe/inference/_rwmh.py:64`) all guard with `isinstance(dist, SupportsLogProb)` rather than `SupportsUnnormalizedLogProb`. Mathematically, MCMC works on unnormalized densities — that's the whole point. Forcing `SupportsLogProb` means callers either implement `_log_prob` as an alias for `_unnormalized_log_prob` (lying about normalization) or fall off the auto-dispatch path entirely.

**Why it matters for sabi.** Every deterministic-target estimator (`expected_target`, future `expected_log_density`, `expected_density`, `median_density`, etc.) returns a `Distribution[Array]` with only an unnormalized log-density available. Forcing them to alias `_log_prob = _unnormalized_log_prob` is semantically wrong and would propagate to anyone who reads the code.

**What we'd want.** Relax the MCMC `check` predicates to `SupportsUnnormalizedLogProb`. The `_build_target_log_prob` path already accesses `dist._log_prob` directly — change to `dist._unnormalized_log_prob` (which by protocol-default delegates to `_log_prob` when the latter exists, so the regular path is unaffected). Same change in `_rwmh`.

**Workaround for v1.2.** Sabi's `_ExpectedTargetDistribution` will alias `_log_prob = _unnormalized_log_prob` with a docstring noting the semantic mismatch and a `# TODO(probpipe-mcmc-unnormalized)` marker for cleanup once this lands.

---

## `WeightedEmpiricalRandomMeasure` as a ProbPipe primitive

**Status:** open.

**Sabi context.** Sabi defines `WeightedEmpiricalRandomMeasure` as a `NumericRandomMeasure` whose draws are degenerate: every draw is the same `NumericEmpiricalDistribution(samples=X, log_weights=Y)`. It's the simplest non-trivial random measure — a Dirac at an empirical distribution — and serves both as a no-GP baseline in sabi's loop and as a reference for any random-measure consumer.

**What we'd want.** Promote `WeightedEmpiricalRandomMeasure[T](NumericRandomMeasure)` (or its non-numeric counterpart `WeightedEmpiricalRandomMeasure[T]`) into ProbPipe directly. Generally useful: it's the natural representation of an SMC particle ensemble's posterior, of any mixture-of-empirical posterior, and of "one inner distribution treated as a Dirac random measure". Would graduate alongside the broader Dirac generalization (`Dirac[T]`, `DiracRandomFunction`, `DiracRandomMeasure`) — the empirical-Dirac case is just `Dirac[Distribution[T]]` instantiated with a `NumericEmpiricalDistribution` as the value.

**Why it matters for sabi.** When ProbPipe ships this, sabi's `WeightedEmpiricalRandomMeasure` becomes a thin re-export (or vanishes entirely if ProbPipe's class is the right shape).

---

## No general `Dirac` distribution abstraction

**Status:** open.

**Sabi context.** v1.2's `WeightedEmpiricalSurrogateDistribution` is conceptually a Dirac random measure (no surrogate uncertainty). Implementing `SupportsRandomLogProb` for it requires a degenerate `RandomFunction` whose `__call__(x)` returns a Dirac `Distribution[Array]` at the deterministic log-density value. Today we'd build this Dirac inside sabi.

**Why it matters for sabi.** Multiple sabi v1.2 paths want "treat a deterministic value as a degenerate distribution for protocol-compatibility purposes": Dirac inner distributions in Dirac random measures, deterministic random functions (degenerate `RandomFunction`s), constant random log-densities, etc. Each instance is a small but fiddly shim.

**What we'd want.** A general `Dirac[T](Distribution[T])` parameterized by a deterministic value — works for any sample type. Provides `_sample` (returns the value), `_log_prob` (`0` at the value, `-inf` elsewhere), `_mean` (the value), and any reasonable defaults for moments. Would naturally generalize to a `DiracRandomFunction(RandomFunction[X, Y])` (a deterministic function wrapped to satisfy the `RandomFunction` protocol with degenerate output distributions) and a `DiracRandomMeasure[T](RandomMeasure[T])` (a fixed `Distribution[T]` wrapped to satisfy the `RandomMeasure` protocol).

**Workaround.** Sabi v1.2 builds local `_DiracArrayRandomFunction` / `_DiracDistribution` shims. They graduate to ProbPipe when this lands.

---

## Pushforward with partial information (mean-only, moment-only, etc.)

**Status:** open. Related to the existing pushforward gap below; calling out as a distinct concept.

**Sabi context.** The random log-density of a surrogate-pushforward posterior — `x ↦ log p̃(x; f) = log_density_form(x, f(x), problem)` for `f ~ Surrogate` — is a pushforward of the surrogate's `RandomFunction` through `log_density_form`. Closed-form availability depends on the form:

- **`Identity`** — pushforward is the surrogate's own random function.
- **`LogLikPlusPrior`** — pushforward is an affine shift of the surrogate's random function. Closed-form.
- **`ForwardModel`** — pushforward through a generally non-linear `log_lik_from_outputs`. Often the *mean* of the marginal at each `x` is computable analytically (or to leading order), even when the full distribution is not.

In the third case we'd ideally produce a `RandomFunction` whose `__call__(x)` returns a `Distribution` parameterized by *whatever moments / quantiles we know* — not necessarily a fully-specified parametric distribution.

**Why it matters for sabi.** sabi v1.2 ships `_random_unnormalized_log_prob` for the `Identity` and `LogLikPlusPrior` cases (closed-form). `ForwardModel` raises until either the user supplies a custom pushforward or a primitive lands.

**What we'd want.** A pushforward / transport-map abstraction that lets distributions carry "partial information" — e.g. a distribution defined only by its mean, or its mean + variance, or a finite set of moments. Composing a function with such a distribution under pushforward yields another partial-information distribution. This unlocks:

1. Generic random-log-density construction for any `LogDensityForm` whose composition with `Surrogate.predict_mean` is computable, even when full marginal pushforward isn't.
2. Estimators like `expected_log_density` (mean of the random log-density at each `x`) and `expected_density` (mean of the unnormalized density at each `x`), each of which is the mean of a different pushforward.
3. Median / quantile estimators that need pointwise pushforward of quantile information.

This is closely related to the long-standing "pushforward not first-class" gap below — but the partial-information aspect is a fresh concept the surrogate-modeling setting motivates.

**Workaround for v1.2.** sabi only handles forms where the closed-form pushforward of the surrogate's random function is straightforward. `ForwardModel` raises; users opt in to custom logic if they need it.

---

## Pushforward not yet a first-class operation

**Status:** open (`origin/dev/pushforward` branch is stale at commit `7fc5153`).

**Sabi context.** sabi's `LogDensityForm` is a deterministic pushforward — composes `f(x) = y` with `prior.log_prob(x)` to produce an unnormalized log-posterior. ProbPipe has *partial* pushforward via `BroadcastDistribution` and `WorkflowFunction` broadcasting, but no `pushforward(f, dist)` operation.

**Why it matters for sabi.** Until ProbPipe ships pushforward, `LogDensityForm` is sabi-local and explicitly described as "sabi's local pushforward operator" in the design doc. The eventual ProbPipe pushforward should subsume it.

**What we'd want.** Land the pushforward PR (with whatever design adjustments sabi's experience suggests). The stale branch is a starting point but probably needs significant updates given the recent record/distribution refactors.
