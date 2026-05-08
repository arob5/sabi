# sabi — Notation

This document defines the shape, symbol, and naming conventions used
throughout sabi. **Code and docs should follow these exactly** — if
you're tempted to use `theta` for an input or `D` for dimension, come
back here first.

## Shape conventions

sabi follows ProbPipe's `ArrayRandomFunction` shape conventions.

A single target map `f : x ↦ y` has

| field | meaning | example |
|---|---|---|
| `input_shape: tuple[int, ...]`  | shape of one input `x`  | `(2,)` for a 2-D parameter vector; `()` for scalar |
| `output_shape: tuple[int, ...]` | shape of one output `y` | `()` for scalar log-density; `(k,)` for a k-output forward model |

Batch dimensions prepend `input_shape` / `output_shape`. A design set of `n`
points has

```
X.shape == (n,) + input_shape
Y.shape == (n,) + output_shape
```

More generally, higher-rank batches are allowed: `X.shape == batch_shape + input_shape`.

## Symbols

| symbol | meaning |
|---|---|
| `f`  | the target map being emulated (`f(x) = y`) |
| `x`  | a single input to `f`, shape `input_shape` |
| `y`  | a single output of `f`, shape `output_shape` |
| `X`  | a batch of inputs, shape `(n,) + input_shape` |
| `Y`  | the corresponding outputs, shape `(n,) + output_shape` |
| `n`  | number of design points currently in `(X, Y)` |
| `d`  | dimensionality of the input space when rank-1, i.e. `input_shape[0]` |
| `p`  | dimensionality of the output space when rank-1, i.e. `output_shape[0]`; `p = 1` conventionally when `output_shape == ()` |
| `q`  | number of points selected per acquisition iteration (batch size) |

`θ` is commonly used in Bayesian-inference math for the parameter we're
inferring; in sabi code and docs we use `x` for this — the input to `f` — and
reserve `θ` for prose where it aids intuition.

## "Target" terms — glossary

The word *target* shows up in several roles in sabi. After the
``DensityDecomposition`` split (issue #65), the math identity and
the algorithmic emulation choice live on different objects:

| term | type | role |
|---|---|---|
| target distribution | ProbPipe `NumericRecordDistribution` | The math identity of the target: ``(name, event_shape, support)`` plus optionally an analytical ``_unnormalized_log_prob`` for benchmarks. Sabi does not subclass it — benchmark targets (``BananaTarget``, ``GaussianTarget``, ``NealsFunnelTarget``) subclass `NumericRecordDistribution` directly. |
| `Problem.target_distribution` | `NumericRecordDistribution` field | The benchmark's target — the math content of a `Problem`. |
| `DensityDecomposition` | abstract `NumericRecordDistribution` subclass at `sabi.density_decomposition` | The algorithmic emulation choice. Composes the emulator's output into log-density via ``link(target_map(x)) + shift(x)``. Many decompositions can pair with one target distribution. Lives on ``Algorithm.density_decomposition``. Concrete subclasses: `LogProbTermTarget`, `LogProbTarget`, `GaussianForwardModelTarget`. |
| `target_map` | `Map` | The batched function the emulator approximates: `f : X ↦ Y`. Abstract method on ``DensityDecomposition``; subclasses implement. |
| `IntermediateTarget` | frozen dataclass | Bookkeeping payload produced by a `TemperingScheme` at one schedule state. Carries ``state`` and ``output_transform``. Lives at `sabi.tempering`. The per-state effective decomposition lives separately, produced by ``TemperingScheme.intermediate_decomposition(base_decomposition, state)``. |
| `AcquisitionTarget` | `Enum` | Which schedule state the acquisition's `SurrogateDistribution` is built at: `CURRENT` / `NEXT` / `TERMINAL`. |
| `target_tempering_state` | opaque PyTree | The state value resolved by `AcquisitionTarget`. Recorded in the per-round metric row for ablation reproducibility. |
| `expected_target` | function `(SurrogateDistribution) -> Distribution` | Deterministic posterior estimator: plug the surrogate's predictive mean into the decomposition. |

## "Emulator" vs. "surrogate"

These two words look interchangeable but mean different things in
sabi:

- **Emulator** — the predictive model fit to evaluations of the
  *target map* `f` (an `ArrayRandomFunction` over the parameter
  space). Concrete in sabi as `Emulator` (`sabi.emulators.base.Emulator`),
  with GP backends in `sabi.emulators.tinygp`, `sabi.emulators.gpjax`.
  Tempering-agnostic: it just consumes `(X, Y_train)`.
- **Surrogate** — any approximate quantity replacing its exact
  analog. The surrogate of the *target distribution* is the
  `SurrogateDistribution` (a `RandomMeasure`, lives in
  `sabi.surrogate`); a "surrogate posterior" in BO literature is the
  same concept under a name biased toward Bayesian use. `Emulator` is
  *one piece* of how a `SurrogateDistribution` is constructed
  (emulator + form via pushforward), but the surrogate could in
  principle be built without an emulator at all (the
  `WeightedEmpiricalRandomMeasure` baseline does exactly this).

Mnemonic: emulator → function (`f`); surrogate → distribution (`π`).

## Per-role distribution fields (post-#65 layout)

The single ``prior`` field that the old ``TargetDistribution``
carried played four distinct roles. After the
``DensityDecomposition`` split, each role lands on its own field:

| Role | New home |
|------|----------|
| Modeling-prior add-on (Bayesian density assembly) | ``DensityDecomposition.shift`` (typically ``LogProb(modeling_prior)``) |
| Initial design distribution | ``Algorithm.initial_design_distribution`` |
| Candidate-set sampler (pointwise optimizers) | ``CandidateSetOptimizer.candidate_distribution`` |
| BFGS seed sampler (continuous optimizer) | ``ContinuousMultiStartOptimizer.seed_distribution`` |
| Parameter-space support definer | ``TargetDistribution.support`` (math) + ``Algorithm.x_support`` (algorithmic search region; defaults equal) |

Each is a plain ``Distribution`` (or ``Constraint`` for
``support`` / ``x_support``). Sampling everywhere routes through
``probpipe.sample(dist, key=..., sample_shape=...)`` directly —
``BatchSampler`` / ``PriorSampler`` are gone (deleted alongside
``sabi.sampling``). For per-dim arrays (e.g. an ``Uniform`` box),
wrap with ``sabi._probpipe_compat.independent_uniform`` to
re-interpret batch dims as event dims (a temporary shim until
ProbPipe ships an ``Independent`` wrapper).

Default-fill behavior (resolved at ``run()`` entry):

- ``x_support`` falls back to ``problem.target_distribution.support``.
- ``initial_design_distribution`` falls back to ``Uniform(x_support)``
  when ``x_support`` is a bounded interval; otherwise ``run()`` raises
  with a pointer to both fields.
- ``density_decomposition`` is required; ``run()`` raises if ``None``.

## Tempering / bridging vocabulary

A `TemperingScheme` is a family of intermediate target distributions
indexed by an opaque state PyTree. The vocabulary:

| term | meaning |
|---|---|
| `TemperingScheme` | The abstraction. Each scheme implements `intermediate_target(base, state) -> IntermediateTarget` (math identity at one state) and `intermediate_decomposition(base_decomposition, state) -> DensityDecomposition` (per-state effective decomposition via Map composition). Subclasses: `NoTempering`, `LikelihoodTemperingViaForm`, `LikelihoodTemperingViaTarget`. |
| `TemperingSchedule` | The state generator: `at(round_idx) -> (state, is_terminal)`. Subclasses: `UntemperedSchedule`, `FixedSchedule`. |
| `IntermediateTarget` | Per-state `TargetDistribution`. Carries `state` and `output_transform`. Math identity only — the per-state link / shift live on the per-state effective `DensityDecomposition`. |
| `tempering_state` | The opaque PyTree produced by the schedule and consumed by the scheme. Type is strategy-specific (scalar `beta` for likelihood tempering, subset index for data tempering, …). |
| `AcquisitionTarget` | Enum that picks *which* state the acquisition's `SurrogateDistribution` is built at — independent of the round's current state. |
| `output_transform` | Value object on `IntermediateTarget` describing how to derive `Y_train` for `f_state` from cached raw evaluations `Y_raw`. Single-axis for the target axis of tempering; identity when only the form varies. |

"Tempering" is sabi's name for the more general **bridging**
abstraction — building a sequence of intermediate distributions that
connect a tractable starting point to a target. Likelihood
tempering, annealed importance sampling, normalising-flow bridges,
score-based bridges, and emulator warm-starts are all instances.
sabi's hierarchy stays as `TemperingScheme` until a non-tempering
bridge actually lands; the abstraction is fully general today.

See [`tempering.md`](tempering.md) for the case analysis with worked
examples.

## Vectorization contract

sabi defers to ProbPipe's vectorization contract for distributions:

- **Public APIs are vectorized.** Class methods that accept design
  data (e.g., `decomposition.target_map(X)`) and ProbPipe ops on
  `Distribution`s (`unnormalized_log_prob(target, X)`,
  `log_prob(dist, X)`, `mean(rm)`, …) accept a leading batch axis
  and return one. Subclasses' `_unnormalized_log_prob(x)` hooks are
  expected to handle batched input natively (sum over the trailing
  event axis with `axis=-1`, index with `x[..., k]`, etc.); the
  ProbPipe ops do not vmap underneath you.
- **`vmap` belongs to the implementation, not the API.** When a
  method must internally apply a single-point computation per
  event, that's an implementation detail of the method. Callers
  pass batched inputs and get batched outputs; the conversion is
  not their concern.
- **Private hooks may be single-event by convention** in code that's
  outside the ProbPipe distribution context (e.g., underscore
  helpers like `_call_single` in older form classes, now mostly
  gone). When sabi exposes such a hook, the public batched method
  vmaps it. We minimize these cases — the goal is for most
  density-bearing objects to be ProbPipe distributions and follow
  the contract above directly.

If something in ProbPipe's distributions doesn't follow this
contract, that's a ProbPipe issue worth fixing upstream — see
[probpipe_issues.md](probpipe_issues.md).

| Class | public vectorized API | notes |
|--|--|--|
| `Emulator` | `__call__(X) -> Distribution` | inherited `predict_*` from `GaussianRandomFunction`, etc. |
| `PointwiseScoredAcquisition` | `score(X, state) -> (n,)` | `_score_single(x, state) -> scalar` is a private vmap-able single-point hook (one of the few exceptions) |
| `TargetDistribution` (subclasses) | `unnormalized_log_prob(target, X)` op (ProbPipe) | subclasses' `_unnormalized_log_prob(x)` hook is itself vectorized |
| `DensityDecomposition` (subclasses) | `unnormalized_log_prob(decomposition, X)` op + `decomposition.target_map(X)` | subclasses' `target_map(x)` is vectorized |

Subclasses override the single-point hook; the batched method is
provided by the base class via `jax.vmap` (or directly when a
batched implementation is more efficient — see e.g. `Identity`'s
batched `__call__` overriding the vmap path).

## `DensityDecomposition` shape contract

``DensityDecomposition`` is a ProbPipe ``NumericRecordDistribution``;
its public surface is the standard ProbPipe ops. The unnormalized
log-density at ``x`` decomposes as
``link(target_map(x)) + shift(x)`` (or just ``link(target_map(x))``
when ``shift is None``). The first term — the **log-prob residual**
— is the contribution attributable to the emulator's output
``y = target_map(x)``, after the link is applied; the shift is the
deterministic x-dependent additive term.

Vectorized:

- ``unnormalized_log_prob(decomposition, X)`` returns ``(n,)`` for
  ``X`` of shape ``(n,) + input_shape``.
- ``decomposition.target_map(X)`` returns ``(n,) + output_shape``.

Pushforward: ``decomposition.pushforward(x, y_dist)`` constructs the
per-``x`` map ``Affine(slope=1.0, intercept=shift(x)) @ link`` (or
just ``link`` when ``shift is None``) and dispatches through
``sabi.maps.pushforward``. Closed-form for
``(Affine, Normal | MultivariateNormal)``; MC fallback for non-affine
links via ``Compose`` recursion.

## Symbol-vs-verbose convention

sabi uses a **mixed** convention. Math symbols are reserved for two
specific contexts; verbose names are used everywhere else.

**Math symbols (`n`, `d`, `p`, `q`, `f`, `x`, `y`, `X`, `Y`)** appear in:

1. **Shape annotations** (today: docstring shape strings; future:
   `jaxtyping.Float[Array, "n d"]`). The symbol set used in shape
   strings *defines* what `n`, `d`, `q` mean everywhere — consistency
   across annotations and runtime code is the point.
2. **Function arguments and local variables where the symbol is
   well-established in math literature.** Acquisition `q` (batch
   size), design count `n` (e.g., `BatchSampler.sample(problem, key,
   n)`), input dim `d` in problem-builder factories
   (`banana(d=2, …)`).

**Verbose names** for everything else: `tempering_state`,
`input_shape`, `target_map`, `output_transform`,
`acquisition_target`, `emulator_factory`, etc. Test that a name is
"well-established" by asking whether a reader who knows the math
literature would recognize the symbol immediately. If not — verbose.

Concrete consequence: `p` as a local variable for a `Problem`
instance is **not** OK (it shadows `p = output_shape[0]`); use
`problem`. Renaming pass landed in #7.

## Naming in code

- **Single-point args:** lowercase `x`, `y`. Used in private hooks like `_call_single(x, y, prior)` and anywhere a function is called on a single parameter setting.
- **Batch args:** uppercase `X`, `Y`. Used in `Emulator.fit(X, Y)`, `Emulator.predict(X)`, the public `LogDensityForm.__call__(X, Y, prior)`, and anywhere a function is called on a collection.
- **JAX PRNG keys:** `key`, `key_init`, `key_loop`, `key_acq`, etc. Never `k_init` or bare `k`.
- **Dimensions:** prefer `problem.target_distribution.input_shape` / `problem.target_distribution.output_shape`. Use `d` / `p` only in math contexts where the scalar dim is unambiguous.
- **Tempering:** `tempering_state` for the opaque state PyTree from `TemperingSchedule`. The per-state effective `DensityDecomposition` is produced by `TemperingScheme.intermediate_decomposition(base_decomposition, state)`.

## Examples

2-D Gaussian benchmark (`input_shape=(2,)`, `output_shape=()`, `d=2`, `p=1`):

```python
from probpipe import sample as pp_sample

problem = gaussian_2d()
decomposition = DensityDecomposition.identity_from_target(problem.target_distribution)
algorithm = Algorithm(density_decomposition=decomposition, ...)

X = pp_sample(algorithm.initial_design_distribution, key=key, sample_shape=(16,))  # (16, 2)
Y = decomposition.target_map(X)  # (16,) — already batched
```

Forward-model benchmark with 5 observables (`input_shape=(3,)`, `output_shape=(5,)`):

```python
def f(x: Array) -> Array:    # x.shape == (3,), returns (5,)
    return observation_operator(x)

X.shape == (n, 3)
Y.shape == (n, 5)
```
