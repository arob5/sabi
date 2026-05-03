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

The word *target* shows up in five distinct roles in sabi. They all
do useful semantic work, but the names won't expose the distinction
unless you read this table:

| term | type | role |
|---|---|---|
| `TargetDistribution` | `Distribution` (subclass of ProbPipe `NumericRecordDistribution`) | The math object the user wants to approximate: an unnormalized log-density `phi(x, f(x); prior)`. Lives at `sabi.target_distribution`. |
| `Problem.target_distribution` | `TargetDistribution` field | The benchmark's `TargetDistribution`. The mathematical content of a `Problem`. |
| `target_map` (was `target_function`) | `Callable[[X], Y]` | The function `f : x ↦ y` the emulator approximates (log-likelihood, log-posterior, forward model — depends on the `LogDensityForm`). Stored as `target_single` (single-point); the batched view `target_map` is derived via `jax.vmap`. |
| `target_single` | `Callable[[x], y]` | Single-point view of `target_map`: maps shape `input_shape` to shape `output_shape`. The primitive contract for hand-written targets. |
| `IntermediateTarget` | `TargetDistribution` subclass | A tempered version of the base `TargetDistribution` at one schedule state. Carries `state`, `output_transform`, and a back-reference to the base `target_map`. Produced by `TemperingScheme.intermediate_target(base, state)`. |
| `AcquisitionTarget` | `Enum` | Which schedule state the acquisition's `SurrogateDistribution` is built at: `CURRENT` (round's state), `NEXT` (one-step look-ahead), `TERMINAL` (final state). Independent of the round's "current" state. |
| `target_tempering_state` | opaque PyTree | The state value resolved by `AcquisitionTarget` — what the acquisition's SP actually sees. Recorded on `AcquisitionState` for ablation reproducibility. |
| `expected_target` | function `(SurrogateDistribution) -> Distribution` | Deterministic posterior estimator: plug the surrogate's predictive mean of `target_map` into the log-density form. The "target" here is the target map under the surrogate's predictive. |

The two math objects `target_distribution` (a Distribution) and
`target_map` (a Callable) are the load-bearing names — keep them
distinct in your head and the rest of the table follows.

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

## Role of `prior`

The `prior` field on a `TargetDistribution` is **required**. It plays
two roles independent of whether the prior also forms part of the
target distribution:

1. **Defines the support of the parameter space.** `support` is not
   a separate field — it's a property delegating to `prior.support`.
   The prior may have unbounded support (a `Normal` over R^d) when
   bounded support isn't desired.
2. **Acts as the design distribution** for initial design, candidate
   sets, and prior-sampling acquisitions.

Whether the prior also enters the unnormalized target is the problem
builder's choice via `log_density_form`: `LogLikPlusPrior` builds it
in (`target = log_lik + log_prior`); `Identity` doesn't. In Bayesian
settings the algorithmic `prior` may be a *truncated* version of the
modeling prior — e.g., a Gaussian Bayesian prior paired with a
uniform-box algorithmic prior used purely to bound sampling.

The `prior` is a multivariate-event Distribution: `prior.event_shape
== input_shape`, `prior.batch_shape == ()`. For per-dim distributions
(e.g., array-valued `Uniform`), wrap with
`sabi._probpipe_compat.independent_uniform` to re-interpret batch
dims as event dims (a temporary shim until ProbPipe ships an
`Independent`-style wrapper).

## Tempering / bridging vocabulary

A `TemperingScheme` is a family of intermediate target distributions
indexed by an opaque state PyTree. The vocabulary:

| term | meaning |
|---|---|
| `TemperingScheme` | The abstraction: maps `(base, state)` to an `IntermediateTarget`. Subclasses: `NoTempering`, `LikelihoodTemperingViaForm`, `LikelihoodTemperingViaTarget`. |
| `TemperingSchedule` | The state generator: `at(round_idx) -> (state, is_terminal)`. Subclasses: `UntemperedSchedule`, `FixedSchedule`. |
| `IntermediateTarget` | Per-state `TargetDistribution`. Carries `state`, `output_transform`, `base_target_map`. |
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

## Public batched / private single-point convention

Across sabi, public methods that accept design data follow the
**batched** convention: arguments have a leading batch axis of size
`n`, return values prepend the same axis. When the public batched
method is a `jax.vmap` of an underlying single-point implementation,
the single-point implementation is **private** — its name is
underscore-prefixed (e.g., `_call_single`), and external callers go
through the batched API (or, for Distributions, through ProbPipe ops
like `unnormalized_log_prob`).

| Class | public batched | private single-point hook |
|--|--|--|
| `Emulator` | `__call__(X) -> Distribution` | inherited `predict_*` from `GaussianRandomFunction`, etc. |
| `PointwiseScoredAcquisition` | `score(X, state) -> (n,)` | `_score_single(x, state) -> scalar` |
| `LogDensityForm` | `__call__(X, Y, *, prior) -> (n,)` | `_call_single(x, y, *, prior) -> scalar` |
| `TargetDistribution` | `target_map(X) -> (n,) + output_shape` | `target_single(x) -> output_shape` (passed at construction) |

Subclasses override the single-point hook; the batched method is
provided by the base class via `jax.vmap` (or directly when a
batched implementation is more efficient — see e.g. `Identity`'s
batched `__call__` overriding the vmap path).

## `LogDensityForm` shape contract

`LogDensityForm.__call__(X, Y, *, prior)` is batched:

- `X.shape == (n,) + input_shape`
- `Y.shape == (n,) + output_shape`
- Returns `(n,)` — one scalar log-density per row of `X`.

Subclasses implement `_call_single(x, y, *, prior) -> scalar` (single
point: `x.shape == input_shape`, `y.shape == output_shape`). Default
`__call__` does `jax.vmap(self._call_single, in_axes=(0, 0, None))(X, Y)`.

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
- **Dimensions:** prefer `problem.input_shape` / `problem.output_shape`. Use `d` / `p` only in math contexts where the scalar dim is unambiguous.
- **Tempering:** `tempering_state` for the opaque state PyTree from `TemperingSchedule`. The per-state `LogDensityForm` is exposed via `IntermediateTarget.log_density_form`.

## Examples

2-D Gaussian benchmark (`input_shape=(2,)`, `output_shape=()`, `d=2`, `p=1`):

```python
def f(x: Array) -> Array:    # x.shape == (2,), returns scalar
    return -0.5 * x @ Sigma_inv @ x

X = PriorSampler().sample(problem, key, n=16)  # X.shape == (16, 2)
Y = problem.target_map(X)                      # Y.shape == (16,) — already batched
```

Forward-model benchmark with 5 observables (`input_shape=(3,)`, `output_shape=(5,)`):

```python
def f(x: Array) -> Array:    # x.shape == (3,), returns (5,)
    return observation_operator(x)

X.shape == (n, 3)
Y.shape == (n, 5)
```
