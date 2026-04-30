# sabi — Notation

This document defines the shape and symbol conventions used throughout sabi.
**Code and docs should follow these exactly** — if you're tempted to use `theta`
for an input or `D` for dimension, come back here first.

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
| `TargetDistribution` | `target_function(X) -> (n,) + output_shape` | `_target_single(x) -> output_shape` (constructed via `from_target_single`) |

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

The `prior` is a multivariate-event Distribution: `prior.event_shape
== input_shape`, `prior.batch_shape == ()`. Then `log_prob(prior, x)`
for `x.shape == input_shape` returns a scalar — exactly what the form
needs. For per-dim distributions (e.g., array-valued `Uniform`), wrap
with `sabi._probpipe_compat.independent_uniform` to re-interpret batch
dims as event dims (a temporary shim until ProbPipe ships an
`Independent`-style wrapper).

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
Y = problem.target_function(X)                 # Y.shape == (16,) — already batched
```

Forward-model benchmark with 5 observables (`input_shape=(3,)`, `output_shape=(5,)`):

```python
def f(x: Array) -> Array:    # x.shape == (3,), returns (5,)
    return observation_operator(x)

X.shape == (n, 3)
Y.shape == (n, 5)
```
