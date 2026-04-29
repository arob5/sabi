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

## Naming in code

- **Single-point args:** lowercase `x`, `y`. Used in `LogDensityForm.__call__(x, y, problem)` and anywhere a function is called on a single parameter setting.
- **Batch args:** uppercase `X`, `Y`. Used in `Surrogate.fit(X, Y)`, `Surrogate.predict(X)`, and anywhere a function is called on a collection.
- **JAX PRNG keys:** `key`, `key_init`, `key_loop`, `key_acq`, etc. Never `k_init` or bare `k`.
- **Dimensions:** prefer `problem.input_shape` / `problem.output_shape`. Use `d` / `p` only in math contexts where the scalar dim is unambiguous.
- **Tempering:** `tempering_state` for the opaque state PyTree; `current_form` for the post-tempering `LogDensityForm`.

## Examples

2-D Gaussian benchmark (`input_shape=(2,)`, `output_shape=()`, `d=2`, `p=1`):

```python
def f(x: Array) -> Array:    # x.shape == (2,), returns scalar
    return -0.5 * x @ Sigma_inv @ x

X = PriorSampler().sample(problem, key, n=16)  # X.shape == (16, 2)
Y = jax.vmap(f)(X)                         # Y.shape == (16,)
```

Forward-model benchmark with 5 observables (`input_shape=(3,)`, `output_shape=(5,)`):

```python
def f(x: Array) -> Array:    # x.shape == (3,), returns (5,)
    return observation_operator(x)

X.shape == (n, 3)
Y.shape == (n, 5)
```
