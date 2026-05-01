# Emulators

sabi ships two GP-backed `Emulator` implementations. Pick the one whose
backend tradeoffs match the experiment.

## Quick table

| Emulator | Backend | Kernel | Hyperparam strategy | Best for |
|---|---|---|---|---|
| `GPEmulator` | tinygp | Matern-5/2 isotropic | data-adaptive lengthscale (median NN distance × factor), fixed noise | low-d, dense data, fast iteration |
| `DSPGPEmulator` | gpjax | RBF or Matern-5/2 ARD | MAP via `gpx.fit_scipy` with dimension-scaled-prior recipe | higher-d (≥ ~5), small n, where overfitting bites |

Both inherit from `Emulator + GaussianRandomFunction` and produce the
same shape of marginal predictions (`predict_mean`, `predict_variance`),
so they're drop-in for each other in the v1.2 marginal-mode pipeline.

## Input / output assumptions

Both emulators standardize inputs and outputs **internally**, so callers
pass raw data:

- **`GPEmulator`** (tinygp): zero-mean unit-variance both inputs and
  outputs.
- **`DSPGPEmulator`** (gpjax): inputs min-max scaled to `[0, 1]^d`,
  outputs zero-mean unit-variance. These match the calibration of the
  Hvarfner et al. (2024) priors.

Predictions are returned in the original (pre-scaling) output space.

## tinygp (`GPEmulator`)

Lightweight, no extra deps. The hyperparameter strategy is intentionally
non-adaptive: a single isotropic Matern-5/2 lengthscale is set from the
median nearest-neighbor distance in the input data, scaled by a factor
and floored. Noise is fixed at a small constant. **No optimization
runs at fit time** — fit is essentially `O(n^3)` for the Cholesky and
nothing more. Good when n is small, d is small, and the user wants
deterministic, fast fits; not great when sample efficiency at higher d
matters.

## gpjax (`DSPGPEmulator`)

Implements the dimension-scaled-prior recipe from
[Hvarfner et al. 2024 — *Vanilla Bayesian Optimization Performs Great in High Dimensions*](https://arxiv.org/abs/2402.02229):

- Lengthscale: ARD, `LogNormal(loc = √2 + 0.5·log(d), scale = √3)` per
  dim. The `+0.5·log(d)` term shifts the prior mass toward larger
  lengthscales as d grows, which counteracts the high-d failure mode
  where unregularized GP-BO collapses lengthscales onto noise.
- Outputscale: pinned at 1.0 (no `ScaleKernel`). Held non-trainable via
  `paramax.NonTrainable`. Calibrated against the unit-variance output
  scaling.
- Noise: `LogNormal(loc = -4, scale = 1)` on the **standard deviation**
  parameter. gpjax exposes `obs_stddev` rather than variance — this
  diverges from the GPyTorch reference in Hvarfner et al., which puts
  the same prior on the variance. Tracked.
- Floors: lengthscale ≥ 2.5e-2, noise stddev ≥ 1e-4. Enforced by the
  `BoundedPositive` paramax wrapper (`lower + softplus(unconstrained)`),
  which guards `fit_scipy` from collapsing onto degenerate solutions.

Fit runs MAP via `gpx.fit_scipy` minimizing
`-(conjugate_mll + Σ log_prior_lengthscale + log_prior_noise)`.

### When to prefer DSP

- d ≥ 5 with active-subspace structure (most relevant inputs are a
  small fraction of the total).
- Small n (say n < 10·d) where ARD lengthscales can otherwise overfit.
- You're willing to pay the extra fit-time cost (L-BFGS-B sweep over
  d+1 hyperparameters; typically dozens of iterations).

### Multi-restart MAP

`DSPGPEmulator(..., n_starts=k, restart_seed=s)` runs `k` MAP
optimizations and keeps the one with the highest objective. The
first start uses the deterministic prior-mode init; the remaining
`k−1` starts sample lengthscale and noise from the priors. With
`n_starts=1` (default) the behavior is bit-equivalent to the
pre-multistart single fit.

Useful when noisier data risks the optimizer settling on a
boundary solution at the noise floor — common in small-n / high-d
benchmarks where the data-likelihood gradient is shallow. Cost
scales linearly: 4 restarts ≈ 4× one fit.

### When NOT to prefer DSP

- d ≤ 2 with dense data — the tinygp default is competitive and 5–10×
  faster.
- You're benchmarking against a vanilla Matern baseline and don't want
  the prior to do work.
- The `gpjax` extra isn't installed and you'd rather not add it.

## Optional dep: gpjax

`DSPGPEmulator` lives under `sabi.emulators.gpjax` and requires the
optional `gpjax` extra. Install with:

```bash
pip install 'sabi[gpjax]'
# or
uv sync --extra gpjax
```

Touching `sabi.emulators.gpjax` does NOT trigger gpjax loading — the
package uses a lazy `__getattr__` import guard. Importing
`DSPGPEmulator` itself raises a clean `ImportError` with the install
command if the extra is missing.

## Config

```yaml
# configs/emulator/gp.yaml — tinygp default
name: gp
ls_factor: 1.5
ls_floor: 0.05
noise: 1.0e-4
jitter: 1.0e-3
```

```yaml
# configs/emulator/dsp_gp.yaml — gpjax DSP-prior
name: dsp_gp
kernel: rbf       # or matern52
max_iters: 500    # L-BFGS-B max iterations
jitter: 1.0e-6
verbose: false
n_starts: 1       # multi-restart MAP; 1 = single deterministic fit
restart_seed: 0
```

Select via Hydra override: `python -m sabi.runner.main emulator=dsp_gp`.

## Latent vs. observation predictive

Sabi emulators report the **latent** posterior (no observation noise):

- `predict_mean(X)`: posterior mean of the latent function `f(X)`.
- `predict_variance(X)`: marginal posterior variance of `f(X)`, with
  no observation noise added.
- `predict_covariance(X, joint_inputs=True)`: full posterior
  covariance of `f(X)`, with no observation noise added. Diagonal
  equals `predict_variance(X)`.

This convention is documented at the `Emulator` base class. Both
`GPEmulator` and `DSPGPEmulator` follow it. Callers that want the
observation-predictive distribution build it explicitly:
`Var[y* | data] = predict_variance(X) + sigma_n²` for whatever obs-
noise model the application has.

## Joint-mode predictions

`DSPGPEmulator` supports joint-input covariance via
`predict_covariance(X, joint_inputs=True)`, which returns the full
`(n, n)` latent covariance. The joint-output case is trivially
supported for the scalar-output case (returns `(n, 1, 1)`). Use
`predict(X, joint_inputs=True)` to get a `MultivariateNormal`
directly via the `GaussianRandomFunction` assembly path.

`GPEmulator` (tinygp) is still marginal-mode only; joint covariance
will land there when a benchmark needs it.

## Performance notes

- `DSPGPEmulator` caches the Cholesky factor `L = chol(K + diag(noise) +
  jitter·I)` and pre-solved `alpha = L⁻¹(y - m(X))` at fit time, so
  each `predict_*` call is `O(n²·m + n·m)` rather than rebuilding the
  factor at `O(n³)`. The joint-input covariance path adds one
  `K(Xt, Xt)` evaluation and one triangular solve over the cached
  factor on top of that. Refitting (calling `fit` again, which returns
  a new emulator instance) invalidates the cache; predicting on the
  unfitted instance raises.
- gpjax stores two independent jitter values on a `ConjugatePosterior`:
  `posterior.prior.jitter` (used by `conjugate_mll` and by
  `posterior.predict`'s test-side covariance) and `posterior.jitter`
  (used by `posterior.predict`'s training-side Cholesky). The
  `prior * likelihood` constructor does not propagate the prior's
  jitter into the posterior. `DSPGPEmulator` constructs the posterior
  directly so both fields take the same user-supplied value — without
  this, the optimized model and the predictive distribution use
  slightly different numerical models, and a strict equivalence
  comparison between the cached and naive predict paths catches the
  divergence.

## Fixed-hyperparameter conditioning

`DSPGPEmulator.condition_on(X_new, Y_new)` returns a new emulator that
appends `(X_new, Y_new)` to the training set **without re-running
hyperparameter optimization**. The kernel lengthscales, observation
noise, and input/output scalers all carry over from the existing fit.
Internally it does a block Cholesky update on the cached factor:
costs `O(n²·m + m³)` for `m` new points, vs `O((n+m)³)` for a
refit-from-scratch. See `_PredictCache.append_rows` for the math.

Use `condition_on` when:
- Hyperparameters were already optimized on a representative dataset
  and adding a few more points won't materially change them.
- You're inside a tight inner loop (e.g., greedy q-batch acquisition)
  where refitting per candidate would dominate runtime.

Use `fit` when:
- The new points push the data into a new regime (e.g., a tempering
  step changed the output scale).
- You haven't fitted yet, or want to refresh the hyperparameter
  estimate.

`DSPGPEmulator` registers three cheap-update handlers with
`sabi.emulators.dispatch.emulator_update_registry`:

- `AppendRows(X_new, Y_new)` — rank-one Cholesky update via
  `condition_on`. O(n²m + m³).
- `RescaleOutputs(factor=β)` — rescales the y-scaler to
  `(loc·β, scale·β)`. The cache is unchanged because z-scoring is
  scale-invariant: standardized y is identical, so `alpha` and
  `L_sigma` carry over verbatim. Predictions in the original output
  space rescale by β (mean) and β² (variance) automatically. O(1).
- `RescaleThenAppend(factor=β, X_new, Y_new)` — composite: rescale
  y-scaler (free), then standardize `Y_new` (already in the new
  state's units) with the new scaler and rank-one append. Total cost
  matches a single append.

All three require the emulator to have been fitted; unfitted
emulators fall through to the refit fallback. Negative or zero
`factor` is reported as infeasible (z-scoring requires positive
scale) and also falls back to refit.

## Forward-look

- ProbPipe `condition_on` integration replaces the bespoke `fit` once
  the primitive lands.
- `_scale_cov_to_output_space` is currently scalar-output only;
  multi-output support requires per-mode Kronecker scaling (the
  helper raises `NotImplementedError` to make this explicit). When
  multi-output emulators ship, that helper is the single edit-site.
