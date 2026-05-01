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
```

Select via Hydra override: `python -m sabi.runner.main emulator=dsp_gp`.

## Forward-look

- v1.5: Cholesky caching for `predict_*`. Currently both emulators
  rebuild the gram per call; the DSP variant especially benefits because
  fit_scipy already produces a converged decomposition that's discarded.
- v1.5: joint covariance support so emulator-level metrics (e.g.,
  posterior MMD with GP-uncertainty propagation) can run.
- ProbPipe `condition_on` integration replaces the bespoke `fit` once the
  primitive lands.
