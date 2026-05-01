"""`DSPGPEmulator` — gpjax-backed dimension-scaled-prior GP emulator.

Implements the Hvarfner et al. (2024) recipe:

  *Vanilla Bayesian Optimization Performs Great in High Dimensions*,
  ICML 2024. https://arxiv.org/abs/2402.02229

Key idea: keep vanilla GP-BO, but tighten the kernel/likelihood priors
and **scale the lengthscale prior with the input dimension**:

- Lengthscale: ``LogNormal(loc = √2 + 0.5·log(d), scale = √3)`` (ARD)
- Outputscale: fixed at 1.0 (no `ScaleKernel` wrapper, kernel `variance`
  parameter is held constant).
- Noise: ``LogNormal(loc = -4.0, scale = 1.0)`` on the **standard
  deviation** — gpjax exposes ``obs_stddev`` rather than variance.
  Tracked separately from the GPyTorch reference, which puts the same
  prior on the variance parameter.
- Lengthscale floor: 2.5e-2; noise floor: 1e-4. Both initialized at
  the prior modes.

Inference: MAP — minimize ``-(conjugate_mll + Σ log_prior)`` via
``gpx.fit_scipy``.

**Input/output scaling.** The DSP recipe is calibrated on inputs
normalized to ``[0, 1]^d`` and outputs standardized to zero-mean
unit-variance. ``DSPGPEmulator.fit(X, Y)`` applies these transforms
internally (min-max for X, z-score for Y) and stores the inverse
transforms for prediction time. Callers do NOT need to pre-scale.

v1.2 caveats:
- Marginal mode only (``supports_joint_inputs=False,
  supports_joint_outputs=False``). Joint covariance lands when emulator
  metrics need it.
- Scalar-output only (``output_shape=()``).
- ``predict_mean`` / ``predict_variance`` reuse a Cholesky cache built
  at fit time (see ``_PredictCache``), so per-call cost is O(n²·m +
  n·m) rather than O(n³). Refit invalidates the cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

# Eager guard — fails fast with a helpful message if the optional
# `gpjax` extra isn't installed. Per-module rather than per-symbol so a
# single import line tells the user how to fix things.
try:
    import gpjax as _gpx  # noqa: F401  (presence check)
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    raise ImportError(
        "DSPGPEmulator requires the optional `gpjax` extra. Install with "
        "`pip install 'sabi[gpjax]'` or `uv sync --extra gpjax`."
    ) from e

import equinox as eqx
import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import lineax as lx
from jax import Array
from probpipe.distributions.gaussian_random_function import GaussianRandomFunction

from sabi.emulators.base import Emulator
from sabi.emulators.gpjax._dsp import (
    DSP_LENGTHSCALE_FLOOR,
    DSP_NOISE_FLOOR,
    BoundedPositive,
    dsp_map_objective,
)


__all__ = ["DSPGPEmulator"]


@dataclass(frozen=True)
class _MinMaxScaler:
    """Affine map: ``x' = (x - lo) / (hi - lo)``, clamped to avoid 0-width.

    For dimensions where ``hi == lo``, the scale is set to 1 so the
    transform reduces to a translation.
    """

    lo: Array
    hi: Array

    @classmethod
    def fit(cls, X: Array) -> "_MinMaxScaler":
        lo = jnp.min(X, axis=0)
        hi = jnp.max(X, axis=0)
        # Where range is degenerate, fall back to scale=1 so we don't divide by 0.
        hi = jnp.where((hi - lo) < 1e-12, lo + 1.0, hi)
        return cls(lo=lo, hi=hi)

    def transform(self, X: Array) -> Array:
        return (X - self.lo) / (self.hi - self.lo)


@dataclass(frozen=True)
class _PredictCache:
    """Pre-solved prediction state for an optimized ConjugatePosterior.

    Caches the bits of the posterior predictive that depend only on the
    training data and fitted hyperparameters — i.e., are constant across
    test inputs:

    - ``L_sigma``: lower Cholesky factor of ``K(X, X) + diag(noise) +
      jitter·I``, shape ``(n, n)``.
    - ``alpha``: ``L_sigma⁻¹ (y - m(X))``, shape ``(n,)``. Combines with
      ``L_sigma⁻¹ K(X, X*)`` to give the predictive mean shift.
    - ``unwrapped_posterior``: the gpjax posterior with paramax
      unwrappables resolved to plain arrays. Stored so we can call
      ``.prior.kernel.cross_covariance``, ``.prior.mean_function``,
      ``.prior.kernel.diagonal``, and ``.likelihood`` without redoing
      the unwrap on every call.
    - ``Xs_train``: training inputs (in scaled coords), shape ``(n, d)``.
    - ``noise_var``: scalar observation noise variance (``obs_stddev²``).

    With this cache, ``predict_*`` cost drops from O(n³) (rebuilding the
    Cholesky every call) to O(n²·m + n·m) per batch of m test points.
    """

    unwrapped_posterior: object
    L_sigma: Array
    alpha: Array
    Xs_train: Array
    noise_var: Array

    @classmethod
    def build(cls, opt_posterior, Xs_train: Array, Ys_train: Array) -> "_PredictCache":
        """Compute the cache from a fitted gpjax posterior + training data.

        ``Ys_train`` must be the (already-standardized) target column of
        shape ``(n,)``; we reshape it to match gpjax's internal
        (n, 1) convention.
        """
        import paramax

        unwrapped = paramax.unwrap(opt_posterior)
        n = Xs_train.shape[0]

        # Mean function over training inputs (Constant returns shape (n, 1)).
        mx = unwrapped.prior.mean_function(Xs_train)
        y_flat, mx_flat = unwrapped.likelihood.prepare_targets(
            Ys_train.reshape(-1, 1), mx
        )
        y_flat = jnp.atleast_1d(y_flat.squeeze())
        mx_flat = jnp.atleast_1d(mx_flat.squeeze())

        # K(X, X) + diag(noise) + jitter·I — same construction as the
        # gpjax `predict` function we're shadowing.
        Kxx = unwrapped.prior.kernel.gram(Xs_train).as_matrix()
        from gpjax.linalg.utils import add_jitter

        Kxx = add_jitter(Kxx, unwrapped.prior.jitter)
        noise_diag = unwrapped.likelihood.noise_vector(n)
        Sigma = Kxx + jnp.diag(noise_diag)
        L_sigma = jnp.linalg.cholesky(Sigma)

        alpha = jsp.linalg.solve_triangular(L_sigma, y_flat - mx_flat, lower=True)

        # Gaussian likelihood: noise_vector returns obs_stddev² · 1ₙ; a
        # single scalar suffices for adding observation noise to test
        # marginal variances.
        noise_var = jnp.asarray(unwrapped.likelihood.obs_stddev) ** 2

        return cls(
            unwrapped_posterior=unwrapped,
            L_sigma=L_sigma,
            alpha=alpha,
            Xs_train=Xs_train,
            noise_var=noise_var,
        )

    def predict_latent(self, Xt: Array) -> tuple[Array, Array]:
        """Latent (noiseless) marginal mean and variance at test inputs.

        Args:
            Xt: ``(m, d)`` test inputs in scaled coordinates.

        Returns:
            ``(mean, var)`` each of shape ``(m,)``. Variance is the
            diagonal of the latent covariance with prior jitter folded
            in (matching gpjax's diagonal-covariance branch).
        """
        kernel = self.unwrapped_posterior.prior.kernel
        mean_fn = self.unwrapped_posterior.prior.mean_function
        prior_jitter = self.unwrapped_posterior.prior.jitter

        Kxt = kernel.cross_covariance(self.Xs_train, Xt)  # (n, m)
        L_inv_Kxt = jsp.linalg.solve_triangular(self.L_sigma, Kxt, lower=True)

        mean_t_raw = mean_fn(Xt)  # (m, 1) for Constant
        mean_t = jnp.atleast_1d(mean_t_raw.squeeze())
        mean = mean_t + L_inv_Kxt.T @ self.alpha  # (m,)

        Ktt_diag = lx.diagonal(kernel.diagonal(Xt))  # (m,)
        var = Ktt_diag - jnp.einsum("ij,ij->j", L_inv_Kxt, L_inv_Kxt)
        var = var + prior_jitter
        return mean, var


@dataclass(frozen=True)
class _ZScoreScaler:
    """Affine map for outputs: ``y' = (y - mean) / std``."""

    loc: Array
    scale: Array

    @classmethod
    def fit(cls, y: Array) -> "_ZScoreScaler":
        loc = jnp.mean(y, axis=0)
        scale = jnp.std(y, axis=0)
        scale = jnp.where(scale < 1e-12, 1.0, scale)
        return cls(loc=loc, scale=scale)

    def transform(self, y: Array) -> Array:
        return (y - self.loc) / self.scale

    def inverse_mean(self, m: Array) -> Array:
        return m * self.scale + self.loc

    def inverse_var(self, v: Array) -> Array:
        return v * (self.scale ** 2)


def _build_dsp_posterior(
    *,
    d: int,
    n: int,
    kernel_name: str,
    init_lengthscale: Array,
    init_obs_stddev: Array,
    jitter: float,
):
    """Construct a fresh ConjugatePosterior with DSP-prior parameter wrappers.

    The kernel ``variance`` (outputscale) is held fixed at 1.0 by
    converting it to a ``paramax.NonTrainable`` after construction —
    this matches the DSP recipe of pinning outputscale=1.
    """
    import paramax

    if kernel_name == "rbf":
        kernel_cls = gpx.kernels.RBF
    elif kernel_name == "matern52":
        kernel_cls = gpx.kernels.Matern52
    else:
        raise ValueError(
            f"DSPGPEmulator: kernel must be 'rbf' or 'matern52', got {kernel_name!r}."
        )

    # Pin outputscale at 1.0: pass a `NonTrainable` wrapping a plain
    # array directly. The kernel's __init__ accepts any AbstractUnwrappable
    # for `variance` and stores it as-is. During fit, `paramax.unwrap`
    # applies stop_gradient to the inner array so scipy can't move it.
    kernel = kernel_cls(
        lengthscale=BoundedPositive(init_lengthscale, lower=DSP_LENGTHSCALE_FLOOR),
        variance=paramax.NonTrainable(jnp.asarray(1.0, dtype=jnp.float64)),
        n_dims=d,
    )

    meanf = gpx.mean_functions.Constant()
    prior = gpx.gps.Prior(
        mean_function=meanf, kernel=kernel, jitter=jitter
    )
    lik = gpx.likelihoods.Gaussian(num_datapoints=n, obs_stddev=init_obs_stddev)
    # Likelihood __init__ re-wraps obs_stddev as NonNegativeReal; swap it
    # for our BoundedPositive after construction.
    lik = eqx.tree_at(
        lambda l: l.obs_stddev,
        lik,
        BoundedPositive(init_obs_stddev, lower=DSP_NOISE_FLOOR),
    )

    return prior * lik


class DSPGPEmulator(Emulator, GaussianRandomFunction):
    """Dimension-scaled-prior GP emulator (Hvarfner et al. 2024).

    Backed by ``gpjax`` for full hyperparameter optimization (MAP via
    ``fit_scipy``) with ARD lengthscales. ``predict_mean`` and
    ``predict_variance`` implement the abstract `GaussianRandomFunction`
    interface; ``predict`` / ``__call__`` come for free from the
    parent.

    Constructor args:
        input_shape: ``(d,)`` — input dimensionality. Joint-mode +
            multi-output not supported in v1.2.
        output_shape: must be ``()`` (scalar output) in v1.2.
        name: optional emulator name (used for repr).
        kernel: ``"rbf"`` or ``"matern52"``. Default ``"rbf"`` matches
            Hvarfner et al. 2024.
        max_iters: max L-BFGS-B iterations for fit_scipy.
        jitter: prior jitter added to the gram diagonal for Cholesky
            stability (separate from the noise floor, which lives on
            ``obs_stddev``).
        verbose: passed through to ``gpx.fit_scipy``.
    """

    # Marginal-only in v1.2.
    supports_joint_inputs: bool = False
    supports_joint_outputs: bool = False

    def __init__(
        self,
        *,
        input_shape: tuple[int, ...] = (2,),
        output_shape: tuple[int, ...] = (),
        name: str | None = None,
        kernel: str = "rbf",
        max_iters: int = 500,
        jitter: float = 1e-6,
        verbose: bool = False,
        # Internal post-fit state (callers don't pass these).
        _x_scaler: _MinMaxScaler | None = None,
        _y_scaler: _ZScoreScaler | None = None,
        _opt_posterior=None,
        _train_dataset=None,
        _predict_cache: _PredictCache | None = None,
    ):
        if output_shape != ():
            raise ValueError(
                f"DSPGPEmulator (v1.2) is scalar-output only; "
                f"got output_shape={output_shape}."
            )
        if len(input_shape) != 1:
            raise ValueError(
                f"DSPGPEmulator expects input_shape=(d,); got {input_shape}."
            )
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            name=name or "DSPGPEmulator",
        )
        self.kernel_name = kernel
        self.max_iters = max_iters
        self.jitter = jitter
        self.verbose = verbose
        self._x_scaler = _x_scaler
        self._y_scaler = _y_scaler
        self._opt_posterior = _opt_posterior
        self._train_dataset = _train_dataset
        self._predict_cache = _predict_cache

    # --- fit ---------------------------------------------------------------

    def fit(self, X: Array, Y: Array) -> Self:
        """Fit the DSP-prior GP to ``(X, Y)`` via MAP optimization.

        Args:
            X: shape ``(n, d)``. Internally rescaled to ``[0, 1]^d`` via
                min-max scaling on the training set.
            Y: shape ``(n,)``. Internally standardized to zero-mean
                unit-variance.

        Returns:
            A new ``DSPGPEmulator`` carrying the optimized posterior
            and the input/output scaling state needed for prediction.
        """
        # gpjax requires float64 throughout — its parameter wrappers
        # default to float64 internally and mismatched dtypes break
        # fit_scipy. Caller is expected to have x64 enabled.
        if not jax.config.jax_enable_x64:
            raise RuntimeError(
                "DSPGPEmulator requires JAX x64 mode. Enable with "
                "`jax.config.update('jax_enable_x64', True)` before fitting."
            )

        d = self.input_shape[0]
        if X.ndim != 2 or X.shape[1] != d:
            raise ValueError(
                f"DSPGPEmulator.fit: expected X.shape=(n, {d}), "
                f"got {tuple(X.shape)}."
            )
        if Y.shape != (X.shape[0],):
            raise ValueError(
                f"DSPGPEmulator.fit: expected Y.shape=({X.shape[0]},), "
                f"got {tuple(Y.shape)}."
            )

        x_scaler = _MinMaxScaler.fit(X)
        y_scaler = _ZScoreScaler.fit(Y)
        Xs = x_scaler.transform(X).astype(jnp.float64)
        Ys = y_scaler.transform(Y).astype(jnp.float64)

        data = gpx.Dataset(X=Xs, y=Ys.reshape(-1, 1))

        # Initialize lengthscales and noise stddev at the prior modes.
        # mode of LogNormal(loc, scale) = exp(loc - scale^2)
        # Lengthscale: loc = √2 + 0.5·log(d), scale = √3
        ls_loc = jnp.sqrt(jnp.asarray(2.0)) + 0.5 * jnp.log(jnp.asarray(float(d)))
        ls_mode = jnp.exp(ls_loc - 3.0)  # exp(loc - scale^2) with scale=√3
        ls_init = jnp.maximum(
            jnp.full((d,), ls_mode, dtype=jnp.float64),
            jnp.asarray(DSP_LENGTHSCALE_FLOOR + 1e-6, dtype=jnp.float64),
        )
        # Noise: loc = -4, scale = 1 → mode = exp(-4 - 1) = exp(-5) ≈ 6.7e-3
        noise_init = jnp.maximum(
            jnp.asarray(jnp.exp(-5.0), dtype=jnp.float64),
            jnp.asarray(DSP_NOISE_FLOOR + 1e-6, dtype=jnp.float64),
        )

        posterior = _build_dsp_posterior(
            d=d,
            n=data.n,
            kernel_name=self.kernel_name,
            init_lengthscale=ls_init,
            init_obs_stddev=noise_init,
            jitter=self.jitter,
        )

        opt_posterior, _history = gpx.fit_scipy(
            model=posterior,
            objective=lambda p, dat: -dsp_map_objective(p, dat),
            train_data=data,
            max_iters=self.max_iters,
            verbose=self.verbose,
        )

        # Pre-solve the Cholesky + alpha vector once. predict_* will
        # reuse these instead of redoing them on every call.
        predict_cache = _PredictCache.build(opt_posterior, Xs, Ys)

        return type(self)(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            name=self.name,
            kernel=self.kernel_name,
            max_iters=self.max_iters,
            jitter=self.jitter,
            verbose=self.verbose,
            _x_scaler=x_scaler,
            _y_scaler=y_scaler,
            _opt_posterior=opt_posterior,
            _train_dataset=data,
            _predict_cache=predict_cache,
        )

    # --- GaussianRandomFunction abstract methods --------------------------

    def predict_mean(self, X: Array) -> Array:
        """Return the predictive mean at each row of `X`.

        Uses the Cholesky cache built at fit time, so cost is O(n²·m +
        n·m) rather than re-Cholesky-ing per call.

        Args:
            X: shape ``(n, d)`` (raw, pre-scaled).

        Returns:
            Shape ``(n,)``. Mean is in the original (pre-standardization)
            output space.
        """
        self._require_fit()
        Xs = self._x_scaler.transform(X).astype(jnp.float64)  # type: ignore[union-attr]
        mean_latent, _ = self._predict_cache.predict_latent(Xs)  # type: ignore[union-attr]
        # Gaussian observation noise has zero mean, so latent and
        # observation predictive means coincide.
        return self._y_scaler.inverse_mean(mean_latent)  # type: ignore[union-attr]

    def predict_variance(self, X: Array) -> Array:
        """Return the marginal predictive variance at each row of `X`.

        Variance includes the observation noise (matches what
        ``posterior.likelihood(latent).variance`` returns in gpjax) so
        downstream code that expects "predictive variance with noise"
        sees the same number.

        Args:
            X: shape ``(n, d)`` (raw, pre-scaled).

        Returns:
            Shape ``(n,)``. Variance is in the original (pre-
            standardization) output space.
        """
        self._require_fit()
        Xs = self._x_scaler.transform(X).astype(jnp.float64)  # type: ignore[union-attr]
        _, latent_var = self._predict_cache.predict_latent(Xs)  # type: ignore[union-attr]
        obs_var = latent_var + self._predict_cache.noise_var  # type: ignore[union-attr]
        return self._y_scaler.inverse_var(jnp.maximum(obs_var, 0.0))  # type: ignore[union-attr]

    def _require_fit(self) -> None:
        if self._opt_posterior is None or self._predict_cache is None:
            raise RuntimeError(
                f"{type(self).__name__} called before fit; conditioning "
                "state is unset."
            )
