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

Caveats:
- Scalar-output only (``output_shape=()``).
- ``predict_mean``, ``predict_variance``, ``predict_covariance`` all
  reuse a single Cholesky cache built at fit time (see
  ``_PredictCache``). Per-call marginal cost is O(n²·m + n·m), and
  the joint-input case adds one ``K(Xt, Xt)`` and a triangular solve.
  Refit returns a new instance with a fresh cache.
- Joint inputs and (trivially) joint outputs are supported via
  ``predict_covariance``; both flags default to True at the class
  level.

How gpjax handles jitter, and what we do
----------------------------------------

gpjax 0.14 stores **two independent** jitter values on a
``ConjugatePosterior`` and uses them in different places. Concretely:

- ``posterior.prior.jitter`` (the prior's jitter field) is used by:
    * ``gpjax.objectives.conjugate_mll`` for the training-side Sigma
      Cholesky, and
    * ``ConjugatePosterior.predict`` for the **test-side** covariance
      (added to ``K(Xt, Xt) - L⁻¹Kxt^T L⁻¹Kxt`` to keep it PSD).
- ``posterior.jitter`` (the posterior's own jitter field, default
  ``1e-6``) is used by:
    * ``ConjugatePosterior.predict`` for the **training-side** Sigma
      Cholesky.

These two values are independent at construction. The ``prior * lik``
shortcut (i.e., ``construct_posterior(prior, likelihood)``) creates the
posterior with its own default ``jitter=1e-6``; it does **not** copy
``prior.jitter`` into the posterior. So passing ``Prior(jitter=X)``
alone leaves the posterior using a different jitter at predict time
than the MLL used at fit time — the optimized hyperparameters are not
the MAP of the model that ``posterior.predict`` evaluates, and a
strict equivalence comparison (cached vs naive predict path) catches
the mismatch immediately.

We follow the gpjax convention of having both fields, but pin them to
the same user-supplied value. ``_build_dsp_posterior`` constructs
``ConjugatePosterior`` directly with ``jitter=jitter``, so both
``posterior.jitter`` and ``posterior.prior.jitter`` agree. The
training-side jitter in ``_PredictCache`` is sourced from
``posterior.jitter`` (matching ``ConjugatePosterior.predict``); the
test-side jitter is sourced from ``posterior.prior.jitter`` (matching
the same gpjax method). With both fields aligned at construction, the
two paths are bit-equivalent — but the cache deliberately mirrors the
gpjax dispatch so that any future divergence between the two fields
would surface as a test failure rather than a silent numerical drift.

Convention: ``predict_variance`` / ``predict_covariance`` return the
**latent** posterior (no observation noise). See the ``Emulator`` base
class for the rationale.
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

        # K(X, X) + diag(noise) + train_jitter·I.
        #
        # gpjax keeps two separate jitter values: `posterior.jitter` is
        # used by `posterior.predict` on the training-side Cholesky;
        # `posterior.prior.jitter` is used by `conjugate_mll` on the
        # same Cholesky AND by `posterior.predict` on the test-side
        # covariance. `_build_dsp_posterior` aligns them so they match,
        # but we still source the training jitter from
        # `posterior.jitter` here to mirror gpjax's `predict` exactly —
        # so that an equivalence test against the naive path would
        # catch any future divergence the moment a future change
        # decoupled the two.
        Kxx = unwrapped.prior.kernel.gram(Xs_train).as_matrix()
        from gpjax.linalg.utils import add_jitter

        train_jitter = unwrapped.jitter
        Kxx = add_jitter(Kxx, train_jitter)
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
            in (matching gpjax's diagonal-covariance branch in
            ``posterior.predict``).
        """
        kernel = self.unwrapped_posterior.prior.kernel
        mean_fn = self.unwrapped_posterior.prior.mean_function
        # Test-side jitter matches gpjax `posterior.predict`'s
        # `_return_diagonal_covariance` branch, which uses
        # `posterior.prior.jitter` on the test-point covariance.
        test_jitter = self.unwrapped_posterior.prior.jitter

        Kxt = kernel.cross_covariance(self.Xs_train, Xt)  # (n, m)
        L_inv_Kxt = jsp.linalg.solve_triangular(self.L_sigma, Kxt, lower=True)

        mean_t_raw = mean_fn(Xt)  # (m, 1) for Constant
        mean_t = jnp.atleast_1d(mean_t_raw.squeeze())
        mean = mean_t + L_inv_Kxt.T @ self.alpha  # (m,)

        Ktt_diag = lx.diagonal(kernel.diagonal(Xt))  # (m,)
        var = Ktt_diag - jnp.einsum("ij,ij->j", L_inv_Kxt, L_inv_Kxt)
        var = var + test_jitter
        return mean, var

    def predict_latent_joint(self, Xt: Array) -> tuple[Array, Array]:
        """Latent (noiseless) joint mean and full covariance at test inputs.

        Used by `DSPGPEmulator.predict_covariance` for the
        ``joint_inputs=True`` mode. Mirrors gpjax's
        ``posterior.predict(...).covariance_matrix`` (the
        ``return_covariance_type='dense'`` branch): the full latent
        covariance with prior jitter on the diagonal.

        Args:
            Xt: ``(m, d)`` test inputs in scaled coordinates.

        Returns:
            ``(mean, cov)`` with shapes ``(m,)`` and ``(m, m)``. Cov is
            symmetric PSD (latent + prior_jitter·I); add ``noise_var·I``
            for the observation-noise-inclusive predictive covariance.
        """
        kernel = self.unwrapped_posterior.prior.kernel
        mean_fn = self.unwrapped_posterior.prior.mean_function
        test_jitter = self.unwrapped_posterior.prior.jitter

        Kxt = kernel.cross_covariance(self.Xs_train, Xt)  # (n, m)
        L_inv_Kxt = jsp.linalg.solve_triangular(self.L_sigma, Kxt, lower=True)

        mean_t_raw = mean_fn(Xt)  # (m, 1) for Constant
        mean_t = jnp.atleast_1d(mean_t_raw.squeeze())
        mean = mean_t + L_inv_Kxt.T @ self.alpha  # (m,)

        Ktt = kernel.gram(Xt).as_matrix()  # (m, m)
        cov = Ktt - L_inv_Kxt.T @ L_inv_Kxt
        cov = cov + test_jitter * jnp.eye(cov.shape[0], dtype=cov.dtype)
        return mean, cov

    def append_rows(
        self, Xs_new: Array, Ys_new: Array
    ) -> "_PredictCache":
        r"""Append new training rows via a block Cholesky / alpha update.

        Conditions the GP on ``(Xs_new, Ys_new)`` while holding the
        kernel hyperparameters and the observation-noise variance
        fixed. Bit-equivalent to building a fresh cache on the
        concatenated training set with the same hyperparameters, but
        runs in O(n²·m + m³) instead of O((n+m)³).

        Math
        ----
        Let

        .. math::

            \Sigma = K(X, X) + \sigma^2 I + \text{jitter} \cdot I = L L^\top

        be the cached training covariance with its lower Cholesky
        factor (n × n). Suppose we condition on a new batch of size m,
        so the augmented training covariance has the symmetric block
        form

        .. math::

            \Sigma' = \begin{bmatrix}
                \Sigma & K(X, X_\text{new}) \\
                K(X_\text{new}, X) & K(X_\text{new}, X_\text{new}) +
                    \sigma^2 I + \text{jitter} \cdot I
            \end{bmatrix}.

        Its lower Cholesky factor admits the block form

        .. math::

            L' = \begin{bmatrix} L & 0 \\ U^\top & L_{22} \end{bmatrix},

        where

        .. math::

            U &= L^{-1} K(X, X_\text{new})
                \quad (\text{forward triangular solve, n} \times \text{m}) \\
            D &= K(X_\text{new}, X_\text{new}) + \sigma^2 I +
                \text{jitter} \cdot I - U^\top U
                \quad (\text{m} \times \text{m, the Schur complement}) \\
            L_{22} &= \text{chol}(D)
                \quad (\text{m} \times \text{m, dense Cholesky}).

        Verification:

        .. math::

            L' (L')^\top = \begin{bmatrix}
                L L^\top & L U \\
                U^\top L^\top & U^\top U + L_{22} L_{22}^\top
            \end{bmatrix}
            = \begin{bmatrix} \Sigma & K(X, X_\text{new}) \\
                K(X_\text{new}, X) & K(X_\text{new}, X_\text{new})
                    + \sigma^2 I + \text{jitter} \cdot I \end{bmatrix}
            = \Sigma',

        using ``L L^\top = \Sigma``, ``L U = K(X, X_\text{new})``, and
        ``U^\top U + D = K(X_\text{new}, X_\text{new}) + \sigma^2 I +
        \text{jitter} \cdot I``.

        The training residual ``α = L^{-1}(y - m(X))`` extends as

        .. math::

            \alpha_\text{new} &= L_{22}^{-1}\bigl(
                y_\text{new} - m(X_\text{new}) - U^\top \alpha
            \bigr), \\
            \alpha' &= [\alpha; \, \alpha_\text{new}],

        which satisfies

        .. math::

            L' \alpha' = \begin{bmatrix} L \alpha \\
                U^\top \alpha + L_{22} \alpha_\text{new} \end{bmatrix}
            = \begin{bmatrix} y - m(X) \\
                y_\text{new} - m(X_\text{new}) \end{bmatrix},

        so ``α'`` is the canonical residual for the augmented training
        set under the same model.

        Cost: one (n × m) triangular solve, one (m × m) gram
        evaluation, one (m × m) Cholesky, and one (m,) triangular
        solve. Memory: a fresh (n+m) × (n+m) lower-triangular factor.
        Refit-from-scratch costs O((n+m)³).

        Args:
            Xs_new: ``(m, d)`` new training inputs in scaled
                coordinates (must already be transformed by the same
                x-scaler used at fit time).
            Ys_new: ``(m,)`` new training outputs in standardized
                coordinates (must already be transformed by the same
                y-scaler used at fit time).

        Returns:
            A new ``_PredictCache`` whose training data is the
            concatenation of the cached data and the new rows, with
            the Cholesky factor and ``alpha`` vector updated.
            Hyperparameters (kernel, mean function, noise) are frozen.
        """
        if Xs_new.ndim != 2 or Xs_new.shape[1] != self.Xs_train.shape[1]:
            raise ValueError(
                f"_PredictCache.append_rows: expected Xs_new.shape=(m, "
                f"{self.Xs_train.shape[1]}), got {tuple(Xs_new.shape)}."
            )
        if Ys_new.shape != (Xs_new.shape[0],):
            raise ValueError(
                f"_PredictCache.append_rows: expected Ys_new.shape="
                f"({Xs_new.shape[0]},), got {tuple(Ys_new.shape)}."
            )

        kernel = self.unwrapped_posterior.prior.kernel
        mean_fn = self.unwrapped_posterior.prior.mean_function
        train_jitter = self.unwrapped_posterior.jitter
        m = Xs_new.shape[0]
        n_old = self.Xs_train.shape[0]

        # K(X_old, X_new): (n_old, m). cross_covariance returns a plain Array.
        K_on = kernel.cross_covariance(self.Xs_train, Xs_new)
        # U = L^{-1} K(X_old, X_new): forward triangular solve, (n_old, m).
        U = jsp.linalg.solve_triangular(self.L_sigma, K_on, lower=True)

        # K(X_new, X_new) + (σ² + jitter)·I - U^T U  → Schur complement (m, m).
        K_nn = kernel.gram(Xs_new).as_matrix()
        D = (
            K_nn
            + (self.noise_var + train_jitter)
            * jnp.eye(m, dtype=K_nn.dtype)
            - U.T @ U
        )
        # Symmetrize for Cholesky stability.
        D = 0.5 * (D + D.T)
        L_22 = jnp.linalg.cholesky(D)

        # Assemble L' = [[L, 0], [U^T, L_22]] of shape (n_old+m, n_old+m).
        n_new = n_old + m
        L_new = jnp.zeros((n_new, n_new), dtype=self.L_sigma.dtype)
        L_new = L_new.at[:n_old, :n_old].set(self.L_sigma)
        L_new = L_new.at[n_old:, :n_old].set(U.T)
        L_new = L_new.at[n_old:, n_old:].set(L_22)

        # Augmented residual:
        #   α_new = L_22^{-1} (y_new - m(X_new) - U^T α)
        mx_new = mean_fn(Xs_new)  # (m, 1) for Constant
        mx_new_flat = jnp.atleast_1d(mx_new.squeeze())
        residual = jnp.atleast_1d(Ys_new) - mx_new_flat - U.T @ self.alpha
        alpha_block = jsp.linalg.solve_triangular(L_22, residual, lower=True)
        alpha_aug = jnp.concatenate([self.alpha, alpha_block])

        Xs_aug = jnp.concatenate([self.Xs_train, Xs_new], axis=0)

        # Defensive: dataclasses.replace would require the import; we
        # rebuild explicitly to keep this file self-contained.
        return _PredictCache(
            unwrapped_posterior=self.unwrapped_posterior,
            L_sigma=L_new,
            alpha=alpha_aug,
            Xs_train=Xs_aug,
            noise_var=self.noise_var,
        )


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

    # gpjax stores jitter on BOTH `posterior.prior.jitter` (used by
    # `conjugate_mll` for the training Cholesky and by `posterior.predict`
    # for the test-side covariance) and `posterior.jitter` (used by
    # `posterior.predict` for the training Cholesky). Defaults are
    # independent — `prior * lik` (i.e. `construct_posterior`) does not
    # propagate `prior.jitter` into the posterior. We construct
    # `ConjugatePosterior` directly so both jitters take the same
    # user-supplied value and the MLL we optimize matches the predictive
    # distribution numerically.
    return gpx.gps.ConjugatePosterior(prior=prior, likelihood=lik, jitter=jitter)


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

    # Joint inputs supported via `predict_covariance`. Joint outputs are
    # trivially supported for the scalar-output case (returns the
    # marginal variance reshaped to (n, 1, 1)).
    supports_joint_inputs: bool = True
    supports_joint_outputs: bool = True

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
        """Return the marginal posterior variance of the **latent** function.

        Per the sabi `Emulator` convention, this is the variance of the
        latent function value f(x*) under the posterior — observation
        noise is **not** added. Equivalent to
        ``posterior.predict(Xt).variance`` in gpjax (the latent path),
        not ``posterior.likelihood(posterior.predict(Xt)).variance``.

        Args:
            X: shape ``(n, d)`` (raw, pre-scaled).

        Returns:
            Shape ``(n,)``. Variance is in the original (pre-
            standardization) output space.
        """
        self._require_fit()
        Xs = self._x_scaler.transform(X).astype(jnp.float64)  # type: ignore[union-attr]
        _, latent_var = self._predict_cache.predict_latent(Xs)  # type: ignore[union-attr]
        return self._y_scaler.inverse_var(jnp.maximum(latent_var, 0.0))  # type: ignore[union-attr]

    def predict_covariance(
        self,
        X: Array,
        *,
        joint_inputs: bool = False,
        joint_outputs: bool = False,
    ) -> Array:
        """Posterior covariance of the **latent** function (no obs noise).

        Per the sabi `Emulator` convention, observation noise is not
        included on the diagonal — the diagonal of the returned matrix
        equals ``predict_variance(X)`` exactly. Reuses the Cholesky
        cache: the joint case adds one ``K(Xt, Xt)`` evaluation and one
        triangular solve over the cached ``L_sigma`` on top of what
        ``predict_mean`` already does.

        For the scalar-output case (``output_shape=()``) the returned
        shapes per the GaussianRandomFunction contract are:

        - ``joint_inputs=True`` (regardless of ``joint_outputs``):
          ``(n, n)`` — full cross-input latent covariance with prior
          jitter on the diagonal (matches ``posterior.predict``'s
          dense covariance).
        - ``joint_inputs=False, joint_outputs=True``: ``(n, 1, 1)`` —
          the marginal latent variance at each input, reshaped (joint
          over outputs is vacuous when the output is scalar).
        - ``joint_inputs=False, joint_outputs=False``: not implemented;
          callers should use ``predict_variance``.

        Args:
            X: shape ``(n, d)`` (raw, pre-scaled).
            joint_inputs: include cross-input covariance.
            joint_outputs: include cross-output covariance (trivial for
                scalar output).

        Returns:
            Latent covariance array in the original (pre-standardization)
            output space; ``y_scaler.scale²`` is folded in here.
        """
        self._require_fit()
        if not joint_inputs and not joint_outputs:
            raise NotImplementedError(
                "predict_covariance with joint_inputs=False, "
                "joint_outputs=False is not implemented; "
                "use predict_variance for the marginal case."
            )

        if not joint_inputs:
            # joint_outputs only: scalar output → reshape variance to (n, 1, 1).
            return self.predict_variance(X)[:, None, None]

        # joint_inputs=True (joint_outputs is trivial here): full (n, n).
        Xs = self._x_scaler.transform(X).astype(jnp.float64)  # type: ignore[union-attr]
        _, latent_cov = self._predict_cache.predict_latent_joint(Xs)  # type: ignore[union-attr]
        # Defensive symmetrization: floating-point asymmetry in
        # `K(Xt, Xt) - A.T @ A` can cause downstream Choleskys to fail
        # on otherwise PSD matrices. Negligible cost, stable result.
        latent_cov = 0.5 * (latent_cov + latent_cov.T)
        # Bring back to original output space. y_scaler.scale is scalar
        # for our scalar-output case, so `scale²` multiplies elementwise.
        return latent_cov * (self._y_scaler.scale ** 2)  # type: ignore[union-attr]

    def _require_fit(self) -> None:
        if self._opt_posterior is None or self._predict_cache is None:
            raise RuntimeError(
                f"{type(self).__name__} called before fit; conditioning "
                "state is unset."
            )

    # --- Fixed-hyperparameter conditioning ---------------------------------

    def condition_on(self, X_new: Array, Y_new: Array) -> Self:
        """Append ``(X_new, Y_new)`` to the training set without refitting.

        Returns a new emulator that conditions on the augmented data
        with **frozen** hyperparameters and **frozen** input/output
        scalers (i.e., the kernel lengthscales, observation noise,
        x-scaler, and y-scaler all carry over from the current fit).
        Internally uses ``_PredictCache.append_rows``, which updates
        the cached Cholesky factor via a block update — see that
        method's docstring for the math and cost analysis.

        For the public dispatch surface (``update_emulator`` +
        ``AppendRows``), the user-facing API isn't fully settled yet.
        This method exposes the low-level capability so it can be
        called directly while the dispatch wiring is designed.
        Crucially it does **not** re-run hyperparameter optimization;
        callers wanting refit semantics should call ``fit`` on the
        concatenated data instead.

        Args:
            X_new: ``(m, d)`` new training inputs in the original
                (raw) input space — they are passed through the same
                ``x-scaler`` used at fit time, so callers don't need
                to pre-scale.
            Y_new: ``(m,)`` new training outputs in the original
                (raw) output space — passed through the same
                ``y-scaler`` used at fit time.

        Returns:
            A new ``DSPGPEmulator`` with the augmented training data
            and an updated cache; the same hyperparameters and scalers.
        """
        self._require_fit()
        d = self.input_shape[0]
        if X_new.ndim != 2 or X_new.shape[1] != d:
            raise ValueError(
                f"DSPGPEmulator.condition_on: expected X_new.shape=(m, {d}), "
                f"got {tuple(X_new.shape)}."
            )
        if Y_new.shape != (X_new.shape[0],):
            raise ValueError(
                f"DSPGPEmulator.condition_on: expected Y_new.shape="
                f"({X_new.shape[0]},), got {tuple(Y_new.shape)}."
            )

        Xs_new = self._x_scaler.transform(X_new).astype(jnp.float64)  # type: ignore[union-attr]
        Ys_new = self._y_scaler.transform(Y_new).astype(jnp.float64)  # type: ignore[union-attr]

        new_cache = self._predict_cache.append_rows(Xs_new, Ys_new)  # type: ignore[union-attr]

        # Augment the stored gpjax Dataset so the (unused-by-predict
        # but available-for-callers) field stays consistent.
        new_dataset = gpx.Dataset(
            X=new_cache.Xs_train,
            y=jnp.concatenate(
                [self._train_dataset.y, Ys_new.reshape(-1, 1)],  # type: ignore[union-attr]
                axis=0,
            ),
        )

        return type(self)(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            name=self.name,
            kernel=self.kernel_name,
            max_iters=self.max_iters,
            jitter=self.jitter,
            verbose=self.verbose,
            _x_scaler=self._x_scaler,
            _y_scaler=self._y_scaler,
            _opt_posterior=self._opt_posterior,
            _train_dataset=new_dataset,
            _predict_cache=new_cache,
        )
