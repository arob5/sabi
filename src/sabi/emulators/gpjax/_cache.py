r"""Cholesky cache for the gpjax-backed DSP-GP emulator.

Holds the parts of a fitted ``ConjugatePosterior``'s predictive
distribution that depend only on the training data and hyperparameters
— so per-call ``predict_*`` cost on test inputs drops from O(n³) to
O(n²·m + n·m). Also exposes a block-Cholesky / alpha update for
fixed-hyperparameter conditioning on new training rows
(``append_rows``).

Standalone of ``DSPGPEmulator``: takes a fitted gpjax posterior and
already-standardized training data; returns a value-typed dataclass.
``DSPGPEmulator`` orchestrates the scaling and reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import jax.scipy as jsp
import lineax as lx
from gpjax.linalg.utils import add_jitter
from jax import Array


@dataclass(frozen=True)
class _PredictCache:
    """Pre-solved prediction state for an optimized ``ConjugatePosterior``.

    Caches the bits of the posterior predictive that depend only on
    the training data and fitted hyperparameters — i.e., are constant
    across test inputs:

    - ``L_sigma``: lower Cholesky factor of ``K(X, X) + diag(noise) +
      jitter·I``, shape ``(n, n)``.
    - ``alpha``: ``L_sigma⁻¹ (y - m(X))``, shape ``(n,)``. Combines
      with ``L_sigma⁻¹ K(X, X*)`` to give the predictive mean shift.
    - ``unwrapped_posterior``: the gpjax posterior with paramax
      unwrappables resolved to plain arrays. Stored so we can call
      ``.prior.kernel.cross_covariance``, ``.prior.mean_function``,
      ``.prior.kernel.diagonal``, and ``.likelihood`` without redoing
      the unwrap on every call.
    - ``Xs_train``: training inputs (in scaled coords), shape ``(n, d)``.
    - ``noise_var``: scalar observation noise variance (``obs_stddev²``).

    With this cache, ``predict_*`` cost drops from O(n³) (rebuilding
    the Cholesky every call) to O(n²·m + n·m) per batch of m test
    points. ``append_rows`` does the block update for adding new
    training rows in O(n²m + m³).
    """

    unwrapped_posterior: object
    L_sigma: Array
    alpha: Array
    Xs_train: Array
    Ys_train: Array
    noise_var: Array

    @classmethod
    def build(
        cls, opt_posterior, Xs_train: Array, Ys_train: Array
    ) -> "_PredictCache":
        """Compute the cache from a fitted gpjax posterior + training data.

        ``Ys_train`` must be the (already-standardized) target column
        of shape ``(n,)``; we reshape it to match gpjax's internal
        ``(n, 1)`` convention. The standardized vector is stored on
        the cache so it is the single source of truth for training
        data — callers don't need to keep a separate gpjax Dataset.
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
        # gpjax keeps two separate jitter values: ``posterior.jitter``
        # is used by ``posterior.predict`` on the training-side
        # Cholesky; ``posterior.prior.jitter`` is used by
        # ``conjugate_mll`` on the same Cholesky AND by
        # ``posterior.predict`` on the test-side covariance.
        # ``_build_dsp_posterior`` aligns them so they match, but we
        # still source the training jitter from ``posterior.jitter``
        # here to mirror gpjax's ``predict`` exactly — so an
        # equivalence test against the naive path catches any future
        # divergence the moment a future change decoupled the two.
        Kxx = unwrapped.prior.kernel.gram(Xs_train).as_matrix()
        train_jitter = unwrapped.jitter
        Kxx = add_jitter(Kxx, train_jitter)
        noise_diag = unwrapped.likelihood.noise_vector(n)
        Sigma = Kxx + jnp.diag(noise_diag)
        L_sigma = jnp.linalg.cholesky(Sigma)

        alpha = jsp.linalg.solve_triangular(
            L_sigma, y_flat - mx_flat, lower=True
        )

        # Gaussian likelihood: noise_vector returns obs_stddev² · 1ₙ;
        # a single scalar suffices for adding observation noise to
        # test marginal variances.
        noise_var = jnp.asarray(unwrapped.likelihood.obs_stddev) ** 2

        return cls(
            unwrapped_posterior=unwrapped,
            L_sigma=L_sigma,
            alpha=alpha,
            Xs_train=Xs_train,
            Ys_train=jnp.atleast_1d(jnp.asarray(Ys_train).squeeze()),
            noise_var=noise_var,
        )

    # ----- prediction -------------------------------------------------------

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

        Used by ``DSPGPEmulator.predict_covariance`` for the
        ``joint_inputs=True`` mode. Mirrors gpjax's
        ``posterior.predict(...).covariance_matrix`` (the
        ``return_covariance_type='dense'`` branch): the full latent
        covariance with prior jitter on the diagonal.

        Returned covariance is symmetrized
        (``0.5 * (cov + cov.T)``) to absorb the floating-point
        asymmetry that arises from ``K(Xt, Xt) - L⁻¹Kxt^T L⁻¹Kxt`` —
        callers downstream can pass the result straight to a Cholesky
        without an extra defensive symmetrization.

        Args:
            Xt: ``(m, d)`` test inputs in scaled coordinates.

        Returns:
            ``(mean, cov)`` with shapes ``(m,)`` and ``(m, m)``. Cov
            is symmetric PSD (latent + prior_jitter·I); add
            ``noise_var·I`` for the observation-noise-inclusive
            predictive covariance.
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
        # Defensive symmetrization: ``Ktt - A.T @ A`` accumulates fp
        # asymmetry that can break a downstream Cholesky on otherwise
        # PSD matrices. Negligible cost, stable result.
        cov = 0.5 * (cov + cov.T)
        return mean, cov

    # ----- block update ----------------------------------------------------

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
            + (self.noise_var + train_jitter) * jnp.eye(m, dtype=K_nn.dtype)
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
        Ys_aug = jnp.concatenate([self.Ys_train, jnp.atleast_1d(Ys_new)])

        return _PredictCache(
            unwrapped_posterior=self.unwrapped_posterior,
            L_sigma=L_new,
            alpha=alpha_aug,
            Xs_train=Xs_aug,
            Ys_train=Ys_aug,
            noise_var=self.noise_var,
        )
