r"""Cholesky cache for the tinygp-backed ``TinyGPEmulator``.

Mirrors ``sabi.emulators.gpjax._cache._PredictCache`` in shape (so it
satisfies the same :class:`PredictCacheProtocol`), but uses tinygp's
kernel API instead of gpjax's posterior. The training mean is taken
to be zero (the emulator z-scores ``Y`` before passing it here, so
the standardized training mean is zero by construction).

Cached state:

- ``L_sigma``: lower Cholesky factor of ``K(X, X) + (noise +
  jitter)·I``, shape ``(n, n)``.
- ``alpha``: ``L_sigma⁻¹ Y`` (no mean subtraction since ``Y`` is
  already z-scored), shape ``(n,)``.
- ``kernel``: the tinygp kernel object — used for
  ``kernel(X1, X2)`` / ``kernel(X)`` evaluations on the test side
  and for the rank-one block update.
- ``noise`` / ``jitter``: scalar variance terms; ``noise`` is the
  emulator's constructor ``noise`` arg and ``jitter`` is the
  Cholesky-stability term. Both go into the diagonal at fit time
  and into the rank-one update at append time.
- ``Xs_train`` / ``Ys_train``: the standardized training data.

With this cache, ``predict_*`` cost drops from O(n³) (rebuilding
the gram + Cholesky every call inside tinygp's
``gp.condition``) to O(n²·m + n·m). ``append_rows`` does the
block update for adding new training rows in O(n²m + m³).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array


@dataclass(frozen=True)
class _TinyGPCache:
    """Pre-solved prediction state for a fixed-hyperparameter tinygp GP.

    Conforms to :class:`sabi.emulators.gp.PredictCacheProtocol`.
    """

    kernel: Any  # tinygp.kernels.Kernel — evaluated via kernel(X1, X2) / kernel(X)
    noise: Array  # observation-noise variance (scalar JAX array)
    jitter: Array  # Cholesky-stability jitter (scalar JAX array)
    L_sigma: Array
    alpha: Array
    Xs_train: Array
    Ys_train: Array

    # ----- noise_var alias for protocol-shape symmetry --------------------

    @property
    def noise_var(self) -> Array:
        """Observation-noise variance, exposed for symmetry with the
        gpjax cache. ``TinyGPEmulator.obs_noise_variance`` returns
        the same value via this alias."""
        return self.noise

    # ----- construction ---------------------------------------------------

    @classmethod
    def build(
        cls,
        kernel: Any,
        Xs_train: Array,
        Ys_train: Array,
        *,
        noise: float,
        jitter: float,
    ) -> "_TinyGPCache":
        """Compute the cache from a tinygp kernel + standardized training data.

        ``Ys_train`` must be the (already-z-scored) target column of
        shape ``(n,)``. The mean is implicitly zero, so the standard
        ``alpha = L⁻¹(y - m(X))`` reduces to ``alpha = L⁻¹ y``.
        """
        n = Xs_train.shape[0]
        # ``jnp.asarray`` (not ``float(...)``) so callers can wrap this
        # in ``jax.jit`` / ``vmap`` / ``grad`` without hitting
        # ``ConcretizationTypeError`` on traced inputs. Mirrors the
        # gpjax ``_PredictCache`` which stores ``noise_var`` as a JAX
        # array for the same reason.
        noise = jnp.asarray(noise)
        jitter = jnp.asarray(jitter)
        Kxx = kernel(Xs_train, Xs_train)  # (n, n)
        Sigma = Kxx + (noise + jitter) * jnp.eye(n, dtype=Kxx.dtype)
        L_sigma = jnp.linalg.cholesky(Sigma)
        # mean(X) ≡ 0 for z-scored Y, so alpha = L⁻¹ y.
        alpha = jsp.linalg.solve_triangular(L_sigma, Ys_train, lower=True)
        return cls(
            kernel=kernel,
            noise=noise,
            jitter=jitter,
            L_sigma=L_sigma,
            alpha=alpha,
            Xs_train=Xs_train,
            Ys_train=jnp.atleast_1d(jnp.asarray(Ys_train).squeeze()),
        )

    # ----- prediction -----------------------------------------------------

    def predict_latent(self, Xt: Array) -> tuple[Array, Array]:
        """Latent (noiseless) marginal mean and variance at test inputs.

        Mirrors the gpjax cache: variance includes ``jitter`` on the
        diagonal (matching the prior-jitter convention). Observation
        noise is **not** added — see :class:`Emulator` for the latent
        convention.
        """
        Kxt = self.kernel(self.Xs_train, Xt)  # (n, m)
        L_inv_Kxt = jsp.linalg.solve_triangular(
            self.L_sigma, Kxt, lower=True
        )

        mean = L_inv_Kxt.T @ self.alpha  # (m,)

        Ktt_diag = self.kernel(Xt)  # (m,)
        var = Ktt_diag - jnp.einsum("ij,ij->j", L_inv_Kxt, L_inv_Kxt)
        var = var + self.jitter
        return mean, var

    def predict_latent_joint(self, Xt: Array) -> tuple[Array, Array]:
        """Latent joint mean and full covariance at test inputs.

        Cov is symmetrized inside (``0.5 * (cov + cov.T)``) so
        downstream Cholesky calls don't trip on fp asymmetry from
        ``Ktt - L⁻¹Kxt^T L⁻¹Kxt``.
        """
        Kxt = self.kernel(self.Xs_train, Xt)  # (n, m)
        L_inv_Kxt = jsp.linalg.solve_triangular(
            self.L_sigma, Kxt, lower=True
        )
        mean = L_inv_Kxt.T @ self.alpha  # (m,)

        Ktt = self.kernel(Xt, Xt)  # (m, m)
        cov = Ktt - L_inv_Kxt.T @ L_inv_Kxt
        cov = cov + self.jitter * jnp.eye(cov.shape[0], dtype=cov.dtype)
        cov = 0.5 * (cov + cov.T)
        return mean, cov

    # ----- block update ---------------------------------------------------

    def append_rows(
        self, Xs_new: Array, Ys_new: Array
    ) -> "_TinyGPCache":
        r"""Append new training rows via a block Cholesky / alpha update.

        Same math as :meth:`sabi.emulators.gpjax._cache._PredictCache.append_rows`
        — see that method's docstring for the verification. Cost:
        O(n²m + m³); refit-from-scratch costs O((n+m)³).

        The training mean is implicitly zero here (z-scored Y), so
        the residual update simplifies to ``α_new = L_22⁻¹(Y_new -
        U^⊤ α)`` (no ``m(X_new)`` term).
        """
        if Xs_new.ndim != 2 or Xs_new.shape[1] != self.Xs_train.shape[1]:
            raise ValueError(
                f"_TinyGPCache.append_rows: expected Xs_new.shape=(m, "
                f"{self.Xs_train.shape[1]}), got {tuple(Xs_new.shape)}."
            )
        if Ys_new.shape != (Xs_new.shape[0],):
            raise ValueError(
                f"_TinyGPCache.append_rows: expected Ys_new.shape="
                f"({Xs_new.shape[0]},), got {tuple(Ys_new.shape)}."
            )

        m = Xs_new.shape[0]
        n_old = self.Xs_train.shape[0]

        # K(X_old, X_new): (n_old, m).
        K_on = self.kernel(self.Xs_train, Xs_new)
        # U = L⁻¹ K(X_old, X_new): forward triangular solve, (n_old, m).
        U = jsp.linalg.solve_triangular(self.L_sigma, K_on, lower=True)

        # K(X_new, X_new) + (σ² + jitter)·I - U^⊤ U: Schur complement (m, m).
        K_nn = self.kernel(Xs_new, Xs_new)
        D = (
            K_nn
            + (self.noise + self.jitter) * jnp.eye(m, dtype=K_nn.dtype)
            - U.T @ U
        )
        # Symmetrize for Cholesky stability.
        D = 0.5 * (D + D.T)
        L_22 = jnp.linalg.cholesky(D)

        # Assemble L' = [[L, 0], [U^⊤, L_22]].
        n_new = n_old + m
        L_new = jnp.zeros((n_new, n_new), dtype=self.L_sigma.dtype)
        L_new = L_new.at[:n_old, :n_old].set(self.L_sigma)
        L_new = L_new.at[n_old:, :n_old].set(U.T)
        L_new = L_new.at[n_old:, n_old:].set(L_22)

        # Augmented residual (mean=0):  α_new = L_22⁻¹ (Y_new - U^⊤ α)
        residual = jnp.atleast_1d(Ys_new) - U.T @ self.alpha
        alpha_block = jsp.linalg.solve_triangular(L_22, residual, lower=True)
        alpha_aug = jnp.concatenate([self.alpha, alpha_block])

        Xs_aug = jnp.concatenate([self.Xs_train, Xs_new], axis=0)
        Ys_aug = jnp.concatenate([self.Ys_train, jnp.atleast_1d(Ys_new)])

        return _TinyGPCache(
            kernel=self.kernel,
            noise=self.noise,
            jitter=self.jitter,
            L_sigma=L_new,
            alpha=alpha_aug,
            Xs_train=Xs_aug,
            Ys_train=Ys_aug,
        )
