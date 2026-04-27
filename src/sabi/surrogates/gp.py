"""tinygp-backed GP surrogate with a Matern-5/2 kernel.

v0 spike: scalar output (the GP is fit to log-posterior values). Inputs and
outputs are standardized before fitting. Hyperparameters are chosen by simple
data-adaptive heuristics rather than log-marginal-likelihood optimization:

- **amplitude**: 1 (correct for standardized outputs; Y has unit std).
- **lengthscale**: a configurable multiple of the median nearest-neighbor
  distance on the standardized inputs. This scales naturally with training-set
  size, which keeps the kernel matrix well-conditioned across the loop as
  data points accumulate.
- **noise**: small fixed value (relative to standardized Y).

Log-marginal-likelihood optimization with optimistix/BFGS was tried first, but
the BFGS compile cost dominated the spike's run time and the optimizer
frequently diverged into singular kernel regimes. A principled hyperparameter
search is deferred to v1 together with the optimization module (§5).

This surrogate does NOT take a tempering parameter — tempering is applied
downstream in `LogDensityForm`. See design doc §4.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Self

import jax.numpy as jnp
from jax import Array
from tinygp import GaussianProcess, kernels

from sabi.surrogates.base import Surrogate, SurrogatePrediction


@dataclass(frozen=True)
class _Standardizer:
    """Affine standardizer: y' = (y - loc) / scale. `scale=1` if input is constant."""

    loc: Array
    scale: Array

    @classmethod
    def fit(cls, x: Array, axis: int = 0) -> "_Standardizer":
        loc = jnp.mean(x, axis=axis)
        scale = jnp.std(x, axis=axis)
        scale = jnp.where(scale < 1e-12, 1.0, scale)
        return cls(loc=loc, scale=scale)

    def transform(self, x: Array) -> Array:
        return (x - self.loc) / self.scale

    def inverse_mean(self, y: Array) -> Array:
        return y * self.scale + self.loc

    def inverse_var(self, v: Array) -> Array:
        return v * (self.scale ** 2)


def _median_nn_distance(X: Array) -> Array:
    """Median of each point's nearest-neighbor Euclidean distance."""
    x2 = jnp.sum(X * X, axis=-1)
    D2 = x2[:, None] + x2[None, :] - 2.0 * X @ X.T
    # Mask the diagonal (self-distance).
    D2 = jnp.where(jnp.eye(X.shape[0], dtype=bool), jnp.inf, jnp.maximum(D2, 0.0))
    nn = jnp.sqrt(jnp.min(D2, axis=-1))
    return jnp.median(nn)


def _choose_lengthscale(X: Array, factor: float, floor: float) -> Array:
    """Lengthscale = `factor` × median NN distance, bounded below by `floor`."""
    if X.shape[0] < 2:
        return jnp.asarray(1.0, dtype=X.dtype)
    ls = factor * _median_nn_distance(X)
    return jnp.maximum(ls, floor)


def _build_gp(X: Array, lengthscale: Array, noise: float, jitter: float) -> GaussianProcess:
    kernel = kernels.Matern52(scale=lengthscale)  # amplitude = 1 on standardized Y
    return GaussianProcess(kernel, X, diag=noise + jitter)


@dataclass(frozen=True)
class GPSurrogate(Surrogate):
    """GP surrogate with Matern-5/2 isotropic kernel.

    Hyperparameters are data-adaptive (see module docstring): amplitude=1,
    lengthscale ∝ median NN distance, fixed small noise + jitter floor.

    Attributes set after `fit`:
        X_train, Y_train: standardized training data kept for conditioning.
        x_standardizer, y_standardizer: the fitted standardizers.
        lengthscale: chosen lengthscale on the standardized input scale.
    """

    # Lengthscale (on standardized inputs) = `ls_factor` × median NN distance,
    # but never below `ls_floor`. 1.5 is a slightly-oversmooth default that keeps
    # the kernel matrix well-conditioned and still picks up the 2-D geometry.
    ls_factor: float = 1.5
    ls_floor: float = 0.05
    # Small observation noise on the standardized scale.
    noise: float = 1e-4
    # Extra jitter added to the diagonal for Cholesky stability. Needed
    # because Matern-5/2 with moderate lengthscales can produce kernel matrices
    # with slightly-negative eigenvalues from roundoff.
    jitter: float = 1e-3

    X_train: Array | None = field(default=None, repr=False)
    Y_train: Array | None = field(default=None, repr=False)
    x_standardizer: _Standardizer | None = field(default=None, repr=False)
    y_standardizer: _Standardizer | None = field(default=None, repr=False)
    lengthscale: Array | None = field(default=None, repr=False)

    def fit(self, X: Array, Y: Array) -> Self:
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if Y.ndim != 1 or Y.shape[0] != X.shape[0]:
            raise ValueError(f"Y must be 1-D with len(X), got shape {Y.shape}.")

        x_std = _Standardizer.fit(X, axis=0)
        y_std = _Standardizer.fit(Y, axis=0)
        Xs = x_std.transform(X)
        Ys = y_std.transform(Y)

        lengthscale = _choose_lengthscale(Xs, self.ls_factor, self.ls_floor)

        return replace(
            self,
            X_train=Xs,
            Y_train=Ys,
            x_standardizer=x_std,
            y_standardizer=y_std,
            lengthscale=lengthscale,
        )

    def predict(self, X: Array) -> SurrogatePrediction:
        if self.X_train is None:
            raise RuntimeError("GPSurrogate.predict called before fit.")
        assert self.lengthscale is not None
        assert self.x_standardizer is not None
        assert self.y_standardizer is not None
        assert self.Y_train is not None

        Xs = self.x_standardizer.transform(X)
        gp = _build_gp(self.X_train, self.lengthscale, self.noise, self.jitter)
        cond = gp.condition(self.Y_train, Xs).gp
        mean = self.y_standardizer.inverse_mean(cond.mean)
        # Clip tiny-negative variances that arise from Cholesky roundoff.
        var = self.y_standardizer.inverse_var(jnp.maximum(cond.variance, 0.0))
        return SurrogatePrediction(mean=mean, variance=var)
