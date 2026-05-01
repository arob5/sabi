"""tinygp-backed GP emulator with a Matern-5/2 kernel.

Inherits from both `Emulator` (sabi-side fittable random function marker)
and ProbPipe's `GaussianRandomFunction`. Diamond inheritance over
`ArrayRandomFunction`, resolved cleanly by Python's C3 MRO.

The class implements the two abstract methods required by
`GaussianRandomFunction`: `predict_mean(X)` and `predict_variance(X)`.
The full `predict` / `__call__` machinery (assembling these into a `Normal`
for marginal mode, `MultivariateNormal` for joint modes) comes from
`GaussianRandomFunction`. v1.2 supports only marginal mode
(`joint_inputs=False, joint_outputs=False`); joint covariance lands in
v1.5 when emulator metrics need it.

Hyperparameter strategy is unchanged from v1.x: data-adaptive lengthscale
(median nearest-neighbor distance × ls_factor, floored), unit amplitude on
standardized outputs, fixed small noise + Cholesky jitter. A principled
hyperparameter search is deferred to v1.4+ (with the optimization module).

v1.2 supports `X.shape == (n,) + input_shape` only (no extra leading batch
axes). Add vmap-over-extra-batch support in v1.5+ if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import jax.numpy as jnp
from jax import Array
from probpipe.distributions.gaussian_random_function import GaussianRandomFunction
from tinygp import GaussianProcess, kernels

from sabi.emulators.base import Emulator


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
    kernel = kernels.Matern52(scale=lengthscale)
    return GaussianProcess(kernel, X, diag=noise + jitter)


class GPEmulator(Emulator, GaussianRandomFunction):
    """GP emulator with Matern-5/2 isotropic kernel.

    `predict_mean` / `predict_variance` implement the abstract `GaussianRandomFunction`
    interface; `predict` / `__call__` come for free from the parent. `fit(X, Y)`
    returns a new `GPEmulator` carrying the conditioned state.
    """

    # Marginal-only in v1.2 — joint covariance lands when emulator metrics need it.
    supports_joint_inputs: bool = False
    supports_joint_outputs: bool = False

    def __init__(
        self,
        *,
        input_shape: tuple[int, ...] = (2,),
        output_shape: tuple[int, ...] = (),
        name: str | None = None,
        ls_factor: float = 1.5,
        ls_floor: float = 0.05,
        noise: float = 1e-4,
        jitter: float = 1e-3,
        # Internal conditioned-on-training state. Users don't pass these on
        # construction; `fit` populates them on the returned instance.
        _x_standardizer: _Standardizer | None = None,
        _y_standardizer: _Standardizer | None = None,
        _X_train: Array | None = None,
        _Y_train: Array | None = None,
        _lengthscale: Array | None = None,
    ):
        # Diamond inheritance: super().__init__ walks the MRO
        # (Surrogate → GaussianRandomFunction → ArrayRandomFunction) until it
        # finds `__init__`. ArrayRandomFunction's `__init__` accepts the args.
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            name=name or "GPEmulator",
        )
        self.ls_factor = ls_factor
        self.ls_floor = ls_floor
        self.noise = noise
        self.jitter = jitter
        self._x_standardizer = _x_standardizer
        self._y_standardizer = _y_standardizer
        self._X_train = _X_train
        self._Y_train = _Y_train
        self._lengthscale = _lengthscale

    @property
    def lengthscale(self) -> Array | None:
        return self._lengthscale

    def fit(self, X: Array, Y: Array) -> Self:
        if X.ndim != 1 + len(self.input_shape):
            raise ValueError(
                f"GPEmulator.fit: expected X.shape == (n,) + input_shape="
                f"{self.input_shape}, got {tuple(X.shape)}."
            )
        if Y.shape != X.shape[: -len(self.input_shape) or None]:
            # For our scalar output case (output_shape=()), Y must be shape (n,).
            expected_y_shape = X.shape[: -len(self.input_shape)] + self.output_shape
            if Y.shape != expected_y_shape:
                raise ValueError(
                    f"GPEmulator.fit: expected Y.shape={expected_y_shape}, "
                    f"got {tuple(Y.shape)}."
                )

        x_std = _Standardizer.fit(X, axis=0)
        y_std = _Standardizer.fit(Y, axis=0)
        Xs = x_std.transform(X)
        Ys = y_std.transform(Y)
        lengthscale = _choose_lengthscale(Xs, self.ls_factor, self.ls_floor)

        return type(self)(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            name=self.name,
            ls_factor=self.ls_factor,
            ls_floor=self.ls_floor,
            noise=self.noise,
            jitter=self.jitter,
            _x_standardizer=x_std,
            _y_standardizer=y_std,
            _X_train=Xs,
            _Y_train=Ys,
            _lengthscale=lengthscale,
        )

    # --- GaussianRandomFunction abstract methods --------------------------

    def predict_mean(self, X: Array) -> Array:
        """Return the predictive mean at each row of `X`.

        Args:
            X: shape `(n,) + input_shape`. (v1.2 doesn't support extra leading
                batch axes.)

        Returns:
            Shape `(n,) + output_shape`. For sabi's scalar-output benchmarks
            this is `(n,)`.
        """
        self._require_fit()
        Xs = self._x_standardizer.transform(X)  # type: ignore[union-attr]
        gp = _build_gp(self._X_train, self._lengthscale, self.noise, self.jitter)  # type: ignore[arg-type]
        cond = gp.condition(self._Y_train, Xs).gp  # type: ignore[arg-type]
        return self._y_standardizer.inverse_mean(cond.mean)  # type: ignore[union-attr]

    def predict_variance(self, X: Array) -> Array:
        """Return the marginal predictive variance at each row of `X`."""
        self._require_fit()
        Xs = self._x_standardizer.transform(X)  # type: ignore[union-attr]
        gp = _build_gp(self._X_train, self._lengthscale, self.noise, self.jitter)  # type: ignore[arg-type]
        cond = gp.condition(self._Y_train, Xs).gp  # type: ignore[arg-type]
        # Clip tiny-negative variances that arise from Cholesky roundoff.
        return self._y_standardizer.inverse_var(jnp.maximum(cond.variance, 0.0))  # type: ignore[union-attr]

    def _require_fit(self) -> None:
        if self._X_train is None:
            raise RuntimeError(
                f"{type(self).__name__} called before fit; conditioning "
                "state is unset."
            )
