"""tinygp-backed GP emulator with a Matern-5/2 kernel.

Inherits from :class:`sabi.emulators.gp.GPEmulator`, which provides
the predict pipeline, ``condition_on``, ``_replace``, and the cheap-
update dispatch. ``TinyGPEmulator`` itself owns:

- The data-adaptive heuristic for the lengthscale (median nearest-
  neighbor distance × ``ls_factor``, floored). No optimization at
  fit time; a principled hyperparameter search is a follow-up.
- A :class:`_TinyGPCache` populated at fit time. The cache exposes
  the same ``predict_latent`` / ``predict_latent_joint`` /
  ``append_rows`` surface as the gpjax cache, so the
  ``GPEmulator`` base treats them interchangeably.

Supports joint-input covariance via ``predict_covariance(X,
joint_inputs=True)`` and fixed-hyperparameter conditioning via
``condition_on(X_new, Y_new)``.

Currently supports ``X.shape == (n,) + input_shape`` only (no extra
leading batch axes); a vmap-over-extra-batch extension is a follow-up
if a benchmark needs it.
"""

from __future__ import annotations

from typing import Self

import jax.numpy as jnp
from jax import Array
from tinygp import kernels

from sabi.emulators._scalers import ZScoreScaler
from sabi.emulators.gp import GPEmulator
from sabi.emulators.tinygp._cache import _TinyGPCache


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


class TinyGPEmulator(GPEmulator):
    """GP emulator with Matern-5/2 isotropic kernel.

    Constructor args:
        input_shape: input dimensionality.
        output_shape: must be ``()`` (scalar output) — multi-output
            is a follow-up.
        name: optional emulator name (used for repr).
        ls_factor: multiplier on the median nearest-neighbor distance
            for the data-adaptive lengthscale.
        ls_floor: hard floor on the chosen lengthscale.
        noise: observation-noise *variance* (not stddev), added to
            the diagonal of the gram at fit time. Fixed; not learned.
        jitter: Cholesky-stability jitter, also added to the
            diagonal. Distinct from ``noise`` so callers can crank it
            without pretending the data is noisier than it is.
    """

    # Joint-input covariance and (trivially for scalar output)
    # joint-output mode are now supported via the cache's
    # ``predict_latent_joint`` and the base's ``predict_covariance``.
    supports_joint_inputs: bool = True
    supports_joint_outputs: bool = True

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
        # Internal post-fit state (callers don't pass these).
        _x_scaler: ZScoreScaler | None = None,
        _y_scaler: ZScoreScaler | None = None,
        _predict_cache: _TinyGPCache | None = None,
    ):
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            name=name or "TinyGPEmulator",
        )
        self.ls_factor = ls_factor
        self.ls_floor = ls_floor
        self.noise = noise
        self.jitter = jitter
        self._x_scaler = _x_scaler
        self._y_scaler = _y_scaler
        self._predict_cache = _predict_cache

    @property
    def lengthscale(self) -> Array | None:
        """Fitted isotropic lengthscale, or ``None`` if not yet fit.

        Read back from the cache's kernel — preserved as a public
        attribute for backwards-compatibility (some tests
        introspect it). Pre-fit returns ``None``.
        """
        if self._predict_cache is None:
            return None
        return self._predict_cache.kernel.scale

    @property
    def obs_noise_variance(self) -> Array:
        """Observation-noise variance — the constructor ``noise`` arg.

        ``TinyGPEmulator`` uses a fixed scalar noise term (passed at
        construction); it is not fit from data. Returned as a JAX
        array so callers can use it in ``jnp`` arithmetic without
        type juggling.
        """
        return jnp.asarray(self.noise)

    def fit(self, X: Array, Y: Array) -> Self:
        """Fit on ``(X, Y)``. Standardizes both, picks an isotropic
        lengthscale heuristically, and builds the Cholesky cache.

        Args:
            X: shape ``(n,) + input_shape``.
            Y: shape ``(n,) + output_shape``. For sabi's scalar-output
                benchmarks this is ``(n,)``.

        Returns:
            A new ``TinyGPEmulator`` carrying the conditioned state.
        """
        if X.ndim != 1 + len(self.input_shape):
            raise ValueError(
                f"TinyGPEmulator.fit: expected X.shape == (n,) + input_shape="
                f"{self.input_shape}, got {tuple(X.shape)}."
            )
        expected_y_shape = X.shape[: -len(self.input_shape)] + self.output_shape
        if Y.shape != expected_y_shape:
            raise ValueError(
                f"TinyGPEmulator.fit: expected Y.shape={expected_y_shape}, "
                f"got {tuple(Y.shape)}."
            )

        x_std = ZScoreScaler.fit(X, axis=0)
        y_std = ZScoreScaler.fit(Y, axis=0)
        Xs = x_std.transform(X)
        Ys = y_std.transform(Y)
        lengthscale = _choose_lengthscale(Xs, self.ls_factor, self.ls_floor)
        kernel = kernels.Matern52(scale=lengthscale)

        cache = _TinyGPCache.build(
            kernel, Xs, Ys, noise=self.noise, jitter=self.jitter
        )
        return self._replace(
            _x_scaler=x_std,
            _y_scaler=y_std,
            _predict_cache=cache,
        )
