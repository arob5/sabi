"""Shared affine input/output scalers used by emulator backends.

Two patterns recur across emulators:

- **Z-score** for outputs: standardize ``Y`` to zero-mean, unit-variance.
  Used by both the tinygp ``TinyGPEmulator`` and the gpjax
  ``DSPGPEmulator`` to keep training outputs on the unit scale that
  hyperparameter priors and unscaled-noise heuristics assume.
- **Min-max** for inputs: rescale ``X`` to ``[0, 1]^d``. Used by
  ``DSPGPEmulator`` (the DSP recipe is calibrated on ``[0, 1]^d``).
  ``TinyGPEmulator`` uses z-score for inputs too — so this module
  exposes both.

Both scalers are immutable (``@dataclass(frozen=True)``), constructed
via classmethod ``fit`` from training data, and provide ``transform``
plus the relevant inverse for prediction-time round-trip.

`ZScoreScaler` additionally provides a ``rescaled_by(factor)`` method:
``loc' = factor·loc, scale' = factor·scale``. Because z-scoring is
scale-invariant (``new_scaler.transform(factor·Y) ==
old_scaler.transform(Y)``), this is the right cheap-path realization
of a uniform multiplicative output rescale — used by emulator-update
handlers for ``RescaleOutputs`` and ``RescaleThenAppend`` plans.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

__all__ = [
    "MinMaxScaler",
    "ZScoreScaler",
]


@dataclass(frozen=True)
class MinMaxScaler:
    """Affine map: ``x' = (x - lo) / (hi - lo)``, clamped to avoid 0-width.

    For dimensions where ``hi == lo`` (a constant feature on the
    training set), the scale is set to 1 so the transform reduces to a
    translation rather than dividing by zero.
    """

    lo: Array
    hi: Array

    @classmethod
    def fit(cls, X: Array) -> "MinMaxScaler":
        lo = jnp.min(X, axis=0)
        hi = jnp.max(X, axis=0)
        # Where range is degenerate, fall back to scale=1 so we don't divide by 0.
        hi = jnp.where((hi - lo) < 1e-12, lo + 1.0, hi)
        return cls(lo=lo, hi=hi)

    def transform(self, X: Array) -> Array:
        return (X - self.lo) / (self.hi - self.lo)


@dataclass(frozen=True)
class ZScoreScaler:
    """Affine standardizer: ``y' = (y - loc) / scale``.

    For axes where the training data has near-zero standard deviation,
    ``scale`` is set to 1 so the transform reduces to a translation.

    Used for output standardization in both GP backends, and (in
    tinygp) for input standardization too.
    """

    loc: Array
    scale: Array

    @classmethod
    def fit(cls, y: Array, axis: int = 0) -> "ZScoreScaler":
        loc = jnp.mean(y, axis=axis)
        scale = jnp.std(y, axis=axis)
        scale = jnp.where(scale < 1e-12, 1.0, scale)
        return cls(loc=loc, scale=scale)

    def transform(self, y: Array) -> Array:
        return (y - self.loc) / self.scale

    def inverse_mean(self, m: Array) -> Array:
        """Map a mean back from standardized to original output space."""
        return m * self.scale + self.loc

    def inverse_var(self, v: Array) -> Array:
        """Map a (marginal) variance back from standardized to original output space.

        Cross-input or cross-output covariances scale by the same
        ``scale²`` for the scalar-output case; for multi-output a
        Kronecker structure is required (see
        ``sabi.emulators.gp._scale_cov_to_output_space``).
        """
        return v * (self.scale ** 2)

    def rescaled_by(self, factor: float) -> "ZScoreScaler":
        r"""Return a scaler with both ``loc`` and ``scale`` multiplied by ``factor``.

        Setting ``loc' = β·loc, scale' = β·scale`` makes
        ``new_scaler.transform(β·Y) == old_scaler.transform(Y)``: the
        standardized y values stay the same under a uniform rescale
        because z-scoring is scale-invariant. Predictions in the
        original output space rescale by ``β`` (mean) and ``β²``
        (variance) via the new scaler's ``inverse_*``.

        Used by the cheap-update dispatch handlers for
        ``RescaleOutputs(factor=β)``: the cache stays unchanged
        (standardized y is invariant), only the y-scaler updates.

        Args:
            factor: positive multiplicative factor.

        Raises:
            ValueError: when ``factor <= 0`` (z-scoring requires a
            positive scale).
        """
        if not (factor > 0):
            raise ValueError(
                f"ZScoreScaler.rescaled_by: factor must be > 0, got {factor!r}."
            )
        return type(self)(loc=self.loc * factor, scale=self.scale * factor)
