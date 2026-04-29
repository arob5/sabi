"""Expected Improvement acquisition.

For a latent function with Gaussian predictive `N(μ, σ²)` at each point
and a current best observed value `f*`, the expected improvement is

    EI(x) = (μ - f*) Φ(z) + σ φ(z),   z = (μ - f*) / σ

EI targets regions likely to exceed the current-best surrogate value.
When the surrogate emulates the log-posterior (`Identity` form), this
hunts for high-posterior-density regions — a reasonable v0+ heuristic.

`ExpectedImprovement` is a `PointwiseScoredAcquisition`; it provides the
single-point `score(x, state)` and delegates batch selection to a
configurable `PointwiseOptimizer`. Default is `CandidateSetOptimizer` for
backwards compatibility with v1.x; switch to `ContinuousMultiStartOptimizer`
for gradient-based maxima or wrap with `GreedyMultiPointOptimizer` for
`q > 1` with in-batch diversity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array
from jax.scipy.stats import norm
from probpipe import mean, variance

from sabi.acquisitions.base import AcquisitionState, PointwiseScoredAcquisition
from sabi.acquisitions.optim import CandidateSetOptimizer, PointwiseOptimizer


@dataclass(frozen=True)
class ExpectedImprovement(PointwiseScoredAcquisition):
    """EI scoring function. Optimization is delegated to `optimizer`.

    Args:
        optimizer: pointwise optimizer that searches the score. Default is
            `CandidateSetOptimizer()` (v1.x behavior). Use
            `ContinuousMultiStartOptimizer()` for gradient-based maxima
            or wrap with `GreedyMultiPointOptimizer(inner=...)` for `q > 1`.
        xi: exploration offset (larger xi ⇒ more exploration). Default
            0.0; small positive for noisy surrogates.
        best_from: ``"data"`` uses `max(state.Y)`; ``"mean"`` uses the
            surrogate's predictive-mean max at `state.X` (more robust for
            noisy labels — unused in the v0/v1 noiseless setting).
    """

    optimizer: PointwiseOptimizer = field(default_factory=CandidateSetOptimizer)
    xi: float = 0.0
    best_from: str = "data"

    def score(self, x: Array, state: AcquisitionState) -> Array:
        """Single-point EI at `x` (shape `input_shape`). Returns scalar."""
        # surrogate.__call__ expects a leading batch axis.
        pred = state.surrogate(x[None])
        mu = jnp.asarray(mean(pred))[0]
        var_ = jnp.asarray(variance(pred))[0]
        std = jnp.sqrt(jnp.maximum(var_, 1e-30))
        best = self._best(state)
        improvement = mu - best - self.xi
        z = improvement / std
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        # Zero EI where variance collapses (already-evaluated points).
        return jnp.where(var_ <= 1e-30, 0.0, ei)

    def _best(self, state: AcquisitionState) -> Array:
        if self.best_from == "data":
            return jnp.max(state.Y)
        if self.best_from == "mean":
            train_pred = state.surrogate(state.X)
            return jnp.max(jnp.asarray(mean(train_pred)))
        raise ValueError(f"Unknown best_from={self.best_from!r}.")
