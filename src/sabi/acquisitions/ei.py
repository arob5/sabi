"""Expected Improvement acquisition (surrogate max).

For a latent function with Gaussian predictive N(μ, σ²) at each point and a
current best observed value f*, the expected improvement is

    EI(x) = (μ - f*) Φ(z) + σ φ(z),   z = (μ - f*) / σ

This targets regions likely to exceed the current-best surrogate value. When
the surrogate emulates the log-posterior (`Identity` form), that corresponds
to hunting for high-posterior-density regions, which is a reasonable v0
heuristic. v1 replaces the candidate-set evaluation with the shared optimizer
in `acquisitions/optim.py` (design doc §5).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.stats import norm

from sabi.acquisitions.base import Acquisition, AcquisitionState
from sabi.initial_designs.base import sample_initial


def _ei(mean: Array, variance: Array, best: Array, xi: float) -> Array:
    std = jnp.sqrt(jnp.maximum(variance, 1e-30))
    improvement = mean - best - xi
    z = improvement / std
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    # Zero EI where variance collapses (already-evaluated points).
    return jnp.where(variance <= 1e-30, 0.0, ei)


@dataclass(frozen=True)
class ExpectedImprovement(Acquisition):
    """EI over a random candidate set.

    Args:
        n_candidates: number of random candidates evaluated per selection.
        xi: exploration offset (larger xi ⇒ more exploration). Standard default
            is 0.0 for noiseless GP, small positive for noisy.
        best_from: 'data' uses max(Y); 'mean' uses surrogate mean at X (useful
            when labels are noisy — unused in v0).
    """

    n_candidates: int = 1024
    xi: float = 0.0
    best_from: str = "data"

    def select_batch(self, state: AcquisitionState, q: int, key: Array) -> Array:
        key_cand, _ = jax.random.split(key)
        candidates = sample_initial(state.problem, key_cand, self.n_candidates)

        pred = state.surrogate.predict(candidates)

        if self.best_from == "data":
            best = jnp.max(state.Y)
        elif self.best_from == "mean":
            train_pred = state.surrogate.predict(state.X)
            best = jnp.max(train_pred.mean)
        else:
            raise ValueError(f"Unknown best_from={self.best_from!r}.")

        scores = _ei(pred.mean, pred.variance, best, self.xi)

        # Greedy top-q (no in-batch diversification for v0; repeats unlikely
        # because the candidate set is random each call).
        top_idx = jnp.argsort(-scores)[:q]
        return candidates[top_idx]
