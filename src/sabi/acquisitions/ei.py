r"""Expected Improvement acquisition.

For a latent function with Gaussian predictive :math:`N(\mu(x), \sigma^2(x))`
at each point and a current best observed value :math:`f^*`, the expected
improvement is

.. math::

    \mathrm{EI}(x) = (\mu(x) - f^* - \xi)\, \Phi(z) + \sigma(x)\, \phi(z),
    \quad z = \frac{\mu(x) - f^* - \xi}{\sigma(x)},

where :math:`\Phi`, :math:`\phi` are the standard-Normal CDF / PDF and
:math:`\xi \ge 0` is an `offset` controlling exploration / exploitation
(larger `offset` → more exploration). Setting :math:`\sigma(x) = 0`
collapses EI to zero (already-evaluated points contribute nothing).

EI targets regions likely to exceed the current-best surrogate value.
When the surrogate emulates the log-posterior (`Identity` form), this
hunts for high-posterior-density regions — a reasonable v0+ heuristic.

**Assumptions.** EI reads only the first two moments of the surrogate
predictive at each query point. Concretely the score requires
`state.surrogate_posterior.surrogate(x[None])` to satisfy ProbPipe's
`SupportsMean` and `SupportsVariance` protocols (TFP-backed `Normal` /
`MultivariateNormal` do this by construction). See the
`PointwiseScoredAcquisition` base docstring for the full contract on
moment availability and JAX-traceability.

Symbols (:math:`x`, :math:`f^*`, :math:`\mu`, :math:`\sigma`) follow
``docs/notation.md``.
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
    """Expected-improvement scoring; optimization delegated to `optimizer`.

    Args:
        optimizer: pointwise optimizer that searches the score. Default is
            `CandidateSetOptimizer()` (v1.x behavior). Use
            `ContinuousMultiStartOptimizer()` for gradient-based maxima
            or wrap with `GreedyMultiPointOptimizer(inner=...)` for `q > 1`.
        offset: exploration offset :math:`\\xi \\ge 0` (larger → more
            exploration). Default `0.0` for noiseless surrogates;
            small positive values can stabilize EI for noisy ones.
        best_from: ``"data"`` uses `max(state.Y)`; ``"mean"`` uses the
            surrogate's predictive-mean max at `state.X` (more robust
            for noisy labels — unused in v0/v1 noiseless setting).
    """

    optimizer: PointwiseOptimizer = field(default_factory=CandidateSetOptimizer)
    offset: float = 0.0
    best_from: str = "data"

    def _score_single(self, x: Array, state: AcquisitionState) -> Array:
        """Single-point EI at `x` (shape `state.problem.input_shape`).
        Returns scalar."""
        surrogate = state.surrogate_posterior.surrogate
        if surrogate is None:
            raise ValueError(
                "ExpectedImprovement requires a non-degenerate surrogate; "
                "got `state.surrogate_posterior.surrogate=None` (this happens "
                "with the weighted-empirical baseline). Switch to a real "
                "surrogate or use a sampling acquisition like PriorSampling."
            )
        # surrogate.__call__ expects a leading batch axis.
        pred = surrogate(x[None])
        mu = jnp.asarray(mean(pred))[0]
        var_ = jnp.asarray(variance(pred))[0]
        std = jnp.sqrt(jnp.maximum(var_, 1e-30))
        best = self._best(state)
        improvement = mu - best - self.offset
        z = improvement / std
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        # Zero EI where variance collapses (already-evaluated points).
        return jnp.where(var_ <= 1e-30, 0.0, ei)

    def _best(self, state: AcquisitionState) -> Array:
        if self.best_from == "data":
            return jnp.max(state.Y)
        if self.best_from == "mean":
            surrogate = state.surrogate_posterior.surrogate
            if surrogate is None:
                raise ValueError(
                    "ExpectedImprovement(best_from='mean') requires a "
                    "non-degenerate surrogate."
                )
            train_pred = surrogate(state.X)
            return jnp.max(jnp.asarray(mean(train_pred)))
        raise ValueError(f"Unknown best_from={self.best_from!r}.")
