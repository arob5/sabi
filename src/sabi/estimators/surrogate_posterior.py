"""`SurrogatePosterior` — abstract bundle representing the posterior estimate
implied by the current surrogate state.

Two concrete subclasses ship in v1.1:

- `WeightedEmpiricalSurrogatePosterior` — the **baseline**, no GP needed.
  Stores `(X, Y)` where `Y` is the observed log-density at `X`, and exposes a
  ProbPipe `NumericEmpiricalDistribution` weighted by `Y` (via ProbPipe's
  `Weights(log_weights=Y)`). Naive — does not de-bias against the design
  distribution; explicitly a baseline for testing the loop without a GP.
- `GPPushforwardSurrogatePosterior` — the v0 plug-in-mean object, refactored.
  Bundles a fitted `Surrogate`, the current `LogDensityForm`, and the
  `Problem`; produces samples / log-density via the GP-pushforward.

In v1.2 these become subclasses of a `RandomMeasure` base class. For v1.1
they're plain dataclasses with no random-measure protocol surface yet.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from functools import cached_property

from jax import Array
from probpipe._weights import Weights
from probpipe.core._empirical import NumericEmpiricalDistribution

from sabi.problems.base import Problem
from sabi.problems.forms import LogDensityForm
from sabi.surrogates.base import Surrogate


class SurrogatePosterior(ABC):
    """Abstract base. Concrete subclasses define how the surrogate state is
    turned into a posterior estimate."""

    problem: Problem


@dataclass(frozen=True)
class WeightedEmpiricalSurrogatePosterior(SurrogatePosterior):
    """No-GP baseline.

    Represents the posterior as a weighted empirical distribution over the
    design points `X`, with weights `softmax(Y)` (where `Y_i = log p̃(X_i)` is
    the observed unnormalized log-density at `X_i`). This ignores the
    biasing introduced by the design distribution — it's intentionally
    simple and used as a sanity check that the loop machinery works without
    a GP.
    """

    X: Array  # (n,) + problem.input_shape
    Y: Array  # (n,) — log unnormalized density at each X_i
    problem: Problem

    @cached_property
    def empirical_distribution(self) -> NumericEmpiricalDistribution:
        """The underlying ProbPipe empirical distribution.

        Weights are `softmax(Y)` via ProbPipe's `Weights(log_weights=Y)` —
        normalization, ESS, etc. are handled inside `Weights`.
        """
        return NumericEmpiricalDistribution(
            samples=self.X,
            weights=Weights(n=self.X.shape[0], log_weights=self.Y),
            name=f"weighted_empirical_{self.problem.name}",
        )


@dataclass(frozen=True)
class GPPushforwardSurrogatePosterior(SurrogatePosterior):
    """Posterior estimate via the surrogate's mean predictor pushed forward
    through the current `LogDensityForm`. The v0 plug-in-mean object."""

    surrogate: Surrogate
    current_form: LogDensityForm
    problem: Problem
