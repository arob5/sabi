"""Surrogate interface.

v0 uses a minimal Gaussian predictive: `predict` returns mean and variance at
test points. The v1 design (§4.3) extends this to `Distribution[Y]` and adds
`sample_function`; for the v0 spike we just need the Gaussian case.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

from jax import Array


@dataclass(frozen=True)
class SurrogatePrediction:
    """Marginal Gaussian predictive at a batch of test points."""

    mean: Array  # (N,)
    variance: Array  # (N,)


class Surrogate(ABC):
    """Stochastic emulator of a scalar target function.

    β-agnostic: the surrogate never sees the tempering parameter — tempering
    lives in `LogDensityForm` / `Tempering` downstream (design doc §4.3, §4.11).
    """

    @abstractmethod
    def fit(self, X: Array, Y: Array) -> Self: ...

    @abstractmethod
    def predict(self, X: Array) -> SurrogatePrediction: ...
