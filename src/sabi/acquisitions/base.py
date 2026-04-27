"""Acquisition interface.

The acquisition sees the current `Surrogate` plus an `AcquisitionState` bundling
everything it might need: the problem (for bounds + forms), the current design
set `(X, Y)`, the post-tempering `current_form`, and the current `tempering_state`
(opaque PyTree — see `docs/notation.md` and tempering/base.py).

v0 acquisitions are simple candidate-set scorers. The shared optimization
machinery (multi-start, greedy batching, support reparameterization) lives in
`acquisitions/optim.py` in v1 (design doc §5).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from jax import Array

from sabi.problems.base import Problem
from sabi.problems.forms import LogDensityForm
from sabi.surrogates.base import Surrogate


@dataclass(frozen=True)
class AcquisitionState:
    """Per-round bundle passed to acquisitions. All fields are read-only."""

    problem: Problem
    surrogate: Surrogate
    X: Array  # (n,) + input_shape
    Y: Array  # (n,) + output_shape
    current_form: LogDensityForm  # form after tempering.apply
    tempering_state: Any  # opaque PyTree; None for untempered


class Acquisition(ABC):
    @abstractmethod
    def select_batch(
        self,
        state: AcquisitionState,
        q: int,
        key: Array,
    ) -> Array:
        """Return `(q,) + problem.input_shape` parameter locations to evaluate next."""
