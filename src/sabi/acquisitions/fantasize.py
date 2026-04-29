"""Fantasy imputers for greedy multi-point acquisition optimization.

When the optimizer picks a batch greedily — pick `x_1`, "observe" it,
pick `x_2` given the new state, and so on — it needs a strategy for what
the hallucinated observation at `x_pending` should be. That strategy is
a `FantasyImputer`.

This is the standard taxonomy from BO literature:

- **Kriging-believer**: `y_pending = surrogate.predict_mean(x_pending)`.
  The surrogate's own best guess; conservative.
- **Constant-liar**: `y_pending = constant`. Pessimistic ("min" of seen
  Y) drives the optimizer toward exploration; optimistic ("max") toward
  exploitation.

`FantasyImputer.impute(x_pending, state) -> Array` is the single
contract; new strategies (e.g., Thompson — sample a function realization
once and reuse) drop in by subclassing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jax import Array
from probpipe import mean

from sabi.acquisitions.base import AcquisitionState


class FantasyImputer(ABC):
    """Strategy for hallucinating observations at pending batch points."""

    @abstractmethod
    def impute(self, x_pending: Array, state: AcquisitionState) -> Array:
        """Return shape `(q_pending,) + state.problem.output_shape` —
        hallucinated `y` values for the pending batch points.
        """


@dataclass(frozen=True)
class KrigingBeliever(FantasyImputer):
    """Use the surrogate's predictive mean at `x_pending` as the imputed y.

    Conservative: the optimizer "believes" the surrogate is right at the
    pending points. Standard default for greedy multi-point BO.
    """

    def impute(self, x_pending: Array, state: AcquisitionState) -> Array:
        pred = state.surrogate(x_pending)
        return jnp.asarray(mean(pred))


@dataclass(frozen=True)
class ConstantLiar(FantasyImputer):
    """Use a constant value (per pending point) as the imputed y.

    Args:
        value: ``"min"`` / ``"max"`` / ``"mean"`` of `state.Y`, or a fixed
            float. ``"min"`` (pessimistic) discourages re-picking nearby
            points (drives exploration); ``"max"`` (optimistic) encourages
            it (drives exploitation).
    """

    value: Literal["min", "max", "mean"] | float = "min"

    def impute(self, x_pending: Array, state: AcquisitionState) -> Array:
        n_pending = x_pending.shape[0]
        if isinstance(self.value, str):
            if self.value == "min":
                v = jnp.min(state.Y)
            elif self.value == "max":
                v = jnp.max(state.Y)
            elif self.value == "mean":
                v = jnp.mean(state.Y)
            else:
                raise ValueError(
                    f"ConstantLiar.value={self.value!r}; expected 'min' / 'max' / 'mean' or a float."
                )
        else:
            v = jnp.asarray(self.value)
        out_shape = state.problem.output_shape
        return jnp.full((n_pending,) + tuple(out_shape), v)
