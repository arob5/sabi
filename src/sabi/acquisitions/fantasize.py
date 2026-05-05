"""Fantasy imputers for greedy multi-point acquisition optimization.

When the optimizer picks a batch greedily — pick `x_1`, "observe" it,
pick `x_2` given the new state, and so on — it needs a strategy for what
the hallucinated observation at `x_pending` should be. That strategy is
a `FantasyImputer`.

This is the standard taxonomy from BO literature:

- **Kriging-believer**: `y_pending = emulator.predict_mean(x_pending)`.
  The emulator's own best guess; conservative.
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
from sabi.surrogate.surrogate_distribution import EmulatedDistribution


class FantasyImputer(ABC):
    """Strategy for hallucinating observations at pending batch points."""

    @abstractmethod
    def impute(self, x_pending: Array, state: AcquisitionState) -> Array:
        """Return shape `(q_pending,) + state.problem.output_shape` —
        hallucinated `y` values for the pending batch points.
        """


@dataclass(frozen=True)
class KrigingBeliever(FantasyImputer):
    r"""Use the emulator's predictive mean at `x_pending` as the imputed y.

    .. math::

        y_{\text{pending}} = \mathbb{E}_f[f(x_{\text{pending}})] = \mu(x_{\text{pending}})

    Conservative: the optimizer "believes" the emulator is right at the
    pending points. Standard default for greedy multi-point BO.

    **Caveat for non-GP / hyperparameter-refit emulators.** For a GP
    with **fixed** hyperparameters, kriging-believer imputation has a
    special property: refitting the GP on data points whose `y` values
    match the predictive mean leaves the predictive mean unchanged —
    only the predictive variance shrinks. So the next pick's score
    sees the same `mu` but tighter `sigma`, and greedy diversity comes
    entirely from variance reduction. **This property does not hold**
    for GPs with hyperparameters re-estimated between picks (sabi's
    GreedyMultiPointOptimizer triggers this via ``emulator.fit``)
    nor for non-Gaussian emulators. The implementation here is general:
    it refits the emulator each greedy step and the predictive mean may
    shift between picks.
    """

    def impute(self, x_pending: Array, state: AcquisitionState) -> Array:
        surrogate_distribution = state.surrogate_distribution
        if not isinstance(surrogate_distribution, EmulatedDistribution):
            raise ValueError(
                f"KrigingBeliever requires an emulator-backed "
                f"`EmulatedDistribution`; got "
                f"{type(surrogate_distribution).__name__}."
            )
        pred = surrogate_distribution.emulator(x_pending)
        return jnp.asarray(mean(pred))


@dataclass(frozen=True)
class ConstantLiar(FantasyImputer):
    r"""Use a constant value (per pending point) as the imputed y.

    .. math::

        y_{\text{pending}} = c,
        \quad c \in \{ \min Y, \max Y, \overline{Y}, \text{user-supplied} \}.

    The constant is broadcast across all pending points (no spatial
    dependence). After refitting the emulator on
    :math:`(X \cup x_{\text{pending}}, Y \cup c\mathbf{1})`, the next
    score call sees a posterior that has been "informed" of pending picks
    via the lie.

    Args:
        value: ``"min"`` / ``"max"`` / ``"mean"`` of ``state.Y_train``,
            or a fixed float. ``"min"`` (pessimistic) discourages
            re-picking nearby points (drives exploration); ``"max"``
            (optimistic) encourages it (drives exploitation). Operates
            on `Y_train` (the values consistent with the emulator's
            training data) — under target-side tempering this differs
            from `Y_raw`.
    """

    value: Literal["min", "max", "mean"] | float = "min"

    def impute(self, x_pending: Array, state: AcquisitionState) -> Array:
        n_pending = x_pending.shape[0]
        if isinstance(self.value, str):
            if self.value == "min":
                v = jnp.min(state.Y_train)
            elif self.value == "max":
                v = jnp.max(state.Y_train)
            elif self.value == "mean":
                v = jnp.mean(state.Y_train)
            else:
                raise ValueError(
                    f"ConstantLiar.value={self.value!r}; expected 'min' / 'max' / 'mean' or a float."
                )
        else:
            v = jnp.asarray(self.value)
        out_shape = state.problem.output_shape
        return jnp.full((n_pending,) + tuple(out_shape), v)
