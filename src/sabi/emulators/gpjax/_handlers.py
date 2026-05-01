r"""Cheap-update dispatch handlers for ``DSPGPEmulator``.

Three handlers cover the three ``EmulatorUpdate`` op types:

- ``_DSPGPAppendRowsHandler``: ``AppendRows(X_new, Y_new)`` →
  rank-one Cholesky / alpha update via
  ``DSPGPEmulator.condition_on``. O(n²m + m³).

- ``_DSPGPRescaleOutputsHandler``: ``RescaleOutputs(factor=β)`` →
  rescale the y-scaler by β. The cache stays unchanged because
  z-scoring is scale-invariant — see ``ZScoreScaler.rescaled_by``
  for the math. O(1).

- ``_DSPGPRescaleThenAppendHandler``: ``RescaleThenAppend(factor=β,
  X_new, Y_new)`` → rescale the y-scaler (free), then standardize
  ``Y_new`` (already in the new state's units) with the new scaler
  and rank-one append. O(n²m + m³), same as ``AppendRows``.

All three require an existing fit (``_predict_cache`` populated) and a
positive ``factor``; otherwise they report infeasible and dispatch
falls back to refit.

Module-bottom: registers all three handlers with
``emulator_update_registry``. The gpjax package's lazy
``__getattr__`` defers loading this module until ``DSPGPEmulator`` is
referenced, so registration only happens when the optional extra is
actually in use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gpjax as gpx
import jax.numpy as jnp
from probpipe.core._registry import MethodInfo

from sabi.emulators.dispatch import (
    EmulatorUpdateMethod,
    emulator_update_registry,
)
from sabi.emulators.updates import AppendRows, RescaleOutputs, RescaleThenAppend

if TYPE_CHECKING:  # pragma: no cover
    from sabi.emulators.gpjax.dsp_gp import DSPGPEmulator


class _DSPGPAppendRowsHandler(EmulatorUpdateMethod):
    """Cheap-path ``AppendRows`` handler for ``DSPGPEmulator``.

    Delegates to ``DSPGPEmulator.condition_on``, which performs the
    rank-one (block) Cholesky update on the cached factor — see
    ``_PredictCache.append_rows`` for the math. Frozen
    hyperparameters are intentional: this is the right cheap path
    *only* when callers have decided the existing fit is good enough
    for the augmented dataset. The dispatcher's natural fallback
    (``factory().fit(X_full, Y_full)``) takes over when the loop
    decides a refit is warranted.

    Feasibility: requires the emulator to have an existing fit (i.e.,
    a populated ``_predict_cache``). An unfitted emulator falls back
    to refit, which is the only correct option there.
    """

    @property
    def name(self) -> str:
        return "dspgp_append_rows_chol_update"

    def supported_types(self) -> tuple[type, ...]:
        from sabi.emulators.gpjax.dsp_gp import DSPGPEmulator

        return (DSPGPEmulator,)

    def check(self, emulator, plan) -> MethodInfo:
        if not isinstance(plan, AppendRows):
            return MethodInfo(
                feasible=False,
                method_name=self.name,
                description="plan is not AppendRows",
            )
        if emulator._predict_cache is None:
            return MethodInfo(
                feasible=False,
                method_name=self.name,
                description="emulator not yet fitted; cache absent",
            )
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(
        self, emulator: "DSPGPEmulator", plan: AppendRows
    ) -> "DSPGPEmulator":
        return emulator.condition_on(plan.X_new, plan.Y_new)


class _DSPGPRescaleOutputsHandler(EmulatorUpdateMethod):
    r"""Cheap-path ``RescaleOutputs`` handler for ``DSPGPEmulator``.

    A pure rescale ``Y_b = β · Y_a`` is essentially a no-op on the
    cache because z-scoring is scale-invariant:

    .. math::

        \text{loc}_b = \beta \cdot \text{loc}_a, \quad
        \text{scale}_b = \beta \cdot \text{scale}_a
        \;\Longrightarrow\;
        \frac{\beta Y_a - \text{loc}_b}{\text{scale}_b}
        = \frac{Y_a - \text{loc}_a}{\text{scale}_a}.

    Standardized y is unchanged, so the cached ``L_sigma`` (depends
    only on x and hyperparameters), ``alpha = L^{-1}(y_std - m(X))``,
    ``Xs_train``, and ``noise_var`` are all invariant. The only thing
    that needs updating is the y-scaler — predictions in the original
    space then come out at the new state's units automatically:
    ``mean_b = β·mean_a``, ``var_b = β²·var_a``.

    Cost: O(1) — construct a new y-scaler.
    """

    @property
    def name(self) -> str:
        return "dspgp_rescale_outputs_yscaler_only"

    def supported_types(self) -> tuple[type, ...]:
        from sabi.emulators.gpjax.dsp_gp import DSPGPEmulator

        return (DSPGPEmulator,)

    def check(self, emulator, plan) -> MethodInfo:
        if not isinstance(plan, RescaleOutputs):
            return MethodInfo(
                feasible=False,
                method_name=self.name,
                description="plan is not RescaleOutputs",
            )
        if emulator._predict_cache is None:
            return MethodInfo(
                feasible=False,
                method_name=self.name,
                description="emulator not yet fitted; cache absent",
            )
        if not (plan.factor > 0):
            return MethodInfo(
                feasible=False,
                method_name=self.name,
                description=(
                    f"RescaleOutputs(factor={plan.factor!r}) is not positive;"
                    " z-scoring requires positive scale."
                ),
            )
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(
        self, emulator: "DSPGPEmulator", plan: RescaleOutputs
    ) -> "DSPGPEmulator":
        if plan.factor == 1.0:
            return emulator  # exact no-op
        new_y_scaler = emulator._y_scaler.rescaled_by(plan.factor)
        return emulator._replace(_y_scaler=new_y_scaler)


class _DSPGPRescaleThenAppendHandler(EmulatorUpdateMethod):
    r"""Cheap-path ``RescaleThenAppend`` handler for ``DSPGPEmulator``.

    Composite of the two single-op handlers, applied in order so the
    new rows are standardized in the new state's coordinate system:

    1. **Rescale step** (O(1)): build ``y_scaler_b`` with
       ``loc_b = β·loc_a, scale_b = β·scale_a``. Cache untouched
       (standardized y is invariant under uniform rescale).
    2. **Append step** (O(n²m + m³)): standardize ``Y_new`` (already
       given at state b) with ``y_scaler_b``, then call
       ``_PredictCache.append_rows`` for the block-Cholesky update.

    Total cost matches a single ``condition_on`` (the rescale step is
    free) — so a tempering round with new rows costs the same as a
    no-tempering round with new rows.

    Note: the rescale step assumes a uniform multiplicative rescale
    of Y (the only shape ``RescaleOutputs`` carries). State-shaped
    transforms with shift terms or per-coordinate scales would need
    a different cheap path.
    """

    @property
    def name(self) -> str:
        return "dspgp_rescale_then_append_chol_update"

    def supported_types(self) -> tuple[type, ...]:
        from sabi.emulators.gpjax.dsp_gp import DSPGPEmulator

        return (DSPGPEmulator,)

    def check(self, emulator, plan) -> MethodInfo:
        if not isinstance(plan, RescaleThenAppend):
            return MethodInfo(
                feasible=False,
                method_name=self.name,
                description="plan is not RescaleThenAppend",
            )
        if emulator._predict_cache is None:
            return MethodInfo(
                feasible=False,
                method_name=self.name,
                description="emulator not yet fitted; cache absent",
            )
        if not (plan.factor > 0):
            return MethodInfo(
                feasible=False,
                method_name=self.name,
                description=(
                    f"RescaleThenAppend(factor={plan.factor!r}) is not "
                    "positive; z-scoring requires positive scale."
                ),
            )
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(
        self, emulator: "DSPGPEmulator", plan: RescaleThenAppend
    ) -> "DSPGPEmulator":
        # Step 1: rescale the y-scaler (free).
        new_y_scaler = emulator._y_scaler.rescaled_by(plan.factor)

        # Step 2: standardize new rows in the *new* coordinate system,
        # then rank-one append. ``Y_new`` is already at state b per
        # the ``RescaleThenAppend`` contract, so the new scaler is
        # the right one to apply.
        d = emulator.input_shape[0]
        if plan.X_new.ndim != 2 or plan.X_new.shape[1] != d:
            raise ValueError(
                f"RescaleThenAppend handler: expected X_new.shape=(m, {d}), "
                f"got {tuple(plan.X_new.shape)}."
            )
        if plan.Y_new.shape != (plan.X_new.shape[0],):
            raise ValueError(
                f"RescaleThenAppend handler: expected Y_new.shape="
                f"({plan.X_new.shape[0]},), got {tuple(plan.Y_new.shape)}."
            )

        Xs_new = emulator._x_scaler.transform(plan.X_new).astype(jnp.float64)
        Ys_new_std = new_y_scaler.transform(plan.Y_new).astype(jnp.float64)
        new_cache = emulator._predict_cache.append_rows(Xs_new, Ys_new_std)

        return emulator._replace(
            _y_scaler=new_y_scaler,
            _predict_cache=new_cache,
        )


# Register at module import time. Registration is idempotent:
# subsequent imports of this module are no-ops on the registry.
for _h in (
    _DSPGPAppendRowsHandler(),
    _DSPGPRescaleOutputsHandler(),
    _DSPGPRescaleThenAppendHandler(),
):
    if _h.name not in emulator_update_registry._name_index:
        emulator_update_registry.register(_h)
