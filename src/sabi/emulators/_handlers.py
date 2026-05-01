r"""Cheap-update dispatch handlers for ``GPEmulator`` subclasses.

Three handlers cover the three ``EmulatorUpdate`` op types. All
three are typed against the abstract :class:`GPEmulator` base, so
they automatically dispatch for any concrete subclass —
``DSPGPEmulator`` today, future tinygp / ProbPipe-backed emulators
once they migrate to inherit from ``GPEmulator``.

Handlers are backend-agnostic because the heavy lifting lives on the
base class:

- ``GPEmulator.condition_on`` does the rank-one append (delegating to
  ``cache.append_rows``).
- ``ZScoreScaler.rescaled_by`` is the scale-invariance trick that
  makes a pure ``RescaleOutputs`` a cache-invariant operation.
- ``GPEmulator._replace`` rebuilds the emulator instance with new
  fields.

The three handlers:

- ``_GPAppendRowsHandler``: ``AppendRows(X_new, Y_new)`` →
  ``emulator.condition_on(X_new, Y_new)``. O(n²m + m³).

- ``_GPRescaleOutputsHandler``: ``RescaleOutputs(factor=β)`` →
  rescale the y-scaler by β. The cache stays unchanged because
  z-scoring is scale-invariant — see ``ZScoreScaler.rescaled_by``
  for the math. O(1).

- ``_GPRescaleThenAppendHandler``: ``RescaleThenAppend(factor=β,
  X_new, Y_new)`` → rescale the y-scaler (free), then standardize
  ``Y_new`` (already in the new state's units per the plan's
  contract) with the new scaler and rank-one append. O(n²m + m³),
  same as ``AppendRows``.

All three require an existing fit (``_predict_cache`` populated) and
a positive ``factor``; otherwise they report infeasible and dispatch
falls back to refit.

Module import (which happens on ``import sabi.emulators``)
unconditionally registers the three handlers with
``emulator_update_registry``. No backend-specific dependency.
"""

from __future__ import annotations

import jax.numpy as jnp
from probpipe.core._registry import MethodInfo

from sabi.emulators.dispatch import (
    EmulatorUpdateMethod,
    emulator_update_registry,
)
from sabi.emulators.gp import GPEmulator
from sabi.emulators.updates import AppendRows, RescaleOutputs, RescaleThenAppend


class _GPAppendRowsHandler(EmulatorUpdateMethod):
    """Cheap-path ``AppendRows`` handler for any ``GPEmulator`` subclass.

    Delegates to ``GPEmulator.condition_on``, which performs the
    rank-one (block) Cholesky update via ``cache.append_rows`` —
    see the per-backend cache for the math. Frozen hyperparameters
    are intentional: this is the right cheap path *only* when
    callers have decided the existing fit is good enough for the
    augmented dataset. The dispatcher's natural fallback
    (``factory().fit(X_full, Y_full)``) takes over when the loop
    decides a refit is warranted.

    Feasibility: requires the emulator to have an existing fit
    (i.e., a populated ``_predict_cache``). An unfitted emulator
    falls back to refit, which is the only correct option there.
    """

    @property
    def name(self) -> str:
        return "gp_append_rows_chol_update"

    def supported_types(self) -> tuple[type, ...]:
        return (GPEmulator,)

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

    def execute(self, emulator: GPEmulator, plan: AppendRows) -> GPEmulator:
        return emulator.condition_on(plan.X_new, plan.Y_new)


class _GPRescaleOutputsHandler(EmulatorUpdateMethod):
    r"""Cheap-path ``RescaleOutputs`` handler for any ``GPEmulator``.

    A pure rescale ``Y_b = β · Y_a`` is essentially a no-op on the
    cache because z-scoring is scale-invariant:

    .. math::

        \text{loc}_b = \beta \cdot \text{loc}_a, \quad
        \text{scale}_b = \beta \cdot \text{scale}_a
        \;\Longrightarrow\;
        \frac{\beta Y_a - \text{loc}_b}{\text{scale}_b}
        = \frac{Y_a - \text{loc}_a}{\text{scale}_a}.

    Standardized y is unchanged, so the cached ``L_sigma``,
    ``alpha``, ``Xs_train``, ``Ys_train``, and ``noise_var`` are all
    invariant. The only thing that needs updating is the y-scaler —
    predictions in the original space then come out at the new
    state's units automatically: ``mean_b = β·mean_a``, ``var_b =
    β²·var_a``.

    Cost: O(1) — construct a new y-scaler.
    """

    @property
    def name(self) -> str:
        return "gp_rescale_outputs_yscaler_only"

    def supported_types(self) -> tuple[type, ...]:
        return (GPEmulator,)

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

    def execute(self, emulator: GPEmulator, plan: RescaleOutputs) -> GPEmulator:
        if plan.factor == 1.0:
            return emulator  # exact no-op
        new_y_scaler = emulator._y_scaler.rescaled_by(plan.factor)
        return emulator._replace(_y_scaler=new_y_scaler)


class _GPRescaleThenAppendHandler(EmulatorUpdateMethod):
    r"""Cheap-path ``RescaleThenAppend`` handler for any ``GPEmulator``.

    Composite of the two single-op handlers, applied in order so the
    new rows are standardized in the new state's coordinate system:

    1. **Rescale step** (O(1)): build ``y_scaler_b`` with
       ``loc_b = β·loc_a, scale_b = β·scale_a``. Cache untouched
       (standardized y is invariant under uniform rescale).
    2. **Append step** (O(n²m + m³)): standardize ``Y_new`` (already
       given at state b per the ``RescaleThenAppend`` contract) with
       ``y_scaler_b``, then call ``cache.append_rows`` for the
       block-Cholesky update.

    Total cost matches a single ``condition_on`` (the rescale step
    is free) — so a tempering round with new rows costs the same as
    a no-tempering round with new rows.

    Note: the rescale step assumes a uniform multiplicative rescale
    of Y (the only shape ``RescaleOutputs`` carries). State-shaped
    transforms with shift terms or per-coordinate scales would need
    a different cheap path.
    """

    @property
    def name(self) -> str:
        return "gp_rescale_then_append_chol_update"

    def supported_types(self) -> tuple[type, ...]:
        return (GPEmulator,)

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
        self, emulator: GPEmulator, plan: RescaleThenAppend
    ) -> GPEmulator:
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
    _GPAppendRowsHandler(),
    _GPRescaleOutputsHandler(),
    _GPRescaleThenAppendHandler(),
):
    if _h.name not in emulator_update_registry._name_index:
        emulator_update_registry.register(_h)
