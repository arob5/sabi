r"""Likelihood-tempering schemes.

Both schemes encode the same intermediate density family but place the
:math:`\beta` factor at different points in the algorithm (see
``docs/tempering.md`` for the full case analysis):

- :class:`LikelihoodTemperingViaForm`: emulator target ``target_map`` is
  unchanged across rounds; the per-state effective decomposition
  rescales ``link`` by :math:`\beta`. The emulator can be reused
  across all states with no refit (``is_invariant_target_map`` returns
  True). Compatible with any base ``link`` / ``shift`` shape.
- :class:`LikelihoodTemperingViaTarget`: emulator target ``target_map``
  is rescaled by :math:`\beta` (via the ``output_transform``); the
  decomposition is unchanged across rounds. Restricted to
  ``link = Identity`` base decompositions.

The state PyTree is the inverse temperature :math:`\beta \in [0, 1]`
(scalar). Pair with ``FixedSchedule(states=(0.1, 0.5, 1.0))`` or similar.

Geometric-bridge interaction
----------------------------

When the base decomposition's ``shift`` is ``None`` (the emulator emits
the full log-density and there is no separate prior shift), the natural
bridge is the *geometric* bridge
:math:`\pi_\beta \propto \pi_0^{1-\beta} \cdot \pi_\mathrm{target}^\beta`.
The geometric bridge requires an external ``initial`` distribution
:math:`\pi_0` (the bridge's left endpoint).

To preserve today's geometric-bridge semantics under the new class
layout — without preempting the bridging-rename refactor that will
introduce a dedicated ``GeometricBridge`` —
:class:`LikelihoodTemperingViaForm` accepts an optional
``initial: Distribution`` field and branches internally on the base
shift shape (``None`` -> geometric-bridge case).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array
from probpipe.core._distribution_base import Distribution
from probpipe.core.constraints import Constraint

from sabi.density_decomposition import DensityDecomposition
from sabi.maps import Affine, Identity, LogProb, Map
from sabi.target_distribution import IntermediateTarget, TargetDistribution
from sabi.tempering.base import TemperingScheme
from sabi.tempering.output_transform import Identity as IdentityTransform
from sabi.tempering.output_transform import Rescale as RescaleTransform


# ---------------------------------------------------------------------------
# Private DensityDecomposition subclass returned by LikelihoodTemperingViaForm
# ---------------------------------------------------------------------------


class _LikelihoodTemperedDecomposition(DensityDecomposition):
    r"""Per-state decomposition under :class:`LikelihoodTemperingViaForm`.

    Rescales the base ``link`` by :math:`\beta`. Shift handling
    branches on ``base.shift``:

    - **Likelihood-tempering case** (``base.shift`` is non-``None``):
      shift unchanged. Encodes
      :math:`\pi_\beta \propto \pi_0 \cdot L^\beta`.
    - **Geometric-bridge case** (``base.shift`` is ``None``): requires
      an ``initial`` distribution. New shift becomes
      ``Affine(slope=(1-β)) @ LogProb(initial)``. Encodes
      :math:`\pi_\beta \propto \pi_0^{1-\beta} \cdot \pi_\mathrm{target}^\beta`.

    Used internally; users construct via
    ``LikelihoodTemperingViaForm.intermediate_decomposition``.
    """

    def __init__(
        self,
        base: DensityDecomposition,
        beta: Array,
        *,
        initial: Distribution | None = None,
    ):
        self._base = base
        self._beta = jnp.asarray(beta)
        self._initial = initial
        super().__init__(
            name=f"{base.name}_tempered_beta{float(self._beta):.3f}",
            input_shape=base.input_shape,
            support=base.support,
        )

    def target_map(self, x: Array) -> Array:
        return self._base.target_map(x)

    @property
    def link(self) -> Map:
        return Affine(slope=self._beta, intercept=jnp.asarray(0.0)) @ self._base.link

    @property
    def shift(self) -> Map | None:
        if self._base.shift is not None:
            return self._base.shift
        # Geometric-bridge case: requires an external initial.
        if self._initial is None:
            raise ValueError(
                "LikelihoodTemperingViaForm: base decomposition has "
                "`shift = None` (the geometric-bridge case), which "
                "requires an external initial distribution. Pass "
                "`LikelihoodTemperingViaForm(initial=...)` with the "
                "bridge's left-endpoint distribution. (Will migrate to "
                "a dedicated `GeometricBridge` class in the bridging-"
                "rename refactor.)"
            )
        return Affine(
            slope=jnp.asarray(1.0) - self._beta, intercept=jnp.asarray(0.0)
        ) @ LogProb(self._initial)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._base.output_shape

    @property
    def output_constraint(self) -> Constraint:
        return self._base.output_constraint


# ---------------------------------------------------------------------------
# Schemes
# ---------------------------------------------------------------------------


class LikelihoodTemperingViaForm(TemperingScheme):
    r"""Likelihood tempering with the emulator target unchanged.

    The per-state effective decomposition rescales the base ``link`` by
    :math:`\beta`. The emulator's ``target_map`` is unchanged across
    rounds, so ``is_invariant_target_map`` returns True and the loop
    can reuse the same fitted emulator across states.

    The shift handling depends on the base decomposition's shape:

    - **Likelihood-tempering case** (``base.shift`` non-``None``):
      shift passes through unchanged.
    - **Geometric-bridge case** (``base.shift`` is ``None``): requires
      the optional ``initial`` field. New shift becomes
      ``Affine(slope=(1-β)) @ LogProb(initial)``. Will migrate to a
      dedicated ``GeometricBridge`` when the bridging-rename refactor
      lands.

    The state PyTree is the scalar :math:`\beta \in [0, 1]`.
    """

    def __init__(self, *, initial: Distribution | None = None):
        self._initial = initial

    @property
    def initial(self) -> Distribution | None:
        return self._initial

    def intermediate_target(
        self,
        base: TargetDistribution,
        state: Any,
    ) -> IntermediateTarget:
        return IntermediateTarget(
            name=base.name,
            input_shape=base.input_shape,
            support=base.support,
            state=state,
            output_transform=IdentityTransform(),
        )

    def intermediate_decomposition(
        self,
        base: DensityDecomposition,
        state: Any,
    ) -> DensityDecomposition:
        return _LikelihoodTemperedDecomposition(
            base, jnp.asarray(state), initial=self._initial
        )

    def is_invariant_target_map(self, state_a: Any, state_b: Any) -> bool:
        return True  # emulator target is invariant under state changes

    def is_invariant_form(self, state_a: Any, state_b: Any) -> bool:
        return state_a == state_b


class LikelihoodTemperingViaTarget(TemperingScheme):
    r"""Likelihood tempering with the emulator target rescaled by :math:`\beta`.

    The per-state effective decomposition is unchanged from the base;
    the emulator is fit on :math:`Y_t = \beta_t \cdot Y_\mathrm{raw}` per
    round. The ``output_transform`` rescales ``Y_raw`` by :math:`\beta`.

    Restricted to base decompositions with ``link = Identity`` — the
    case where the emulator's :math:`y` is the log-likelihood
    directly. Raises at ``intermediate_decomposition`` time when
    applied to non-``Identity`` links.

    The state PyTree is the scalar :math:`\beta \in [0, 1]`.
    """

    def intermediate_target(
        self,
        base: TargetDistribution,
        state: Any,
    ) -> IntermediateTarget:
        return IntermediateTarget(
            name=base.name,
            input_shape=base.input_shape,
            support=base.support,
            state=state,
            output_transform=RescaleTransform(),
        )

    def intermediate_decomposition(
        self,
        base: DensityDecomposition,
        state: Any,  # noqa: ARG002 — Y_train carries the beta factor; decomposition is unchanged
    ) -> DensityDecomposition:
        if not isinstance(base.link, Identity):
            raise ValueError(
                "LikelihoodTemperingViaTarget requires `base.link` to be "
                "`Identity` (so y is log-likelihood and scaling y by "
                f"beta is equivalent to scaling the link by beta). Got "
                f"link={type(base.link).__name__}. Use "
                "`LikelihoodTemperingViaForm` for non-Identity links."
            )
        return base

    def is_invariant_target_map(self, state_a: Any, state_b: Any) -> bool:
        return state_a == state_b

    def is_invariant_form(self, state_a: Any, state_b: Any) -> bool:
        return True  # decomposition is invariant under state changes
