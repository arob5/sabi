r"""Likelihood-tempering schemes.

Both schemes encode the same intermediate density family but place the
:math:`\beta` factor at different points in the algorithm (see
``docs/tempering.md`` for the full case analysis):

- :class:`LikelihoodTemperingViaForm`: emulator target ``target_single``
  is unchanged across rounds; the per-state effective decomposition
  rescales ``link`` by :math:`\beta`. The emulator can be reused
  across all states with no refit (``is_invariant_target_map`` returns
  True). Compatible with any base ``link`` / ``shift`` shape — the
  tempered link is always ``Affine(slope=beta) @ base.link``.
- :class:`LikelihoodTemperingViaTarget`: emulator target
  ``target_single``  is rescaled by :math:`\beta` (via the
  ``output_transform``); the decomposition is unchanged across rounds.
  The emulator must be refit (or rescaled — see issue #4 for
  cheap-update dispatch) per round. Restricted to ``link = Identity``
  base decompositions (the case where the emulator's :math:`y` is
  log-likelihood directly).

The state PyTree is the inverse temperature :math:`\beta \in [0, 1]`
(scalar). Pair with ``FixedSchedule(states=(0.1, 0.5, 1.0))`` or similar.

Geometric-bridge interaction
----------------------------

When the base decomposition has ``shift = Constant(0)`` — i.e., the
emulator emits the full log-density and there is no separate
modeling-prior shift — the natural bridge is the *geometric* bridge
:math:`\pi_\beta \propto \pi_0^{1-\beta} \cdot \pi_\mathrm{target}^\beta`,
not likelihood tempering. The geometric bridge requires an external
``initial`` distribution :math:`\pi_0` (the bridge's left endpoint),
which the base decomposition does not carry.

To preserve today's ``_IdentityTempered`` behavior under the new
class layout — without preempting the bridging-rename refactor (#YY)
that introduces a dedicated ``GeometricBridge`` —
:class:`LikelihoodTemperingViaForm` accepts an optional ``initial:
Distribution`` field and branches internally on the base shift shape.
The branching will migrate cleanly to ``GeometricBridge`` when that
class lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
from jax import Array
from probpipe.core._distribution_base import Distribution

from sabi.density_decomposition import DensityDecomposition, ScalarConstant
from sabi.maps import Affine, Constant, Identity, LogProb
from sabi.target_distribution import IntermediateTarget, TargetDistribution
from sabi.tempering.base import TemperingScheme
from sabi.tempering.output_transform import Identity as IdentityTransform
from sabi.tempering.output_transform import Rescale as RescaleTransform


def _is_zero_shift(shift) -> bool:
    """True if ``shift`` is the geometric-bridge sentinel (a scalar zero).

    Detects either :class:`sabi.maps.Constant` or
    :class:`sabi.density_decomposition.ScalarConstant` carrying a value
    that's pointwise-zero. The latter is the default ``shift`` on a
    fresh ``DensityDecomposition``.
    """
    if not isinstance(shift, (Constant, ScalarConstant)):
        return False
    return bool(jnp.all(jnp.asarray(shift.c) == 0))


@dataclass(frozen=True)
class LikelihoodTemperingViaForm(TemperingScheme):
    r"""Likelihood tempering with the emulator target unchanged.

    The per-state effective decomposition rescales the base ``link`` by
    :math:`\beta` (via ``Affine(slope=beta) @ base.link``). The
    emulator's ``target_single`` is unchanged across rounds, so
    ``is_invariant_target_map`` returns True and the loop can reuse
    the same fitted emulator across states.

    The shift handling depends on the base decomposition's shape:

    - **Likelihood-tempering case** (``base.shift`` non-zero, typically
      ``LogProb(modeling_prior)``): shift is unchanged. Encodes
      :math:`\pi_\beta \propto \pi_0 \cdot L^\beta`.
    - **Geometric-bridge case** (``base.shift`` is ``Constant(0)``):
      requires the optional ``initial`` field. New shift becomes
      ``Affine(slope=(1-beta)) @ LogProb(initial)``. Encodes
      :math:`\pi_\beta \propto \pi_0^{1-\beta} \cdot \pi_\mathrm{target}^\beta`.
      Mirrors today's ``_IdentityTempered``; will migrate to a dedicated
      ``GeometricBridge`` when the bridging-rename refactor lands.

    The state PyTree is the scalar :math:`\beta \in [0, 1]`.
    """

    initial: Distribution | None = None

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
            unnormalized_log_prob=base._analytical_unnormalized_log_prob,
        )

    def intermediate_decomposition(
        self,
        base: DensityDecomposition,
        state: Any,
    ) -> DensityDecomposition:
        beta = jnp.asarray(state)
        new_link = Affine(slope=beta, intercept=jnp.asarray(0.0)) @ base.link
        if _is_zero_shift(base.shift):
            # Geometric-bridge case: needs the external initial distribution.
            if self.initial is None:
                raise ValueError(
                    "LikelihoodTemperingViaForm: base decomposition has "
                    "`shift = Constant(0)` (the geometric-bridge case), "
                    "which requires an external initial distribution. "
                    "Pass `LikelihoodTemperingViaForm(initial=...)` with "
                    "the bridge's left-endpoint distribution. (Will "
                    "migrate to a dedicated `GeometricBridge` class in "
                    "the bridging-rename refactor.)"
                )
            new_shift = Affine(
                slope=jnp.asarray(1.0) - beta,
                intercept=jnp.asarray(0.0),
            ) @ LogProb(self.initial)
        else:
            new_shift = base.shift
        return DensityDecomposition(
            target_single=base.target_single,
            output_shape=base.output_shape,
            link=new_link,
            shift=new_shift,
            constraint=base.constraint,
        )

    def is_invariant_target_map(self, state_a: Any, state_b: Any) -> bool:
        return True  # emulator target is invariant under state changes

    def is_invariant_form(self, state_a: Any, state_b: Any) -> bool:
        return state_a == state_b


class LikelihoodTemperingViaTarget(TemperingScheme):
    r"""Likelihood tempering with the emulator target rescaled by :math:`\beta`.

    The per-state effective decomposition is unchanged from the base;
    the emulator is fit on :math:`Y_t = \beta_t \cdot Y_\mathrm{raw}` per
    round (or rescaled cheaply — issue #4). The
    ``output_transform`` rescales ``Y_raw`` by :math:`\beta`.

    Restricted to base decompositions with ``link = Identity()`` — the
    case where the emulator's :math:`y` is the log-likelihood
    directly. With non-``Identity`` links (e.g., ``GaussianLogLik``,
    ``LogSoftplus``), scaling :math:`y` by :math:`\beta` does NOT
    correspond to scaling the link's output by :math:`\beta`, so the
    optimization is unsound. Raises at
    ``intermediate_decomposition`` time when applied to non-``Identity``
    links; use :class:`LikelihoodTemperingViaForm` for those cases.

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
            unnormalized_log_prob=base._analytical_unnormalized_log_prob,
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


# Output transforms (Identity / Rescale) live in
# `sabi.tempering.output_transform` as value-typed `OutputTransform`
# subclasses; both schemes above import them via aliases at the top of
# this module to avoid colliding with `sabi.maps.Identity`.
