"""`TemperingScheme` — family of intermediate target distributions.

A `TemperingScheme` plays two roles per round:

1. ``intermediate_target(base, state)`` produces an
   :class:`IntermediateTarget` carrying the per-state ``output_transform``
   (drives ``Y_train`` derivation from cached ``Y_raw``).
2. ``intermediate_decomposition(base_decomposition, state)`` produces
   the per-state effective :class:`DensityDecomposition` — typically a
   ``Map`` composition on the base decomposition's ``link`` / ``shift``.

"Tempering" here is the general bridging abstraction; the name is kept
because likelihood tempering is the current concrete instance. See
``docs/design.md §4.11`` and ``docs/tempering.md`` for the two-axes
(target / form) decomposition and worked examples.

Concrete schemes:

- `NoTempering`: identity on both axes. ``intermediate_target`` wraps
  ``state`` and an identity ``output_transform``;
  ``intermediate_decomposition`` returns the base decomposition
  unchanged. Default.
- `LikelihoodTemperingViaForm` and `LikelihoodTemperingViaTarget` (in
  ``sabi.tempering.likelihood``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from probpipe.core._numeric_record_distribution import NumericRecordDistribution

from sabi.density_decomposition import DensityDecomposition
from sabi.tempering.output_transform import Identity

if TYPE_CHECKING:
    from sabi.tempering.output_transform import OutputTransform


@dataclass(frozen=True)
class IntermediateTarget:
    """Per-state metadata produced by a tempering scheme.

    Carries the schedule's ``state`` and the
    :class:`~sabi.tempering.output_transform.OutputTransform` that
    derives ``Y_train`` for this intermediate from cached ``Y_raw``.
    Math identity (event_shape, support, analytical density) lives on
    the base ``NumericRecordDistribution`` passed into
    ``intermediate_target(base, state)`` — this dataclass intentionally
    does not duplicate it. The per-state effective
    :class:`DensityDecomposition` lives separately, produced by
    ``intermediate_decomposition``.
    """

    state: Any
    output_transform: "OutputTransform"


class InvarianceFlags(NamedTuple):
    """Per-axis invariance flags returned by `TemperingScheme.invariance`.

    Attributes:
        target_map: True iff the per-state effective target map is
            unchanged (so the emulator's training data is unchanged).
        form: True iff the per-state effective ``DensityDecomposition``'s
            ``link`` / ``shift`` is unchanged.
        both: True iff both axes are invariant.
    """

    target_map: bool
    form: bool
    both: bool


class TemperingScheme(ABC):
    """A family of intermediate target distributions indexed by state.

    Subclasses implement:

    - :meth:`intermediate_target` — produces the per-state
      :class:`IntermediateTarget` (state + output_transform metadata).
    - :meth:`intermediate_decomposition` — produces the per-state
      effective :class:`DensityDecomposition` from the base decomposition.

    `is_invariant_target_map` and `is_invariant_form` are optimization
    hints used by the loop. Conservative defaults
    (``state_a == state_b``) work for any scheme; subclasses can
    override with stronger guarantees.
    """

    @abstractmethod
    def intermediate_target(
        self,
        base: NumericRecordDistribution,
        state: Any,
    ) -> IntermediateTarget:
        """Build the `IntermediateTarget` at ``state``."""

    @abstractmethod
    def intermediate_decomposition(
        self,
        base: DensityDecomposition,
        state: Any,
    ) -> DensityDecomposition:
        """Produce the per-state effective ``DensityDecomposition``.

        Subclasses compose ``Map``s on ``base.link`` / ``base.shift`` to
        encode the per-state intermediate density. ``target_map`` and
        ``output_shape`` typically pass through unchanged: bridging
        modifies how the emulator's output composes into log-density,
        not what the emulator approximates.
        """

    def is_invariant_target_map(
        self,
        state_a: Any,
        state_b: Any,
    ) -> bool:
        """True iff the effective target map is unchanged from a to b."""
        return state_a == state_b

    def is_invariant_form(self, state_a: Any, state_b: Any) -> bool:
        """True iff the effective ``DensityDecomposition``'s ``link`` /
        ``shift`` is unchanged from a to b.
        """
        return state_a == state_b

    def invariance(self, state_a: Any, state_b: Any) -> InvarianceFlags:
        """Return per-axis invariance flags between two states."""
        target_map = self.is_invariant_target_map(state_a, state_b)
        form = self.is_invariant_form(state_a, state_b)
        return InvarianceFlags(
            target_map=target_map,
            form=form,
            both=target_map and form,
        )


class NoTempering(TemperingScheme):
    """Identity tempering: the intermediate equals the base.

    ``intermediate_target`` returns an :class:`IntermediateTarget`
    with the identity ``output_transform``;
    ``intermediate_decomposition`` returns the base decomposition
    unchanged. Both axes are invariant under state changes.
    """

    def intermediate_target(
        self,
        base: NumericRecordDistribution,  # noqa: ARG002 — base unused under NoTempering
        state: Any,
    ) -> IntermediateTarget:
        return IntermediateTarget(state=state, output_transform=Identity())

    def intermediate_decomposition(
        self,
        base: DensityDecomposition,
        state: Any,  # noqa: ARG002 — state unused under NoTempering
    ) -> DensityDecomposition:
        return base

    def is_invariant_target_map(
        self,
        state_a: Any,  # noqa: ARG002
        state_b: Any,  # noqa: ARG002
    ) -> bool:
        return True

    def is_invariant_form(
        self,
        state_a: Any,  # noqa: ARG002
        state_b: Any,  # noqa: ARG002
    ) -> bool:
        return True
