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
  the base with ``state=state``; ``intermediate_decomposition`` returns
  the base decomposition unchanged. Default.
- `LikelihoodTemperingViaForm` and `LikelihoodTemperingViaTarget` (in
  ``sabi.tempering.likelihood``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, NamedTuple

from sabi.density_decomposition import DensityDecomposition
from sabi.target_distribution import IntermediateTarget, TargetDistribution
from sabi.tempering.output_transform import Identity


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
      :class:`IntermediateTarget` (carrying ``output_transform``).
    - :meth:`intermediate_decomposition` — produces the per-state
      effective :class:`DensityDecomposition` from the base decomposition.

    `is_invariant_target_map` and `is_invariant_form` are
    optimization hints used by the loop to skip redundant emulator
    refits / decomposition rebuilds when consecutive states yield the
    same effective target map / link+shift. Conservative defaults
    (``state_a == state_b``) work for any scheme; subclasses can
    override with stronger guarantees (e.g., the no-op scheme returns
    True regardless of state).
    """

    @abstractmethod
    def intermediate_target(
        self,
        base: TargetDistribution,
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
        encode the per-state intermediate density. ``target_single`` and
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
        """Return per-axis invariance flags between two states.

        Combines the two `is_invariant_*` checks plus a ``both`` flag
        for the common "nothing changed" branch in the loop.
        """
        target_map = self.is_invariant_target_map(state_a, state_b)
        form = self.is_invariant_form(state_a, state_b)
        return InvarianceFlags(
            target_map=target_map,
            form=form,
            both=target_map and form,
        )


class NoTempering(TemperingScheme):
    """Identity tempering: the intermediate equals the base.

    For any state, ``intermediate_target`` returns an
    :class:`IntermediateTarget` whose ``output_transform`` is the
    identity; ``intermediate_decomposition`` returns the base
    decomposition unchanged. Both axes are invariant under state changes.
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
            output_transform=Identity(),
        )

    def intermediate_decomposition(
        self,
        base: DensityDecomposition,
        state: Any,  # noqa: ARG002 — state is unused for the no-op scheme
    ) -> DensityDecomposition:
        return base

    def is_invariant_target_map(
        self,
        state_a: Any,
        state_b: Any,
    ) -> bool:
        return True

    def is_invariant_form(self, state_a: Any, state_b: Any) -> bool:
        return True
