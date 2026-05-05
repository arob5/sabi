"""`TemperingScheme` — family of intermediate target distributions.

A `TemperingScheme` maps each state from a `TemperingSchedule` to an
`IntermediateTarget` (a `TargetDistribution` carrying ``f_state``,
``phi_state``, and an ``output_transform`` adapter for cheap derivation
of training data from cached raw evaluations).

"Tempering" here is the general bridging abstraction; the name is kept
because likelihood tempering is the current concrete instance. See
``docs/design.md`` §4.11 and ``docs/tempering.md`` for the two-axes
(target / form) decomposition and worked examples.

Concrete schemes:

- `NoTempering`: identity on both axes. The intermediate equals the
  base target wrapped with ``state=state``. Default.
- `LikelihoodTemperingViaForm` and `LikelihoodTemperingViaTarget` (in
  ``sabi.tempering.likelihood``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, NamedTuple

from sabi.target_distribution import IntermediateTarget, TargetDistribution
from sabi.tempering.output_transform import Identity


class InvarianceFlags(NamedTuple):
    """Per-axis invariance flags returned by `TemperingScheme.invariance`.

    Attributes:
        target_map: True iff `f_state_a == f_state_b` (so the
            emulator's training data is unchanged).
        form: True iff `phi_state_a == phi_state_b` (so the round's
            log-density form is unchanged).
        both: True iff both axes are invariant. Convenience for the
            common "is anything different at all?" check.
    """

    target_map: bool
    form: bool
    both: bool


class TemperingScheme(ABC):
    """A family of intermediate target distributions indexed by state.

    Subclasses implement :meth:`intermediate_target`, which produces an
    `IntermediateTarget` given the base `TargetDistribution` and a
    state from the schedule.

    `is_invariant_target_map` and `is_invariant_form` are
    optimization hints used by the loop to skip redundant emulator
    refits / form rebuilds when consecutive states yield the same
    `f_state` or `phi_state`. Conservative defaults
    (``state_a == state_b``) work for any scheme; subclasses can
    override with stronger guarantees (e.g., the no-op scheme returns
    True regardless of state).

    :meth:`invariance` is a convenience that bundles both per-axis
    flags plus a combined ``both`` flag.
    """

    @abstractmethod
    def intermediate_target(
        self,
        base: TargetDistribution,
        state: Any,
    ) -> IntermediateTarget:
        """Build the `IntermediateTarget` at ``state``."""

    def is_invariant_target_map(
        self,
        state_a: Any,
        state_b: Any,
    ) -> bool:
        """True iff ``f_state_a == f_state_b`` (so the emulator's
        training data is unchanged under a state change from a to b).
        """
        return state_a == state_b

    def is_invariant_form(self, state_a: Any, state_b: Any) -> bool:
        """True iff ``phi_state_a == phi_state_b``."""
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
    """Identity tempering: the intermediate target equals the base.

    For any state, returns an `IntermediateTarget` whose
    ``target_map`` and ``log_density_form`` are the base's
    (unchanged) and whose ``output_transform`` is the identity. The
    state is recorded but has no effect on the math.

    Both axes are invariant under state changes.
    """

    def intermediate_target(
        self,
        base: TargetDistribution,
        state: Any,
    ) -> IntermediateTarget:
        return IntermediateTarget(
            name=base.name,
            input_shape=base.input_shape,
            output_shape=base.output_shape,
            target_single=base.target_single,
            log_density_form=base.log_density_form,
            state=state,
            output_transform=Identity(),
            base_target_map=base.target_map,
            prior=base.prior,
        )

    def is_invariant_target_map(
        self,
        state_a: Any,
        state_b: Any,
    ) -> bool:
        return True

    def is_invariant_form(self, state_a: Any, state_b: Any) -> bool:
        return True
