"""Tests for ``TemperingScheme``, ``NoTempering``, and the per-state
``IntermediateTarget`` they produce."""

import jax.numpy as jnp
import pytest

from sabi.density_decomposition import LogProbTarget
from sabi.maps import Identity as MapIdentity
from sabi.problems.base import Problem
from sabi.target_distribution import IntermediateTarget, TargetDistribution
from sabi.tempering.base import NoTempering, TemperingScheme
from sabi.tempering.schedule import FixedSchedule, TemperingSchedule, UntemperedSchedule

from tests._targets import OpaqueTarget, QuadraticTarget, box_support, quadratic_log_density


# -------------------------------------------------------------------------
# NoTempering
# -------------------------------------------------------------------------


def test_no_tempering_returns_intermediate_target_subclass():
    target = QuadraticTarget()
    intermediate = NoTempering().intermediate_target(target, state=None)
    assert isinstance(intermediate, IntermediateTarget)
    assert isinstance(intermediate, TargetDistribution)


def test_no_tempering_intermediate_decomposition_returns_base_unchanged():
    target = QuadraticTarget()
    decomp = LogProbTarget(target)
    out = NoTempering().intermediate_decomposition(decomp, state=0.5)
    assert out is decomp


def test_no_tempering_records_state_but_ignores_it():
    target = QuadraticTarget()
    intermediate_a = NoTempering().intermediate_target(target, state="a")
    intermediate_b = NoTempering().intermediate_target(target, state=42.0)
    assert intermediate_a.state == "a"
    assert intermediate_b.state == 42.0


def test_no_tempering_output_transform_is_identity():
    target = QuadraticTarget()
    intermediate = NoTempering().intermediate_target(target, state=None)
    X = jnp.asarray([[0.0, 0.0], [1.0, -0.5]])
    Y_raw = quadratic_log_density(X)
    out = intermediate.output_transform(intermediate.state, X, Y_raw)
    assert jnp.allclose(out, Y_raw)


def test_no_tempering_invariance_flags_are_true():
    """Both axes are invariant under any state change."""
    scheme = NoTempering()
    assert scheme.is_invariant_target_map("a", "b")
    assert scheme.is_invariant_target_map(0.1, 0.9)
    assert scheme.is_invariant_form("a", "b")


def test_intermediate_target_is_metadata_only():
    """``IntermediateTarget`` is metadata-only — no analytical density.
    The per-state effective decomposition lives elsewhere."""
    target = QuadraticTarget()
    intermediate = NoTempering().intermediate_target(target, state=None)
    # Intermediate has no `_unnormalized_log_prob` of its own — it does
    # not satisfy SupportsUnnormalizedLogProb (the base TargetDistribution
    # is abstract with respect to the density).
    from probpipe.core.protocols import SupportsUnnormalizedLogProb
    assert not isinstance(intermediate, SupportsUnnormalizedLogProb)


def test_no_tempering_intermediate_propagates_opaque_target():
    """An intermediate built from an opaque target is also opaque
    (metadata-only)."""
    target = OpaqueTarget()
    intermediate = NoTempering().intermediate_target(target, state=None)
    from probpipe.core.protocols import SupportsUnnormalizedLogProb
    assert not isinstance(intermediate, SupportsUnnormalizedLogProb)


# -------------------------------------------------------------------------
# TemperingScheme ABC
# -------------------------------------------------------------------------


def test_tempering_scheme_default_invariance_uses_equality():
    """Default ``is_invariant_*`` returns True iff states are equal."""

    class _NullScheme(TemperingScheme):
        def intermediate_target(self, base, state):
            return NoTempering().intermediate_target(base, state)

        def intermediate_decomposition(self, base, state):
            return base

    scheme = _NullScheme()
    assert scheme.is_invariant_target_map(0.5, 0.5)
    assert not scheme.is_invariant_target_map(0.5, 0.6)
    assert scheme.is_invariant_form(None, None)
    assert not scheme.is_invariant_form(None, "a")


# -------------------------------------------------------------------------
# Schedule
# -------------------------------------------------------------------------


def test_untempered_schedule_returns_none_state_and_terminal():
    sched = UntemperedSchedule()
    state, is_terminal_state = sched.at(0)
    assert state is None
    assert is_terminal_state is True


def test_fixed_schedule_iterates_and_marks_terminal():
    sched = FixedSchedule(states=(0.1, 0.5, 1.0))
    assert sched.at(0) == (0.1, False)
    assert sched.at(1) == (0.5, False)
    assert sched.at(2) == (1.0, True)
    assert sched.at(42) == (1.0, True)


def test_fixed_schedule_accepts_non_scalar_states():
    sched = FixedSchedule(states=((0, 1), (0, 1, 2), (0, 1, 2, 3)))
    state, is_terminal_state = sched.at(1)
    assert state == (0, 1, 2)
    assert is_terminal_state is False


def test_fixed_schedule_rejects_empty():
    with pytest.raises(ValueError):
        FixedSchedule(states=())


# -------------------------------------------------------------------------
# TemperingSchedule.terminal_state default fallback
# -------------------------------------------------------------------------


def test_terminal_state_default_success_path_for_converging_subclass():
    class _ConvergingSchedule(TemperingSchedule):
        def at(self, round_idx: int):
            if round_idx == 0:
                return 0.5, False
            return 1.0, True

    sched = _ConvergingSchedule()
    assert sched.terminal_state() == 1.0


def test_terminal_state_default_raise_path_for_nonconverging_subclass():
    class _NeverTerminalSchedule(TemperingSchedule):
        def at(self, round_idx: int):
            return float(round_idx), False

    sched = _NeverTerminalSchedule()
    with pytest.raises(NotImplementedError, match="did not"):
        sched.terminal_state()


# -------------------------------------------------------------------------
# Loop integration: NoTempering preserves untempered semantics
# -------------------------------------------------------------------------


def test_no_tempering_decomposition_link_is_identity_map():
    target = QuadraticTarget()
    decomp = LogProbTarget(target)
    out = NoTempering().intermediate_decomposition(decomp, state=None)
    assert isinstance(out.link, MapIdentity)


def test_problem_target_distribution_round_trips_via_no_tempering():
    """Wraps a target in a Problem; NoTempering's intermediate carries
    matching support and input_shape (math identity preserved)."""
    target = QuadraticTarget()
    problem = Problem(target_distribution=target, name="quad_problem")
    intermediate = NoTempering().intermediate_target(
        problem.target_distribution, state=None
    )
    assert intermediate.input_shape == target.input_shape
    assert intermediate.support is target.support
