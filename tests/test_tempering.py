"""Tests for `TemperingScheme`, `NoTempering`, and the per-state
`IntermediateTarget` they produce.
"""

import jax.numpy as jnp
import pytest

from sabi._probpipe_compat import independent_uniform
from sabi.density_decomposition import DensityDecomposition
from sabi.maps import Identity as MapIdentity
from sabi.problems.base import Problem
from sabi.target_distribution import IntermediateTarget, TargetDistribution
from sabi.tempering.base import NoTempering, TemperingScheme
from sabi.tempering.schedule import FixedSchedule, TemperingSchedule, UntemperedSchedule


def _box_support():
    return independent_uniform(
        low=jnp.full((2,), -5.0), high=jnp.full((2,), 5.0), name="p"
    ).support


def _quad(x):
    return -0.5 * jnp.sum(x * x)


def _target() -> TargetDistribution:
    """Quadratic log-density on R^2."""
    return TargetDistribution(
        name="quadratic",
        input_shape=(2,),
        support=_box_support(),
        unnormalized_log_prob=_quad,
    )


def _decomposition(target: TargetDistribution) -> DensityDecomposition:
    return DensityDecomposition.identity_from_target(target)


# -------------------------------------------------------------------------
# NoTempering
# -------------------------------------------------------------------------


def test_no_tempering_returns_intermediate_target_subclass():
    target = _target()
    intermediate = NoTempering().intermediate_target(target, state=None)
    assert isinstance(intermediate, IntermediateTarget)
    assert isinstance(intermediate, TargetDistribution)


def test_no_tempering_intermediate_decomposition_returns_base_unchanged():
    target = _target()
    decomp = _decomposition(target)
    out = NoTempering().intermediate_decomposition(decomp, state=0.5)
    assert out is decomp


def test_no_tempering_records_state_but_ignores_it():
    target = _target()
    intermediate_a = NoTempering().intermediate_target(target, state="a")
    intermediate_b = NoTempering().intermediate_target(target, state=42.0)
    assert intermediate_a.state == "a"
    assert intermediate_b.state == 42.0
    x = jnp.asarray([0.5, -0.3])
    assert float(intermediate_a._unnormalized_log_prob(x)) == pytest.approx(
        float(intermediate_b._unnormalized_log_prob(x)), abs=1e-6
    )


def test_no_tempering_output_transform_is_identity():
    target = _target()
    intermediate = NoTempering().intermediate_target(target, state=None)
    X = jnp.asarray([[0.0, 0.0], [1.0, -0.5]])
    Y_raw = jnp.asarray([_quad(x) for x in X])
    out = intermediate.output_transform(intermediate.state, X, Y_raw)
    assert jnp.allclose(out, Y_raw)


def test_no_tempering_invariance_flags_are_true():
    """Both axes are invariant under any state change."""
    scheme = NoTempering()
    assert scheme.is_invariant_target_map("a", "b")
    assert scheme.is_invariant_target_map(0.1, 0.9)
    assert scheme.is_invariant_form("a", "b")


def test_no_tempering_intermediate_acts_as_distribution():
    """The intermediate is a Distribution; its `_unnormalized_log_prob`
    matches the base's."""
    target = _target()
    intermediate = NoTempering().intermediate_target(target, state=None)
    x = jnp.asarray([0.5, -0.3])
    expected = float(_quad(x))
    assert float(intermediate._unnormalized_log_prob(x)) == pytest.approx(expected, abs=1e-6)


def test_no_tempering_intermediate_propagates_opaque_target():
    """An intermediate built from an opaque target also raises
    `NotImplementedError` (no analytical density was supplied)."""
    target = TargetDistribution(name="opaque", input_shape=(2,), support=_box_support())
    intermediate = NoTempering().intermediate_target(target, state=None)
    with pytest.raises(NotImplementedError):
        intermediate._unnormalized_log_prob(jnp.zeros((2,)))


# -------------------------------------------------------------------------
# TemperingScheme ABC
# -------------------------------------------------------------------------


def test_tempering_scheme_default_invariance_uses_equality():
    """Default `is_invariant_*` returns True iff states are equal."""

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
# Schedule (existing — exercised at the new abstraction's seams)
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
# Loop integration: NoTempering preserves untempered-loop semantics
# -------------------------------------------------------------------------


def test_problem_target_distribution_round_trips_via_no_tempering():
    target = _target()
    problem = Problem(target_distribution=target, name="quad_problem")
    intermediate = NoTempering().intermediate_target(
        problem.target_distribution, state=None
    )
    x = jnp.asarray([0.4, -0.2])
    assert float(intermediate._unnormalized_log_prob(x)) == pytest.approx(
        float(problem.target_distribution._unnormalized_log_prob(x)), abs=1e-6
    )


def test_no_tempering_decomposition_link_is_identity_map():
    """The base decomposition's `link` round-trips through NoTempering."""
    target = _target()
    decomp = _decomposition(target)
    out = NoTempering().intermediate_decomposition(decomp, state=None)
    assert isinstance(out.link, MapIdentity)
