"""Tests for `TemperingScheme`, `NoTempering`, and the per-state
`IntermediateTarget` they produce.

The form-axis dispatch (`LikelihoodTemperingViaForm` and friends) lands
in Step 4; this file covers the Step 3 surface only.
"""

import jax
import jax.numpy as jnp
import pytest

from sabi.problems.base import Problem
from sabi.problems.forms import Identity
from sabi.problems.target_distribution import IntermediateTarget, TargetDistribution
from sabi.tempering.base import NoTempering, TemperingScheme
from sabi.tempering.schedule import FixedSchedule, UntemperedSchedule


def _target() -> TargetDistribution:
    """Quadratic log-density on R^2."""
    return TargetDistribution.from_target_single(
        target_single=lambda x: -0.5 * jnp.sum(x * x),
        name="quadratic",
        input_shape=(2,),
        output_shape=(),
        log_density_form=Identity(),
    )


# -------------------------------------------------------------------------
# NoTempering
# -------------------------------------------------------------------------


def test_no_tempering_returns_intermediate_target_subclass():
    target = _target()
    intermediate = NoTempering().intermediate_target(target, state=None)
    assert isinstance(intermediate, IntermediateTarget)
    assert isinstance(intermediate, TargetDistribution)


def test_no_tempering_preserves_target_function_and_form():
    target = _target()
    intermediate = NoTempering().intermediate_target(target, state=0.5)
    # Same callables, same form, same prior, same support.
    assert intermediate.target_function is target.target_function
    assert intermediate.target_single is target.target_single
    assert intermediate.log_density_form is target.log_density_form
    assert intermediate.prior is target.prior
    assert intermediate.support is target.support


def test_no_tempering_records_state_but_ignores_it():
    target = _target()
    intermediate_a = NoTempering().intermediate_target(target, state="a")
    intermediate_b = NoTempering().intermediate_target(target, state=42.0)
    # State is recorded.
    assert intermediate_a.state == "a"
    assert intermediate_b.state == 42.0
    # But the math is identical.
    x = jnp.asarray([0.5, -0.3])
    assert float(intermediate_a.target_single(x)) == float(intermediate_b.target_single(x))


def test_no_tempering_output_transform_is_identity():
    target = _target()
    intermediate = NoTempering().intermediate_target(target, state=None)
    X = jnp.asarray([[0.0, 0.0], [1.0, -0.5]])
    Y_raw = jax.vmap(lambda x: -0.5 * jnp.sum(x * x))(X)
    out = intermediate.output_transform(intermediate.state, X, Y_raw)
    assert jnp.allclose(out, Y_raw)


def test_no_tempering_invariance_flags_are_true():
    """Both axes are invariant under any state change."""
    scheme = NoTempering()
    assert scheme.is_invariant_target_function("a", "b")
    assert scheme.is_invariant_target_function(0.1, 0.9)
    assert scheme.is_invariant_form("a", "b")


def test_no_tempering_intermediate_acts_as_distribution():
    """The intermediate is a Distribution; its
    `_unnormalized_log_prob` works at the un-tempered target's value."""
    target = _target()
    intermediate = NoTempering().intermediate_target(target, state=None)
    x = jnp.asarray([0.5, -0.3])
    out = float(intermediate._unnormalized_log_prob(x))
    expected = float(-0.5 * jnp.sum(x * x))
    assert out == pytest.approx(expected, abs=1e-6)


# -------------------------------------------------------------------------
# TemperingScheme ABC
# -------------------------------------------------------------------------


def test_tempering_scheme_default_invariance_uses_equality():
    """Default `is_invariant_*` returns True iff states are equal."""

    class _NullScheme(TemperingScheme):
        def intermediate_target(self, base, state):
            return NoTempering().intermediate_target(base, state)

    scheme = _NullScheme()
    assert scheme.is_invariant_target_function(0.5, 0.5)
    assert not scheme.is_invariant_target_function(0.5, 0.6)
    assert scheme.is_invariant_form(None, None)
    assert not scheme.is_invariant_form(None, "a")


# -------------------------------------------------------------------------
# Schedule (existing — exercised at the new abstraction's seams)
# -------------------------------------------------------------------------


def test_untempered_schedule_returns_none_state_and_final():
    sched = UntemperedSchedule()
    state, final = sched.next(0, None)
    assert state is None
    assert final is True


def test_fixed_schedule_iterates_and_marks_final():
    sched = FixedSchedule(states=(0.1, 0.5, 1.0))
    assert sched.next(0, None) == (0.1, False)
    assert sched.next(1, None) == (0.5, False)
    assert sched.next(2, None) == (1.0, True)
    # Past the end clamps to the last entry and stays final.
    assert sched.next(42, None) == (1.0, True)


def test_fixed_schedule_accepts_non_scalar_states():
    """States are opaque PyTrees — e.g. subset indices for data tempering."""
    sched = FixedSchedule(states=((0, 1), (0, 1, 2), (0, 1, 2, 3)))
    state, final = sched.next(1, None)
    assert state == (0, 1, 2)
    assert final is False


def test_fixed_schedule_rejects_empty():
    with pytest.raises(ValueError):
        FixedSchedule(states=())


# -------------------------------------------------------------------------
# Loop integration: NoTempering preserves untempered-loop semantics
# -------------------------------------------------------------------------


def test_problem_target_distribution_round_trips_via_no_tempering():
    """Constructing an intermediate via NoTempering on a Problem's
    target_distribution should produce a distribution whose log-density
    matches `Problem.log_posterior` at the same input."""
    problem = Problem.from_target_single(
        target_single=lambda x: -0.5 * jnp.sum(x * x),
        name="quad_problem",
        input_shape=(2,),
        output_shape=(),
        log_density_form=Identity(),
    )
    intermediate = NoTempering().intermediate_target(
        problem.target_distribution, state=None
    )
    x = jnp.asarray([0.4, -0.2])
    assert float(intermediate._unnormalized_log_prob(x)) == pytest.approx(
        float(problem.log_posterior(x)), abs=1e-6
    )
