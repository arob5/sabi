import jax.numpy as jnp
import pytest

from sabi.problems.base import Problem
from sabi.problems.forms import Identity, LogDensityForm, LogLikPlusPrior
from sabi.tempering.base import NoTempering, Tempering, register_tempering
from sabi.tempering.schedule import FixedSchedule, UntemperedSchedule


def _problem():
    return Problem.from_target_single(
        name="dummy",
        input_shape=(1,),
        output_shape=(),
        target_single=lambda x: jnp.asarray(0.0),
        log_density_form=Identity(),
    )


def test_no_tempering_returns_form_unchanged():
    form = Identity()
    # State is ignored (opaque PyTree) — pass arbitrary state to confirm.
    out = NoTempering().apply(form, state=(1, 2, 3))
    assert out is form


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


def test_dispatch_registry_raises_for_unknown_pair():
    class DummyTempering(Tempering):
        pass

    with pytest.raises(NotImplementedError, match="No tempering registered"):
        DummyTempering().apply(Identity(), state=0.5)


def test_dispatch_registry_allows_registration():
    """Registering a new (Tempering, Form) pair works end-to-end."""

    class _DemoTempering(Tempering):
        pass

    class _ScaledLogLikPlusPrior(LogLikPlusPrior):
        def __init__(self, beta: float):
            object.__setattr__(self, "_beta", beta)

        def __call__(self, x, y, problem):
            return self._beta * y  # skip prior to keep test hermetic

    @register_tempering(_DemoTempering, LogLikPlusPrior)
    def _demo_loglik(temp, form: LogDensityForm, state: float) -> LogDensityForm:
        return _ScaledLogLikPlusPrior(beta=state)

    tempered = _DemoTempering().apply(LogLikPlusPrior(), state=0.4)
    assert float(tempered(jnp.asarray([0.0]), jnp.asarray(2.0), _problem())) == pytest.approx(0.8)
