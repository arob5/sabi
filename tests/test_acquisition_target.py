"""Tests for `AcquisitionTarget` look-ahead in the loop.

Verifies:
- The default (`CURRENT`) is bit-for-bit equivalent to the
  no-acquisition-target loop (i.e., behavior preserved).
- `NEXT` and `TERMINAL` build the SP at a different state than the
  round's current state.
- Per-round metric rows reflect the resolved
  `(tempering_state, target_tempering_state)` for each
  `acquisition_target` value.
- Schedule's `terminal_state()` returns the correct state for both
  built-in schedules.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sabi._probpipe_compat import independent_uniform
from sabi.acquisitions.base import AcquisitionTarget, resolve_state
from sabi.acquisitions.random import DistributionSampling
from sabi.algorithms import Algorithm, run
from sabi.density_decomposition import DensityDecomposition, LogProbTarget
from sabi.emulators import TinyGPEmulator
from sabi.maps import Identity, LogProb
from sabi.problems.base import Problem
from sabi.problems.benchmarks import gaussian_2d
from probpipe.core._numeric_record_distribution import NumericRecordDistribution
from sabi.tempering.likelihood import (
    LikelihoodTemperingViaForm,
    LikelihoodTemperingViaTarget,
)
from sabi.tempering.schedule import (
    FixedSchedule,
    UntemperedSchedule,
)


# -------------------------------------------------------------------------
# Schedule.terminal_state
# -------------------------------------------------------------------------


def test_untempered_schedule_terminal_state_is_none():
    assert UntemperedSchedule().terminal_state() is None


def test_fixed_schedule_terminal_state_is_last_entry():
    sched = FixedSchedule(states=(0.1, 0.5, 1.0))
    assert sched.terminal_state() == 1.0


def test_default_terminal_state_probes_via_large_round_idx():
    from sabi.tempering.schedule import TemperingSchedule

    sched = FixedSchedule(states=(0.1, 0.5, 1.0))
    base_terminal = TemperingSchedule.terminal_state(sched)
    assert base_terminal == 1.0


# -------------------------------------------------------------------------
# resolve_state unit tests — direct, no loop.
# -------------------------------------------------------------------------


def test_resolve_state_current_returns_current_state():
    schedule = FixedSchedule(states=(0.1, 0.5, 1.0))
    out = resolve_state(
        AcquisitionTarget.CURRENT, schedule, round_idx=0, current_state=0.1
    )
    assert out == 0.1


def test_resolve_state_next_advances_one_round():
    schedule = FixedSchedule(states=(0.1, 0.5, 1.0))
    out = resolve_state(
        AcquisitionTarget.NEXT, schedule, round_idx=0, current_state=0.1
    )
    assert out == 0.5


def test_resolve_state_next_clamps_at_last_round():
    schedule = FixedSchedule(states=(0.1, 0.5, 1.0))
    last_round = len(schedule.states) - 1
    out = resolve_state(
        AcquisitionTarget.NEXT,
        schedule,
        round_idx=last_round,
        current_state=schedule.at(last_round)[0],
    )
    assert out == schedule.terminal_state()
    assert out == 1.0


def test_resolve_state_next_clamps_for_untempered_schedule():
    schedule = UntemperedSchedule()
    out = resolve_state(
        AcquisitionTarget.NEXT, schedule, round_idx=99, current_state=None
    )
    assert out is None
    assert out == schedule.terminal_state()


def test_resolve_state_terminal_returns_terminal_state():
    schedule = FixedSchedule(states=(0.1, 0.5, 1.0))
    for round_idx in (0, 1, 2, 5):
        out = resolve_state(
            AcquisitionTarget.TERMINAL,
            schedule,
            round_idx=round_idx,
            current_state=0.1,
        )
        assert out == schedule.terminal_state()


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _acq_round_states(result) -> list[tuple]:
    """Return ``[(tempering_state, target_tempering_state), ...]`` for
    each acquisition round (skipping round 0)."""
    return [
        (row["tempering_state"], row["target_tempering_state"])
        for row in result.per_round_metrics
        if row["round"] >= 1
    ]


# -------------------------------------------------------------------------
# CURRENT (default): target == current
# -------------------------------------------------------------------------


def test_current_default_target_state_equals_current():
    """Untempered loop: target_tempering_state == tempering_state == None."""
    problem = gaussian_2d()
    decomposition = LogProbTarget(problem.target_distribution)
    alg = Algorithm(
        emulator_factory=lambda: TinyGPEmulator(input_shape=(2,)),
        acquisition=DistributionSampling(),
        density_decomposition=decomposition,
        n_initial=8,
        n_rounds=3,
        q=1,
        acquisition_target=AcquisitionTarget.CURRENT,
    )
    result = run(problem, alg, jax.random.key(0))

    rounds = _acq_round_states(result)
    assert len(rounds) == 2
    for current, target in rounds:
        assert current is None
        assert target is None


# -------------------------------------------------------------------------
# NEXT: target_tempering_state is one step ahead of current
# -------------------------------------------------------------------------


def test_next_with_fixed_schedule_advances_one_step():
    """Round 1 acquisition: current=0.5, target=1.0. Round 2: current=1.0, target=1.0 (clamped)."""
    problem = gaussian_2d()
    decomposition = LogProbTarget(problem.target_distribution)
    schedule = FixedSchedule(states=(0.1, 0.5, 1.0))
    # Identity-base decomposition: tempering needs an `initial`
    # distribution (the geometric-bridge case). Use the target's box
    # support as a uniform initial.
    box = problem.target_distribution.support
    initial = independent_uniform(low=jnp.asarray(box.low), high=jnp.asarray(box.high), name="init")
    alg = Algorithm(
        emulator_factory=lambda: TinyGPEmulator(input_shape=(2,)),
        acquisition=DistributionSampling(),
        density_decomposition=decomposition,
        n_initial=8,
        n_rounds=3,
        q=1,
        tempering_scheme=LikelihoodTemperingViaForm(initial=initial),
        schedule=schedule,
        acquisition_target=AcquisitionTarget.NEXT,
    )
    result = run(problem, alg, jax.random.key(1))

    rounds = _acq_round_states(result)
    assert len(rounds) == 2
    assert rounds[0] == (0.5, 1.0)
    assert rounds[1] == (1.0, 1.0)


# -------------------------------------------------------------------------
# TERMINAL: target_tempering_state is always the terminal state
# -------------------------------------------------------------------------


def test_terminal_target_state_is_terminal_for_every_round():
    problem = gaussian_2d()
    decomposition = LogProbTarget(problem.target_distribution)
    schedule = FixedSchedule(states=(0.1, 0.5, 1.0))
    box = problem.target_distribution.support
    initial = independent_uniform(low=jnp.asarray(box.low), high=jnp.asarray(box.high), name="init")
    alg = Algorithm(
        emulator_factory=lambda: TinyGPEmulator(input_shape=(2,)),
        acquisition=DistributionSampling(),
        density_decomposition=decomposition,
        n_initial=8,
        n_rounds=3,
        q=1,
        tempering_scheme=LikelihoodTemperingViaForm(initial=initial),
        schedule=schedule,
        acquisition_target=AcquisitionTarget.TERMINAL,
    )
    result = run(problem, alg, jax.random.key(2))

    rounds = _acq_round_states(result)
    assert len(rounds) == 2
    for _, target in rounds:
        assert target == 1.0
    assert [current for current, _ in rounds] == [0.5, 1.0]


# -------------------------------------------------------------------------
# Default (CURRENT) preserves untempered behavior
# -------------------------------------------------------------------------


def test_default_acquisition_target_preserves_untempered_metrics():
    problem = gaussian_2d()
    decomposition = LogProbTarget(problem.target_distribution)
    alg = Algorithm(
        emulator_factory=lambda: TinyGPEmulator(input_shape=(2,)),
        acquisition=DistributionSampling(),
        density_decomposition=decomposition,
        n_initial=8,
        n_rounds=3,
        q=1,
    )
    result = run(problem, alg, jax.random.key(0))
    assert result.X.shape == (10,) + problem.target_distribution.event_shape


# -------------------------------------------------------------------------
# Look-ahead with TARGET-axis tempering causes emulator refit
# -------------------------------------------------------------------------


def test_via_target_with_next_lookahead_runs_to_completion():
    """`LikelihoodTemperingViaTarget` + `NEXT` look-ahead — refit path."""
    from sabi.density_decomposition import LogProbTermTarget

    prior = independent_uniform(
        low=jnp.full((2,), -3.0), high=jnp.full((2,), 3.0), name="p"
    )

    class _QuadLogLikTarget(NumericRecordDistribution):
        # Math identity only (no analytical density); the loop doesn't
        # need it under `LikelihoodTemperingViaTarget`.
        @property
        def event_shape(self):
            return (2,)

        @property
        def support(self):
            return prior.support

    class _LogLikDecomp(LogProbTermTarget):
        @property
        def event_shape(self):
            return (2,)

        def target_map(self, x):
            return -0.5 * jnp.sum(x * x, axis=-1)

    target = _QuadLogLikTarget(name="quad_loglik_target")
    problem = Problem(target_distribution=target, name="quad_loglik")
    decomposition = _LogLikDecomp(
        name="quad_loglik_decomp",
        support=prior.support,
        prior=prior,
    )
    alg = Algorithm(
        emulator_factory=lambda: TinyGPEmulator(input_shape=(2,)),
        acquisition=DistributionSampling(),
        density_decomposition=decomposition,
        initial_design_distribution=prior,
        x_support=prior.support,
        n_initial=8,
        n_rounds=3,
        q=1,
        tempering_scheme=LikelihoodTemperingViaTarget(),
        schedule=FixedSchedule(states=(0.5, 1.0)),
        acquisition_target=AcquisitionTarget.NEXT,
    )
    result = run(problem, alg, jax.random.key(3))
    assert result.X.shape == (10, 2)
    assert result.per_round_metrics[0]["round"] == 0
    assert result.per_round_metrics[0]["target_tempering_state"] is None
    assert result.per_round_metrics[1]["target_tempering_state"] == 1.0
    assert result.per_round_metrics[2]["target_tempering_state"] == 1.0
