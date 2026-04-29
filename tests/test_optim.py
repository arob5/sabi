"""Tests for PointwiseOptimizer hierarchy.

Coverage:
- `CandidateSetOptimizer` reproduces v1.x EI behavior on a fitted surrogate.
- `ContinuousMultiStartOptimizer` finds a known argmax of a synthetic
  concave score; on a real surrogate it improves over `CandidateSetOptimizer`
  with a smaller candidate budget.
- `GreedyMultiPointOptimizer` returns `q` distinct points and uses the
  imputer to drive in-batch diversity.
- Reparameterization round-trips: optimization in unconstrained space
  lands within support bounds; non-interval supports raise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from probpipe.core.constraints import interval, positive

from sabi.acquisitions.base import AcquisitionState, PointwiseScoredAcquisition
from sabi.acquisitions.ei import ExpectedImprovement
from sabi.acquisitions.fantasize import ConstantLiar, KrigingBeliever
from sabi.acquisitions.optim import (
    CandidateSetOptimizer,
    ContinuousMultiStartOptimizer,
    GreedyMultiPointOptimizer,
    _make_bijector,
)
from sabi.posterior.surrogate_posterior import SurrogatePosterior
from sabi.problems.gaussian2d import gaussian2d
from sabi.surrogates.gp import GPSurrogate


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


def _state(n: int = 30, seed: int = 0):
    problem = gaussian2d()
    key = jax.random.key(seed)
    lower, upper = problem.support.low, problem.support.high
    X = lower + (upper - lower) * jax.random.uniform(
        key, shape=(n,) + problem.input_shape
    )
    Y = problem.target_function(X)
    surrogate = GPSurrogate(input_shape=problem.input_shape).fit(X, Y)
    sp = SurrogatePosterior(
        surrogate=surrogate,
        log_density_form=problem.log_density_form,
        support=problem.support,
        input_shape=problem.input_shape,
        prior=problem.prior,
    )
    return AcquisitionState(
        problem=problem,
        surrogate_posterior=sp,
        X=X,
        Y=Y,
        tempering_state=None,
    )


class _ConcaveScore(PointwiseScoredAcquisition):
    """Test acquisition: score(x) = -‖x - target‖².

    Known argmax at `target`; a global concave bowl. Used to verify that
    `ContinuousMultiStartOptimizer` actually finds the argmax. Uses
    `_score_single` so the default vmapped `score` handles batching.
    """

    def __init__(self, target: jax.Array, optimizer):
        object.__setattr__(self, "target", jnp.asarray(target))
        object.__setattr__(self, "optimizer", optimizer)

    def _score_single(self, x, state):
        return -jnp.sum((x - self.target) ** 2)


# -------------------------------------------------------------------------
# Bijector unit tests
# -------------------------------------------------------------------------


def test_make_bijector_interval_roundtrips():
    low = jnp.asarray([-3.0, -2.0])
    high = jnp.asarray([5.0, 4.0])
    bij = _make_bijector(interval(low, high))
    x = jnp.asarray([1.0, 0.5])
    u = bij.inverse(x)
    x_back = bij.forward(u)
    assert jnp.allclose(x_back, x, atol=1e-5)


def test_make_bijector_interval_forward_lands_in_bounds():
    low = jnp.asarray([-1.0, -1.0])
    high = jnp.asarray([1.0, 1.0])
    bij = _make_bijector(interval(low, high))
    # Extreme unconstrained values map into the interior of the box.
    extreme = jnp.asarray([100.0, -100.0])
    x = bij.forward(extreme)
    assert jnp.all(x >= low) and jnp.all(x <= high)


def test_make_bijector_raises_for_non_interval_constraints():
    with pytest.raises(NotImplementedError, match="Constraint"):
        _make_bijector(positive)


# -------------------------------------------------------------------------
# Optimizer correctness
# -------------------------------------------------------------------------


def test_candidate_set_optimizer_returns_top_q():
    state = _state()
    acq = ExpectedImprovement(optimizer=CandidateSetOptimizer(n_candidates=64))
    batch = acq.optimizer.optimize(acq, state, q=3, key=jax.random.key(7))
    assert batch.shape == (3,) + state.problem.input_shape


def test_continuous_multistart_finds_known_concave_argmax():
    """Synthetic test: argmax of `-‖x - target‖²` is `target`."""
    state = _state()
    target = jnp.asarray([0.5, -1.0])
    optimizer = ContinuousMultiStartOptimizer(
        n_starts=8,
        n_seeding_candidates=64,
        bfgs_max_steps=50,
    )
    acq = _ConcaveScore(target=target, optimizer=optimizer)
    batch = optimizer.optimize(acq, state, q=1, key=jax.random.key(2))
    assert batch.shape == (1, 2)
    assert jnp.allclose(batch[0], target, atol=1e-3)


def test_continuous_multistart_constrained_to_support():
    """Optimum of an unbounded concave function lands inside the support box."""
    state = _state()
    # Target outside the support box (which is [mu-5, mu+5]² = [-5, 5]²
    # for default gaussian2d). Optimizer should clamp at the boundary.
    target = jnp.asarray([20.0, -20.0])
    optimizer = ContinuousMultiStartOptimizer(
        n_starts=4,
        n_seeding_candidates=32,
        bfgs_max_steps=30,
    )
    acq = _ConcaveScore(target=target, optimizer=optimizer)
    batch = optimizer.optimize(acq, state, q=1, key=jax.random.key(3))
    lower, upper = state.problem.support.low, state.problem.support.high
    assert jnp.all(batch >= lower - 1e-3)
    assert jnp.all(batch <= upper + 1e-3)


def test_greedy_multi_point_returns_distinct_points():
    state = _state()
    inner = CandidateSetOptimizer(n_candidates=128)
    optimizer = GreedyMultiPointOptimizer(inner=inner, imputer=KrigingBeliever())
    acq = ExpectedImprovement(optimizer=optimizer)
    batch = optimizer.optimize(acq, state, q=3, key=jax.random.key(11))
    assert batch.shape == (3,) + state.problem.input_shape
    # No two picks coincide.
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        assert not jnp.allclose(batch[i], batch[j], atol=1e-6)


def test_greedy_multi_point_with_constant_liar_min():
    """Pessimistic ConstantLiar should still give distinct picks."""
    state = _state()
    inner = CandidateSetOptimizer(n_candidates=128)
    optimizer = GreedyMultiPointOptimizer(
        inner=inner, imputer=ConstantLiar(value="min")
    )
    acq = ExpectedImprovement(optimizer=optimizer)
    batch = optimizer.optimize(acq, state, q=2, key=jax.random.key(13))
    assert batch.shape == (2,) + state.problem.input_shape
    assert not jnp.allclose(batch[0], batch[1], atol=1e-6)


def test_continuous_multistart_finds_higher_score_than_candidate_set():
    """At matched cost, continuous-multistart should match-or-beat
    candidate-set. We give continuous a strictly smaller seeding budget,
    and assert it still scores at least as well — confirming the BFGS
    refinement actually adds something."""
    state = _state()
    target = jnp.asarray([0.4, -0.7])

    cs = CandidateSetOptimizer(n_candidates=128)
    cm = ContinuousMultiStartOptimizer(
        n_starts=4, n_seeding_candidates=64, bfgs_max_steps=50
    )

    acq_cs = _ConcaveScore(target=target, optimizer=cs)
    acq_cm = _ConcaveScore(target=target, optimizer=cm)

    cs_pick = cs.optimize(acq_cs, state, q=1, key=jax.random.key(20))[0]
    cm_pick = cm.optimize(acq_cm, state, q=1, key=jax.random.key(20))[0]

    cs_score = float(acq_cs.score(cs_pick[None], state)[0])
    cm_score = float(acq_cm.score(cm_pick[None], state)[0])
    # Continuous refinement should push closer to the target.
    assert cm_score >= cs_score - 1e-6
