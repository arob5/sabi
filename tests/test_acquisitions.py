import jax
import jax.numpy as jnp
import pytest
from probpipe import mean
from probpipe.distributions.continuous import Normal

from sabi.acquisitions.base import AcquisitionState
from sabi.acquisitions.ei import ExpectedImprovement
from sabi.acquisitions.optim import CandidateSetOptimizer
from sabi.acquisitions.random import PriorSampling
from sabi.emulators.base import Emulator
from sabi.surrogate.surrogate_distribution import SurrogateDistribution
from sabi.surrogate.weighted_empirical import WeightedEmpiricalRandomMeasure
from sabi.problems.benchmarks import gaussian_2d
from sabi.sampling import PriorSampler
from sabi.emulators import TinyGPEmulator

from tests.conftest import make_acquisition_state


def test_prior_sampling_acquisition_shape_and_bounds():
    problem = gaussian_2d()
    state = make_acquisition_state(problem=problem)
    batch = PriorSampling().select_batch(state, q=4, key=jax.random.key(7))
    assert batch.shape == (4,) + problem.input_shape
    # support is interval(low, high) per element; check membership
    assert jnp.all(jnp.asarray(problem.support.check(batch)))


def test_ei_acquisition_shape():
    problem = gaussian_2d()
    state = make_acquisition_state(problem=problem)
    batch = ExpectedImprovement(optimizer=CandidateSetOptimizer(n_candidates=512)).select_batch(
        state, q=3, key=jax.random.key(11)
    )
    assert batch.shape == (3,) + problem.input_shape


def test_ei_picks_points_with_higher_emulator_mean_than_random():
    """EI is defined to prefer points with high emulator mean + variance.
    Verify directly against the emulator (decouples from GP fit quality)."""
    problem = gaussian_2d()
    state = make_acquisition_state(problem=problem, n=60)

    ei_batch = ExpectedImprovement(optimizer=CandidateSetOptimizer(n_candidates=4096)).select_batch(
        state, q=16, key=jax.random.key(2)
    )
    random_batch = PriorSampling().select_batch(state, q=16, key=jax.random.key(3))

    emulator = state.surrogate_distribution.emulator
    ei_pred = emulator(ei_batch)
    rand_pred = emulator(random_batch)

    ei_mean = jnp.asarray(mean(ei_pred))
    rand_mean = jnp.asarray(mean(rand_pred))
    assert float(jnp.mean(ei_mean)) > float(jnp.mean(rand_mean))


def test_ei_average_best_beats_random_average_best_across_seeds():
    problem = gaussian_2d()
    state = make_acquisition_state(problem=problem, n=60)

    ei_bests = []
    rand_bests = []
    for seed in range(8):
        ei_batch = ExpectedImprovement(optimizer=CandidateSetOptimizer(n_candidates=2048)).select_batch(
            state, q=8, key=jax.random.key(100 + seed)
        )
        rand_batch = PriorSampling().select_batch(state, q=8, key=jax.random.key(200 + seed))
        ei_bests.append(float(jnp.max(problem.target_map(ei_batch))))
        rand_bests.append(float(jnp.max(problem.target_map(rand_batch))))
    assert sum(ei_bests) / len(ei_bests) > sum(rand_bests) / len(rand_bests)


# -------------------------------------------------------------------------
# Edge cases: σ=0 collapse and degenerate (emulator=None) surrogates.
# -------------------------------------------------------------------------


class _ConstantEmulator(Emulator):
    """Stub emulator returning a Normal with caller-supplied loc / scale.

    Used to drive `ExpectedImprovement._score_single` to the σ ≤ 1e-30
    branch. The real `TinyGPEmulator` floors variance at `jitter`
    (default 1e-3) so it can't reach the σ=0 path naturally.
    """

    def __init__(self, *, loc: float, scale: float, input_shape=(2,)):
        super().__init__(input_shape=input_shape, output_shape=(), name="constant_em")
        self._loc = float(loc)
        self._scale = float(scale)

    def fit(self, X, Y):
        return self

    def predict(self, X, *, joint_inputs=False, joint_outputs=False):
        n = X.shape[0]
        return Normal(
            loc=jnp.full((n,), self._loc),
            scale=jnp.full((n,), self._scale),
            name="constant_pred",
        )


def test_ei_collapses_to_zero_at_zero_variance():
    """`EI(x) = 0` whenever `σ(x) ≤ 1e-30` — the docstring promise at
    [src/sabi/acquisitions/ei.py:14]. Uses a stub emulator since the
    real GP's jitter floor keeps σ orders of magnitude above 1e-30."""
    problem = gaussian_2d()
    # Train Y_train so `best = max(Y_train)` is a finite scalar.
    X = PriorSampler().sample(problem, jax.random.key(0), 4)
    Y = problem.target_map(X)
    emulator = _ConstantEmulator(loc=0.0, scale=0.0, input_shape=problem.input_shape)
    sp = SurrogateDistribution(
        emulator=emulator,
        log_density_form=problem.log_density_form,
        support=problem.support,
        input_shape=problem.input_shape,
        prior=problem.prior,
    )
    state = AcquisitionState(
        problem=problem,
        surrogate_distribution=sp,
        X=X,
        Y_raw=Y,
        Y_train=Y,
    )
    x = jnp.zeros(problem.input_shape)
    score = float(ExpectedImprovement()._score_single(x, state))
    assert score == pytest.approx(0.0, abs=1e-12)


def test_ei_raises_on_degenerate_surrogate_emulator():
    """`ExpectedImprovement` requires a non-None emulator. Using a
    `WeightedEmpiricalRandomMeasure` (the no-emulator baseline) at
    `state.surrogate_distribution` should raise with a message that
    points the user at the swap."""
    problem = gaussian_2d()
    X = PriorSampler().sample(problem, jax.random.key(0), 8)
    Y = problem.target_map(X)
    sp = WeightedEmpiricalRandomMeasure(
        X=X,
        log_weights=Y,
        support=problem.support,
        input_shape=problem.input_shape,
    )
    state = AcquisitionState(
        problem=problem,
        surrogate_distribution=sp,
        X=X,
        Y_raw=Y,
        Y_train=Y,
    )
    with pytest.raises(ValueError, match="non-degenerate emulator"):
        ExpectedImprovement().select_batch(state, q=1, key=jax.random.key(0))
