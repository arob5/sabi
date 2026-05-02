import jax
import jax.numpy as jnp
from probpipe import mean

from sabi.acquisitions.base import AcquisitionState
from sabi.acquisitions.ei import ExpectedImprovement
from sabi.acquisitions.optim import CandidateSetOptimizer
from sabi.acquisitions.random import PriorSampling
from sabi.posterior.surrogate_distribution import SurrogateDistribution
from sabi.problems.gaussian import gaussian2d
from sabi.sampling import PriorSampler
from sabi.emulators import TinyGPEmulator


def _state(problem, key_seed=0, n=20):
    """Build an AcquisitionState seeded by samples from problem.prior — that's
    where acquisition candidates will also be drawn from, so the GP gets
    trained on the same support."""
    X = PriorSampler().sample(problem, jax.random.key(key_seed), n)
    Y = problem.target_map(X)
    gp = TinyGPEmulator(input_shape=problem.input_shape).fit(X, Y)
    sp = SurrogateDistribution(
        emulator=gp,
        log_density_form=problem.log_density_form,
        support=problem.support,
        input_shape=problem.input_shape,
        prior=problem.prior,
    )
    return AcquisitionState(
        problem=problem,
        surrogate_distribution=sp,
        X=X,
        Y_raw=Y,
        Y_train=Y,
        tempering_state=None,
        target_tempering_state=None,
    )


def test_prior_sampling_acquisition_shape_and_bounds():
    problem = gaussian2d()
    state = _state(problem)
    batch = PriorSampling().select_batch(state, q=4, key=jax.random.key(7))
    assert batch.shape == (4,) + problem.input_shape
    # support is interval(low, high) per element; check membership
    assert jnp.all(jnp.asarray(problem.support.check(batch)))


def test_ei_acquisition_shape():
    problem = gaussian2d()
    state = _state(problem)
    batch = ExpectedImprovement(optimizer=CandidateSetOptimizer(n_candidates=512)).select_batch(
        state, q=3, key=jax.random.key(11)
    )
    assert batch.shape == (3,) + problem.input_shape


def test_ei_picks_points_with_higher_emulator_mean_than_random():
    """EI is defined to prefer points with high emulator mean + variance.
    Verify directly against the emulator (decouples from GP fit quality)."""
    problem = gaussian2d()
    state = _state(problem, n=60)

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
    problem = gaussian2d()
    state = _state(problem, n=60)

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
