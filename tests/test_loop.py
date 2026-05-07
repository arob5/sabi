import jax
import jax.numpy as jnp
from probpipe import sample as pp_sample
from probpipe.core._distribution_base import Distribution
from probpipe.core._empirical import NumericEmpiricalDistribution
from probpipe.core.protocols import SupportsSampling

from sabi.acquisitions.ei import ExpectedImprovement
from sabi.acquisitions.optim import (
    CandidateSetOptimizer,
    ContinuousMultiStartOptimizer,
)
from sabi.acquisitions.random import DistributionSampling
from sabi.algorithms import (
    Algorithm,
    emulator_pushforward_factory,
    run,
    weighted_empirical_factory,
)
from sabi.density_decomposition import DensityDecomposition
from sabi.metrics.mmd import MMD
from sabi.problems.banana import banana
from sabi.problems.benchmarks import gaussian_2d
from sabi.emulators import TinyGPEmulator


def _algorithm(
    problem,
    acquisition,
    n_rounds: int = 5,
    metrics=(MMD(n_estimate_samples=512, n_reference_samples=512),),
    surrogate_distribution_factory=emulator_pushforward_factory,
):
    decomposition = DensityDecomposition.identity_from_target(problem.target_distribution)
    return Algorithm(
        emulator_factory=lambda: TinyGPEmulator(input_shape=problem.target_distribution.input_shape),
        acquisition=acquisition,
        density_decomposition=decomposition,
        n_initial=16,
        n_rounds=n_rounds,
        q=1,
        surrogate_distribution_factory=surrogate_distribution_factory,
        metrics=metrics,
    )


def test_loop_runs_on_gaussian_2d_with_distribution_sampling_acq():
    problem = gaussian_2d()
    alg = _algorithm(problem, DistributionSampling(), n_rounds=5)
    result = run(problem, alg, jax.random.key(0))
    assert result.X.shape == (16 + 4,) + problem.target_distribution.input_shape
    assert len(result.per_round_metrics) == 5
    assert all(m["tempering_state"] is None for m in result.per_round_metrics)
    assert "mmd2" in result.final_metrics
    assert result.final_metrics["mmd2"] >= -1e-6
    assert isinstance(result.final_estimate, Distribution)


def test_loop_runs_on_banana_with_ei_acq():
    problem = banana()
    alg = _algorithm(
        problem,
        ExpectedImprovement(optimizer=CandidateSetOptimizer(n_candidates=512)),
        n_rounds=5,
    )
    result = run(problem, alg, jax.random.key(1))
    assert result.X.shape == (16 + 4,) + problem.target_distribution.input_shape
    assert "mmd2" in result.final_metrics


def test_loop_grows_dataset_and_records_metrics():
    problem = gaussian_2d()
    alg = _algorithm(
        problem,
        ExpectedImprovement(optimizer=CandidateSetOptimizer(n_candidates=512)),
        n_rounds=4,
    )
    result = run(problem, alg, jax.random.key(3))
    assert result.X.shape == (19, 2)
    assert result.Y_raw.shape == (19,)
    assert result.Y_train.shape == (19,)
    assert [m["round"] for m in result.per_round_metrics] == [0, 1, 2, 3]
    assert [m["n_evals"] for m in result.per_round_metrics] == [16, 17, 18, 19]


def test_loop_with_no_metrics_skips_estimator():
    problem = gaussian_2d()
    alg = _algorithm(problem, DistributionSampling(), n_rounds=3, metrics=())
    result = run(problem, alg, jax.random.key(4))
    assert result.final_metrics == {}
    for row in result.per_round_metrics:
        assert "mmd2" not in row


def test_loop_with_weighted_empirical_baseline():
    """No-GP baseline path: WeightedEmpiricalSurrogateDistribution produces
    a NumericEmpiricalDistribution as the estimate."""
    problem = gaussian_2d()
    alg = _algorithm(
        problem,
        DistributionSampling(),
        n_rounds=3,
        surrogate_distribution_factory=weighted_empirical_factory,
    )
    result = run(problem, alg, jax.random.key(5))
    assert isinstance(result.final_estimate, NumericEmpiricalDistribution)
    assert isinstance(result.final_estimate, SupportsSampling)
    assert "mmd2" in result.final_metrics
    samples = jnp.asarray(
        pp_sample(result.final_estimate, key=jax.random.key(0), sample_shape=(128,))
    )
    matches = jnp.any(jnp.all(samples[:, None, :] == result.X[None, :, :], axis=-1), axis=-1)
    assert bool(jnp.all(matches))


def test_loop_continuous_ei_beats_candidate_set_ei_on_gaussian_2d():
    """Continuous EI should be no worse than candidate-set EI by more than 5%."""
    problem = gaussian_2d()

    cs_acq = ExpectedImprovement(optimizer=CandidateSetOptimizer(n_candidates=512))
    cm_acq = ExpectedImprovement(
        optimizer=ContinuousMultiStartOptimizer(
            n_starts=8,
            n_seeding_candidates=128,
            bfgs_max_steps=40,
        )
    )

    cs_alg = _algorithm(problem, cs_acq, n_rounds=5)
    cm_alg = _algorithm(problem, cm_acq, n_rounds=5)

    cs_result = run(problem, cs_alg, jax.random.key(0))
    cm_result = run(problem, cm_alg, jax.random.key(0))

    cs_mmd2 = float(cs_result.final_metrics["mmd2"])
    cm_mmd2 = float(cm_result.final_metrics["mmd2"])
    assert cm_mmd2 <= cs_mmd2 * 1.05, (
        f"Continuous EI MMD² ({cm_mmd2:.4f}) > CandidateSet EI MMD² "
        f"({cs_mmd2:.4f}) × 1.05; continuous optimization regressed."
    )
