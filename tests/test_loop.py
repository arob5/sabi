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
from sabi.acquisitions.random import Random
from sabi.algorithms.loop import (
    Algorithm,
    gp_pushforward_factory,
    run,
    weighted_empirical_factory,
)
from sabi.metrics.posterior_mmd import ReferenceMMD
from sabi.problems.banana import banana
from sabi.problems.gaussian2d import gaussian2d
from sabi.surrogates.gp import GPSurrogate


def _algorithm(
    acquisition,
    n_rounds: int = 4,
    metrics=(ReferenceMMD(n_estimate_samples=512, n_reference_samples=512),),
    surrogate_posterior_factory=gp_pushforward_factory,
):
    return Algorithm(
        surrogate_factory=lambda: GPSurrogate(),
        acquisition=acquisition,
        n_initial=16,
        n_rounds=n_rounds,
        q=1,
        surrogate_posterior_factory=surrogate_posterior_factory,
        metrics=metrics,
    )


def test_loop_runs_on_gaussian2d_with_random_acq():
    problem = gaussian2d()
    alg = _algorithm(Random(), n_rounds=4)
    result = run(problem, alg, jax.random.key(0))
    assert result.X.shape == (16 + 4,) + problem.input_shape
    assert len(result.per_round_metrics) == 4
    assert all(m["tempering_state"] is None for m in result.per_round_metrics)
    assert "mmd2" in result.final_metrics
    assert result.final_metrics["mmd2"] >= -1e-6
    assert isinstance(result.final_estimate, Distribution)


def test_loop_runs_on_banana_with_ei_acq():
    problem = banana()
    alg = _algorithm(ExpectedImprovement(optimizer=CandidateSetOptimizer(n_candidates=512)), n_rounds=4)
    result = run(problem, alg, jax.random.key(1))
    assert result.X.shape == (16 + 4,) + problem.input_shape
    assert "mmd2" in result.final_metrics


def test_loop_grows_dataset_and_records_metrics():
    problem = gaussian2d()
    alg = _algorithm(ExpectedImprovement(optimizer=CandidateSetOptimizer(n_candidates=512)), n_rounds=3)
    result = run(problem, alg, jax.random.key(3))
    assert result.X.shape == (19, 2)
    assert result.Y.shape == (19,)
    assert [m["round"] for m in result.per_round_metrics] == [0, 1, 2]
    assert [m["n_evals"] for m in result.per_round_metrics] == [17, 18, 19]


def test_loop_with_no_metrics_skips_estimator():
    problem = gaussian2d()
    alg = _algorithm(Random(), n_rounds=2, metrics=())
    result = run(problem, alg, jax.random.key(4))
    assert result.final_metrics == {}
    for row in result.per_round_metrics:
        assert "mmd2" not in row


def test_loop_with_weighted_empirical_baseline():
    """No-GP baseline path: WeightedEmpiricalSurrogatePosterior produces
    a NumericEmpiricalDistribution as the estimate, which satisfies
    SupportsSampling, so ReferenceMMD runs end-to-end."""
    problem = gaussian2d()
    alg = _algorithm(
        Random(),
        n_rounds=3,
        surrogate_posterior_factory=weighted_empirical_factory,
    )
    result = run(problem, alg, jax.random.key(5))
    assert isinstance(result.final_estimate, NumericEmpiricalDistribution)
    assert isinstance(result.final_estimate, SupportsSampling)
    assert "mmd2" in result.final_metrics
    # Final samples must come from design points (X) — that's what the
    # weighted-empirical baseline represents.
    samples = jnp.asarray(
        pp_sample(result.final_estimate, key=jax.random.key(0), sample_shape=(128,))
    )
    # Each sampled row should match some row in result.X.
    matches = jnp.any(jnp.all(samples[:, None, :] == result.X[None, :, :], axis=-1), axis=-1)
    assert bool(jnp.all(matches))


def test_loop_continuous_ei_beats_candidate_set_ei_on_gaussian2d():
    """v1.4 exit criterion: ContinuousMultiStartOptimizer-backed EI should
    yield at-or-below MMD compared to CandidateSetOptimizer-backed EI at
    matched evaluation budgets, on gaussian2d.

    Tolerance: continuous EI must be no worse than candidate-set EI by
    more than 5 % of the candidate-set MMD². This is loose enough to
    survive seed-dependent variance with 4 acquisition rounds, but tight
    enough that a regression in the continuous optimizer would surface.
    """
    problem = gaussian2d()

    cs_acq = ExpectedImprovement(optimizer=CandidateSetOptimizer(n_candidates=512))
    cm_acq = ExpectedImprovement(
        optimizer=ContinuousMultiStartOptimizer(
            n_starts=8,
            n_seeding_candidates=128,
            bfgs_max_steps=40,
        )
    )

    cs_alg = _algorithm(cs_acq, n_rounds=4)
    cm_alg = _algorithm(cm_acq, n_rounds=4)

    cs_result = run(problem, cs_alg, jax.random.key(0))
    cm_result = run(problem, cm_alg, jax.random.key(0))

    cs_mmd2 = float(cs_result.final_metrics["mmd2"])
    cm_mmd2 = float(cm_result.final_metrics["mmd2"])
    assert cm_mmd2 <= cs_mmd2 * 1.05, (
        f"Continuous EI MMD² ({cm_mmd2:.4f}) > CandidateSet EI MMD² "
        f"({cs_mmd2:.4f}) × 1.05; continuous optimization regressed."
    )
