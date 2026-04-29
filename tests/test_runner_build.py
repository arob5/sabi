from omegaconf import OmegaConf

from sabi.algorithms.loop import (
    Algorithm,
    gp_pushforward_factory,
    weighted_empirical_factory,
)
from sabi.metrics.posterior_mmd import ReferenceMMD
from sabi.problems.base import Problem
from sabi.runner.build import build_algorithm, build_problem


def _cfg(with_metric=True):
    d = {
        "problem": {
            "name": "gaussian2d",
            "mean": [0.0, 0.0],
            "cov": [[1.0, 0.0], [0.0, 1.0]],
            "bounds_radius": 5.0,
        },
        "surrogate": {"name": "gp"},
        "acquisition": {"name": "prior_sampling"},
        "algorithm": {"n_initial": 4, "n_rounds": 2, "q": 1},
        "seed": 0,
    }
    if with_metric:
        d["metrics"] = [{"name": "reference_mmd"}]
    return OmegaConf.create(d)


def test_build_problem_returns_problem():
    problem = build_problem(_cfg().problem)
    assert isinstance(problem, Problem)
    assert problem.name == "gaussian2d"
    assert problem.input_shape == (2,)
    assert problem.output_shape == ()


def test_build_algorithm_wires_components():
    cfg = _cfg()
    alg = build_algorithm(cfg, problem=build_problem(cfg.problem))
    assert isinstance(alg, Algorithm)
    assert alg.n_initial == 4
    assert alg.n_rounds == 2
    from sabi.surrogates.gp import GPSurrogate

    assert isinstance(alg.surrogate_factory(), GPSurrogate)
    assert len(alg.metrics) == 1
    assert isinstance(alg.metrics[0], ReferenceMMD)


def test_build_algorithm_without_metrics_yields_empty_tuple():
    cfg = _cfg(with_metric=False)
    alg = build_algorithm(cfg, problem=build_problem(cfg.problem))
    assert alg.metrics == ()


def test_build_algorithm_default_surrogate_posterior_is_gp_pushforward():
    cfg = _cfg()
    alg = build_algorithm(cfg, problem=build_problem(cfg.problem))
    assert alg.surrogate_posterior_factory is gp_pushforward_factory


def test_build_algorithm_can_select_weighted_empirical_factory():
    cfg = _cfg()
    cfg.algorithm.surrogate_posterior = "weighted_empirical"
    alg = build_algorithm(cfg, problem=build_problem(cfg.problem))
    assert alg.surrogate_posterior_factory is weighted_empirical_factory
