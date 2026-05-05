import pytest
from omegaconf import OmegaConf

from sabi.algorithms import (
    Algorithm,
    emulator_pushforward_factory,
    weighted_empirical_factory,
)
from sabi.metrics import MetricTarget, ScheduledMetric
from sabi.metrics.mmd import MMD
from sabi.problems.base import Problem
from sabi.runner.build import build_algorithm, build_problem


def _cfg(with_metric=True):
    d = {
        "problem": {
            "name": "gaussian",
            "d": 2,
            "mean": [0.0, 0.0],
            "cov": [[1.0, 0.0], [0.0, 1.0]],
            "bounds_radius": 5.0,
        },
        "emulator": {"name": "gp"},
        "acquisition": {"name": "prior_sampling"},
        "algorithm": {"n_initial": 4, "n_rounds": 2, "q": 1},
        "seed": 0,
    }
    if with_metric:
        d["metrics"] = [{"name": "mmd"}]
    return OmegaConf.create(d)


def test_build_problem_returns_problem():
    problem = build_problem(_cfg().problem)
    assert isinstance(problem, Problem)
    assert problem.name == "gaussian"
    assert problem.input_shape == (2,)
    assert problem.output_shape == ()


def test_build_algorithm_wires_components():
    cfg = _cfg()
    alg = build_algorithm(cfg, problem=build_problem(cfg.problem))
    assert isinstance(alg, Algorithm)
    assert alg.n_initial == 4
    assert alg.n_rounds == 2
    from sabi.emulators import TinyGPEmulator

    assert isinstance(alg.emulator_factory(), TinyGPEmulator)
    assert len(alg.metrics) == 1
    assert isinstance(alg.metrics[0], MMD)


def test_build_algorithm_without_metrics_yields_empty_tuple():
    cfg = _cfg(with_metric=False)
    alg = build_algorithm(cfg, problem=build_problem(cfg.problem))
    assert alg.metrics == ()


def test_build_algorithm_default_surrogate_distribution_is_emulator_pushforward():
    cfg = _cfg()
    alg = build_algorithm(cfg, problem=build_problem(cfg.problem))
    assert alg.surrogate_distribution_factory is emulator_pushforward_factory


def test_build_algorithm_can_select_weighted_empirical_factory():
    cfg = _cfg()
    cfg.algorithm.surrogate_distribution = "weighted_empirical"
    alg = build_algorithm(cfg, problem=build_problem(cfg.problem))
    assert alg.surrogate_distribution_factory is weighted_empirical_factory


def test_build_algorithm_parses_scheduled_metric_fields():
    cfg = _cfg()
    # Replace bare metric with one carrying the new scheduling fields.
    cfg.metrics = [
        {
            "name": "mmd",
            "every": 5,
            "target": "terminal",
            "final": False,
            "name_suffix": "term",
        }
    ]
    alg = build_algorithm(cfg, problem=build_problem(cfg.problem))
    assert len(alg.metrics) == 1
    sm = alg.metrics[0]
    assert isinstance(sm, ScheduledMetric)
    assert isinstance(sm.metric, MMD)
    assert sm.every == 5
    assert sm.target == MetricTarget.TERMINAL
    assert sm.final is False
    assert sm.name_suffix == "term"


def test_build_algorithm_bare_metric_returns_metric_not_scheduled():
    """A YAML entry without scheduling fields stays a bare Metric (the
    loop auto-wraps with defaults at run time)."""
    cfg = _cfg()
    alg = build_algorithm(cfg, problem=build_problem(cfg.problem))
    # The default _cfg uses just `- name: mmd` — bare.
    assert isinstance(alg.metrics[0], MMD)


# -------------------------------------------------------------------------
# Error paths: each `raise ValueError` in build.py exercised directly.
# Catches typos in YAML configs at the test stage rather than at run time.
# -------------------------------------------------------------------------


def test_build_problem_unknown_name_raises():
    cfg = _cfg()
    cfg.problem.name = "not_a_real_problem"
    with pytest.raises(ValueError, match="problem.name"):
        build_problem(cfg.problem)


def test_build_algorithm_unknown_emulator_name_raises():
    cfg = _cfg()
    problem = build_problem(cfg.problem)
    cfg.emulator.name = "not_a_real_emulator"
    with pytest.raises(ValueError, match="emulator.name"):
        build_algorithm(cfg, problem=problem)


def test_build_algorithm_unknown_acquisition_name_raises():
    cfg = _cfg()
    problem = build_problem(cfg.problem)
    cfg.acquisition.name = "not_a_real_acquisition"
    with pytest.raises(ValueError, match="acquisition.name"):
        build_algorithm(cfg, problem=problem)


def test_build_algorithm_unknown_optimizer_name_raises():
    """`acquisition.optimizer.name` is a sub-dispatch under EI; a bad
    name should raise from `_build_optimizer`."""
    cfg = _cfg()
    cfg.acquisition = OmegaConf.create(
        {
            "name": "ei",
            "optimizer": {"name": "not_a_real_optimizer"},
        }
    )
    problem = build_problem(cfg.problem)
    with pytest.raises(ValueError, match="acquisition.optimizer.name"):
        build_algorithm(cfg, problem=problem)


def test_build_algorithm_unknown_metric_name_raises():
    cfg = _cfg()
    cfg.metrics = [{"name": "not_a_real_metric"}]
    problem = build_problem(cfg.problem)
    with pytest.raises(ValueError, match="metric.name"):
        build_algorithm(cfg, problem=problem)


def test_build_algorithm_unknown_surrogate_distribution_factory_raises():
    cfg = _cfg()
    cfg.algorithm.surrogate_distribution = "not_a_real_factory"
    problem = build_problem(cfg.problem)
    with pytest.raises(ValueError, match="surrogate_distribution"):
        build_algorithm(cfg, problem=problem)
