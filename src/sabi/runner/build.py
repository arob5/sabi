"""Factories mapping resolved Hydra configs to sabi objects.

v0 uses plain dispatch on a `name` field rather than hydra.utils.instantiate so
that the entry-point code is easy to read; we can migrate to `_target_` strings
in v1 once the component tree stabilizes.
"""

from __future__ import annotations

from omegaconf import DictConfig

from sabi.acquisitions.base import Acquisition
from sabi.acquisitions.ei import ExpectedImprovement
from sabi.acquisitions.random import Random
from sabi.algorithms.loop import (
    Algorithm,
    SurrogatePosteriorFactory,
    gp_pushforward_factory,
    weighted_empirical_factory,
)
from sabi.metrics.base import PosteriorMetric
from sabi.metrics.posterior_mmd import ReferenceMMD
from sabi.problems.banana import banana
from sabi.problems.base import Problem
from sabi.problems.gaussian2d import gaussian2d
from sabi.surrogates.gp import GPSurrogate


def build_problem(cfg: DictConfig) -> Problem:
    name = cfg.name
    if name == "gaussian2d":
        return gaussian2d(
            mean=tuple(cfg.get("mean", (0.0, 0.0))),
            cov=tuple(tuple(row) for row in cfg.get("cov", ((1.0, 0.5), (0.5, 1.0)))),
            bounds_radius=float(cfg.get("bounds_radius", 5.0)),
        )
    if name == "banana":
        return banana(
            a=float(cfg.get("a", 1.0)),
            b=float(cfg.get("b", 4.0)),
            bounds=tuple(tuple(b) for b in cfg.get("bounds", ((-4.0, 4.0), (-10.0, 4.0)))),
        )
    raise ValueError(f"Unknown problem.name={name!r}.")


def _build_surrogate_factory(cfg: DictConfig):
    name = cfg.name
    if name == "gp":
        def factory() -> GPSurrogate:
            return GPSurrogate(
                ls_factor=float(cfg.get("ls_factor", 1.5)),
                ls_floor=float(cfg.get("ls_floor", 0.05)),
                noise=float(cfg.get("noise", 1e-4)),
                jitter=float(cfg.get("jitter", 1e-3)),
            )
        return factory
    raise ValueError(f"Unknown surrogate.name={name!r}.")


def _build_acquisition(cfg: DictConfig) -> Acquisition:
    name = cfg.name
    if name == "random":
        return Random()
    if name == "ei":
        return ExpectedImprovement(
            n_candidates=int(cfg.get("n_candidates", 1024)),
            xi=float(cfg.get("xi", 0.0)),
            best_from=str(cfg.get("best_from", "data")),
        )
    raise ValueError(f"Unknown acquisition.name={name!r}.")


def _build_metric(cfg: DictConfig) -> PosteriorMetric:
    name = cfg.name
    if name == "reference_mmd":
        bw = cfg.get("bandwidth", None)
        return ReferenceMMD(bandwidth=None if bw is None else float(bw))
    raise ValueError(f"Unknown metric.name={name!r}.")


def _build_metrics(cfg: DictConfig) -> tuple[PosteriorMetric, ...]:
    metrics_cfg = cfg.get("metrics", None)
    if metrics_cfg is None:
        return ()
    return tuple(_build_metric(m) for m in metrics_cfg)


def _build_surrogate_posterior_factory(name: str) -> SurrogatePosteriorFactory:
    if name == "gp_pushforward":
        return gp_pushforward_factory
    if name == "weighted_empirical":
        return weighted_empirical_factory
    raise ValueError(f"Unknown surrogate_posterior factory: {name!r}.")


def build_algorithm(cfg: DictConfig) -> Algorithm:
    return Algorithm(
        surrogate_factory=_build_surrogate_factory(cfg.surrogate),
        acquisition=_build_acquisition(cfg.acquisition),
        surrogate_posterior_factory=_build_surrogate_posterior_factory(
            str(cfg.algorithm.get("surrogate_posterior", "gp_pushforward"))
        ),
        n_initial=int(cfg.algorithm.n_initial),
        n_rounds=int(cfg.algorithm.n_rounds),
        q=int(cfg.algorithm.get("q", 1)),
        metrics=_build_metrics(cfg),
    )
