"""Factories mapping resolved Hydra configs to sabi objects.

v0 uses plain dispatch on a `name` field rather than hydra.utils.instantiate so
that the entry-point code is easy to read; we can migrate to `_target_` strings
in v1 once the component tree stabilizes.
"""

from __future__ import annotations

from omegaconf import DictConfig

from sabi.acquisitions.base import Acquisition
from sabi.acquisitions.ei import ExpectedImprovement
from sabi.acquisitions.optim import (
    CandidateSetOptimizer,
    ContinuousMultiStartOptimizer,
    GreedyMultiPointOptimizer,
    PointwiseOptimizer,
)
from sabi.acquisitions.random import PriorSampling
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
from sabi.problems.neals_funnel import neals_funnel
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
    if name == "neals_funnel":
        return neals_funnel(
            d=int(cfg.get("d", 2)),
            sigma_v=float(cfg.get("sigma_v", 3.0)),
            v_bound=float(cfg.get("v_bound", 9.0)),
            x_bound=float(cfg.get("x_bound", 30.0)),
            num_results=int(cfg.get("num_results", 2000)),
            num_warmup=int(cfg.get("num_warmup", 2000)),
            num_chains=int(cfg.get("num_chains", 4)),
            random_seed=int(cfg.get("random_seed", 0)),
        )
    raise ValueError(f"Unknown problem.name={name!r}.")


def _build_surrogate_factory(cfg: DictConfig, *, input_shape: tuple[int, ...]):
    """Build a no-arg factory that constructs a `Surrogate` with the
    problem's input_shape baked in."""
    name = cfg.name
    if name == "gp":
        def factory() -> GPSurrogate:
            return GPSurrogate(
                input_shape=input_shape,
                ls_factor=float(cfg.get("ls_factor", 1.5)),
                ls_floor=float(cfg.get("ls_floor", 0.05)),
                noise=float(cfg.get("noise", 1e-4)),
                jitter=float(cfg.get("jitter", 1e-3)),
            )
        return factory
    raise ValueError(f"Unknown surrogate.name={name!r}.")


def _build_optimizer(cfg: DictConfig | None) -> PointwiseOptimizer:
    """Build a `PointwiseOptimizer` from a config subsection. None →
    `CandidateSetOptimizer()` for backwards compatibility."""
    if cfg is None:
        return CandidateSetOptimizer()
    name = cfg.get("name", "candidate_set")
    if name == "candidate_set":
        return CandidateSetOptimizer(n_candidates=int(cfg.get("n_candidates", 1024)))
    if name == "continuous_multistart":
        return ContinuousMultiStartOptimizer(
            n_starts=int(cfg.get("n_starts", 16)),
            n_seeding_candidates=int(cfg.get("n_seeding_candidates", 256)),
            bfgs_max_steps=int(cfg.get("bfgs_max_steps", 50)),
            bfgs_rtol=float(cfg.get("bfgs_rtol", 1e-5)),
            bfgs_atol=float(cfg.get("bfgs_atol", 1e-5)),
        )
    if name == "greedy":
        # Greedy wraps an inner optimizer. Inner config under `cfg.inner`.
        inner = _build_optimizer(cfg.get("inner", None))
        # Imputer wiring is left minimal in v1.4: kriging_believer is the
        # default; richer config support lands when v1.5 needs it.
        return GreedyMultiPointOptimizer(inner=inner)
    raise ValueError(f"Unknown acquisition.optimizer.name={name!r}.")


def _build_acquisition(cfg: DictConfig) -> Acquisition:
    name = cfg.name
    if name == "prior_sampling":
        return PriorSampling()
    if name == "ei":
        return ExpectedImprovement(
            optimizer=_build_optimizer(cfg.get("optimizer", None)),
            offset=float(cfg.get("offset", cfg.get("xi", 0.0))),
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


def build_algorithm(cfg: DictConfig, *, problem: Problem) -> Algorithm:
    """Build the `Algorithm` from config, baking in problem shape metadata
    where downstream components need it (e.g., the surrogate factory)."""
    return Algorithm(
        surrogate_factory=_build_surrogate_factory(
            cfg.surrogate, input_shape=problem.input_shape
        ),
        acquisition=_build_acquisition(cfg.acquisition),
        surrogate_posterior_factory=_build_surrogate_posterior_factory(
            str(cfg.algorithm.get("surrogate_posterior", "gp_pushforward"))
        ),
        n_initial=int(cfg.algorithm.n_initial),
        n_rounds=int(cfg.algorithm.n_rounds),
        q=int(cfg.algorithm.get("q", 1)),
        metrics=_build_metrics(cfg),
    )
