"""Factories mapping resolved Hydra configs to sabi objects.

Uses plain dispatch on a `name` field rather than `hydra.utils.instantiate`
so that the entry-point code is easy to read; migration to `_target_`
strings is a follow-up once the component tree stabilizes.
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
from sabi.acquisitions.random import DistributionSampling
from sabi.algorithms import (
    Algorithm,
    SurrogateDistributionFactory,
    emulator_pushforward_factory,
    weighted_empirical_factory,
)
from sabi.density_decomposition import DensityDecomposition, LogProbTarget
from sabi.metrics.base import Metric
from sabi.metrics.mmd import MMD
from sabi.metrics.scheduling import MetricTarget, ScheduledMetric
from sabi.problems.banana import banana
from sabi.problems.base import Problem
from sabi.problems.gaussian import gaussian
from sabi.problems.neals_funnel import neals_funnel
from sabi.emulators import TinyGPEmulator


def build_problem(cfg: DictConfig) -> Problem:
    name = cfg.name
    if name == "gaussian":
        mean_cfg = cfg.get("mean", None)
        cov_cfg = cfg.get("cov", None)
        return gaussian(
            d=int(cfg.get("d", 2)),
            mean=tuple(mean_cfg) if mean_cfg is not None else None,
            cov=(
                tuple(tuple(row) for row in cov_cfg) if cov_cfg is not None else None
            ),
            bounds_radius=float(cfg.get("bounds_radius", 5.0)),
        )
    if name == "banana":
        bounds_cfg = cfg.get("bounds", None)
        bounds = (
            tuple(tuple(b) for b in bounds_cfg) if bounds_cfg is not None else None
        )
        return banana(
            d=int(cfg.get("d", 2)),
            a=float(cfg.get("a", 1.0)),
            b=float(cfg.get("b", 4.0)),
            c=float(cfg.get("c", 1.0)),
            bounds=bounds,
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


def build_density_decomposition(
    cfg: DictConfig, *, problem: Problem
) -> DensityDecomposition:
    """Build the algorithm's ``DensityDecomposition`` from config.

    Required Hydra block ``density_decomposition:`` — there is no
    default. Per ``docs/density_decomposition.md`` the user must
    explicitly pick how the emulator composes into log-density.

    Currently dispatched on a ``kind`` field:

    - ``identity_from_target`` — the recommended idiom for benchmark
      problems whose ``TargetDistribution`` carries an analytical
      ``_unnormalized_log_prob``. Equivalent to
      ``DensityDecomposition.identity_from_target(problem.target_distribution)``.

    Future kinds (``likelihood_with_prior``, ``forward_model``) wire in
    here when their config schemas stabilize.
    """
    kind = cfg.get("kind")
    if kind is None:
        raise ValueError(
            "density_decomposition.kind is required. Choose `identity_from_target` "
            "for benchmark problems with analytical density."
        )
    if kind == "identity_from_target":
        # Trivial decomposition: emulator approximates the target's
        # analytical unnormalized log-density directly.
        return LogProbTarget(problem.target_distribution)
    raise ValueError(f"Unknown density_decomposition.kind={kind!r}.")


def _build_emulator_factory(cfg: DictConfig, *, input_shape: tuple[int, ...]):
    """Build a no-arg factory that constructs an `Emulator` with the
    problem's input_shape baked in."""
    name = cfg.name
    if name == "gp":
        def factory() -> TinyGPEmulator:
            return TinyGPEmulator(
                input_shape=input_shape,
                ls_factor=float(cfg.get("ls_factor", 1.5)),
                ls_floor=float(cfg.get("ls_floor", 0.05)),
                noise=float(cfg.get("noise", 1e-4)),
                jitter=float(cfg.get("jitter", 1e-3)),
            )
        return factory
    if name == "dsp_gp":
        from sabi.emulators.gpjax import DSPGPEmulator

        def factory() -> "DSPGPEmulator":
            return DSPGPEmulator(
                input_shape=input_shape,
                kernel=str(cfg.get("kernel", "rbf")),
                max_iters=int(cfg.get("max_iters", 500)),
                jitter=float(cfg.get("jitter", 1e-6)),
                verbose=bool(cfg.get("verbose", False)),
                n_starts=int(cfg.get("n_starts", 1)),
                restart_seed=int(cfg.get("restart_seed", 0)),
            )
        return factory
    raise ValueError(f"Unknown emulator.name={name!r}.")


def _build_optimizer(cfg: DictConfig | None) -> PointwiseOptimizer:
    """Build a `PointwiseOptimizer` from a config subsection."""
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
        inner = _build_optimizer(cfg.get("inner", None))
        return GreedyMultiPointOptimizer(inner=inner)
    raise ValueError(f"Unknown acquisition.optimizer.name={name!r}.")


def _build_acquisition(cfg: DictConfig) -> Acquisition:
    name = cfg.name
    if name == "distribution_sampling":
        return DistributionSampling()
    if name == "ei":
        return ExpectedImprovement(
            optimizer=_build_optimizer(cfg.get("optimizer", None)),
            offset=float(cfg.get("offset", 0.0)),
            best_from=str(cfg.get("best_from", "data")),
        )
    raise ValueError(f"Unknown acquisition.name={name!r}.")


def _build_bare_metric(cfg: DictConfig) -> Metric:
    name = cfg.name
    if name == "mmd":
        bw = cfg.get("bandwidth", None)
        return MMD(bandwidth=None if bw is None else float(bw))
    raise ValueError(f"Unknown metric.name={name!r}.")


_SCHEDULING_FIELDS = frozenset({"every", "target", "final", "name_suffix"})


def _build_metric(cfg: DictConfig) -> Metric | ScheduledMetric:
    """Build a single metric entry from YAML."""
    metric = _build_bare_metric(cfg)
    if not any(f in cfg for f in _SCHEDULING_FIELDS):
        return metric
    target_cfg = cfg.get("target", "current")
    target = MetricTarget(str(target_cfg).lower())
    return ScheduledMetric(
        metric=metric,
        every=int(cfg.get("every", 1)),
        target=target,
        final=bool(cfg.get("final", True)),
        name_suffix=str(cfg.get("name_suffix", "")),
    )


def _build_metrics(cfg: DictConfig) -> tuple[Metric | ScheduledMetric, ...]:
    metrics_cfg = cfg.get("metrics", None)
    if metrics_cfg is None:
        return ()
    return tuple(_build_metric(m) for m in metrics_cfg)


def _build_surrogate_distribution_factory(name: str) -> SurrogateDistributionFactory:
    if name == "emulator_pushforward":
        return emulator_pushforward_factory
    if name == "weighted_empirical":
        return weighted_empirical_factory
    raise ValueError(f"Unknown surrogate_distribution factory: {name!r}.")


def build_algorithm(cfg: DictConfig, *, problem: Problem) -> Algorithm:
    """Build the `Algorithm` from config.

    Wires in the required ``DensityDecomposition`` (via
    :func:`build_density_decomposition`) and leaves
    ``initial_design_distribution`` / ``x_support`` at ``None`` so the
    loop's resolver fills them from
    ``problem.target_distribution.support`` at run time.
    """
    decomposition = build_density_decomposition(
        cfg.density_decomposition, problem=problem
    )
    return Algorithm(
        emulator_factory=_build_emulator_factory(
            cfg.emulator, input_shape=problem.target_distribution.event_shape
        ),
        acquisition=_build_acquisition(cfg.acquisition),
        density_decomposition=decomposition,
        surrogate_distribution_factory=_build_surrogate_distribution_factory(
            str(cfg.algorithm.get("surrogate_distribution", "emulator_pushforward"))
        ),
        n_initial=int(cfg.algorithm.n_initial),
        n_rounds=int(cfg.algorithm.n_rounds),
        q=int(cfg.algorithm.get("q", 1)),
        metrics=_build_metrics(cfg),
    )
