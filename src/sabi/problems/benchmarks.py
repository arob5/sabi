"""Validated `BenchmarkProblem` factories — the curated benchmark suite.

Each factory in this module returns a `BenchmarkProblem` with locked-in
parameters and a trusted reference posterior. These are the entries the
regression test suite runs against; metric values from runs on these
benchmarks are comparable across commits.

Adding a new validated benchmark:

1. Build a flexible `Problem` factory under `sabi/problems/` (e.g.
   `banana(d=…, a=…, b=…)`).
2. Pick a canonical parameter set; the resulting posterior is what the
   benchmark name refers to forever.
3. Wrap it here in a no-arg factory that returns a `BenchmarkProblem`.
   Bumping a benchmark's `artifact_version` is reserved for fixes to
   the reference generation pipeline; changing the posterior itself
   should introduce a new name (posteriordb invariant).

Form variants (e.g., emulating a forward model vs. the log-posterior
directly) are tracked as a follow-up — see issue #11.
"""

from __future__ import annotations

from sabi.problems.banana import banana
from sabi.problems.base import BenchmarkProblem
from sabi.problems.gaussian import gaussian
from sabi.problems.neals_funnel import neals_funnel


def banana_2d() -> BenchmarkProblem:
    """Validated 2-D banana benchmark (a=1, b=4).

    The canonical small-d banana from Haario et al. — analytic reference
    samples, ~4σ default bounds, log-density emulation (`Identity` form).
    Used by visualization tutorials and as a sanity-check benchmark.
    """
    p = banana(d=2, a=1.0, b=4.0)
    return BenchmarkProblem(
        target_distribution=p.target_distribution,
        reference_distribution=p.reference_distribution,
        name="banana_2d",
        artifact_version="v1",
    )


def banana_10d() -> BenchmarkProblem:
    """Validated 10-D banana benchmark (a=1, b=4, c=1).

    Same banana shape on (x_1, x_2); eight independent N(0, 1) filler
    dimensions on top. Tests scaling of the loop in moderate dimension
    while keeping the analytic reference distribution exact.
    """
    p = banana(d=10, a=1.0, b=4.0, c=1.0)
    return BenchmarkProblem(
        target_distribution=p.target_distribution,
        reference_distribution=p.reference_distribution,
        name="banana_10d",
        artifact_version="v1",
    )


def gaussian_2d() -> BenchmarkProblem:
    """Validated 2-D Gaussian benchmark (mean=0, cov=[[1, 0.5], [0.5, 1]]).

    Preserves the historical `gaussian2d` defaults from the pre-refactor
    factory: zero mean, unit marginal variances, off-diagonal correlation
    0.5. Reference is the analytic ProbPipe `MultivariateNormal` itself.
    """
    p = gaussian(d=2, mean=(0.0, 0.0), cov=((1.0, 0.5), (0.5, 1.0)))
    return BenchmarkProblem(
        target_distribution=p.target_distribution,
        reference_distribution=p.reference_distribution,
        name="gaussian_2d",
        artifact_version="v1",
    )


def gaussian_10d() -> BenchmarkProblem:
    """Validated 10-D Gaussian benchmark (mean=0, cov=I_10).

    Isotropic moderate-d Gaussian — a sanity benchmark whose analytic
    posterior is trivially samplable. Useful for emulator-fidelity
    ablations where the curse of dimension matters but the geometry
    doesn't.
    """
    p = gaussian(d=10)
    return BenchmarkProblem(
        target_distribution=p.target_distribution,
        reference_distribution=p.reference_distribution,
        name="gaussian_10d",
        artifact_version="v1",
    )


def neals_funnel_3d() -> BenchmarkProblem:
    """Validated 3-D Neal's funnel (1 v dim + 2 x dims; sigma_v=3).

    Reference samples are loaded from the committed on-disk artifact at
    `reference_posteriors/neals_funnel/d2_sv3.0_vb9.0_xb30.0_*.parquet`,
    generated under the canonical NUTS configuration and the funnel's
    relaxed quality thresholds (max_rhat=1.15, min_ess=30,
    max_divergence_rate=0.10). Those sampler-side knobs are part of the
    benchmark's identity and are pinned here.
    """
    p = neals_funnel(
        d=2,
        sigma_v=3.0,
        v_bound=9.0,
        x_bound=30.0,
        num_results=2000,
        num_warmup=2000,
        num_chains=4,
        random_seed=0,
    )
    return BenchmarkProblem(
        target_distribution=p.target_distribution,
        reference_distribution=p.reference_distribution,
        name="neals_funnel_3d",
        artifact_version="v1",
    )
