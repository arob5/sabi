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
