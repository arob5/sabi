"""`Problem` — a named target distribution with an optional reference solution.

A `Problem` is the *identity* layer: what defines the inference problem
mathematically, plus a human-readable name and (optionally) a reference
solution used by reference-based metrics. The mathematical content
itself — target function, form, design prior, support, batched vs.
per-point views — lives on ``target_distribution: TargetDistribution``.

Algorithmic-pipeline choices (which form, which design prior, vmapped
vs. per-point views of the target) are reachable via
``problem.target_distribution.X``. That extra hop is the point: it keeps
``Problem`` honest about what's a problem-defining fact versus a
consumer-side configuration.

This split — math vs. benchmark identity — was the right framing
because:

- A `TargetDistribution` is a self-contained mathematical object that
  can be consumed by ProbPipe ops directly (`condition_on`,
  `unnormalized_log_prob`, etc.). No `Problem` wrapping required.
- `TemperingScheme` operates on `TargetDistribution` to produce
  intermediate targets, with no awareness of `Problem`-level metadata.
- `BenchmarkProblem` extends `Problem` with validated reference
  artifacts (locked-in name, ``artifact_version``) without touching
  the math layer.

Shape conventions follow ProbPipe's `ArrayRandomFunction` (see
`docs/notation.md`): a single input has shape ``input_shape``, a single
output has shape ``output_shape``, and design sets ``X`` / ``Y``
prepend a batch dimension. Reach through
``problem.target_distribution`` for the batched (``target_map``) or
per-point (``target_single``) view. See
`sabi.target_distribution.TargetDistribution`.
"""

from __future__ import annotations

from dataclasses import dataclass

from probpipe.core._distribution_base import Distribution

from sabi.target_distribution import TargetDistribution


@dataclass(frozen=True)
class Problem:
    """A named target distribution with an optional reference solution.

    Attributes:
        target_distribution: the mathematical target — a
            `TargetDistribution` carrying the target function, form,
            design prior, and support.
        reference_distribution: optional ground-truth target distribution
            used by reference-based metrics (analytic when one fits
            naturally, otherwise an `EmpiricalDistribution` over
            precomputed samples).
        name: human-readable benchmark name (e.g. ``"gaussian"``,
            ``"banana"``). Used for cache keys and metadata.
    """

    target_distribution: TargetDistribution
    reference_distribution: Distribution | None = None
    name: str = ""


@dataclass(frozen=True)
class BenchmarkProblem(Problem):
    """A `Problem` with a locked-in, validated reference solution.

    Each named `BenchmarkProblem` corresponds to a single fixed
    configuration of a problem family (parameters baked into the
    factory that produced it). Changing the parameters yields a *new*
    benchmark, not a mutation — the posteriordb invariant: a name
    refers to one log-density up to a normalizing constant.

    Validation invariants enforced at construction:

    - ``reference_distribution`` must be non-None — that's the whole
      point of the validated tier. Use bare `Problem` if you want a
      flexible instance without a reference.
    - ``name`` must be non-empty — names are the identity.
    - The frozen dataclass blocks post-hoc parameter mutation.

    Attributes:
        artifact_version: opaque tag (e.g. ``"v1"``) bumped when the
            target distribution or its reference artifact genuinely
            changes. The canonical operation when a benchmark needs to
            evolve is to *introduce a new name* (posteriordb-style).
            The version field exists for the rare case where a fix to
            a reference generation pipeline is rolled into the same
            name and tests need a way to declare which artifact they
            trust.
    """

    artifact_version: str = "v1"

    def __post_init__(self):
        if self.reference_distribution is None:
            raise ValueError(
                f"BenchmarkProblem {self.name!r} requires a non-None "
                "reference_distribution. Use Problem for flexible "
                "instances without a validated reference."
            )
        if not self.name:
            raise ValueError(
                "BenchmarkProblem requires a non-empty name; the name "
                "is the benchmark's identity."
            )

    @classmethod
    def from_problem(
        cls,
        problem: Problem,
        *,
        name: str,
        artifact_version: str = "v1",
    ) -> BenchmarkProblem:
        """Promote a flexible `Problem` to a validated `BenchmarkProblem`.

        Copies ``target_distribution`` and ``reference_distribution``
        from the source `Problem`, attaches the locked-in ``name`` and
        ``artifact_version``. The source's own ``name`` is intentionally
        ignored — flexible factories may leave it empty, and the
        benchmark identity is set here.
        """
        return cls(
            target_distribution=problem.target_distribution,
            reference_distribution=problem.reference_distribution,
            name=name,
            artifact_version=artifact_version,
        )
