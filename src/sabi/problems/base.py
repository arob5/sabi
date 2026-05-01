"""`Problem` — a benchmark inference problem bundle.

The mathematical content of a problem (target function + form + prior +
support) lives in `target_distribution: TargetDistribution`. The
`Problem` layer adds benchmark-suite metadata: a name, an optional
reference posterior, and any reproducibility info. Convenience
``@property`` accessors mirror the inner target distribution so existing
callers (acquisitions, metrics, the loop) read the same fields they
always have.

This split — math vs. benchmark identity — was the right framing
because:

- A `TargetDistribution` is a self-contained mathematical object that
  can be consumed by ProbPipe ops directly (`condition_on`,
  `unnormalized_log_prob`, etc.). No `Problem` wrapping required.
- `TemperingScheme` (Step 3) operates on `TargetDistribution` to produce
  intermediate targets, with no awareness of `Problem`-level metadata.
- A future `BenchmarkProblem` (issue #2) can subclass / extend `Problem`
  with validated reference artifacts without touching the math layer.

Shape conventions follow ProbPipe's `ArrayRandomFunction` (see
`docs/notation.md`): a single input has shape `input_shape`, a single
output has shape `output_shape`, and design sets `X` / `Y` prepend a
batch dimension. ``target_function`` is the **batched** view;
``target_single`` is the per-point view. See
`sabi.problems.target_distribution.TargetDistribution`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from jax import Array
from probpipe.core._distribution_base import Distribution
from probpipe.core.constraints import Constraint

from sabi.problems.forms import LogDensityForm
from sabi.target_distribution import TargetDistribution


@dataclass(frozen=True)
class Problem:
    """A benchmark inference problem.

    Attributes:
        target_distribution: the mathematical target — a
            `TargetDistribution` carrying the target function, form,
            prior, and support.
        reference_distribution: optional ground-truth posterior used by
            reference-based metrics (analytic when one fits naturally,
            otherwise an `EmpiricalDistribution` over precomputed
            samples).
        name: human-readable benchmark name (e.g. ``"gaussian2d"``,
            ``"banana"``). Used for cache keys and metadata.
    """

    target_distribution: TargetDistribution
    reference_distribution: Distribution | None = None
    name: str = ""

    # ------------------------------------------------------------------------
    # Convenience accessors mirroring the inner target_distribution
    # ------------------------------------------------------------------------

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.target_distribution.input_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.target_distribution.output_shape

    @property
    def target_function(self) -> Callable[[Array], Array]:
        return self.target_distribution.target_function

    @property
    def target_single(self) -> Callable[[Array], Array]:
        return self.target_distribution.target_single

    @property
    def log_density_form(self) -> LogDensityForm:
        return self.target_distribution.log_density_form

    @property
    def prior(self) -> Distribution:
        return self.target_distribution.prior

    @property
    def support(self) -> Constraint:
        """Support of the parameter space (delegates to ``prior.support``)."""
        return self.target_distribution.support

    def log_posterior(self, x: Array) -> Array:
        """Single-point unnormalized log-posterior at ``x`` (shape ``input_shape``).

        Convenience for tests / debugging — not used in the hot loop.
        Equivalent to ``target_distribution.unnormalized_log_prob(x)``
        via the ProbPipe op.
        """
        y = self.target_single(x)
        return self.log_density_form(x, y, prior=self.prior)


@dataclass(frozen=True)
class BenchmarkProblem(Problem):
    """A `Problem` with a locked-in, validated reference posterior.

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
            posterior or its reference artifact genuinely changes. The
            canonical operation when a benchmark needs to evolve is to
            *introduce a new name* (posteriordb-style). The version
            field exists for the rare case where a fix to a reference
            generation pipeline is rolled into the same name and tests
            need a way to declare which artifact they trust.
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

