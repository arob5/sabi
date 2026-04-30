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
from typing import Any

from jax import Array
from probpipe.core._distribution_base import Distribution
from probpipe.core.constraints import Constraint

from sabi.problems.forms import LogDensityForm
from sabi.problems.target_distribution import TargetDistribution


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
    def prior(self) -> Distribution | None:
        return self.target_distribution.prior

    @property
    def support(self) -> Constraint | None:
        return self.target_distribution.support

    def log_posterior(self, x: Array) -> Array:
        """Single-point unnormalized log-posterior at ``x`` (shape ``input_shape``).

        Convenience for tests / debugging — not used in the hot loop.
        Equivalent to ``target_distribution.unnormalized_log_prob(x)``
        via the ProbPipe op.
        """
        y = self.target_single(x)
        return self.log_density_form(x, y, prior=self.prior)

    # ------------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------------

    @classmethod
    def from_target_single(
        cls,
        *,
        target_single: Callable[[Array], Array],
        name: str,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        log_density_form: LogDensityForm,
        prior: Distribution | None = None,
        support: Constraint | None = None,
        reference_distribution: Distribution | None = None,
        target_distribution_name: str | None = None,
    ) -> Problem:
        """Build a `Problem` from a single-point ``target_single`` callable.

        Constructs the underlying `TargetDistribution` via
        :meth:`TargetDistribution.from_target_single`. Returns the
        `Problem` bundle.

        Args:
            target_single: single-point target ``input_shape -> output_shape``.
            name: benchmark name (also used as the default
                target_distribution name).
            input_shape, output_shape: shape conventions.
            log_density_form: how the target output composes into the
                unnormalized log-posterior.
            prior, support: passed to the `TargetDistribution`.
            reference_distribution: optional ground-truth posterior.
            target_distribution_name: optional override for the inner
                target's name; defaults to ``f"{name}_target"``.
        """
        td = TargetDistribution.from_target_single(
            target_single=target_single,
            name=target_distribution_name or f"{name}_target",
            input_shape=input_shape,
            output_shape=output_shape,
            log_density_form=log_density_form,
            prior=prior,
            support=support,
        )
        return cls(
            target_distribution=td,
            reference_distribution=reference_distribution,
            name=name,
        )

    # Legacy constructor support — for tests / call sites that bypass
    # `from_target_single` and build a Problem from individual
    # components. New code should prefer `from_target_single` or
    # construct a `TargetDistribution` directly.
    @classmethod
    def from_components(
        cls,
        *,
        name: str,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        target_function: Callable[[Array], Array],
        target_single: Callable[[Array], Array],
        log_density_form: LogDensityForm,
        prior: Distribution | None = None,
        support: Constraint | None = None,
        reference_distribution: Distribution | None = None,
        target_distribution_name: str | None = None,
    ) -> Problem:
        """Build a `Problem` from already-constructed batched + single
        target callables.

        Use this when you have explicit batched and single-point
        functions that aren't a simple `jax.vmap` of each other (e.g.,
        the batched version uses a more efficient implementation).
        """
        td = TargetDistribution(
            name=target_distribution_name or f"{name}_target",
            input_shape=input_shape,
            output_shape=output_shape,
            target_function=target_function,
            target_single=target_single,
            log_density_form=log_density_form,
            prior=prior,
            support=support,
        )
        return cls(
            target_distribution=td,
            reference_distribution=reference_distribution,
            name=name,
        )
