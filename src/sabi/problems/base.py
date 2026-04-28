"""`Problem` — a benchmark bundle.

Shape conventions follow ProbPipe's `ArrayRandomFunction` (see
`docs/notation.md`): a single target input has shape `input_shape`, a single
target output has shape `output_shape`, and design sets `X` / `Y` prepend a
batch dimension.

Schema is built around ProbPipe primitives:

- `prior` is a ProbPipe `Distribution` — used for initial-design and
  random-acquisition sampling. For benchmarks where there's no Bayesian prior
  (e.g. analytic posteriors expressed directly via `target_function`),
  `prior` doubles as the design distribution.
- `support` is a ProbPipe `Constraint` — metadata describing the parameter
  space; consumed by acquisitions / metrics that want to check feasibility.
- `reference_distribution` is a ProbPipe `Distribution` representing the
  ground-truth posterior used by reference-based metrics (analytic when one
  fits naturally, otherwise an `EmpiricalDistribution` over precomputed
  samples).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from jax import Array
from probpipe.core._distribution_base import Distribution
from probpipe.core.constraints import Constraint

from sabi.problems.forms import LogDensityForm


@dataclass(frozen=True)
class Problem:
    """A benchmark inference problem.

    The emulator fits `target_function`. The unnormalized log-posterior at `x`
    is reconstructed by applying `log_density_form` to `(x, target_function(x))`.
    """

    name: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    target_function: Callable[[Array], Array]
    log_density_form: LogDensityForm
    prior: Distribution | None = None
    support: Constraint | None = None
    reference_distribution: Distribution | None = None

    def log_posterior(self, x: Array) -> Array:
        y = self.target_function(x)
        return self.log_density_form(x, y, prior=self.prior)
