"""`Problem` — a benchmark inference problem bundle.

Shape conventions follow ProbPipe's `ArrayRandomFunction` (see
`docs/notation.md`): a single target input has shape `input_shape`, a
single target output has shape `output_shape`, and design sets `X` / `Y`
prepend a batch dimension.

Vectorization. ``target_function`` is **batched** — it takes
``X`` of shape ``(n,) + input_shape`` and returns ``Y`` of shape
``(n,) + output_shape``. Mirrors `PointwiseScoredAcquisition.score`'s
batched contract, so callers (the loop, tests) never need
``jax.vmap(...)`` at the call site. Builders that have a natural
single-point implementation should `jax.vmap` it before storing —
the convenience helper :meth:`Problem.from_target_single` does this.

Schema is built around ProbPipe primitives:

- `prior` is a ProbPipe `Distribution` — used for initial-design and
  random-acquisition sampling. For benchmarks where there's no Bayesian
  prior (e.g. analytic posteriors expressed directly via
  `target_function`), `prior` doubles as the design distribution.
- `support` is a ProbPipe `Constraint` — metadata describing the
  parameter space; consumed by acquisitions / metrics that want to check
  feasibility.
- `reference_distribution` is a ProbPipe `Distribution` representing the
  ground-truth posterior used by reference-based metrics (analytic when
  one fits naturally, otherwise an `EmpiricalDistribution` over
  precomputed samples).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
from jax import Array
from probpipe.core._distribution_base import Distribution
from probpipe.core.constraints import Constraint

from sabi.problems.forms import LogDensityForm


@dataclass(frozen=True)
class Problem:
    """A benchmark inference problem.

    The emulator fits `target_function`. The unnormalized log-posterior at
    `x` is reconstructed by applying `log_density_form` to
    `(x, target_function(x[None])[0])`.

    `target_function` is **batched**: ``(n,) + input_shape -> (n,) + output_shape``.
    """

    name: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    target_function: Callable[[Array], Array]
    log_density_form: LogDensityForm
    prior: Distribution | None = None
    support: Constraint | None = None
    reference_distribution: Distribution | None = None

    @classmethod
    def from_target_single(
        cls,
        *,
        target_single: Callable[[Array], Array],
        **kwargs: Any,
    ) -> Problem:
        """Build a `Problem` from a single-point ``target_single`` callable.

        Wraps ``target_single`` with `jax.vmap` to produce the batched
        ``target_function``. The wrapped single-point function is also
        cached on the instance as ``_target_single`` for paths that need
        per-point evaluation (e.g., NUTS-based reference generation in
        ``sabi.reference.nuts``).
        """
        target_function = jax.vmap(target_single)
        problem = cls(target_function=target_function, **kwargs)
        # Stash the single-point function on the instance — bypasses the
        # frozen dataclass, but is the natural place for it since callers
        # that want per-point access shouldn't reach back into the
        # benchmark builder.
        object.__setattr__(problem, "_target_single", target_single)
        return problem

    @property
    def target_single(self) -> Callable[[Array], Array]:
        """Single-point view of ``target_function``: ``input_shape -> output_shape``.

        If the problem was built via :meth:`from_target_single`, returns
        the original single-point callable. Otherwise wraps
        ``target_function`` to extract one point at a time (slower; a
        thin convenience for benchmarks that only define the batched
        version).
        """
        cached = self.__dict__.get("_target_single")
        if cached is not None:
            return cached
        return lambda x: self.target_function(x[None])[0]

    def log_posterior(self, x: Array) -> Array:
        """Single-point unnormalized log-posterior at ``x`` (shape ``input_shape``).

        Convenience for tests / debugging — not used in the hot loop.
        """
        y = self.target_single(x)
        return self.log_density_form(x, y, prior=self.prior)
