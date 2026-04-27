"""Initial designs: draw the first batch of parameter locations.

v0 ships one strategy — sample from `problem.prior`. Sobol and LHS are
straightforward additions for v1.4+.

Output shape follows the convention in `docs/notation.md`:
`X.shape == (n,) + problem.input_shape`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import probpipe.core.ops as pp_ops
from jax import Array

from sabi.problems.base import Problem


@dataclass(frozen=True)
class InitialDesign:
    """Marker type; subclasses implement `sample(problem, key, n)`."""

    def sample(self, problem: Problem, key: Array, n: int) -> Array:
        raise NotImplementedError


@dataclass(frozen=True)
class FromPrior(InitialDesign):
    """Sample `n` points from `problem.prior`."""

    def sample(self, problem: Problem, key: Array, n: int) -> Array:
        return sample_initial(problem, key, n, design=None)


def sample_initial(
    problem: Problem,
    key: Array,
    n: int,
    design: InitialDesign | None = None,
) -> Array:
    """Draw `n` initial-design points for `problem`.

    Defaults to sampling from `problem.prior` (the design distribution). An
    explicit `design` overrides this.
    """
    if design is not None:
        return design.sample(problem, key, n)
    if problem.prior is None:
        raise ValueError(
            f"Problem {problem.name!r} has no prior and no explicit InitialDesign."
        )
    samples = pp_ops.sample(problem.prior, key=key, sample_shape=(n,))
    return jnp.asarray(samples)
