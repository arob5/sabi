"""Batch samplers: draw `n` parameter-space points for the loop and acquisitions.

A `BatchSampler` is the single abstraction for "give me `n` points in
parameter space" — used by:

- the loop's initial-design step (`Algorithm.initial_sampler.sample(...)`)
- `PriorSampling.select_batch` (the prior-sampling acquisition)
- pointwise optimizers' candidate / seed sets (`CandidateSetOptimizer`,
  `ContinuousMultiStartOptimizer`)

Each call site previously rolled its own logic on top of `sample_initial`.
Unifying under `BatchSampler` lets users swap in Sobol, LHS, or any other
sampling strategy at the same field they would swap a prior sampler — no
changes to the loop or the optimizers.

v1.4.1 ships one concrete sampler: `PriorSampler`, which draws i.i.d.
samples from `problem.prior`. Sobol and LHS land alongside the first
benchmark that needs deterministic / low-discrepancy sequences.

Output shape follows `docs/notation.md`:
``X.shape == (n,) + problem.input_shape``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from probpipe import sample as pp_sample

from sabi.problems.base import Problem


class BatchSampler(ABC):
    """Strategy for drawing `n` parameter-space points.

    Implementers return shape ``(n,) + problem.input_shape``. The sampler
    sees the full `Problem` so subclasses can read whichever fields they
    need (`prior` for prior-based sampling, `support` for low-discrepancy
    sequences, `input_shape` for shape, etc.).
    """

    @abstractmethod
    def sample(self, problem: Problem, key: Array, n: int) -> Array:
        """Return `(n,) + problem.input_shape` samples."""


@dataclass(frozen=True)
class PriorSampler(BatchSampler):
    """Draw `n` i.i.d. samples from ``problem.prior``.

    Requires ``problem.prior is not None``; raises ``ValueError`` otherwise.
    The default sampler everywhere a `BatchSampler` is needed.
    """

    def sample(self, problem: Problem, key: Array, n: int) -> Array:
        if problem.prior is None:
            raise ValueError(
                f"PriorSampler requires problem.prior, but problem "
                f"{problem.name!r} has prior=None. Provide an explicit "
                "BatchSampler."
            )
        drawn = pp_sample(problem.prior, key=key, sample_shape=(n,))
        return jnp.asarray(drawn)
