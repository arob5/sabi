"""Batch samplers: draw `n` parameter-space points for the loop and acquisitions.

A `BatchSampler` is the single abstraction for "give me `n` points in
parameter space" — used by:

- the loop's initial-design step (`Algorithm.initial_sampler.sample(...)`)
- `PriorSampling.select_batch` (the prior-sampling acquisition)
- pointwise optimizers' candidate / seed sets (`CandidateSetOptimizer`,
  `ContinuousMultiStartOptimizer`)

Unifying under `BatchSampler` lets users swap in Sobol, LHS, or any
other sampling strategy at the same field they would swap a prior
sampler — no changes to the loop or the optimizers.

Currently ships one concrete sampler: `PriorSampler`, which draws
i.i.d. samples from ``problem.target_distribution.prior``. Sobol and
LHS land alongside the first benchmark that needs deterministic /
low-discrepancy sequences.

Output shape follows `docs/notation.md`:
``X.shape == (n,) + problem.target_distribution.input_shape``.
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

    Implementers return shape
    ``(n,) + problem.target_distribution.input_shape``. The sampler
    sees the full `Problem` so subclasses can read whichever fields
    they need (`prior` for prior-based sampling, `prior.support` for
    low-discrepancy sequences, `input_shape` for shape, etc.) via
    ``problem.target_distribution``.
    """

    @abstractmethod
    def sample(self, problem: Problem, key: Array, n: int) -> Array:
        """Return `(n,) + problem.target_distribution.input_shape` samples."""


@dataclass(frozen=True)
class PriorSampler(BatchSampler):
    """Draw `n` i.i.d. samples from ``problem.target_distribution.prior``.

    The default sampler everywhere a `BatchSampler` is needed. The
    underlying `TargetDistribution` requires a non-None ``prior``, so
    this sampler always works.
    """

    def sample(self, problem: Problem, key: Array, n: int) -> Array:
        drawn = pp_sample(problem.target_distribution.prior, key=key, sample_shape=(n,))
        return jnp.asarray(drawn)
