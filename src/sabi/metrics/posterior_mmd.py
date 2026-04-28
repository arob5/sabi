"""`ReferenceMMD` — `PosteriorMetric` comparing samples from the posterior
estimate to samples from the problem's `reference_distribution`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
from jax import Array
from probpipe import sample
from probpipe.core._distribution_base import Distribution
from probpipe.core.protocols import SupportsSampling

from sabi.metrics.base import PosteriorMetric
from sabi.metrics.mmd import mmd2_unbiased
from sabi.problems.base import Problem


@dataclass(frozen=True)
class ReferenceMMD(PosteriorMetric):
    """MMD² (and MMD) against `problem.reference_distribution` under an RBF kernel.

    Bandwidth defaults to the median heuristic (inside `mmd2_unbiased`).
    Returns an empty dict if the problem has no `SupportsSampling` reference.
    """

    bandwidth: float | None = None
    n_estimate_samples: int = 2048
    n_reference_samples: int = 2048

    requires: ClassVar[tuple[type, ...]] = (SupportsSampling,)

    def __call__(
        self,
        posterior: Distribution,
        problem: Problem,
        *,
        key: Array,
    ) -> dict[str, float]:
        ref = problem.reference_distribution
        if ref is None or not isinstance(ref, SupportsSampling):
            return {}

        key_est, key_ref = jax.random.split(key)
        est_samples = jnp.asarray(
            sample(posterior, key=key_est, sample_shape=(self.n_estimate_samples,))
        )
        ref_samples = jnp.asarray(
            sample(ref, key=key_ref, sample_shape=(self.n_reference_samples,))
        )

        mmd2 = mmd2_unbiased(est_samples, ref_samples, bandwidth=self.bandwidth)
        mmd = jnp.sqrt(jnp.maximum(mmd2, 0.0))
        return {"mmd2": float(mmd2), "mmd": float(mmd)}
