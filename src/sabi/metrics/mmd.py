r"""Maximum Mean Discrepancy (MMD) with an RBF kernel.

For samples :math:`X = \{x_1, \dots, x_m\}` and :math:`Y = \{y_1, \dots, y_n\}`
with kernel :math:`k(x, y) = \exp\!\left(-\|x - y\|^2 / (2 h^2)\right)`,
the unbiased U-statistic estimator of squared MMD (Gretton et al., 2012) is

.. math::

    \widehat{\mathrm{MMD}}_u^2 =
        \frac{1}{m(m-1)} \sum_{i \ne j} k(x_i, x_j)
      + \frac{1}{n(n-1)} \sum_{i \ne j} k(y_i, y_j)
      - \frac{2}{m n} \sum_{i, j} k(x_i, y_j).

Bandwidth :math:`h` defaults to the median heuristic on the pooled sample
:math:`X \cup Y` — the standard off-the-shelf choice for
posterior-comparison MMD.

Note: the unbiased estimator can be slightly negative for finite samples.
We return the raw value; callers that want a non-negative distance should
take ``sqrt(max(mmd2, 0))`` explicitly (or use :func:`mmd_rbf`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
from jax import Array
from probpipe import sample
from probpipe.core.protocols import SupportsSampling

from sabi.metrics.base import Metric, MetricContext


def _pairwise_sq_dists(X: Array, Y: Array) -> Array:
    """Pairwise squared Euclidean distances, shape (m, n)."""
    x2 = jnp.sum(X * X, axis=-1, keepdims=True)
    y2 = jnp.sum(Y * Y, axis=-1, keepdims=True).T
    return jnp.maximum(x2 + y2 - 2.0 * X @ Y.T, 0.0)


def median_heuristic_bandwidth(X: Array, Y: Array) -> Array:
    r"""Median of pairwise distances across the pooled sample.

    .. math::

        h = \mathrm{median}\big( \{ \|z_i - z_j\| : i \ne j \} \big),
        \quad Z = X \cup Y.

    Returns a scalar lengthscale :math:`h` for use in
    :math:`k(x, y) = \exp\!\left(-\|x - y\|^2 / (2 h^2)\right)`.
    """
    Z = jnp.concatenate([X, Y], axis=0)
    sq = _pairwise_sq_dists(Z, Z)
    n = Z.shape[0]
    # Use only off-diagonal entries.
    mask = ~jnp.eye(n, dtype=bool)
    dists = jnp.sqrt(sq[mask])
    med = jnp.median(dists)
    return jnp.where(med > 0, med, 1.0)


def mmd2_unbiased(X: Array, Y: Array, bandwidth: float | Array | None = None) -> Array:
    r"""Unbiased MMD² estimator with RBF kernel.

    Computes :math:`\widehat{\mathrm{MMD}}_u^2(X, Y)` per the formula in
    the module docstring, with :math:`k(x, y) = \exp(-\|x-y\|^2 / (2 h^2))`.

    Args:
        X: shape ``(m, d)`` sample array.
        Y: shape ``(n, d)`` sample array.
        bandwidth: kernel bandwidth :math:`h`. If ``None``, uses the
            median heuristic (:func:`median_heuristic_bandwidth`).

    Returns:
        Scalar MMD² estimate (may be slightly negative for finite samples).
    """
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError(f"Shape mismatch: X {X.shape}, Y {Y.shape}.")
    m, n = X.shape[0], Y.shape[0]
    if m < 2 or n < 2:
        raise ValueError(f"Unbiased MMD² needs m,n ≥ 2, got m={m}, n={n}.")

    h = median_heuristic_bandwidth(X, Y) if bandwidth is None else jnp.asarray(bandwidth)
    inv_2h2 = 1.0 / (2.0 * h * h)

    Kxx = jnp.exp(-_pairwise_sq_dists(X, X) * inv_2h2)
    Kyy = jnp.exp(-_pairwise_sq_dists(Y, Y) * inv_2h2)
    Kxy = jnp.exp(-_pairwise_sq_dists(X, Y) * inv_2h2)

    # Unbiased: remove diagonals from within-sample sums.
    sum_xx = jnp.sum(Kxx) - jnp.sum(jnp.diag(Kxx))
    sum_yy = jnp.sum(Kyy) - jnp.sum(jnp.diag(Kyy))
    term_xx = sum_xx / (m * (m - 1))
    term_yy = sum_yy / (n * (n - 1))
    term_xy = jnp.sum(Kxy) / (m * n)

    return term_xx + term_yy - 2.0 * term_xy


def mmd_rbf(X: Array, Y: Array, bandwidth: float | Array | None = None) -> Array:
    r"""Non-negative MMD distance: :math:`\sqrt{\max(\widehat{\mathrm{MMD}}_u^2, 0)}`."""
    return jnp.sqrt(jnp.maximum(mmd2_unbiased(X, Y, bandwidth=bandwidth), 0.0))


@dataclass(frozen=True)
class MMD(Metric):
    """MMD² (and MMD) against `problem.reference_distribution` under an RBF kernel.

    Bandwidth defaults to the median heuristic (inside `mmd2_unbiased`).
    Returns an empty dict if the problem has no `SupportsSampling` reference.
    """

    bandwidth: float | None = None
    n_estimate_samples: int = 2048
    n_reference_samples: int = 2048

    requires: ClassVar[tuple[type, ...]] = (SupportsSampling,)
    keys: ClassVar[tuple[str, ...]] = ("mmd", "mmd2")

    def __call__(
        self,
        ctx: MetricContext,
        *,
        key: Array,
    ) -> dict[str, float]:
        ref = ctx.problem.reference_distribution
        if ref is None or not isinstance(ref, SupportsSampling):
            return {}

        key_est, key_ref = jax.random.split(key)
        est_samples = jnp.asarray(
            sample(ctx.estimate, key=key_est, sample_shape=(self.n_estimate_samples,))
        )
        ref_samples = jnp.asarray(
            sample(ref, key=key_ref, sample_shape=(self.n_reference_samples,))
        )

        mmd2 = mmd2_unbiased(est_samples, ref_samples, bandwidth=self.bandwidth)
        mmd = jnp.sqrt(jnp.maximum(mmd2, 0.0))
        return {"mmd2": float(mmd2), "mmd": float(mmd)}
