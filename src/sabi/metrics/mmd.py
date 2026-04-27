"""Maximum Mean Discrepancy (MMD) with an RBF kernel.

We implement the unbiased U-statistic estimator of MMD² (Gretton et al., 2012):

    MMD²_u = (1/(m(m-1))) Σ_{i≠j} k(xᵢ, xⱼ)
           + (1/(n(n-1))) Σ_{i≠j} k(yᵢ, yⱼ)
           - (2/(m n))    Σ_{i,j} k(xᵢ, yⱼ)

Bandwidth defaults to the median heuristic on the pooled sample, which is the
standard off-the-shelf choice for posterior-comparison MMD.

Note: the unbiased estimator can be slightly negative for finite samples. We
return the raw value; callers that want a non-negative distance should take
`sqrt(max(mmd2, 0))` explicitly.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def _pairwise_sq_dists(X: Array, Y: Array) -> Array:
    """Pairwise squared Euclidean distances, shape (m, n)."""
    x2 = jnp.sum(X * X, axis=-1, keepdims=True)
    y2 = jnp.sum(Y * Y, axis=-1, keepdims=True).T
    return jnp.maximum(x2 + y2 - 2.0 * X @ Y.T, 0.0)


def median_heuristic_bandwidth(X: Array, Y: Array) -> Array:
    """Median of pairwise distances across the pooled sample.

    Returns a scalar lengthscale `h` such that `k(x, y) = exp(-||x-y||² / (2 h²))`
    uses the median pairwise Euclidean distance as its bandwidth.
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
    """Unbiased MMD² estimator with RBF kernel.

    Args:
        X, Y: (m, d) and (n, d) sample arrays.
        bandwidth: kernel bandwidth `h` (RBF: k(x,y) = exp(-||x-y||²/(2 h²))).
            If None, uses the median heuristic.

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
    """Non-negative MMD distance = sqrt(max(mmd², 0))."""
    return jnp.sqrt(jnp.maximum(mmd2_unbiased(X, Y, bandwidth=bandwidth), 0.0))
