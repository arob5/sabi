import jax
import jax.numpy as jnp

from sabi.metrics.mmd import median_heuristic_bandwidth, mmd2_unbiased, mmd_rbf


def test_mmd2_small_for_same_distribution():
    key1, key2 = jax.random.split(jax.random.key(0))
    X = jax.random.normal(key1, shape=(300, 2))
    Y = jax.random.normal(key2, shape=(300, 2))
    mmd2 = float(mmd2_unbiased(X, Y))
    assert abs(mmd2) < 5e-2


def test_mmd2_large_for_different_distributions():
    key1, key2 = jax.random.split(jax.random.key(1))
    X = jax.random.normal(key1, shape=(300, 2))
    Y = jax.random.normal(key2, shape=(300, 2)) + 3.0  # large mean shift
    mmd2 = float(mmd2_unbiased(X, Y))
    # Should be clearly positive and much larger than the same-distribution case.
    assert mmd2 > 0.1


def test_mmd_rbf_non_negative():
    key1, key2 = jax.random.split(jax.random.key(2))
    X = jax.random.normal(key1, shape=(200, 3))
    Y = jax.random.normal(key2, shape=(250, 3)) + 0.5
    assert float(mmd_rbf(X, Y)) >= 0.0


def test_median_heuristic_positive_and_finite():
    X = jax.random.normal(jax.random.key(3), shape=(100, 2))
    Y = jax.random.normal(jax.random.key(4), shape=(100, 2))
    h = float(median_heuristic_bandwidth(X, Y))
    assert jnp.isfinite(h)
    assert h > 0.0


def test_mmd2_grows_with_mean_shift():
    key1, key2 = jax.random.split(jax.random.key(5))
    X = jax.random.normal(key1, shape=(300, 2))
    shifts = [0.0, 0.5, 1.5, 3.0]
    values = []
    for s in shifts:
        Y = jax.random.normal(key2, shape=(300, 2)) + s
        values.append(float(mmd2_unbiased(X, Y)))
    # Monotonic in the mean shift (up to sampling noise).
    for v_prev, v_next in zip(values[:-1], values[1:]):
        assert v_next >= v_prev - 1e-3
