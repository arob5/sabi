import jax
import jax.numpy as jnp
import pytest

from sabi.surrogates.gp import GPSurrogate


def _sample_2d_gp_data(n: int = 40, key_seed: int = 0):
    key = jax.random.key(key_seed)
    X = jax.random.uniform(key, shape=(n, 2), minval=-2.0, maxval=2.0)
    Y = jnp.sum(jnp.sin(X), axis=-1) + 0.5 * X[:, 0] * X[:, 1]
    return X, Y


def test_gp_predict_close_to_training_data():
    X, Y = _sample_2d_gp_data()
    gp = GPSurrogate().fit(X, Y)
    pred = gp.predict(X)
    # With small noise + jitter, mean is close to (but not exactly) training Y.
    rmse = float(jnp.sqrt(jnp.mean((pred.mean - Y) ** 2)))
    y_std = float(jnp.std(Y))
    assert rmse < 0.1 * y_std
    assert jnp.all(pred.variance >= 0.0)


def test_gp_predict_returns_correct_shapes():
    X, Y = _sample_2d_gp_data(n=30)
    gp = GPSurrogate().fit(X, Y)
    Xtest = jax.random.uniform(jax.random.key(1), shape=(7, 2), minval=-2.0, maxval=2.0)
    pred = gp.predict(Xtest)
    assert pred.mean.shape == (7,)
    assert pred.variance.shape == (7,)


def test_gp_predict_before_fit_raises():
    gp = GPSurrogate()
    with pytest.raises(RuntimeError, match="before fit"):
        gp.predict(jnp.zeros((3, 2)))


def test_gp_generalization_better_than_mean_predictor():
    """With fixed hyperparameters (v0 defaults), the GP should still beat the
    naive mean-predictor baseline on held-out points."""
    n_train = 60
    key = jax.random.key(42)
    ktrain, ktest = jax.random.split(key)
    X = jax.random.uniform(ktrain, shape=(n_train, 2), minval=-2.0, maxval=2.0)
    Y = jnp.sum(jnp.sin(X), axis=-1) + 0.5 * X[:, 0] * X[:, 1]

    gp = GPSurrogate().fit(X, Y)

    Xtest = jax.random.uniform(ktest, shape=(30, 2), minval=-1.5, maxval=1.5)
    Ytest = jnp.sum(jnp.sin(Xtest), axis=-1) + 0.5 * Xtest[:, 0] * Xtest[:, 1]
    pred = gp.predict(Xtest)

    gp_rmse = float(jnp.sqrt(jnp.mean((pred.mean - Ytest) ** 2)))
    baseline_rmse = float(jnp.sqrt(jnp.mean((jnp.mean(Y) - Ytest) ** 2)))
    assert gp_rmse < baseline_rmse


def test_lengthscale_shrinks_with_more_data():
    """Median-NN-based lengthscale should decrease as n grows on a fixed box."""
    key = jax.random.key(0)
    X_small = jax.random.uniform(key, shape=(20, 2), minval=-2.0, maxval=2.0)
    X_large = jax.random.uniform(key, shape=(200, 2), minval=-2.0, maxval=2.0)
    Y_small = jnp.zeros(20)
    Y_large = jnp.zeros(200)

    gp_small = GPSurrogate().fit(X_small, Y_small)
    gp_large = GPSurrogate().fit(X_large, Y_large)
    assert float(gp_large.lengthscale) < float(gp_small.lengthscale)
