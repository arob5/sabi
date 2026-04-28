import jax
import jax.numpy as jnp
import pytest
from probpipe import mean, variance
from probpipe.distributions.continuous import Normal

from sabi.surrogates.gp import GPSurrogate


def _sample_2d_gp_data(n: int = 40, key_seed: int = 0):
    key = jax.random.key(key_seed)
    X = jax.random.uniform(key, shape=(n, 2), minval=-2.0, maxval=2.0)
    Y = jnp.sum(jnp.sin(X), axis=-1) + 0.5 * X[:, 0] * X[:, 1]
    return X, Y


def test_gp_call_returns_normal_with_correct_shape():
    """GPSurrogate is an ArrayRandomFunction; __call__(X) returns a Normal
    with batch_shape=(n,) and event_shape=()."""
    X, Y = _sample_2d_gp_data(n=30)
    gp = GPSurrogate(input_shape=(2,)).fit(X, Y)
    Xtest = jax.random.uniform(jax.random.key(1), shape=(7, 2), minval=-2.0, maxval=2.0)
    pred = gp(Xtest)
    assert isinstance(pred, Normal)
    assert pred.batch_shape == (7,)
    assert pred.event_shape == ()


def test_gp_predict_close_to_training_data():
    X, Y = _sample_2d_gp_data()
    gp = GPSurrogate(input_shape=(2,)).fit(X, Y)
    pred = gp(X)
    pred_mean = jnp.asarray(mean(pred))
    rmse = float(jnp.sqrt(jnp.mean((pred_mean - Y) ** 2)))
    y_std = float(jnp.std(Y))
    assert rmse < 0.1 * y_std
    assert jnp.all(jnp.asarray(variance(pred)) >= 0.0)


def test_gp_predict_before_fit_raises():
    gp = GPSurrogate(input_shape=(2,))
    with pytest.raises(RuntimeError, match="before fit"):
        gp(jnp.zeros((3, 2)))


def test_gp_generalization_better_than_mean_predictor():
    """With fixed hyperparameters (v0 defaults), the GP should still beat the
    naive mean-predictor baseline on held-out points."""
    n_train = 60
    key = jax.random.key(42)
    ktrain, ktest = jax.random.split(key)
    X = jax.random.uniform(ktrain, shape=(n_train, 2), minval=-2.0, maxval=2.0)
    Y = jnp.sum(jnp.sin(X), axis=-1) + 0.5 * X[:, 0] * X[:, 1]

    gp = GPSurrogate(input_shape=(2,)).fit(X, Y)

    Xtest = jax.random.uniform(ktest, shape=(30, 2), minval=-1.5, maxval=1.5)
    Ytest = jnp.sum(jnp.sin(Xtest), axis=-1) + 0.5 * Xtest[:, 0] * Xtest[:, 1]
    pred = gp(Xtest)
    pred_mean = jnp.asarray(mean(pred))

    gp_rmse = float(jnp.sqrt(jnp.mean((pred_mean - Ytest) ** 2)))
    baseline_rmse = float(jnp.sqrt(jnp.mean((jnp.mean(Y) - Ytest) ** 2)))
    assert gp_rmse < baseline_rmse


def test_lengthscale_shrinks_with_more_data():
    key = jax.random.key(0)
    X_small = jax.random.uniform(key, shape=(20, 2), minval=-2.0, maxval=2.0)
    X_large = jax.random.uniform(key, shape=(200, 2), minval=-2.0, maxval=2.0)
    Y_small = jnp.zeros(20)
    Y_large = jnp.zeros(200)

    gp_small = GPSurrogate(input_shape=(2,)).fit(X_small, Y_small)
    gp_large = GPSurrogate(input_shape=(2,)).fit(X_large, Y_large)
    assert float(gp_large.lengthscale) < float(gp_small.lengthscale)
