"""End-to-end tests for `DSPGPEmulator`.

Exercises:
- 2-d fit + predict shapes
- fit actually moves the lengthscale away from its init (optimizer ran)
- sample-efficiency: at d=10 with small n, the DSP-prior emulator beats
  the tinygp-backed `GPEmulator` (which has no high-d-aware prior) on
  held-out MSE.

All tests skip cleanly when gpjax is not installed.
"""

from __future__ import annotations

import pytest


def _enable_x64() -> None:
    import jax

    jax.config.update("jax_enable_x64", True)


# --- Basic fit + predict ------------------------------------------------------


def test_dspgp_fit_predict_shapes_on_2d_toy():
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp
    import jax.random as jr

    from sabi.emulators.gpjax import DSPGPEmulator

    key = jr.key(0)
    n, d = 30, 2
    X = jr.uniform(key, (n, d))
    y = jnp.sin(2 * jnp.pi * X[:, 0]) + 0.3 * X[:, 1]

    em = DSPGPEmulator(input_shape=(d,))
    fitted = em.fit(X, y)

    X_test = jr.uniform(jr.key(1), (5, d))
    mean = fitted.predict_mean(X_test)
    var = fitted.predict_variance(X_test)
    assert mean.shape == (5,)
    assert var.shape == (5,)
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(var))
    assert jnp.all(var >= 0.0)


def test_dspgp_predict_before_fit_raises():
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp

    from sabi.emulators.gpjax import DSPGPEmulator

    em = DSPGPEmulator(input_shape=(2,))
    with pytest.raises(RuntimeError, match="before fit"):
        em.predict_mean(jnp.zeros((3, 2)))


def test_dspgp_fit_changes_lengthscale_from_init():
    """Sanity check that fit_scipy actually optimizes — the post-fit
    lengthscale should differ from the prior-mode init."""
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp
    import jax.random as jr
    import paramax

    from sabi.emulators.gpjax import DSPGPEmulator

    key = jr.key(2)
    n, d = 40, 3
    X = jr.uniform(key, (n, d))
    # Strong dependence on x[:, 0]; near-flat in others. A trained
    # lengthscale should grow large for dims 1, 2.
    y = jnp.sin(3 * X[:, 0]) + 0.05 * jr.normal(jr.key(3), (n,))

    em = DSPGPEmulator(input_shape=(d,))
    fitted = em.fit(X, y)
    unwrapped = paramax.unwrap(fitted._opt_posterior)
    ls = unwrapped.prior.kernel.lengthscale
    # Init was set to the prior mode `exp(loc - 3)`. Any non-degenerate
    # fit should move at least one lengthscale meaningfully.
    init_loc = jnp.sqrt(jnp.asarray(2.0)) + 0.5 * jnp.log(jnp.asarray(float(d)))
    init_mode = float(jnp.exp(init_loc - 3.0))
    moved = bool(jnp.any(jnp.abs(ls - init_mode) > 1e-2))
    assert moved, f"Lengthscales unchanged from init mode {init_mode}; got {ls}"


def test_dspgp_constructor_rejects_multi_output():
    pytest.importorskip("gpjax")
    from sabi.emulators.gpjax import DSPGPEmulator

    with pytest.raises(ValueError, match="scalar-output"):
        DSPGPEmulator(input_shape=(2,), output_shape=(3,))


def test_dspgp_constructor_rejects_non_1d_input_shape():
    pytest.importorskip("gpjax")
    from sabi.emulators.gpjax import DSPGPEmulator

    with pytest.raises(ValueError, match=r"input_shape=\(d,\)"):
        DSPGPEmulator(input_shape=(2, 3))


def test_dspgp_fit_rejects_wrong_x_shape():
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp

    from sabi.emulators.gpjax import DSPGPEmulator

    em = DSPGPEmulator(input_shape=(2,))
    X = jnp.zeros((10, 3))  # wrong d
    Y = jnp.zeros((10,))
    with pytest.raises(ValueError, match=r"X.shape=\(n, 2\)"):
        em.fit(X, Y)


# --- Cholesky cache equivalence ----------------------------------------------


def test_dspgp_cached_predict_matches_naive_gpjax_predict():
    """The Cholesky-cached predict path should be numerically equivalent
    (within fp slop) to the naive ``posterior.predict + likelihood``
    path that gpjax exposes. The cache only changes *how much work*
    happens per call, not *what it computes*."""
    pytest.importorskip("gpjax")
    _enable_x64()
    import gpjax as gpx
    import jax.numpy as jnp
    import jax.random as jr

    from sabi.emulators.gpjax import DSPGPEmulator

    key = jr.key(11)
    n, d = 25, 3
    X = jr.uniform(key, (n, d))
    y = jnp.sin(2 * X[:, 0]) + 0.3 * X[:, 1] - 0.1 * X[:, 2]

    em = DSPGPEmulator(input_shape=(d,)).fit(X, y)

    X_test = jr.uniform(jr.key(12), (8, d))
    cached_mean = em.predict_mean(X_test)
    cached_var = em.predict_variance(X_test)

    # Recompute the naive way: scale X_test, run posterior.predict +
    # likelihood, undo the y standardization. This is what predict_*
    # used to do before caching.
    Xs_test = em._x_scaler.transform(X_test).astype(jnp.float64)
    latent = em._opt_posterior.predict(Xs_test, train_data=em._train_dataset)
    pred = em._opt_posterior.likelihood(latent)
    naive_mean = em._y_scaler.inverse_mean(pred.mean)
    naive_var = em._y_scaler.inverse_var(jnp.maximum(pred.variance, 0.0))

    assert jnp.allclose(cached_mean, naive_mean, rtol=1e-6, atol=1e-8)
    assert jnp.allclose(cached_var, naive_var, rtol=1e-5, atol=1e-7)


# --- Sample efficiency at high d ---------------------------------------------


def test_dspgp_beats_tinygp_in_high_d_small_n():
    """At d=10 with n=30, the DSP-prior emulator should achieve lower
    held-out MSE than the tinygp-backed `GPEmulator`. The DSP prior
    biases lengthscales upward with d, which prevents the high-d
    overfitting failure mode of generic GP-BO.
    """
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp
    import jax.random as jr

    from sabi.emulators import GPEmulator
    from sabi.emulators.gpjax import DSPGPEmulator

    key = jr.key(7)
    d = 10
    n_train, n_test = 30, 200

    # Active-subspace target: only the first 2 dimensions matter, the
    # rest are nuisance noise. This is the classic Hvarfner setup where
    # the DSP prior shines.
    def f(X):
        return jnp.sin(2 * X[:, 0]) + 0.5 * jnp.cos(X[:, 1])

    X_train = jr.uniform(key, (n_train, d))
    y_train = f(X_train)
    X_test = jr.uniform(jr.key(8), (n_test, d))
    y_test = f(X_test)

    dsp = DSPGPEmulator(input_shape=(d,)).fit(X_train, y_train)
    tg = GPEmulator(input_shape=(d,)).fit(X_train, y_train)

    dsp_mse = float(jnp.mean((dsp.predict_mean(X_test) - y_test) ** 2))
    tg_mse = float(jnp.mean((tg.predict_mean(X_test) - y_test) ** 2))

    assert dsp_mse <= tg_mse, (
        f"DSP MSE ({dsp_mse:.4f}) should be <= tinygp MSE ({tg_mse:.4f}) "
        f"on a high-d (d={d}) active-subspace target with n={n_train}."
    )
