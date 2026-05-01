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


def _naive_predict_mean_var(em, X_test):
    """Recompute predictions the way the pre-cache code did: run the
    gpjax `posterior.predict` -> `likelihood` pipeline, then undo the
    output scaling. The cache must match this pointwise."""
    import jax.numpy as jnp

    Xs_test = em._x_scaler.transform(X_test).astype(jnp.float64)
    latent = em._opt_posterior.predict(Xs_test, train_data=em._train_dataset)
    pred = em._opt_posterior.likelihood(latent)
    naive_mean = em._y_scaler.inverse_mean(pred.mean)
    naive_var = em._y_scaler.inverse_var(jnp.maximum(pred.variance, 0.0))
    return naive_mean, naive_var


def test_dspgp_cached_predict_matches_naive_gpjax_predict():
    """The Cholesky-cached predict path should be numerically equivalent
    (within fp slop) to the naive ``posterior.predict + likelihood``
    path that gpjax exposes. The cache only changes *how much work*
    happens per call, not *what it computes*."""
    pytest.importorskip("gpjax")
    _enable_x64()
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
    naive_mean, naive_var = _naive_predict_mean_var(em, X_test)

    assert jnp.allclose(cached_mean, naive_mean, rtol=1e-6, atol=1e-8)
    assert jnp.allclose(cached_var, naive_var, rtol=1e-5, atol=1e-7)


def test_dspgp_cached_predict_matches_naive_with_inflated_jitter():
    """Stronger version of the equivalence check: crank prior jitter up
    to 1e-2 (10000x the default 1e-6). Any miscount of `prior_jitter`
    or any double-add/missing-add would now show up as an
    O(1e-2) absolute mismatch in the variance — well outside fp slop.
    """
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp
    import jax.random as jr

    from sabi.emulators.gpjax import DSPGPEmulator

    key = jr.key(21)
    n, d = 20, 3
    X = jr.uniform(key, (n, d))
    y = jnp.cos(3 * X[:, 0]) - 0.4 * X[:, 2]

    big_jitter = 1e-2
    em = DSPGPEmulator(input_shape=(d,), jitter=big_jitter).fit(X, y)
    assert float(em._opt_posterior.prior.jitter) == big_jitter

    X_test = jr.uniform(jr.key(22), (10, d))
    cached_mean = em.predict_mean(X_test)
    cached_var = em.predict_variance(X_test)
    naive_mean, naive_var = _naive_predict_mean_var(em, X_test)

    # If the jitter were being missed in the cache path, abs diff in
    # variance would be of order `big_jitter * y_scale^2`. Tight bounds
    # here will catch any such bug.
    assert jnp.allclose(cached_mean, naive_mean, rtol=1e-7, atol=1e-9)
    assert jnp.allclose(cached_var, naive_var, rtol=1e-7, atol=1e-9)


def test_dspgp_cached_predict_variance_includes_obs_noise():
    """Independent ground-truth check: predict_variance should equal
    `latent_var + obs_stddev^2 * y_scale^2` (the noise scales with the
    output standardizer). This pins down the noise layer separately
    from the jitter layer.
    """
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp
    import jax.random as jr
    import paramax

    from sabi.emulators.gpjax import DSPGPEmulator

    key = jr.key(31)
    n, d = 18, 2
    X = jr.uniform(key, (n, d))
    y = jnp.sin(X[:, 0]) + 0.2 * jr.normal(jr.key(32), (n,))

    em = DSPGPEmulator(input_shape=(d,)).fit(X, y)

    # Reach into the cache: predict the raw latent variance (no noise),
    # then assert that predict_variance matches `latent + sigma^2`
    # in the standardized space.
    X_test = jr.uniform(jr.key(33), (6, d))
    Xs_test = em._x_scaler.transform(X_test).astype(jnp.float64)
    _, latent_var_std = em._predict_cache.predict_latent(Xs_test)

    obs_stddev = float(paramax.unwrap(em._opt_posterior).likelihood.obs_stddev)
    # In the (still standardized) y-space:
    expected_obs_var_std = latent_var_std + obs_stddev ** 2
    # Bring back to original output space:
    expected_obs_var = em._y_scaler.inverse_var(expected_obs_var_std)
    assert jnp.allclose(em.predict_variance(X_test), expected_obs_var, rtol=1e-9, atol=1e-12)


# --- predict_covariance ------------------------------------------------------


def test_dspgp_predict_covariance_shape_joint_inputs():
    """`predict_covariance(X, joint_inputs=True)` returns an (n, n)
    PSD matrix whose diagonal equals `predict_variance(X)`."""
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp
    import jax.random as jr

    from sabi.emulators.gpjax import DSPGPEmulator

    key = jr.key(41)
    n_train, d = 20, 3
    X = jr.uniform(key, (n_train, d))
    y = jnp.sin(X[:, 0] + X[:, 1])
    em = DSPGPEmulator(input_shape=(d,)).fit(X, y)

    X_test = jr.uniform(jr.key(42), (7, d))
    cov = em.predict_covariance(X_test, joint_inputs=True)
    assert cov.shape == (7, 7)

    # Symmetric.
    assert jnp.allclose(cov, cov.T, atol=1e-12)

    # Diagonal equals predict_variance — the only place predict_*
    # routines should disagree about a single test point's marginal
    # is fp slop.
    assert jnp.allclose(jnp.diag(cov), em.predict_variance(X_test), rtol=1e-9, atol=1e-12)

    # PSD: smallest eigenvalue ≥ 0 (allow a tiny negative slop).
    w = jnp.linalg.eigvalsh(cov)
    assert float(jnp.min(w)) >= -1e-9


def test_dspgp_predict_covariance_matches_naive_gpjax_dense():
    """The cached joint-input covariance must match the gpjax
    `posterior.predict + likelihood` dense path: same numerical model,
    just routed through the cache. Inflate jitter to put any layering
    error well above fp slop.
    """
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp
    import jax.random as jr

    from sabi.emulators.gpjax import DSPGPEmulator

    key = jr.key(51)
    n_train, d = 15, 2
    X = jr.uniform(key, (n_train, d))
    y = jnp.cos(X[:, 0]) + 0.3 * X[:, 1]

    em = DSPGPEmulator(input_shape=(d,), jitter=1e-2).fit(X, y)
    X_test = jr.uniform(jr.key(52), (6, d))

    cached = em.predict_covariance(X_test, joint_inputs=True)

    # Naive gpjax dense path.
    Xs_test = em._x_scaler.transform(X_test).astype(jnp.float64)
    latent = em._opt_posterior.predict(Xs_test, train_data=em._train_dataset)
    pred = em._opt_posterior.likelihood(latent)
    naive_cov_std = pred.covariance_matrix
    # Apply the y-standardizer's inverse_var (scales by scale²) to bring
    # back to the original output space.
    naive_cov = naive_cov_std * (em._y_scaler.scale ** 2)

    assert jnp.allclose(cached, naive_cov, rtol=1e-7, atol=1e-9)


def test_dspgp_predict_covariance_joint_outputs_only_returns_n_1_1_marginal():
    """For scalar output, joint over outputs is trivial — reshape
    of the marginal variance to (n, 1, 1)."""
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp
    import jax.random as jr

    from sabi.emulators.gpjax import DSPGPEmulator

    key = jr.key(61)
    n_train, d = 18, 2
    X = jr.uniform(key, (n_train, d))
    y = jnp.sin(X[:, 0])
    em = DSPGPEmulator(input_shape=(d,)).fit(X, y)

    X_test = jr.uniform(jr.key(62), (5, d))
    cov = em.predict_covariance(X_test, joint_inputs=False, joint_outputs=True)
    assert cov.shape == (5, 1, 1)
    expected = em.predict_variance(X_test)[:, None, None]
    assert jnp.allclose(cov, expected, rtol=1e-12, atol=1e-14)


def test_dspgp_predict_covariance_joint_inputs_outputs_returns_n_n():
    """For scalar output, joint_inputs+joint_outputs collapses to the
    same (n, n) as joint_inputs alone (no extra output dimension to
    cross with itself)."""
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp
    import jax.random as jr

    from sabi.emulators.gpjax import DSPGPEmulator

    key = jr.key(71)
    n_train, d = 12, 2
    X = jr.uniform(key, (n_train, d))
    y = X[:, 0] - 0.5 * X[:, 1]
    em = DSPGPEmulator(input_shape=(d,)).fit(X, y)

    X_test = jr.uniform(jr.key(72), (4, d))
    cov_jj = em.predict_covariance(X_test, joint_inputs=True, joint_outputs=True)
    cov_j = em.predict_covariance(X_test, joint_inputs=True, joint_outputs=False)
    assert cov_jj.shape == (4, 4)
    assert jnp.allclose(cov_jj, cov_j, rtol=1e-12, atol=1e-14)


def test_dspgp_predict_covariance_marginal_only_raises():
    """When neither flag is set, predict_covariance is undefined
    (callers should use predict_variance)."""
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp
    import jax.random as jr

    from sabi.emulators.gpjax import DSPGPEmulator

    key = jr.key(81)
    X = jr.uniform(key, (8, 2))
    y = X[:, 0]
    em = DSPGPEmulator(input_shape=(2,)).fit(X, y)
    with pytest.raises(NotImplementedError, match="predict_variance"):
        em.predict_covariance(X[:3], joint_inputs=False, joint_outputs=False)


def test_dspgp_supports_joint_inputs_flag():
    """The class-level joint-mode flags advertise the new capability."""
    pytest.importorskip("gpjax")
    from sabi.emulators.gpjax import DSPGPEmulator

    assert DSPGPEmulator.supports_joint_inputs is True
    assert DSPGPEmulator.supports_joint_outputs is True


def test_dspgp_predict_joint_assembles_multivariate_normal():
    """End-to-end: `predict(X, joint_inputs=True)` (the parent's
    assembly path) returns a MultivariateNormal whose covariance is a
    Cholesky factor of `predict_covariance(...)`.
    """
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp
    import jax.random as jr

    from sabi.emulators.gpjax import DSPGPEmulator

    key = jr.key(91)
    X = jr.uniform(key, (10, 2))
    y = jnp.sin(X[:, 0]) + 0.2 * X[:, 1]
    em = DSPGPEmulator(input_shape=(2,)).fit(X, y)

    X_test = jr.uniform(jr.key(92), (5, 2))
    dist = em.predict(X_test, joint_inputs=True)
    # Defensive: only check the contract bits that the GRF assembly
    # exposes. `scale_tril @ scale_tril.T` should equal predict_covariance.
    cov_from_dist = dist.scale_tril @ dist.scale_tril.T
    cov_from_predict = em.predict_covariance(X_test, joint_inputs=True)
    assert jnp.allclose(cov_from_dist, cov_from_predict, rtol=1e-9, atol=1e-12)
    # Mean shape (n,) for scalar output joint_inputs case.
    assert dist.loc.shape == (5,)
    assert jnp.allclose(dist.loc, em.predict_mean(X_test), rtol=1e-9, atol=1e-12)


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
