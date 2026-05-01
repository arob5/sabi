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
    """Recompute predictions the naive way: run gpjax's
    ``posterior.predict`` (latent path — no obs noise), then undo the
    output scaling. The cached predict must match this pointwise.

    Sabi's `Emulator` convention is **latent** posterior — no
    observation noise on the diagonal — so we deliberately stop at
    ``posterior.predict`` and do NOT call ``posterior.likelihood`` on
    top of it.
    """
    import jax.numpy as jnp

    Xs_test = em._x_scaler.transform(X_test).astype(jnp.float64)
    latent = em._opt_posterior.predict(Xs_test, train_data=em._train_dataset)
    naive_mean = em._y_scaler.inverse_mean(latent.mean)
    naive_var = em._y_scaler.inverse_var(jnp.maximum(latent.variance, 0.0))
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


def test_dspgp_predict_variance_is_latent_only_not_observation():
    """Per the sabi `Emulator` convention, ``predict_variance`` returns
    the **latent** posterior variance (no observation noise on the
    diagonal). To make the latent-vs-obs distinction crisp regardless
    of what the optimizer found for ``obs_stddev``, we inject a known
    non-trivial obs_stddev into the cache post-hoc via ``eqx.tree_at``
    and check predict_variance against both candidates.
    """
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp
    import jax.random as jr

    from sabi.emulators.gpjax import DSPGPEmulator

    key = jr.key(31)
    n, d = 18, 2
    X = jr.uniform(key, (n, d))
    y = jnp.sin(X[:, 0])

    em = DSPGPEmulator(input_shape=(d,)).fit(X, y)
    # Crank cache.noise_var up to a known appreciable value so
    # `latent` and `latent + noise_var` differ by ~0.5 in standardized
    # space, well above fp slop. This isolates the convention test
    # from whatever the MAP optimizer found. _PredictCache is a plain
    # frozen dataclass (not eqx.Module) so we use dataclasses.replace.
    import dataclasses

    em._predict_cache = dataclasses.replace(
        em._predict_cache, noise_var=jnp.asarray(0.5)
    )

    X_test = jr.uniform(jr.key(33), (6, d))
    Xs_test = em._x_scaler.transform(X_test).astype(jnp.float64)
    _, latent_var_std = em._predict_cache.predict_latent(Xs_test)

    expected_latent_var = em._y_scaler.inverse_var(latent_var_std)
    expected_obs_var = em._y_scaler.inverse_var(
        latent_var_std + em._predict_cache.noise_var
    )

    pred_var = em.predict_variance(X_test)
    # Latent: matches.
    assert jnp.allclose(pred_var, expected_latent_var, rtol=1e-9, atol=1e-12)
    # Obs: deliberately doesn't match — we'd be off by `noise_var * scale²`.
    noise_gap = float(em._predict_cache.noise_var * em._y_scaler.scale ** 2)
    assert noise_gap > 1e-3
    assert jnp.max(jnp.abs(pred_var - expected_obs_var)) > 0.5 * noise_gap


def test_dspgp_predict_covariance_is_latent_only_not_observation():
    """Companion to the variance check: ``predict_covariance(joint_inputs=True)``
    must return latent covariance. Diagonal of the joint covariance
    must equal ``predict_variance`` (so neither has obs noise added)."""
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp
    import jax.random as jr

    from sabi.emulators.gpjax import DSPGPEmulator

    key = jr.key(34)
    n, d = 16, 2
    X = jr.uniform(key, (n, d))
    y = jnp.sin(X[:, 0]) + 0.4 * jr.normal(jr.key(35), (n,))

    em = DSPGPEmulator(input_shape=(d,)).fit(X, y)

    X_test = jr.uniform(jr.key(36), (5, d))
    cov = em.predict_covariance(X_test, joint_inputs=True)
    var = em.predict_variance(X_test)
    # Latent convention: diag(cov) == var. Both are latent.
    assert jnp.allclose(jnp.diag(cov), var, rtol=1e-9, atol=1e-12)


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
    """The cached joint-input latent covariance must match gpjax's
    ``posterior.predict`` (latent path) dense covariance, output-scaled.
    Inflate jitter to put any layering error well above fp slop.
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

    # Naive gpjax dense path — latent only (do NOT call likelihood
    # on top), to match the sabi latent convention.
    Xs_test = em._x_scaler.transform(X_test).astype(jnp.float64)
    latent = em._opt_posterior.predict(Xs_test, train_data=em._train_dataset)
    naive_cov_std = latent.covariance_matrix
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


# --- Fixed-hyperparameter conditioning (rank-one Cholesky update) ------------


def test_dspgp_condition_on_interpolates_through_appended_training_data():
    """Sanity check: after ``condition_on(X_new, Y_new)``, the new
    points are training data, so the latent posterior mean at those
    inputs should pass approximately through ``Y_new`` (within
    obs_stddev — typically tiny for the default fit)."""
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp
    import jax.random as jr

    from sabi.emulators.gpjax import DSPGPEmulator

    key = jr.key(101)
    d = 2
    X_part = jr.uniform(key, (12, d))
    Y_part = jnp.sin(X_part[:, 0]) + 0.2 * X_part[:, 1]
    em = DSPGPEmulator(input_shape=(d,)).fit(X_part, Y_part)

    X_extra = jr.uniform(jr.key(102), (4, d))
    Y_extra = jnp.sin(X_extra[:, 0]) + 0.2 * X_extra[:, 1]

    em_cond = em.condition_on(X_extra, Y_extra)
    # Predicting at appended training inputs returns near-perfect fit.
    pred_at_appended = em_cond.predict_mean(X_extra)
    assert jnp.max(jnp.abs(pred_at_appended - Y_extra)) < 1e-2


def test_dspgp_condition_on_predicts_identically_to_partial_when_no_new_rows():
    """``condition_on(empty)`` should be a no-op (sanity check)."""
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp
    import jax.random as jr

    from sabi.emulators.gpjax import DSPGPEmulator

    key = jr.key(111)
    X = jr.uniform(key, (12, 2))
    Y = jnp.sin(X[:, 0])
    em = DSPGPEmulator(input_shape=(2,)).fit(X, Y)

    X_test = jr.uniform(jr.key(112), (5, 2))
    pred_before = em.predict_mean(X_test)
    var_before = em.predict_variance(X_test)

    em_cond = em.condition_on(jnp.empty((0, 2)), jnp.empty((0,)))
    assert jnp.allclose(em_cond.predict_mean(X_test), pred_before, rtol=1e-12, atol=1e-14)
    assert jnp.allclose(em_cond.predict_variance(X_test), var_before, rtol=1e-12, atol=1e-14)


def test_dspgp_condition_on_matches_partition_then_condition_strategy():
    """The cleanest equivalence test for the rank-one update:

    Two paths must yield the same cache, hence the same predictions:
    1. Build cache from full data: ``cache_full = build(opt_posterior, X_full, Y_full)``
    2. Build cache from partial data, then condition on the rest:
       ``cache_partial.append_rows(X_extra_scaled, Y_extra_scaled)``

    Both should produce the same ``L_sigma``, ``alpha``, and the same
    predict_mean/variance values — they're the same mathematical
    object via two different computational paths.

    This test bypasses the scaler by going directly through the cache
    on already-standardized inputs, isolating the rank-one update math.
    """
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp
    import jax.random as jr

    from sabi.emulators.gpjax import DSPGPEmulator
    from sabi.emulators.gpjax.dsp_gp import _PredictCache

    key = jr.key(121)
    n_total, d = 18, 3
    X = jr.uniform(key, (n_total, d))
    Y = jnp.sin(X[:, 0]) + 0.3 * X[:, 1] - 0.1 * X[:, 2]

    em = DSPGPEmulator(input_shape=(d,)).fit(X, Y)

    # Standardize using the emulator's own scalers so the same numbers
    # flow through both paths.
    Xs = em._x_scaler.transform(X).astype(jnp.float64)
    Ys = em._y_scaler.transform(Y).astype(jnp.float64)

    # Path 1: build cache from full standardized data.
    cache_full = _PredictCache.build(em._opt_posterior, Xs, Ys)

    # Path 2: build cache from prefix, append rest via rank-one update.
    n_part = 11
    cache_partial = _PredictCache.build(em._opt_posterior, Xs[:n_part], Ys[:n_part])
    cache_appended = cache_partial.append_rows(Xs[n_part:], Ys[n_part:])

    # The L_sigma factors should match (up to fp slop) — same target
    # matrix, same Cholesky algorithm.
    assert jnp.allclose(cache_full.L_sigma, cache_appended.L_sigma, rtol=1e-9, atol=1e-12)
    # alpha vectors equal.
    assert jnp.allclose(cache_full.alpha, cache_appended.alpha, rtol=1e-9, atol=1e-12)
    # And consequently predictions agree.
    Xt = jr.uniform(jr.key(122), (8, d))
    Xst = em._x_scaler.transform(Xt).astype(jnp.float64)
    m_full, v_full = cache_full.predict_latent(Xst)
    m_app, v_app = cache_appended.predict_latent(Xst)
    assert jnp.allclose(m_full, m_app, rtol=1e-9, atol=1e-12)
    assert jnp.allclose(v_full, v_app, rtol=1e-9, atol=1e-12)


def test_dspgp_condition_on_is_faster_than_refit_in_principle():
    """End-to-end: ``condition_on`` should NOT call fit_scipy.

    We don't measure wall-clock time (flaky in CI), but we verify the
    optimized posterior pytree is the same object after conditioning
    — it can't have been re-optimized."""
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp
    import jax.random as jr

    from sabi.emulators.gpjax import DSPGPEmulator

    key = jr.key(131)
    X = jr.uniform(key, (10, 2))
    Y = jnp.cos(X[:, 0])
    em = DSPGPEmulator(input_shape=(2,)).fit(X, Y)

    X_extra = jr.uniform(jr.key(132), (5, 2))
    Y_extra = jnp.cos(X_extra[:, 0])
    em_cond = em.condition_on(X_extra, Y_extra)

    # Same opt_posterior: condition_on did not re-fit.
    assert em_cond._opt_posterior is em._opt_posterior
    # Same scalers.
    assert em_cond._x_scaler is em._x_scaler
    assert em_cond._y_scaler is em._y_scaler
    # New cache, augmented training set.
    assert em_cond._predict_cache.Xs_train.shape == (15, 2)
    assert em_cond._predict_cache.alpha.shape == (15,)
    assert em_cond._predict_cache.L_sigma.shape == (15, 15)


def test_dspgp_condition_on_rejects_wrong_shapes():
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp

    from sabi.emulators.gpjax import DSPGPEmulator

    em = DSPGPEmulator(input_shape=(2,)).fit(
        jnp.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]),
        jnp.array([0.0, 1.0, 0.5]),
    )
    # Wrong d
    with pytest.raises(ValueError, match=r"X_new.shape=\(m, 2\)"):
        em.condition_on(jnp.zeros((3, 5)), jnp.zeros((3,)))
    # Wrong Y length
    with pytest.raises(ValueError, match=r"Y_new.shape="):
        em.condition_on(jnp.zeros((3, 2)), jnp.zeros((4,)))


# --- AppendRows dispatch handler ---------------------------------------------


def test_dspgp_append_rows_handler_is_registered_on_module_import():
    """Importing the dsp_gp module registers the AppendRows handler
    in the global emulator_update_registry."""
    pytest.importorskip("gpjax")
    # Trigger the lazy import.
    from sabi.emulators.gpjax import DSPGPEmulator  # noqa: F401
    from sabi.emulators.dispatch import emulator_update_registry

    assert "dspgp_append_rows_chol_update" in emulator_update_registry._name_index


def test_dspgp_update_emulator_dispatches_through_handler_not_refit():
    """``update_emulator`` with an AppendRows plan on a fitted
    DSPGPEmulator must hit the handler — not the refit fallback. We
    prove this by passing an exploding factory that raises if it's
    called."""
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp
    import jax.random as jr

    from sabi.emulators.dispatch import update_emulator
    from sabi.emulators.gpjax import DSPGPEmulator
    from sabi.emulators.updates import AppendRows

    key = jr.key(201)
    X = jr.uniform(key, (12, 2))
    Y = jnp.sin(X[:, 0])
    em = DSPGPEmulator(input_shape=(2,)).fit(X, Y)

    X_extra = jr.uniform(jr.key(202), (4, 2))
    Y_extra = jnp.sin(X_extra[:, 0])
    plan = AppendRows(X_new=X_extra, Y_new=Y_extra)

    def _exploding_factory():
        raise AssertionError(
            "factory should not be called: cheap path is feasible"
        )

    X_full = jnp.concatenate([X, X_extra], axis=0)
    Y_full = jnp.concatenate([Y, Y_extra], axis=0)
    out = update_emulator(
        em, plan, factory=_exploding_factory, X_full=X_full, Y_full=Y_full
    )
    assert isinstance(out, DSPGPEmulator)
    # Same _opt_posterior object: handler routed through condition_on,
    # which preserves the hyperparameters.
    assert out._opt_posterior is em._opt_posterior
    # Cache grew by exactly the number of new rows.
    assert out._predict_cache.Xs_train.shape == (16, 2)


def test_dspgp_update_emulator_dispatch_matches_direct_condition_on():
    """The dispatch path and a direct ``condition_on`` call must
    produce numerically identical predictions."""
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp
    import jax.random as jr

    from sabi.emulators.dispatch import update_emulator
    from sabi.emulators.gpjax import DSPGPEmulator
    from sabi.emulators.updates import AppendRows

    key = jr.key(211)
    X = jr.uniform(key, (10, 2))
    Y = jnp.cos(X[:, 0])
    em = DSPGPEmulator(input_shape=(2,)).fit(X, Y)

    X_extra = jr.uniform(jr.key(212), (3, 2))
    Y_extra = jnp.cos(X_extra[:, 0])
    plan = AppendRows(X_new=X_extra, Y_new=Y_extra)

    direct = em.condition_on(X_extra, Y_extra)
    dispatched = update_emulator(
        em,
        plan,
        factory=lambda: DSPGPEmulator(input_shape=(2,)),
        X_full=jnp.concatenate([X, X_extra], axis=0),
        Y_full=jnp.concatenate([Y, Y_extra], axis=0),
    )

    X_test = jr.uniform(jr.key(213), (5, 2))
    assert jnp.allclose(
        direct.predict_mean(X_test),
        dispatched.predict_mean(X_test),
        rtol=1e-12,
        atol=1e-14,
    )
    assert jnp.allclose(
        direct.predict_variance(X_test),
        dispatched.predict_variance(X_test),
        rtol=1e-12,
        atol=1e-14,
    )


def test_dspgp_loop_planner_collapses_trivial_rescale_to_append_rows():
    """End-to-end: when there's no tempering (rescale factor == 1.0)
    plus new rows, the loop's planner should produce a plain
    `AppendRows` plan — which our DSPGPEmulator handler accepts. This
    is the integration that makes the cheap path actually fire in
    runner runs."""
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp

    from sabi.algorithms.loop import _plan_round_update
    from sabi.emulators.updates import AppendRows, RescaleOutputs

    class _IdentityRescaleTransform:
        """Stub OutputTransform that always reports a no-op rescale diff."""

        def diff(self, state_prev, state_new):
            return RescaleOutputs(factor=1.0)

    plan = _plan_round_update(
        _IdentityRescaleTransform(),
        state_prev=None,
        state_new=None,
        X_new=jnp.zeros((3, 2)),
        Y_new_at_new_state=jnp.zeros((3,)),
    )
    assert isinstance(plan, AppendRows)
    assert plan.X_new.shape == (3, 2)


def test_dspgp_loop_planner_returns_none_for_trivial_rescale_no_new_rows():
    """factor=1.0 with no new rows → nothing to do → `None` (skips
    the dispatch entirely, or — at present — falls through to the
    refit fallback which the caller can short-circuit)."""
    pytest.importorskip("gpjax")
    _enable_x64()

    from sabi.algorithms.loop import _plan_round_update
    from sabi.emulators.updates import RescaleOutputs

    class _IdentityRescaleTransform:
        def diff(self, state_prev, state_new):
            return RescaleOutputs(factor=1.0)

    plan = _plan_round_update(
        _IdentityRescaleTransform(),
        state_prev=None,
        state_new=None,
        X_new=None,
        Y_new_at_new_state=None,
    )
    assert plan is None


def test_dspgp_update_emulator_falls_back_to_refit_when_unfitted():
    """An unfitted DSPGPEmulator can't take the cheap path (no cache
    to update). Dispatch must fall back to a fresh ``factory().fit``."""
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp
    import jax.random as jr

    from sabi.emulators.dispatch import update_emulator
    from sabi.emulators.gpjax import DSPGPEmulator
    from sabi.emulators.updates import AppendRows

    em_unfitted = DSPGPEmulator(input_shape=(2,))
    assert em_unfitted._predict_cache is None

    key = jr.key(221)
    X_full = jr.uniform(key, (8, 2))
    Y_full = jnp.sin(X_full[:, 0])
    plan = AppendRows(X_new=X_full, Y_new=Y_full)

    factory_calls = {"n": 0}

    def _counting_factory():
        factory_calls["n"] += 1
        return DSPGPEmulator(input_shape=(2,))

    out = update_emulator(
        em_unfitted, plan, factory=_counting_factory, X_full=X_full, Y_full=Y_full
    )
    assert factory_calls["n"] == 1, "factory should have been called once for refit"
    # Output is a freshly fitted emulator (has a cache).
    assert out._predict_cache is not None
    assert out._predict_cache.Xs_train.shape == (8, 2)


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
