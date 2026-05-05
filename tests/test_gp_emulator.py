import jax
import jax.numpy as jnp
import pytest
from probpipe import mean, variance
from probpipe.distributions.continuous import Normal

from sabi.emulators import TinyGPEmulator


def _sample_2d_gp_data(n: int = 40, key_seed: int = 0):
    key = jax.random.key(key_seed)
    X = jax.random.uniform(key, shape=(n, 2), minval=-2.0, maxval=2.0)
    Y = jnp.sum(jnp.sin(X), axis=-1) + 0.5 * X[:, 0] * X[:, 1]
    return X, Y


def test_gp_call_returns_normal_with_correct_shape():
    """TinyGPEmulator is an ArrayRandomFunction; __call__(X) returns a Normal
    with batch_shape=(n,) and event_shape=()."""
    X, Y = _sample_2d_gp_data(n=30)
    gp = TinyGPEmulator(input_shape=(2,)).fit(X, Y)
    Xtest = jax.random.uniform(jax.random.key(1), shape=(7, 2), minval=-2.0, maxval=2.0)
    pred = gp(Xtest)
    assert isinstance(pred, Normal)
    assert pred.batch_shape == (7,)
    assert pred.event_shape == ()


def test_gp_predict_close_to_training_data():
    X, Y = _sample_2d_gp_data()
    gp = TinyGPEmulator(input_shape=(2,)).fit(X, Y)
    pred = gp(X)
    pred_mean = jnp.asarray(mean(pred))
    rmse = float(jnp.sqrt(jnp.mean((pred_mean - Y) ** 2)))
    y_std = float(jnp.std(Y))
    assert rmse < 0.1 * y_std
    assert jnp.all(jnp.asarray(variance(pred)) >= 0.0)


def test_gp_predict_before_fit_raises():
    gp = TinyGPEmulator(input_shape=(2,))
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

    gp = TinyGPEmulator(input_shape=(2,)).fit(X, Y)

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

    gp_small = TinyGPEmulator(input_shape=(2,)).fit(X_small, Y_small)
    gp_large = TinyGPEmulator(input_shape=(2,)).fit(X_large, Y_large)
    assert float(gp_large.lengthscale) < float(gp_small.lengthscale)


def test_tinygp_obs_noise_variance_returns_constructor_value():
    """``obs_noise_variance`` exposes the constructor ``noise`` arg as
    an Array, both pre- and post-fit (TinyGPEmulator's noise is
    fixed, not fit from data)."""
    em_pre = TinyGPEmulator(input_shape=(2,), noise=2.5e-3)
    pre = em_pre.obs_noise_variance
    assert pre is not None
    assert float(pre) == pytest.approx(2.5e-3)

    X, Y = _sample_2d_gp_data(n=20)
    em_post = em_pre.fit(X, Y)
    post = em_post.obs_noise_variance
    assert post is not None
    assert float(post) == pytest.approx(2.5e-3)


# --- Post-GPEmulator-migration: joint covariance + condition_on -------------


def test_tinygp_supports_joint_inputs():
    """``TinyGPEmulator`` advertises joint-input support after the
    GPEmulator migration."""
    assert TinyGPEmulator.supports_joint_inputs is True
    assert TinyGPEmulator.supports_joint_outputs is True


def test_tinygp_predict_covariance_joint_inputs_shape_and_diag_matches_variance():
    """``predict_covariance(X, joint_inputs=True)`` returns an
    (n, n) symmetric PSD matrix whose diagonal equals
    ``predict_variance(X)``."""
    X, Y = _sample_2d_gp_data(n=30)
    em = TinyGPEmulator(input_shape=(2,)).fit(X, Y)

    X_test = jax.random.uniform(
        jax.random.key(99), shape=(7, 2), minval=-2.0, maxval=2.0
    )
    cov = em.predict_covariance(X_test, joint_inputs=True)
    assert cov.shape == (7, 7)
    assert jnp.allclose(cov, cov.T, atol=1e-12)
    assert jnp.allclose(jnp.diag(cov), em.predict_variance(X_test), rtol=1e-9, atol=1e-12)
    w = jnp.linalg.eigvalsh(cov)
    assert float(jnp.min(w)) >= -1e-9


def test_tinygp_predict_joint_assembles_multivariate_normal():
    """End-to-end: ``predict(X, joint_inputs=True)`` (the parent's
    assembly path) returns a MultivariateNormal whose Cholesky
    matches ``predict_covariance(...)``."""
    X, Y = _sample_2d_gp_data(n=20)
    em = TinyGPEmulator(input_shape=(2,)).fit(X, Y)

    X_test = jax.random.uniform(
        jax.random.key(101), shape=(5, 2), minval=-2.0, maxval=2.0
    )
    dist = em.predict(X_test, joint_inputs=True)
    cov_from_dist = dist.scale_tril @ dist.scale_tril.T
    cov_from_predict = em.predict_covariance(X_test, joint_inputs=True)
    assert jnp.allclose(cov_from_dist, cov_from_predict, rtol=1e-9, atol=1e-12)
    assert jnp.allclose(dist.loc, em.predict_mean(X_test), rtol=1e-9, atol=1e-12)


def test_tinygp_predict_covariance_joint_outputs_only_returns_n_1_1():
    """For scalar output, joint over outputs is trivial — reshape of
    the marginal variance to (n, 1, 1)."""
    X, Y = _sample_2d_gp_data(n=15)
    em = TinyGPEmulator(input_shape=(2,)).fit(X, Y)

    X_test = jax.random.uniform(
        jax.random.key(110), shape=(4, 2), minval=-2.0, maxval=2.0
    )
    cov = em.predict_covariance(X_test, joint_inputs=False, joint_outputs=True)
    assert cov.shape == (4, 1, 1)
    expected = em.predict_variance(X_test)[:, None, None]
    assert jnp.allclose(cov, expected, rtol=1e-12, atol=1e-14)


def test_tinygp_condition_on_interpolates_through_appended_training_data():
    """After ``condition_on(X_new, Y_new)``, the new points are
    training data, so the latent posterior mean at those inputs
    passes approximately through ``Y_new`` (within obs_stddev — for
    TinyGPEmulator's tiny default noise, the fit is near-perfect)."""
    X, Y = _sample_2d_gp_data(n=20)
    em = TinyGPEmulator(input_shape=(2,)).fit(X, Y)

    key = jax.random.key(120)
    X_extra = jax.random.uniform(key, shape=(4, 2), minval=-2.0, maxval=2.0)
    Y_extra = jnp.sum(jnp.sin(X_extra), axis=-1) + 0.5 * X_extra[:, 0] * X_extra[:, 1]

    em_cond = em.condition_on(X_extra, Y_extra)
    pred = em_cond.predict_mean(X_extra)
    assert jnp.max(jnp.abs(pred - Y_extra)) < 5e-2


def test_tinygp_condition_on_partition_then_condition_matches_full_fit():
    """Cache equivalence: building from full data vs. from a prefix +
    conditioning on the rest must produce the same L_sigma, alpha,
    and predictions."""
    X, Y = _sample_2d_gp_data(n=24)
    em = TinyGPEmulator(input_shape=(2,)).fit(X, Y)

    Xs = em._x_scaler.transform(X)
    Ys = em._y_scaler.transform(Y)

    from sabi.emulators.tinygp._cache import _TinyGPCache

    full = _TinyGPCache.build(
        em._predict_cache.kernel, Xs, Ys, noise=em.noise, jitter=em.jitter
    )

    n_part = 14
    partial = _TinyGPCache.build(
        em._predict_cache.kernel,
        Xs[:n_part],
        Ys[:n_part],
        noise=em.noise,
        jitter=em.jitter,
    )
    appended = partial.append_rows(Xs[n_part:], Ys[n_part:])

    assert jnp.allclose(full.L_sigma, appended.L_sigma, rtol=1e-9, atol=1e-12)
    assert jnp.allclose(full.alpha, appended.alpha, rtol=1e-9, atol=1e-12)

    # And predictions match.
    Xt = jax.random.uniform(
        jax.random.key(130), shape=(6, 2), minval=-2.0, maxval=2.0
    )
    m_full, v_full = full.predict_latent(Xt)
    m_app, v_app = appended.predict_latent(Xt)
    assert jnp.allclose(m_full, m_app, rtol=1e-9, atol=1e-12)
    assert jnp.allclose(v_full, v_app, rtol=1e-9, atol=1e-12)


def test_tinygp_dispatch_handler_fires_for_append_rows():
    """The shared ``GPEmulator``-typed dispatch handlers should now
    fire for ``TinyGPEmulator`` automatically — verifying the Tier B
    promise that backend migration to ``GPEmulator`` is enough to
    inherit cheap-path dispatch."""
    X, Y = _sample_2d_gp_data(n=16)
    em = TinyGPEmulator(input_shape=(2,)).fit(X, Y)

    from sabi.emulators.dispatch import update_emulator
    from sabi.emulators.updates import AppendRows

    key = jax.random.key(140)
    X_new = jax.random.uniform(key, shape=(3, 2), minval=-2.0, maxval=2.0)
    Y_new = jnp.sum(jnp.sin(X_new), axis=-1)
    plan = AppendRows(X_new=X_new, Y_new=Y_new)

    def _exploding_factory():
        raise AssertionError("cheap path should fire; factory should not be called")

    out = update_emulator(
        em,
        plan,
        factory=_exploding_factory,
        X_full=jnp.concatenate([X, X_new], axis=0),
        Y_full=jnp.concatenate([Y, Y_new], axis=0),
    )
    assert isinstance(out, TinyGPEmulator)
    assert out._predict_cache.Xs_train.shape == (19, 2)


# --- Hand-computed Cholesky pinning (issue #32) -----------------------------
#
# Bit-equivalence between ``_TinyGPCache.build`` and ``append_rows`` (above)
# catches regressions where one path drifts from the other, but it can't catch
# bugs where *both* paths share the same flawed math (e.g. a sign flip in the
# Schur complement). The tests below pin the cache's outputs to values
# derived independently from the cache's own code path:
#
# - Kernel: ``Matern52(scale=1.0)``, ``noise=0``, ``jitter=0``.
# - Dataset: 1-D inputs ``X = [0.0, 1.0, 2.0]^T`` (column vector,
#   ``shape=(3, 1)``), targets ``Y = [0.5, -1.0, 2.0]``.
# - Test point: ``Xt = [0.5]`` (``shape=(1, 1)``).
#
# Reference values were computed offline via ``numpy.linalg.cholesky`` +
# ``scipy.linalg.solve_triangular`` on a hand-built kernel matrix using
# the closed-form Matern52 expression, NOT via ``jnp.linalg.cholesky``. This
# keeps the reference path independent of the cache's internal solver.
#
# Closed-form Matern52 with scale 1:
#
# .. math::
#     k(d) = \left(1 + \sqrt{5}\, d + \tfrac{5}{3} d^2\right) \exp(-\sqrt{5}\, d).
#
# Hand-computed:
#
# - ``k(0) = 1``
# - ``k(0.5) ~ 0.8286491424181253``
# - ``k(1)   ~ 0.5239941088318203``
# - ``k(1.5) ~ 0.2831632713397992``
# - ``k(2)   ~ 0.13866021913850426``
#
# Resulting full-fit values:
#
# - ``L_sigma`` = ``[[1, 0, 0],
#                   [k(1), sqrt(1 - k(1)^2), 0],
#                   [k(2), (k(1) - k(1) k(2)) / sqrt(1 - k(1)^2),
#                          sqrt(1 - k(2)^2 - ((k(1) - k(1) k(2))/sqrt(1-k(1)^2))^2)]]``
#   ~ ``[[1, 0, 0], [0.5239941, 0.8517219, 0],
#         [0.1386602, 0.5299112, 0.8366406]]``
# - ``alpha`` = ``L_sigma^{-1} Y`` ~ ``[0.5, -1.4817009, 3.2461249]``
# - ``predict_latent([[0.5]])`` mean ~ ``[-0.5711871]``,
#   var ~ ``[0.0903663]``.

_HAND_COMPUTED_L = jnp.array(
    [
        [1.0, 0.0, 0.0],
        [0.5239941088318203, 0.8517218876543836, 0.0],
        [0.13866021913850426, 0.5299112038988257, 0.8366405797061],
    ],
    dtype=jnp.float64,
)
_HAND_COMPUTED_ALPHA = jnp.array(
    [0.5, -1.4817008611712585, 3.2461248515413543], dtype=jnp.float64
)
_HAND_COMPUTED_MEAN_AT_HALF = jnp.array([-0.5711870801337865], dtype=jnp.float64)
_HAND_COMPUTED_VAR_AT_HALF = jnp.array([0.09036633090275048], dtype=jnp.float64)
# 2-point partial-fit values (X = [0, 1], Y = [0.5, -1]) used by the
# append_rows test as the starting cache.
_HAND_COMPUTED_L_PARTIAL = jnp.array(
    [[1.0, 0.0], [0.5239941088318203, 0.8517218876543836]], dtype=jnp.float64
)
_HAND_COMPUTED_ALPHA_PARTIAL = jnp.array(
    [0.5, -1.4817008611712585], dtype=jnp.float64
)


def _cholesky_pinning_dataset():
    """Inputs/targets/test-point used by all hand-computed Cholesky tests."""
    X = jnp.array([[0.0], [1.0], [2.0]], dtype=jnp.float64)
    Y = jnp.array([0.5, -1.0, 2.0], dtype=jnp.float64)
    Xt = jnp.array([[0.5]], dtype=jnp.float64)
    return X, Y, Xt


def test_tinygp_cache_build_matches_hand_computed_cholesky():
    """``_TinyGPCache.build`` on a 3-point Matern52 dataset must produce
    ``L_sigma``, ``alpha``, and ``predict_latent`` outputs that match
    closed-form values derived independently of the cache's own solver
    (numpy + scipy on a hand-built kernel matrix). This pins the math
    against sign-flip / transpose regressions that bit-equivalence
    against a refit can't catch."""
    from tinygp import kernels

    from sabi.emulators.tinygp._cache import _TinyGPCache

    X, Y, Xt = _cholesky_pinning_dataset()
    kernel = kernels.Matern52(scale=1.0)
    cache = _TinyGPCache.build(kernel, X, Y, noise=0.0, jitter=0.0)

    assert jnp.allclose(cache.L_sigma, _HAND_COMPUTED_L, rtol=0, atol=1e-10)
    assert jnp.allclose(cache.alpha, _HAND_COMPUTED_ALPHA, rtol=0, atol=1e-10)

    mean, var = cache.predict_latent(Xt)
    assert jnp.allclose(mean, _HAND_COMPUTED_MEAN_AT_HALF, rtol=0, atol=1e-10)
    assert jnp.allclose(var, _HAND_COMPUTED_VAR_AT_HALF, rtol=0, atol=1e-10)


def test_tinygp_cache_append_rows_matches_hand_computed_cholesky():
    """Starting from the 2-point partial fit (``X = [0, 1]``,
    ``Y = [0.5, -1]``) and appending the third row ``X_new = [2]``,
    ``Y_new = 2``, the augmented cache must equal the from-scratch
    3-point Cholesky / alpha — pinned to hand-computed reference values
    (not just to a refit). This catches Schur-complement sign flips,
    ``L_22`` transposition errors, and residual-update mistakes that
    bit-equivalence between two flawed paths would absorb."""
    from tinygp import kernels

    from sabi.emulators.tinygp._cache import _TinyGPCache

    X, Y, Xt = _cholesky_pinning_dataset()
    kernel = kernels.Matern52(scale=1.0)

    partial = _TinyGPCache.build(
        kernel, X[:2], Y[:2], noise=0.0, jitter=0.0
    )
    # Pin the partial fit too — guards the build path on the prefix
    # used as input to append_rows.
    assert jnp.allclose(
        partial.L_sigma, _HAND_COMPUTED_L_PARTIAL, rtol=0, atol=1e-10
    )
    assert jnp.allclose(
        partial.alpha, _HAND_COMPUTED_ALPHA_PARTIAL, rtol=0, atol=1e-10
    )

    appended = partial.append_rows(X[2:], Y[2:])

    assert jnp.allclose(
        appended.L_sigma, _HAND_COMPUTED_L, rtol=0, atol=1e-10
    )
    assert jnp.allclose(
        appended.alpha, _HAND_COMPUTED_ALPHA, rtol=0, atol=1e-10
    )

    mean, var = appended.predict_latent(Xt)
    assert jnp.allclose(mean, _HAND_COMPUTED_MEAN_AT_HALF, rtol=0, atol=1e-10)
    assert jnp.allclose(var, _HAND_COMPUTED_VAR_AT_HALF, rtol=0, atol=1e-10)


# --- (end issue #32 hand-computed pinning) ----------------------------------


def test_tinygp_dispatch_handler_fires_for_rescale_outputs():
    """``RescaleOutputs(factor=β)`` should rescale the y-scaler in
    O(1); ``predict_mean`` should return β times the original
    predictions."""
    X, Y = _sample_2d_gp_data(n=16)
    em = TinyGPEmulator(input_shape=(2,)).fit(X, Y)

    from sabi.emulators.dispatch import update_emulator
    from sabi.emulators.updates import RescaleOutputs

    X_test = jax.random.uniform(
        jax.random.key(150), shape=(5, 2), minval=-2.0, maxval=2.0
    )
    pred_before = em.predict_mean(X_test)

    beta = 2.0
    em_b = update_emulator(
        em,
        RescaleOutputs(factor=beta),
        factory=lambda: TinyGPEmulator(input_shape=(2,)),
        X_full=X,
        Y_full=Y,
    )
    pred_after = em_b.predict_mean(X_test)
    assert jnp.allclose(pred_after, beta * pred_before, rtol=1e-12, atol=1e-14)
    # Cache untouched: L_sigma reused.
    assert em_b._predict_cache.L_sigma is em._predict_cache.L_sigma
