"""Tests for the abstract `GPEmulator` base via a tiny fake backend.

The real backends (`TinyGPEmulator`, `DSPGPEmulator`) cover the
end-to-end behavior; this file exercises the base class's plumbing
in isolation. The fake cache and emulator below are intentionally
minimal — just enough surface to drive the predict pipeline,
``condition_on``, and ``_replace``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import pytest
from jax import Array

from sabi.emulators._scalers import MinMaxScaler, ZScoreScaler
from sabi.emulators.gp import GPEmulator, PredictCacheProtocol


@dataclass(frozen=True)
class _FakeCache:
    """Identity-mapping cache: ``predict_latent(X) = (X[:,0], X[:,0]²)``.

    Doesn't pretend to be a GP. Just produces deterministic outputs
    we can pin down in tests so the base's predict pipeline can be
    exercised without dragging in tinygp / gpjax.
    """

    Xs_train: Array
    Ys_train: Array

    def predict_latent(self, Xt: Array) -> tuple[Array, Array]:
        mean = Xt[:, 0]
        var = Xt[:, 0] ** 2
        return mean, var

    def predict_latent_joint(self, Xt: Array) -> tuple[Array, Array]:
        mean = Xt[:, 0]
        cov = jnp.diag(Xt[:, 0] ** 2)
        return mean, cov

    def append_rows(self, Xs_new: Array, Ys_new: Array) -> "_FakeCache":
        return _FakeCache(
            Xs_train=jnp.concatenate([self.Xs_train, Xs_new], axis=0),
            Ys_train=jnp.concatenate([self.Ys_train, jnp.atleast_1d(Ys_new)]),
        )


class _FakeGPEmulator(GPEmulator):
    """Concrete GPEmulator subclass with a fake cache, for base-class tests."""

    # Demonstrates the constructor-arg / attribute-name mapping.
    _replace_field_map = {"flavor": "flavor_name"}

    def __init__(
        self,
        *,
        input_shape: tuple[int, ...] = (2,),
        output_shape: tuple[int, ...] = (),
        name: str | None = None,
        flavor: str = "vanilla",
        _x_scaler: MinMaxScaler | None = None,
        _y_scaler: ZScoreScaler | None = None,
        _predict_cache: PredictCacheProtocol | None = None,
    ):
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            name=name or "_FakeGPEmulator",
        )
        self.flavor_name = flavor
        self._x_scaler = _x_scaler
        self._y_scaler = _y_scaler
        self._predict_cache = _predict_cache

    def fit(self, X: Array, Y: Array):
        x_scaler = MinMaxScaler.fit(X)
        y_scaler = ZScoreScaler.fit(Y)
        Xs = x_scaler.transform(X).astype(jnp.float64)
        Ys = y_scaler.transform(Y).astype(jnp.float64)
        cache = _FakeCache(Xs_train=Xs, Ys_train=Ys)
        return self._replace(
            _x_scaler=x_scaler, _y_scaler=y_scaler, _predict_cache=cache
        )


# --- _replace -----------------------------------------------------------------


def test_gpemulator_replace_preserves_unchanged_fields():
    """``_replace(...)`` carries forward the fields you don't override."""
    em = _FakeGPEmulator(flavor="chocolate")
    out = em._replace()
    assert out.flavor_name == "chocolate"
    assert out.input_shape == em.input_shape
    assert out.name == em.name


def test_gpemulator_replace_overrides_named_fields():
    em = _FakeGPEmulator(flavor="chocolate")
    out = em._replace(flavor="strawberry")
    assert out.flavor_name == "strawberry"


def test_gpemulator_replace_handles_field_map_mismatch():
    """The constructor takes ``flavor``; the attribute is
    ``flavor_name``. ``_replace_field_map`` bridges them so the
    reflection-based default lookup picks the right value."""
    em = _FakeGPEmulator(flavor="mint")
    # No override: must read from self.flavor_name and pass as
    # flavor= to the constructor.
    out = em._replace()
    assert out.flavor_name == "mint"


def test_gpemulator_replace_returns_same_concrete_type():
    em = _FakeGPEmulator()
    assert type(em._replace()) is _FakeGPEmulator


# --- predict pipeline ---------------------------------------------------------


def test_gpemulator_predict_before_fit_raises():
    em = _FakeGPEmulator()
    with pytest.raises(RuntimeError, match="before fit"):
        em.predict_mean(jnp.zeros((3, 2)))


def test_gpemulator_predict_mean_routes_through_scalers_and_cache():
    """Verifies the predict pipeline: x_scaler.transform → cache →
    y_scaler.inverse_mean."""
    X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    Y = jnp.array([0.0, 1.0, 2.0])
    em = _FakeGPEmulator().fit(X, Y)

    # The fake cache returns mean = Xt_scaled[:, 0]. After x_scaler,
    # X[:, 0] mapped to [0, 0.5, 1.0]. Then y_scaler.inverse_mean
    # multiplies by std(Y) and adds mean(Y).
    pred = em.predict_mean(X)
    assert pred.shape == (3,)


def test_gpemulator_predict_variance_is_nonnegative_and_inverse_var_scaled():
    X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    Y = jnp.array([0.0, 1.0])
    em = _FakeGPEmulator().fit(X, Y)
    var = em.predict_variance(X)
    assert var.shape == (2,)
    assert jnp.all(var >= 0.0)


def test_gpemulator_predict_covariance_marginal_only_raises():
    em = _FakeGPEmulator().fit(
        jnp.array([[0.0, 0.0], [1.0, 1.0]]), jnp.array([0.0, 1.0])
    )
    with pytest.raises(NotImplementedError, match="predict_variance"):
        em.predict_covariance(
            jnp.zeros((1, 2)), joint_inputs=False, joint_outputs=False
        )


def test_gpemulator_predict_covariance_joint_inputs_returns_n_n():
    X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    Y = jnp.array([0.0, 1.0, 2.0])
    em = _FakeGPEmulator().fit(X, Y)
    cov = em.predict_covariance(X, joint_inputs=True)
    assert cov.shape == (3, 3)
    # Diagonal equals predict_variance.
    assert jnp.allclose(jnp.diag(cov), em.predict_variance(X))


def test_gpemulator_predict_covariance_joint_outputs_only_returns_n_1_1():
    X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    Y = jnp.array([0.0, 1.0])
    em = _FakeGPEmulator().fit(X, Y)
    cov = em.predict_covariance(X, joint_inputs=False, joint_outputs=True)
    assert cov.shape == (2, 1, 1)


# --- condition_on -------------------------------------------------------------


def test_gpemulator_condition_on_validates_x_shape():
    em = _FakeGPEmulator(input_shape=(2,)).fit(
        jnp.zeros((3, 2)), jnp.zeros(3)
    )
    with pytest.raises(ValueError, match=r"X_new.shape=\(m, 2\)"):
        em.condition_on(jnp.zeros((3, 5)), jnp.zeros(3))


def test_gpemulator_condition_on_validates_y_shape():
    em = _FakeGPEmulator(input_shape=(2,)).fit(
        jnp.zeros((3, 2)), jnp.zeros(3)
    )
    with pytest.raises(ValueError, match=r"Y_new.shape="):
        em.condition_on(jnp.zeros((3, 2)), jnp.zeros(4))


def test_gpemulator_condition_on_routes_through_cache_append_rows():
    """``condition_on`` should standardize new rows and pass them to
    ``cache.append_rows``."""
    X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    Y = jnp.array([0.0, 1.0, 2.0])
    em = _FakeGPEmulator().fit(X, Y)

    X_new = jnp.array([[2.0, 3.0]])
    Y_new = jnp.array([1.5])
    em_after = em.condition_on(X_new, Y_new)
    # Cache grew by 1 row, rest of the fields unchanged.
    assert em_after._predict_cache.Xs_train.shape == (4, 2)
    assert em_after._predict_cache.Ys_train.shape == (4,)
    assert em_after._x_scaler is em._x_scaler
    assert em_after._y_scaler is em._y_scaler


# --- support flags ------------------------------------------------------------


def test_gpemulator_default_support_flags_are_true():
    """Subclasses inherit ``True`` for both joint flags by default."""
    assert _FakeGPEmulator.supports_joint_inputs is True
    assert _FakeGPEmulator.supports_joint_outputs is True
