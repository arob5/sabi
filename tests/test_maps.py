"""Tests for the `Map` ABC and concrete `Map` subclasses.

The shape contract for `Map.__call__`: given input of shape
``batch_shape + event_shape_in``, returns ``batch_shape + event_shape_out``.
``Compose(f, g)`` requires ``f.event_shape_in == g.event_shape_out``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from probpipe.distributions.continuous import Normal
from probpipe.distributions.multivariate import MultivariateNormal

from sabi.maps import (
    Affine,
    Compose,
    Constant,
    Exp,
    GaussianLogLik,
    Identity,
    Log,
    LogProb,
    LogSoftplus,
    LogSquare,
    Softplus,
    Square,
)

# -------------------------------------------------------------------------
# Per-map __call__ correctness
# -------------------------------------------------------------------------


def test_identity_call_returns_input():
    f = Identity()
    z = jnp.asarray(0.7)
    assert float(f(z)) == pytest.approx(0.7)


def test_constant_call_returns_c_regardless_of_input():
    c = jnp.asarray([1.0, -2.0])
    f = Constant(c=c)
    assert jnp.allclose(f(jnp.asarray(0.0)), c)
    assert jnp.allclose(f(jnp.asarray(99.0)), c)


def test_affine_call_matches_formula():
    f = Affine(slope=2.0, intercept=-3.0)
    z = jnp.asarray(5.0)
    assert float(f(z)) == pytest.approx(2.0 * 5.0 - 3.0)


def test_exp_call_matches_jnp_exp():
    f = Exp()
    z = jnp.asarray(1.5)
    assert float(f(z)) == pytest.approx(float(jnp.exp(z)))


def test_log_call_matches_jnp_log():
    f = Log()
    z = jnp.asarray(2.0)
    assert float(f(z)) == pytest.approx(float(jnp.log(z)))


def test_softplus_call_matches_jax_softplus():
    f = Softplus()
    z = jnp.asarray(-0.5)
    assert float(f(z)) == pytest.approx(float(jax.nn.softplus(z)))


def test_log_softplus_call_matches_log_of_softplus():
    f = LogSoftplus()
    z = jnp.asarray(0.5)
    assert float(f(z)) == pytest.approx(float(jnp.log(jax.nn.softplus(z))))


def test_square_call_matches_z_squared():
    f = Square()
    z = jnp.asarray(-3.0)
    assert float(f(z)) == pytest.approx(9.0)


def test_log_square_call_matches_2_log_abs_z():
    f = LogSquare()
    z = jnp.asarray(-2.0)
    assert float(f(z)) == pytest.approx(2.0 * float(jnp.log(2.0)))


def test_gaussian_log_lik_call_matches_normal_log_prob():
    """`GaussianLogLik(obs, cov)(z) = log N(obs | z, cov)` agrees with
    a TFP MVN's log_prob computed at obs with mean z."""
    obs = jnp.asarray([1.0, 0.5])
    cov = jnp.asarray([[2.0, 0.3], [0.3, 1.0]])
    f = GaussianLogLik(obs=obs, cov=cov)
    z = jnp.asarray([0.2, -0.4])
    # Reference via MVN(z, cov).log_prob(obs)
    ref = MultivariateNormal(loc=z, scale_tril=jnp.linalg.cholesky(cov), name="ref")
    from probpipe import log_prob as pp_log_prob
    expected = float(pp_log_prob(ref, obs))
    assert float(f(z)) == pytest.approx(expected, abs=1e-5)


def test_log_prob_call_matches_distribution_log_prob():
    dist = Normal(loc=jnp.asarray(0.0), scale=jnp.asarray(1.0), name="p")
    f = LogProb(dist=dist)
    z = jnp.asarray(1.5)
    expected = -0.5 * 1.5 ** 2 - 0.5 * float(jnp.log(2 * jnp.pi))
    assert float(f(z)) == pytest.approx(expected, abs=1e-5)


# -------------------------------------------------------------------------
# event_shape_in / event_shape_out
# -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        Identity,
        Exp,
        Log,
        Softplus,
        LogSoftplus,
        Square,
        LogSquare,
        lambda: Affine(slope=1.0, intercept=0.0),
    ],
)
def test_scalar_maps_event_shapes_are_unit(factory):
    f = factory()
    assert f.event_shape_in == ()
    assert f.event_shape_out == ()


def test_constant_event_shapes_match_c_shape():
    c = jnp.zeros((3, 2))
    f = Constant(c=c)
    assert f.event_shape_in == ()
    assert f.event_shape_out == (3, 2)


def test_gaussian_log_lik_event_shapes():
    f = GaussianLogLik(obs=jnp.zeros(4), cov=jnp.eye(4))
    assert f.event_shape_in == (4,)
    assert f.event_shape_out == ()


def test_log_prob_event_shapes_match_dist_event_shape():
    dist = MultivariateNormal(loc=jnp.zeros(3), scale_tril=jnp.eye(3), name="p")
    f = LogProb(dist=dist)
    assert f.event_shape_in == (3,)
    assert f.event_shape_out == ()


# -------------------------------------------------------------------------
# Batched-input broadcasting
# -------------------------------------------------------------------------


def test_scalar_map_broadcasts_over_leading_batch():
    """Per the shape contract, calling ``f`` on shape
    ``batch + event_shape_in`` returns ``batch + event_shape_out``."""
    f = Affine(slope=3.0, intercept=1.0)
    z = jnp.asarray([1.0, 2.0, 3.0])  # batch_shape == (3,)
    out = f(z)
    assert out.shape == (3,)
    assert jnp.allclose(out, jnp.asarray([4.0, 7.0, 10.0]))


def test_gaussian_log_lik_broadcasts_over_leading_batch():
    """Map with ``event_shape_in == (d,)`` collapses each row."""
    f = GaussianLogLik(obs=jnp.zeros(2), cov=jnp.eye(2))
    Z = jnp.asarray([[0.0, 0.0], [1.0, 1.0], [-1.0, 0.5]])
    out = jax.vmap(f)(Z)
    assert out.shape == (3,)


def test_constant_broadcasts_against_arbitrary_input_shape():
    c = jnp.asarray([1.0, 2.0])
    f = Constant(c=c)
    z_batch = jnp.zeros((4,))
    out = f(z_batch)
    assert out.shape == (4, 2)
    assert jnp.allclose(out[0], c)


# -------------------------------------------------------------------------
# Compose / @ operator
# -------------------------------------------------------------------------


def test_matmul_returns_compose():
    f = Exp()
    g = Affine(slope=2.0)
    composed = f @ g
    assert isinstance(composed, Compose)
    assert composed.f is f
    assert composed.g is g


def test_compose_call_equals_f_of_g():
    """``(f @ g)(z) == f(g(z))``."""
    f = Exp()
    g = Affine(slope=2.0, intercept=-1.0)
    z = jnp.asarray(0.5)
    expected = float(f(g(z)))
    assert float((f @ g)(z)) == pytest.approx(expected, abs=1e-6)


def test_compose_with_identity_is_no_op():
    f = Affine(slope=3.0, intercept=2.0)
    z = jnp.asarray(0.7)
    assert float((f @ Identity())(z)) == pytest.approx(float(f(z)))
    assert float((Identity() @ f)(z)) == pytest.approx(float(f(z)))


def test_compose_associativity():
    f = Exp()
    g = Affine(slope=2.0, intercept=1.0)
    h = LogSoftplus()
    z = jnp.asarray(0.3)
    left = f @ (g @ h)
    right = (f @ g) @ h
    assert float(left(z)) == pytest.approx(float(right(z)), abs=1e-6)


def test_compose_event_shape_propagation():
    g = GaussianLogLik(obs=jnp.zeros(3), cov=jnp.eye(3))
    f = Affine(slope=2.0)
    composed = f @ g
    assert composed.event_shape_in == (3,)
    assert composed.event_shape_out == ()


def test_compose_shape_mismatch_raises():
    """``f.event_shape_in == g.event_shape_out`` is required at construction."""
    g = GaussianLogLik(obs=jnp.zeros(3), cov=jnp.eye(3))  # (3,) -> ()
    f_bad = LogProb(
        dist=MultivariateNormal(loc=jnp.zeros(2), scale_tril=jnp.eye(2), name="p")
    )  # (2,) -> ()
    with pytest.raises(ValueError, match="shape mismatch"):
        Compose(f_bad, g)
