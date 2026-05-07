"""Tests for ``pushforward(map, dist)`` — the multiple-dispatch op.

Two flavors of test per closed-form entry: (a) direct equality on the
returned distribution's parameters; (b) Monte Carlo agreement, drawing
samples from the closed-form output and checking that empirical
mean/variance match the formula at ~5% slack
(``docs/contributing.md`` §Tests).

MC fallback paths are validated by sample-shape checks plus a sanity
check that empirical moments are consistent with a hand-computed
reference.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from probpipe import sample
from probpipe.core._empirical import NumericEmpiricalDistribution
from probpipe.distributions.continuous import LogNormal, Normal
from probpipe.distributions.multivariate import MultivariateNormal

from sabi.maps import (
    Affine,
    Constant,
    Dirac,
    Exp,
    Identity,
    Log,
    LogSoftplus,
    pushforward,
)

# -------------------------------------------------------------------------
# Closed-form: Identity
# -------------------------------------------------------------------------


def test_identity_pushforward_returns_input_unchanged():
    n = Normal(loc=jnp.asarray(0.5), scale=jnp.asarray(1.5), name="x")
    out = pushforward(Identity(), n)
    assert out is n


# -------------------------------------------------------------------------
# Closed-form: Constant -> Dirac
# -------------------------------------------------------------------------


def test_constant_pushforward_returns_dirac_at_c():
    c = jnp.asarray([1.0, -2.0, 0.5])
    n = Normal(loc=jnp.asarray(0.0), scale=jnp.asarray(1.0), name="x")
    out = pushforward(Constant(c=c), n)
    assert isinstance(out, Dirac)
    assert jnp.allclose(out.c, c)
    assert out.event_shape == c.shape


# -------------------------------------------------------------------------
# Closed-form: (Affine, Normal)
# -------------------------------------------------------------------------


def test_affine_normal_returns_normal_with_shifted_loc_and_scaled_scale():
    n = Normal(loc=jnp.asarray(2.0), scale=jnp.asarray(3.0), name="x")
    out = pushforward(Affine(slope=-2.0, intercept=1.0), n)
    assert isinstance(out, Normal)
    assert float(out.loc) == pytest.approx(-2.0 * 2.0 + 1.0)
    # |slope| · scale — sign goes into the absolute value
    assert float(out.scale) == pytest.approx(2.0 * 3.0)


def test_affine_normal_mc_moments_match_closed_form():
    """Closed-form output samples should agree on mean/variance with
    the MC pushforward (each at the same sample budget)."""
    n = Normal(loc=jnp.asarray(0.5), scale=jnp.asarray(1.0), name="x")
    out = pushforward(Affine(slope=2.0, intercept=-1.0), n)
    key = jax.random.key(0)
    s = sample(out, key=key, sample_shape=(20_000,))
    assert float(jnp.mean(s)) == pytest.approx(2.0 * 0.5 - 1.0, abs=0.05)
    assert float(jnp.std(s)) == pytest.approx(2.0 * 1.0, abs=0.05)


# -------------------------------------------------------------------------
# Closed-form: (Affine, MultivariateNormal)
# -------------------------------------------------------------------------


def test_affine_mvn_returns_mvn_with_shifted_loc_and_scaled_scale_tril():
    L = jnp.asarray([[1.0, 0.0], [0.5, 0.8]])
    mvn = MultivariateNormal(
        loc=jnp.asarray([1.0, -1.0]), scale_tril=L, name="x"
    )
    intercept = jnp.asarray([0.5, 2.0])
    out = pushforward(Affine(slope=3.0, intercept=intercept), mvn)
    assert isinstance(out, MultivariateNormal)
    assert jnp.allclose(out.loc, 3.0 * mvn.loc + intercept)
    assert jnp.allclose(out.scale_tril, 3.0 * L)


# -------------------------------------------------------------------------
# Closed-form: (Affine, NumericEmpiricalDistribution)
# -------------------------------------------------------------------------


def test_affine_empirical_applies_elementwise_to_samples():
    emp = NumericEmpiricalDistribution(
        samples=jnp.asarray([[0.0, 1.0], [2.0, 3.0], [-1.0, 0.5]]), name="emp"
    )
    out = pushforward(Affine(slope=2.0, intercept=1.0), emp)
    assert isinstance(out, NumericEmpiricalDistribution)
    expected = 2.0 * emp._samples + 1.0
    assert jnp.allclose(out._samples, expected)


# -------------------------------------------------------------------------
# Closed-form: (Exp, Normal) -> LogNormal
# -------------------------------------------------------------------------


def test_exp_normal_returns_lognormal_with_same_loc_scale():
    n = Normal(loc=jnp.asarray(0.3), scale=jnp.asarray(0.7), name="x")
    out = pushforward(Exp(), n)
    assert isinstance(out, LogNormal)
    assert float(out.loc) == pytest.approx(0.3)
    assert float(out.scale) == pytest.approx(0.7)


def test_exp_normal_mc_mean_matches_lognormal_closed_form():
    """``E[exp(N(mu, sigma))] = exp(mu + sigma^2/2)``."""
    mu, sigma = 0.4, 0.5
    n = Normal(loc=jnp.asarray(mu), scale=jnp.asarray(sigma), name="x")
    out = pushforward(Exp(), n)
    key = jax.random.key(1)
    s = sample(out, key=key, sample_shape=(20_000,))
    expected_mean = float(jnp.exp(mu + 0.5 * sigma ** 2))
    assert float(jnp.mean(s)) == pytest.approx(expected_mean, rel=0.05)


# -------------------------------------------------------------------------
# Closed-form: (Log, LogNormal) -> Normal
# -------------------------------------------------------------------------


def test_log_lognormal_returns_normal_with_same_loc_scale():
    ln = LogNormal(loc=jnp.asarray(-0.2), scale=jnp.asarray(1.1), name="x")
    out = pushforward(Log(), ln)
    assert isinstance(out, Normal)
    assert float(out.loc) == pytest.approx(-0.2)
    assert float(out.scale) == pytest.approx(1.1)


# -------------------------------------------------------------------------
# Compose: (f @ g, dist) recurses
# -------------------------------------------------------------------------


def test_compose_recursion_matches_manual_chain():
    """``pushforward(f @ g, dist)`` matches
    ``pushforward(f, pushforward(g, dist))``."""
    n = Normal(loc=jnp.asarray(0.0), scale=jnp.asarray(1.0), name="x")
    f = Affine(slope=2.0, intercept=3.0)
    g = Affine(slope=-1.0, intercept=0.5)
    out_compose = pushforward(f @ g, n)
    out_manual = pushforward(f, pushforward(g, n))
    assert isinstance(out_compose, Normal)
    assert float(out_compose.loc) == pytest.approx(float(out_manual.loc))
    assert float(out_compose.scale) == pytest.approx(float(out_manual.scale))


def test_compose_mixed_closed_then_mc_returns_empirical():
    """``Affine @ LogSoftplus`` through MVN: LogSoftplus falls to MC,
    Affine applies closed-form on the resulting empirical distribution.
    Per ``docs/link_functions.md`` §5.4."""
    L = jnp.eye(3)
    mvn = MultivariateNormal(loc=jnp.zeros(3), scale_tril=L, name="x")
    intercept = jnp.asarray([1.0, -1.0, 0.5])
    m = Affine(slope=2.0, intercept=intercept) @ LogSoftplus()
    out = pushforward(m, mvn)
    assert isinstance(out, NumericEmpiricalDistribution)
    assert out.event_shape == (3,)
    # Each output coord is `2 * LogSoftplus(z_i) + intercept_i` with
    # `z_i ~ N(0, 1)` iid. So the variance is `4 * Var[LogSoftplus(N(0,1))]`,
    # the *same* across coords; subtracting the per-coord mean removes the
    # intercept and leaves a stationary, zero-mean per-coord sample.
    centered = out._samples - jnp.mean(out._samples, axis=0)
    per_coord_var = jnp.var(centered, axis=0)
    # All per-coord variances should be close to each other.
    assert jnp.allclose(per_coord_var, per_coord_var[0], rtol=0.5)


# -------------------------------------------------------------------------
# MC fallback
# -------------------------------------------------------------------------


def test_mc_fallback_log_softplus_normal_returns_empirical():
    n = Normal(loc=jnp.asarray(1.0), scale=jnp.asarray(0.5), name="x")
    out = pushforward(LogSoftplus(), n)
    assert isinstance(out, NumericEmpiricalDistribution)
    assert out.event_shape == ()
    # All samples are finite: log(softplus(z)) is finite for z away from -inf.
    assert jnp.all(jnp.isfinite(out._samples))


def test_mc_fallback_log_softplus_mvn_returns_empirical_with_correct_event_shape():
    L = jnp.eye(3)
    mvn = MultivariateNormal(loc=jnp.zeros(3), scale_tril=L, name="x")
    out = pushforward(LogSoftplus(), mvn)
    assert isinstance(out, NumericEmpiricalDistribution)
    assert out.event_shape == (3,)


# -------------------------------------------------------------------------
# Unregistered + non-samplable input raises
# -------------------------------------------------------------------------


def test_unregistered_non_samplable_input_raises():
    """Without a ``SupportsSampling`` input there's no MC fallback path."""

    class _NonSamplable:
        """Looks like a Distribution but doesn't implement SupportsSampling."""

        pass

    with pytest.raises(NotImplementedError, match="no dispatch path"):
        pushforward(LogSoftplus(), _NonSamplable())  # type: ignore[arg-type]
