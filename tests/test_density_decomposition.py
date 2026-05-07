"""Tests for `DensityDecomposition` and `is_consistent_with`.

Coverage:
- Field contracts: required vs default fields, frozen-dataclass identity.
- ``target_map`` derivation via ``jax.vmap``.
- ``__call__`` and ``density_at`` evaluations under common ``(link, shift)``
  shapes (Identity / Constant, Identity / LogProb, Affine link, etc.).
- ``pushforward(x, y_dist)`` against a closed-form Gaussian-affine path
  and an MC fallback (via a non-default link).
- The four classmethod helpers (``identity_from_target``,
  ``likelihood_with_prior``, ``forward_model``,
  ``gaussian_forward_model``).
- ``is_consistent_with`` in both ``strict=True`` and ``strict=False``
  modes, plus the ``NotImplementedError`` short-circuit for user
  inverse problems.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from probpipe.core._distribution_base import Distribution
from probpipe.core.constraints import non_negative, real
from probpipe.distributions.continuous import Normal

from sabi._probpipe_compat import independent_uniform
from sabi.density_decomposition import (
    DensityDecomposition,
    ScalarConstant,
    is_consistent_with,
)
from sabi.maps import (
    Affine,
    Constant,
    GaussianLogLik,
    Identity,
    LogProb,
    LogSquare,
)
from sabi.target_distribution import TargetDistribution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _box_prior(d: int = 2):
    return independent_uniform(
        low=jnp.full((d,), -5.0),
        high=jnp.full((d,), 5.0),
        name=f"box_prior_d{d}",
    )


def _quad_log_density(x):
    return -0.5 * jnp.sum(x * x)


def _benchmark_target(d: int = 2) -> TargetDistribution:
    """Analytical-density target: ``_unnormalized_log_prob(x) = -0.5 * x.x``."""
    return TargetDistribution(
        name=f"quad_d{d}",
        input_shape=(d,),
        support=_box_prior(d).support,
        unnormalized_log_prob=_quad_log_density,
    )


def _opaque_target(d: int = 2) -> TargetDistribution:
    """User-style target: ``_unnormalized_log_prob`` raises ``NotImplementedError``."""
    return TargetDistribution(
        name=f"opaque_d{d}",
        input_shape=(d,),
        support=_box_prior(d).support,
    )


# ---------------------------------------------------------------------------
# Field contracts
# ---------------------------------------------------------------------------


def test_default_shift_returns_scalar_zero():
    dd = DensityDecomposition(
        target_single=_quad_log_density,
        output_shape=(),
        link=Identity(),
    )
    # Default shift returns scalar zero regardless of input shape.
    out_single = dd.shift(jnp.asarray([1.0, 2.0]))
    assert out_single.shape == ()
    assert float(out_single) == 0.0
    out_batched = dd.shift(jnp.zeros((4, 2)))
    assert out_batched.shape == (4,)
    assert float(jnp.sum(out_batched)) == 0.0


def test_default_constraint_is_real():
    dd = DensityDecomposition(
        target_single=_quad_log_density,
        output_shape=(),
        link=Identity(),
    )
    assert dd.constraint is real


def test_target_map_derived_via_vmap():
    dd = DensityDecomposition(
        target_single=_quad_log_density,
        output_shape=(),
        link=Identity(),
    )
    X = jnp.asarray([[0.0, 0.0], [1.0, -1.0], [2.0, 0.5]])
    expected = jax.vmap(_quad_log_density)(X)
    assert jnp.allclose(dd.target_map(X), expected)


# ---------------------------------------------------------------------------
# __call__ and density_at
# ---------------------------------------------------------------------------


def test_call_identity_link_zero_shift():
    """``link = Identity, shift = Constant(0)``: ``__call__`` returns ``y``."""
    dd = DensityDecomposition(
        target_single=_quad_log_density,
        output_shape=(),
        link=Identity(),
    )
    x = jnp.asarray([1.0, 0.5])
    y = jnp.asarray(2.5)
    assert float(dd(x, y)) == pytest.approx(2.5)


def test_call_identity_link_logprob_shift():
    """``shift = LogProb(prior)``: ``__call__`` returns ``y + log_prior(x)``."""
    prior = _box_prior(2)
    dd = DensityDecomposition(
        target_single=_quad_log_density,
        output_shape=(),
        link=Identity(),
        shift=LogProb(prior),
    )
    from probpipe import log_prob

    x = jnp.asarray([1.0, 0.5])
    y = jnp.asarray(2.5)
    expected = 2.5 + float(log_prob(prior, x))
    assert float(dd(x, y)) == pytest.approx(expected, abs=1e-6)


def test_call_affine_link():
    """``link = Affine(slope=2, intercept=1)``: ``__call__`` returns ``2y + 1``."""
    dd = DensityDecomposition(
        target_single=_quad_log_density,
        output_shape=(),
        link=Affine(slope=jnp.asarray(2.0), intercept=jnp.asarray(1.0)),
    )
    x = jnp.asarray([0.0, 0.0])
    y = jnp.asarray(3.0)
    assert float(dd(x, y)) == pytest.approx(7.0)


def test_density_at_routes_through_target_single():
    """``density_at(x) == link(target_single(x)) + shift(x)``."""
    dd = DensityDecomposition(
        target_single=_quad_log_density,
        output_shape=(),
        link=Identity(),
    )
    x = jnp.asarray([1.0, 0.5])
    expected = float(_quad_log_density(x))
    assert float(dd.density_at(x)) == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# pushforward(x, y_dist)
# ---------------------------------------------------------------------------


def test_pushforward_closed_form_gaussian_affine():
    """``link=Identity, shift=LogProb(prior)`` × ``Normal`` → closed-form ``Normal``.

    The pushforward composes ``Affine(slope=1, intercept=shift(x)) @ Identity ==
    Affine(intercept=shift(x))``. Pushing ``Normal(loc, scale)`` through this
    gives ``Normal(loc + shift(x), scale)``.
    """
    prior = _box_prior(2)
    dd = DensityDecomposition(
        target_single=_quad_log_density,
        output_shape=(),
        link=Identity(),
        shift=LogProb(prior),
    )
    from probpipe import log_prob

    x = jnp.asarray([1.0, 0.5])
    y_dist = Normal(loc=jnp.asarray(2.0), scale=jnp.asarray(0.5), name="y_dist")
    pushed = dd.pushforward(x, y_dist)
    assert isinstance(pushed, Normal)
    expected_loc = 2.0 + float(log_prob(prior, x))
    assert float(pushed.loc) == pytest.approx(expected_loc, abs=1e-6)
    assert float(pushed.scale) == pytest.approx(0.5, abs=1e-6)


def test_pushforward_mc_fallback_for_non_affine_link():
    """``link=LogSquare`` × ``Normal`` falls to MC (no closed-form pair)."""
    dd = DensityDecomposition(
        target_single=_quad_log_density,
        output_shape=(),
        link=LogSquare(),
    )
    x = jnp.asarray([0.0, 0.0])
    y_dist = Normal(loc=jnp.asarray(1.0), scale=jnp.asarray(0.5), name="y_dist")
    pushed = dd.pushforward(x, y_dist)
    # MC fallback returns a NumericEmpiricalDistribution-shaped object;
    # we just assert it's a Distribution and not the raw Normal.
    assert isinstance(pushed, Distribution)
    assert not isinstance(pushed, Normal)


# ---------------------------------------------------------------------------
# Classmethod helpers
# ---------------------------------------------------------------------------


def test_identity_from_target_recovers_analytical_density():
    target = _benchmark_target(2)
    dd = DensityDecomposition.identity_from_target(target)
    x = jnp.asarray([0.5, -0.3])
    expected = float(_quad_log_density(x))
    assert float(dd.density_at(x)) == pytest.approx(expected, abs=1e-6)
    assert isinstance(dd.link, Identity)


def test_likelihood_with_prior_default_link_is_identity():
    prior = _box_prior(2)

    def log_lik(x):
        return -0.5 * jnp.sum(x * x)

    dd = DensityDecomposition.likelihood_with_prior(
        log_lik,
        output_shape=(),
        modeling_prior=prior,
    )
    assert isinstance(dd.link, Identity)
    assert isinstance(dd.shift, LogProb)
    assert dd.shift.dist is prior

    from probpipe import log_prob

    x = jnp.asarray([0.5, -0.3])
    expected = float(log_lik(x)) + float(log_prob(prior, x))
    assert float(dd.density_at(x)) == pytest.approx(expected, abs=1e-6)


def test_forward_model_uses_provided_link_map():
    prior = _box_prior(2)
    obs = jnp.asarray([0.0, 0.0])
    cov = jnp.eye(2)
    log_lik_map = GaussianLogLik(obs=obs, cov=cov)

    def fmodel(x):
        return x  # simulated observation = x itself

    dd = DensityDecomposition.forward_model(
        fmodel,
        output_shape=(2,),
        log_lik_from_outputs=log_lik_map,
        modeling_prior=prior,
    )
    assert dd.link is log_lik_map
    assert isinstance(dd.shift, LogProb)


def test_gaussian_forward_model_constructs_gaussian_log_lik():
    prior = _box_prior(2)
    dd = DensityDecomposition.gaussian_forward_model(
        forward_model=lambda x: x,
        output_shape=(2,),
        obs=jnp.asarray([0.0, 0.0]),
        cov=jnp.eye(2),
        modeling_prior=prior,
    )
    assert isinstance(dd.link, GaussianLogLik)
    assert isinstance(dd.shift, LogProb)


# ---------------------------------------------------------------------------
# is_consistent_with
# ---------------------------------------------------------------------------


def test_is_consistent_with_strict_passes_for_identity_from_target():
    target = _benchmark_target(2)
    dd = DensityDecomposition.identity_from_target(target)
    x_test = jnp.asarray([[0.0, 0.0], [0.5, -0.3], [1.0, 1.0]])
    assert is_consistent_with(dd, target, x_test=x_test, strict=True)


def test_is_consistent_with_strict_rejects_constant_offset():
    target = _benchmark_target(2)
    # Add a +1.0 shift — pointwise mismatch.
    dd = DensityDecomposition(
        target_single=_quad_log_density,
        output_shape=(),
        link=Identity(),
        shift=ScalarConstant(jnp.asarray(1.0)),
    )
    x_test = jnp.asarray([[0.0, 0.0], [0.5, -0.3]])
    assert not is_consistent_with(dd, target, x_test=x_test, strict=True)


def test_is_consistent_with_loose_accepts_constant_offset():
    """``strict=False`` allows an additive constant."""
    target = _benchmark_target(2)
    dd = DensityDecomposition(
        target_single=_quad_log_density,
        output_shape=(),
        link=Identity(),
        shift=ScalarConstant(jnp.asarray(1.0)),
    )
    x_test = jnp.asarray([[0.0, 0.0], [0.5, -0.3], [1.0, 1.0]])
    assert is_consistent_with(dd, target, x_test=x_test, strict=False)


def test_is_consistent_with_loose_rejects_x_dependent_offset():
    target = _benchmark_target(2)
    # An x-dependent shift — uniform's log_prob is constant inside the
    # box, so we need a Normal-style prior whose log_prob actually
    # varies with x.
    from probpipe.distributions.multivariate import MultivariateNormal

    normal_prior = MultivariateNormal(
        loc=jnp.zeros(2), cov=jnp.eye(2), name="normal_prior"
    )
    dd = DensityDecomposition(
        target_single=_quad_log_density,
        output_shape=(),
        link=Identity(),
        shift=LogProb(normal_prior),  # x-dependent shift
    )
    x_test = jnp.asarray([[0.0, 0.0], [0.5, -0.3], [1.0, 1.0]])
    assert not is_consistent_with(dd, target, x_test=x_test, strict=False)


def test_is_consistent_with_returns_true_for_opaque_target():
    """Targets without analytical density are trivially consistent."""
    target = _opaque_target(2)
    dd = DensityDecomposition(
        target_single=_quad_log_density,
        output_shape=(),
        link=Identity(),
    )
    x_test = jnp.asarray([[0.0, 0.0], [1.0, 1.0]])
    assert is_consistent_with(dd, target, x_test=x_test, strict=True)


def test_is_consistent_with_loose_requires_two_rows():
    target = _benchmark_target(2)
    dd = DensityDecomposition.identity_from_target(target)
    with pytest.raises(ValueError, match="n >= 2"):
        is_consistent_with(
            dd, target, x_test=jnp.asarray([[0.0, 0.0]]), strict=False
        )


def test_is_consistent_with_rejects_rank_one_x_test():
    target = _benchmark_target(2)
    dd = DensityDecomposition.identity_from_target(target)
    with pytest.raises(ValueError, match="at least rank-2"):
        is_consistent_with(dd, target, x_test=jnp.asarray([0.0, 0.0]))


# ---------------------------------------------------------------------------
# Constraint metadata round-trip
# ---------------------------------------------------------------------------


def test_constraint_round_trips():
    dd = DensityDecomposition(
        target_single=_quad_log_density,
        output_shape=(),
        link=LogSquare(),
        constraint=non_negative,
    )
    assert dd.constraint is non_negative
