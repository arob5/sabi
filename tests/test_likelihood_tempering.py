r"""Tests for `LikelihoodTemperingViaForm` and `LikelihoodTemperingViaTarget`.

After the ``DensityDecomposition`` split, tempering schemes operate on
the algorithm's :class:`DensityDecomposition` (via
``intermediate_decomposition``) rather than rewriting form classes.
The math is unchanged: both schemes encode

.. math::

    \pi_\beta(x) \propto \pi_0(x) \cdot L(x)^\beta

(or the geometric bridge ``π_0^(1-β) · π_target^β`` when ``base.shift``
is ``Constant(0)`` and ``initial`` is supplied).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from probpipe import log_prob

from sabi._probpipe_compat import independent_uniform
from sabi.density_decomposition import DensityDecomposition
from sabi.maps import Affine, Constant, GaussianLogLik, Identity, LogProb
from sabi.target_distribution import IntermediateTarget, TargetDistribution
from sabi.tempering.likelihood import (
    LikelihoodTemperingViaForm,
    LikelihoodTemperingViaTarget,
)


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


def _flat_prior():
    return independent_uniform(
        low=jnp.full((2,), -5.0),
        high=jnp.full((2,), 5.0),
        name="flat_prior",
    )


def _quadratic_log_lik(x):
    return -0.5 * jnp.sum(x * x)


def _box_support():
    return _flat_prior().support


def _target_with_density(unnormalized_log_prob=None) -> TargetDistribution:
    return TargetDistribution(
        name="quadratic_target",
        input_shape=(2,),
        support=_box_support(),
        unnormalized_log_prob=unnormalized_log_prob or _quadratic_log_lik,
    )


def _log_lik_plus_prior_decomposition(prior) -> DensityDecomposition:
    """``link = Identity, shift = LogProb(prior)`` — emulator emits log_lik."""
    return DensityDecomposition(
        target_single=_quadratic_log_lik,
        output_shape=(),
        link=Identity(),
        shift=LogProb(prior),
    )


def _identity_decomposition(target) -> DensityDecomposition:
    """``link = Identity, shift = Constant(0)`` — emulator emits full log-density."""
    return DensityDecomposition.identity_from_target(target)


# -------------------------------------------------------------------------
# LikelihoodTemperingViaForm
# -------------------------------------------------------------------------


def test_via_form_returns_intermediate_target():
    target = _target_with_density()
    intermediate = LikelihoodTemperingViaForm().intermediate_target(target, state=0.4)
    assert isinstance(intermediate, IntermediateTarget)


def test_via_form_decomposition_target_single_unchanged():
    """``target_single`` and ``output_shape`` pass through unchanged."""
    prior = _flat_prior()
    base = _log_lik_plus_prior_decomposition(prior)
    out = LikelihoodTemperingViaForm().intermediate_decomposition(base, state=0.4)
    assert out.target_single is base.target_single
    assert out.output_shape == base.output_shape


def test_via_form_log_lik_plus_prior_scales_likelihood():
    """``link_β = Affine(slope=β) @ Identity``; ``shift`` unchanged."""
    prior = _flat_prior()
    base = _log_lik_plus_prior_decomposition(prior)
    beta = 0.4
    out = LikelihoodTemperingViaForm().intermediate_decomposition(base, state=beta)
    x = jnp.asarray([0.5, -0.3])
    y = _quadratic_log_lik(x)
    actual = float(out(x, y))
    expected = beta * float(y) + float(jnp.asarray(log_prob(prior, x)))
    assert actual == pytest.approx(expected, abs=1e-6)


def test_via_form_forward_model_scales_link():
    """A ``GaussianLogLik`` forward-model decomposition under tempering:
    the link is wrapped in ``Affine(slope=β) @ link``."""
    prior = _flat_prior()
    obs = jnp.asarray([1.0, 1.0])
    cov = jnp.eye(2)
    base = DensityDecomposition.gaussian_forward_model(
        forward_model=lambda x: x,
        output_shape=(2,),
        obs=obs,
        cov=cov,
        modeling_prior=prior,
    )
    beta = 0.3
    out = LikelihoodTemperingViaForm().intermediate_decomposition(base, state=beta)
    x = jnp.asarray([0.5, -0.3])
    y = jnp.asarray([1.5, 0.5])
    actual = float(out(x, y))
    expected = (
        beta * float(GaussianLogLik(obs=obs, cov=cov)(y))
        + float(jnp.asarray(log_prob(prior, x)))
    )
    assert actual == pytest.approx(expected, abs=1e-6)


def test_via_form_identity_geometric_bridge_requires_initial():
    """``shift = Constant(0)`` is the geometric-bridge case; without
    ``initial`` the scheme raises."""
    target = _target_with_density()
    base = _identity_decomposition(target)
    with pytest.raises(ValueError, match="geometric-bridge case"):
        LikelihoodTemperingViaForm().intermediate_decomposition(base, state=0.4)


def test_via_form_identity_geometric_bridge_with_initial():
    """``(1-β)·log_initial + β·y`` when ``initial`` is supplied."""
    prior = _flat_prior()
    target = _target_with_density()
    base = _identity_decomposition(target)
    beta = 0.4
    out = LikelihoodTemperingViaForm(initial=prior).intermediate_decomposition(
        base, state=beta
    )
    x = jnp.asarray([0.5, -0.3])
    y = jnp.asarray(3.14)
    actual = float(out(x, y))
    expected = (1.0 - beta) * float(jnp.asarray(log_prob(prior, x))) + beta * float(y)
    assert actual == pytest.approx(expected, abs=1e-6)


def test_via_form_invariance_flags():
    scheme = LikelihoodTemperingViaForm()
    assert scheme.is_invariant_target_map(0.1, 0.5)
    assert not scheme.is_invariant_form(0.1, 0.5)
    assert scheme.is_invariant_form(0.5, 0.5)


def test_via_form_output_transform_is_identity():
    target = _target_with_density()
    intermediate = LikelihoodTemperingViaForm().intermediate_target(target, state=0.4)
    X = jnp.asarray([[0.5, -0.3], [1.0, 0.0]])
    Y_raw = jnp.asarray([1.0, -2.0])
    out = intermediate.output_transform(intermediate.state, X, Y_raw)
    assert jnp.allclose(out, Y_raw)


def test_via_form_terminal_state_recovers_base_decomposition():
    """At β=1, ``Affine(1.0) @ Identity == Identity`` (numerically)."""
    prior = _flat_prior()
    base = _log_lik_plus_prior_decomposition(prior)
    out = LikelihoodTemperingViaForm().intermediate_decomposition(base, state=1.0)
    x = jnp.asarray([0.5, -0.3])
    y = _quadratic_log_lik(x)
    assert float(out(x, y)) == pytest.approx(float(base(x, y)), abs=1e-6)


# -------------------------------------------------------------------------
# LikelihoodTemperingViaTarget
# -------------------------------------------------------------------------


def test_via_target_returns_intermediate_target():
    target = _target_with_density()
    intermediate = LikelihoodTemperingViaTarget().intermediate_target(target, state=0.4)
    assert isinstance(intermediate, IntermediateTarget)


def test_via_target_decomposition_unchanged():
    """The decomposition is unchanged across states (β factor enters via
    ``Y_train`` rescaling, not via the decomposition)."""
    prior = _flat_prior()
    base = _log_lik_plus_prior_decomposition(prior)
    out = LikelihoodTemperingViaTarget().intermediate_decomposition(base, state=0.4)
    assert out is base


def test_via_target_output_transform_scales_y_raw():
    target = _target_with_density()
    beta = 0.4
    intermediate = LikelihoodTemperingViaTarget().intermediate_target(target, state=beta)
    X = jnp.asarray([[0.5, -0.3], [1.0, 0.0]])
    Y_raw = jnp.asarray([1.0, -2.0])
    out = intermediate.output_transform(intermediate.state, X, Y_raw)
    expected = beta * Y_raw
    assert jnp.allclose(out, expected, atol=1e-6)


def test_via_target_invariance_flags():
    scheme = LikelihoodTemperingViaTarget()
    assert not scheme.is_invariant_target_map(0.1, 0.5)
    assert scheme.is_invariant_target_map(0.5, 0.5)
    assert scheme.is_invariant_form(0.1, 0.5)


def test_via_target_rejects_non_identity_link():
    """A non-``Identity`` link breaks the ``β·Y_raw == Affine(β) @ link``
    equivalence."""
    prior = _flat_prior()
    base = DensityDecomposition.gaussian_forward_model(
        forward_model=lambda x: x,
        output_shape=(2,),
        obs=jnp.asarray([1.0, 1.0]),
        cov=jnp.eye(2),
        modeling_prior=prior,
    )
    with pytest.raises(ValueError, match="Identity"):
        LikelihoodTemperingViaTarget().intermediate_decomposition(base, state=0.4)


# -------------------------------------------------------------------------
# β=0 endpoint: prior limit. Both schemes should collapse to log_prior.
# -------------------------------------------------------------------------


def test_via_form_at_beta_zero_recovers_prior():
    prior = _flat_prior()
    base = _log_lik_plus_prior_decomposition(prior)
    out = LikelihoodTemperingViaForm().intermediate_decomposition(base, state=0.0)
    x = jnp.asarray([0.5, -0.3])
    y = _quadratic_log_lik(x)
    expected = float(jnp.asarray(log_prob(prior, x)))
    assert float(out(x, y)) == pytest.approx(expected, abs=1e-6)


def test_via_form_geometric_bridge_at_beta_zero_recovers_initial():
    """At β=0: ``(1-0)·log_initial + 0·y == log_initial(x)``."""
    prior = _flat_prior()
    target = _target_with_density()
    base = _identity_decomposition(target)
    out = LikelihoodTemperingViaForm(initial=prior).intermediate_decomposition(
        base, state=0.0
    )
    x = jnp.asarray([0.5, -0.3])
    y = jnp.asarray(3.14)
    expected = float(jnp.asarray(log_prob(prior, x)))
    assert float(out(x, y)) == pytest.approx(expected, abs=1e-6)
