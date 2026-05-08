r"""Tests for ``LikelihoodTemperingViaForm`` and
``LikelihoodTemperingViaTarget``.

After the ``DensityDecomposition`` rework, both schemes operate on
:class:`DensityDecomposition` instances via
``intermediate_decomposition(base, state)``. The math is unchanged.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from probpipe import log_prob, unnormalized_log_prob

from sabi._probpipe_compat import independent_uniform
from sabi.density_decomposition import (
    DensityDecomposition,
    GaussianForwardModelTarget,
    LogProbTermTarget,
)
from sabi.maps import GaussianLogLik
from sabi.target_distribution import IntermediateTarget
from sabi.tempering.likelihood import (
    LikelihoodTemperingViaForm,
    LikelihoodTemperingViaTarget,
)

from tests._targets import (
    QuadraticLogProbDecomposition,
    QuadraticTarget,
    box_support,
    box_uniform,
    quadratic_log_density,
)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _log_lik_plus_prior_decomposition(prior) -> DensityDecomposition:
    """``link = Identity, shift = LogProb(prior)`` — emulator emits log_lik."""
    return QuadraticLogProbDecomposition(prior=prior)


def _identity_decomposition() -> DensityDecomposition:
    """``link = Identity, shift = None`` — emulator emits full log-density."""
    return QuadraticLogProbDecomposition(prior=None)


class _IdentityForwardModelGaussian(GaussianForwardModelTarget):
    """Gaussian forward model with f(x) = x — for tempering tests."""

    def target_map(self, x):
        return x


# -------------------------------------------------------------------------
# LikelihoodTemperingViaForm
# -------------------------------------------------------------------------


def test_via_form_returns_intermediate_target():
    target = QuadraticTarget()
    intermediate = LikelihoodTemperingViaForm().intermediate_target(target, state=0.4)
    assert isinstance(intermediate, IntermediateTarget)


def test_via_form_decomposition_target_map_unchanged():
    """``target_map`` and ``output_shape`` pass through unchanged."""
    prior = box_uniform()
    base = _log_lik_plus_prior_decomposition(prior)
    out = LikelihoodTemperingViaForm().intermediate_decomposition(base, state=0.4)
    x = jnp.asarray([0.5, -0.3])
    # target_map outputs match
    assert jnp.allclose(out.target_map(x), base.target_map(x))
    assert out.output_shape == base.output_shape


def test_via_form_log_lik_plus_prior_scales_likelihood():
    """``link_β = Affine(slope=β) @ Identity``; ``shift`` unchanged."""
    prior = box_uniform()
    base = _log_lik_plus_prior_decomposition(prior)
    beta = 0.4
    out = LikelihoodTemperingViaForm().intermediate_decomposition(base, state=beta)
    x = jnp.asarray([0.5, -0.3])
    actual = float(jnp.asarray(unnormalized_log_prob(out, x)))
    expected = beta * float(quadratic_log_density(x)) + float(log_prob(prior, x))
    assert actual == pytest.approx(expected, abs=1e-6)


def test_via_form_forward_model_scales_link():
    """A ``GaussianLogLik`` forward-model decomposition under tempering:
    the link is wrapped in ``Affine(slope=β) @ link``."""
    prior = box_uniform()
    obs = jnp.asarray([1.0, 1.0])
    cov = jnp.eye(2)
    base = _IdentityForwardModelGaussian(
        name="fmtest",
        input_shape=(2,),
        support=box_support(),
        obs=obs,
        cov=cov,
        prior=prior,
    )
    beta = 0.3
    out = LikelihoodTemperingViaForm().intermediate_decomposition(base, state=beta)
    x = jnp.asarray([0.5, -0.3])
    actual = float(jnp.asarray(unnormalized_log_prob(out, x)))
    expected = (
        beta * float(GaussianLogLik(obs=obs, cov=cov)(base.target_map(x)))
        + float(log_prob(prior, x))
    )
    assert actual == pytest.approx(expected, abs=1e-6)


def test_via_form_identity_geometric_bridge_requires_initial():
    """``shift = None`` is the geometric-bridge case; without
    ``initial`` the scheme raises when the shift is materialized."""
    base = _identity_decomposition()
    out = LikelihoodTemperingViaForm().intermediate_decomposition(base, state=0.4)
    x = jnp.asarray([0.5, -0.3])
    with pytest.raises(ValueError, match="geometric-bridge case"):
        unnormalized_log_prob(out, x)


def test_via_form_identity_geometric_bridge_with_initial():
    """``(1-β)·log_initial + β·target_map(x)`` when ``initial`` is supplied."""
    prior = box_uniform()
    base = _identity_decomposition()
    beta = 0.4
    out = LikelihoodTemperingViaForm(initial=prior).intermediate_decomposition(
        base, state=beta
    )
    x = jnp.asarray([0.5, -0.3])
    actual = float(jnp.asarray(unnormalized_log_prob(out, x)))
    expected = (
        (1.0 - beta) * float(log_prob(prior, x))
        + beta * float(quadratic_log_density(x))
    )
    assert actual == pytest.approx(expected, abs=1e-6)


def test_via_form_invariance_flags():
    scheme = LikelihoodTemperingViaForm()
    assert scheme.is_invariant_target_map(0.1, 0.5)
    assert not scheme.is_invariant_form(0.1, 0.5)
    assert scheme.is_invariant_form(0.5, 0.5)


def test_via_form_output_transform_is_identity():
    target = QuadraticTarget()
    intermediate = LikelihoodTemperingViaForm().intermediate_target(target, state=0.4)
    X = jnp.asarray([[0.5, -0.3], [1.0, 0.0]])
    Y_raw = jnp.asarray([1.0, -2.0])
    out = intermediate.output_transform(intermediate.state, X, Y_raw)
    assert jnp.allclose(out, Y_raw)


def test_via_form_terminal_state_recovers_base_decomposition():
    """At β=1, scaled-by-1 link agrees with base."""
    prior = box_uniform()
    base = _log_lik_plus_prior_decomposition(prior)
    out = LikelihoodTemperingViaForm().intermediate_decomposition(base, state=1.0)
    x = jnp.asarray([0.5, -0.3])
    actual = float(jnp.asarray(unnormalized_log_prob(out, x)))
    expected = float(jnp.asarray(unnormalized_log_prob(base, x)))
    assert actual == pytest.approx(expected, abs=1e-6)


# -------------------------------------------------------------------------
# LikelihoodTemperingViaTarget
# -------------------------------------------------------------------------


def test_via_target_returns_intermediate_target():
    target = QuadraticTarget()
    intermediate = LikelihoodTemperingViaTarget().intermediate_target(target, state=0.4)
    assert isinstance(intermediate, IntermediateTarget)


def test_via_target_decomposition_unchanged():
    """The decomposition is unchanged across states (β factor enters via
    ``Y_train`` rescaling, not via the decomposition)."""
    prior = box_uniform()
    base = _log_lik_plus_prior_decomposition(prior)
    out = LikelihoodTemperingViaTarget().intermediate_decomposition(base, state=0.4)
    assert out is base


def test_via_target_output_transform_scales_y_raw():
    target = QuadraticTarget()
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
    """A non-Identity link breaks the ``β·Y_raw == Affine(β) @ link``
    equivalence."""
    prior = box_uniform()
    base = _IdentityForwardModelGaussian(
        name="fmtest",
        input_shape=(2,),
        support=box_support(),
        obs=jnp.asarray([1.0, 1.0]),
        cov=jnp.eye(2),
        prior=prior,
    )
    with pytest.raises(ValueError, match="Identity"):
        LikelihoodTemperingViaTarget().intermediate_decomposition(base, state=0.4)


# -------------------------------------------------------------------------
# β=0 endpoint
# -------------------------------------------------------------------------


def test_via_form_at_beta_zero_recovers_prior():
    prior = box_uniform()
    base = _log_lik_plus_prior_decomposition(prior)
    out = LikelihoodTemperingViaForm().intermediate_decomposition(base, state=0.0)
    x = jnp.asarray([0.5, -0.3])
    actual = float(jnp.asarray(unnormalized_log_prob(out, x)))
    expected = float(log_prob(prior, x))
    assert actual == pytest.approx(expected, abs=1e-6)


def test_via_form_geometric_bridge_at_beta_zero_recovers_initial():
    """At β=0: ``(1-0)·log_initial + 0·y == log_initial(x)``."""
    prior = box_uniform()
    base = _identity_decomposition()
    out = LikelihoodTemperingViaForm(initial=prior).intermediate_decomposition(
        base, state=0.0
    )
    x = jnp.asarray([0.5, -0.3])
    actual = float(jnp.asarray(unnormalized_log_prob(out, x)))
    expected = float(log_prob(prior, x))
    assert actual == pytest.approx(expected, abs=1e-6)
