"""Tests for `LikelihoodTemperingViaForm` and `LikelihoodTemperingViaTarget`.

Both schemes encode the same likelihood-tempering family
(:math:`\\ell_t(x) = \\log\\pi_0 + \\lambda_t \\log L`); the tests
verify they produce numerically equivalent intermediate distributions
when applied to the same problem.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from probpipe import log_prob

from sabi._probpipe_compat import independent_uniform
from sabi.problems.forms import (
    ForwardModel,
    Identity,
    LogLikPlusPrior,
)
from sabi.problems.target_distribution import IntermediateTarget, TargetDistribution
from sabi.tempering.likelihood import (
    LikelihoodTemperingViaForm,
    LikelihoodTemperingViaTarget,
)


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


def _flat_prior():
    """Multivariate-event uniform on [-5, 5]^2."""
    return independent_uniform(
        low=jnp.full((2,), -5.0),
        high=jnp.full((2,), 5.0),
        name="flat_prior",
    )


def _quadratic_log_lik(x):
    """Stand-in for log p(data | x): a quadratic in x."""
    return -0.5 * jnp.sum(x * x)


def _log_lik_plus_prior_target() -> TargetDistribution:
    """f = log_lik; phi = LogLikPlusPrior. The standard log-likelihood
    emulation setup."""
    prior = _flat_prior()
    return TargetDistribution.from_target_single(
        target_single=_quadratic_log_lik,
        name="log_lik_target",
        input_shape=(2,),
        output_shape=(),
        log_density_form=LogLikPlusPrior(),
        prior=prior,
    )


def _identity_target() -> TargetDistribution:
    """f = full unnormalized log-posterior; phi = Identity."""
    prior = _flat_prior()

    def full_log_posterior(x):
        return _quadratic_log_lik(x) + jnp.asarray(log_prob(prior, x))

    return TargetDistribution.from_target_single(
        target_single=full_log_posterior,
        name="identity_target",
        input_shape=(2,),
        output_shape=(),
        log_density_form=Identity(),
        prior=prior,
    )


# -------------------------------------------------------------------------
# LikelihoodTemperingViaForm
# -------------------------------------------------------------------------


def test_via_form_returns_intermediate_target():
    target = _log_lik_plus_prior_target()
    intermediate = LikelihoodTemperingViaForm().intermediate_target(target, state=0.4)
    assert isinstance(intermediate, IntermediateTarget)


def test_via_form_target_function_unchanged():
    target = _log_lik_plus_prior_target()
    intermediate = LikelihoodTemperingViaForm().intermediate_target(target, state=0.4)
    # f_state == f for the via-form scheme.
    assert intermediate.target_function is target.target_function
    assert intermediate.target_single is target.target_single


def test_via_form_log_lik_plus_prior_scales_likelihood():
    target = _log_lik_plus_prior_target()
    beta = 0.4
    intermediate = LikelihoodTemperingViaForm().intermediate_target(target, state=beta)
    x = jnp.asarray([0.5, -0.3])
    y = _quadratic_log_lik(x)  # log-likelihood
    # Single-point access goes through the form's per-point hook.
    out = float(intermediate.log_density_form._call_single(x, y, prior=target.prior))
    expected = beta * float(y) + float(jnp.asarray(log_prob(target.prior, x)))
    assert out == pytest.approx(expected, abs=1e-6)


def test_via_form_forward_model_scales_likelihood():
    """ForwardModel: tempered form scales `log_lik_from_outputs` by beta,
    keeps full prior."""
    prior = _flat_prior()

    def log_lik(x, y):
        return -0.5 * jnp.sum((y - 1.0) ** 2)

    target = TargetDistribution.from_target_single(
        target_single=lambda x: x,  # forward model = identity
        name="fm_target",
        input_shape=(2,),
        output_shape=(2,),
        log_density_form=ForwardModel(log_lik_from_outputs=log_lik),
        prior=prior,
    )
    beta = 0.3
    intermediate = LikelihoodTemperingViaForm().intermediate_target(target, state=beta)
    x = jnp.asarray([0.5, -0.3])
    y = jnp.asarray([1.5, 0.5])
    out = float(intermediate.log_density_form._call_single(x, y, prior=prior))
    expected = beta * float(log_lik(x, y)) + float(jnp.asarray(log_prob(prior, x)))
    assert out == pytest.approx(expected, abs=1e-6)


def test_via_form_identity_uses_geometric_bridge():
    """Identity + likelihood tempering = (1-beta)*log_prior + beta*y."""
    target = _identity_target()
    beta = 0.4
    intermediate = LikelihoodTemperingViaForm().intermediate_target(target, state=beta)
    x = jnp.asarray([0.5, -0.3])
    y = jnp.asarray(3.14)  # arbitrary "full log-posterior" value
    out = float(intermediate.log_density_form._call_single(x, y, prior=target.prior))
    expected = (
        (1.0 - beta) * float(jnp.asarray(log_prob(target.prior, x)))
        + beta * float(y)
    )
    assert out == pytest.approx(expected, abs=1e-6)


def test_via_form_invariance_flags():
    scheme = LikelihoodTemperingViaForm()
    # Target is invariant under any state change.
    assert scheme.is_invariant_target_function(0.1, 0.5)
    # Form depends on state.
    assert not scheme.is_invariant_form(0.1, 0.5)
    assert scheme.is_invariant_form(0.5, 0.5)


def test_via_form_output_transform_is_identity():
    target = _log_lik_plus_prior_target()
    intermediate = LikelihoodTemperingViaForm().intermediate_target(target, state=0.4)
    X = jnp.asarray([[0.5, -0.3], [1.0, 0.0]])
    Y_raw = jnp.asarray([1.0, -2.0])
    out = intermediate.output_transform(intermediate.state, X, Y_raw)
    assert jnp.allclose(out, Y_raw)


def test_via_form_terminal_state_recovers_base_distribution():
    """At beta=1, the tempered form should give the same result as the
    base log-density form."""
    target = _log_lik_plus_prior_target()
    intermediate = LikelihoodTemperingViaForm().intermediate_target(target, state=1.0)
    x = jnp.asarray([0.5, -0.3])
    y = _quadratic_log_lik(x)
    out_tempered = float(
        intermediate.log_density_form._call_single(x, y, prior=target.prior)
    )
    out_base = float(target.log_density_form._call_single(x, y, prior=target.prior))
    assert out_tempered == pytest.approx(out_base, abs=1e-6)


# -------------------------------------------------------------------------
# LikelihoodTemperingViaTarget
# -------------------------------------------------------------------------


def test_via_target_returns_intermediate_target():
    target = _log_lik_plus_prior_target()
    intermediate = LikelihoodTemperingViaTarget().intermediate_target(target, state=0.4)
    assert isinstance(intermediate, IntermediateTarget)


def test_via_target_form_unchanged():
    target = _log_lik_plus_prior_target()
    intermediate = LikelihoodTemperingViaTarget().intermediate_target(target, state=0.4)
    assert intermediate.log_density_form is target.log_density_form


def test_via_target_target_function_scaled_by_state():
    target = _log_lik_plus_prior_target()
    beta = 0.4
    intermediate = LikelihoodTemperingViaTarget().intermediate_target(target, state=beta)
    x = jnp.asarray([0.5, -0.3])
    y_state = float(intermediate.target_single(x))
    y_base = float(target.target_single(x))
    assert y_state == pytest.approx(beta * y_base, abs=1e-6)


def test_via_target_output_transform_scales_y_raw():
    target = _log_lik_plus_prior_target()
    beta = 0.4
    intermediate = LikelihoodTemperingViaTarget().intermediate_target(target, state=beta)
    X = jnp.asarray([[0.5, -0.3], [1.0, 0.0]])
    Y_raw = jnp.asarray([1.0, -2.0])
    out = intermediate.output_transform(intermediate.state, X, Y_raw)
    expected = beta * Y_raw
    assert jnp.allclose(out, expected, atol=1e-6)


def test_via_target_invariance_flags():
    scheme = LikelihoodTemperingViaTarget()
    # Target depends on state.
    assert not scheme.is_invariant_target_function(0.1, 0.5)
    assert scheme.is_invariant_target_function(0.5, 0.5)
    # Form is invariant.
    assert scheme.is_invariant_form(0.1, 0.5)


def test_via_target_rejects_non_log_lik_plus_prior_form():
    """Identity / ForwardModel base forms aren't supported — the
    likelihood-tempering interpretation requires `f` to be the
    log-likelihood directly."""
    target = _identity_target()
    with pytest.raises(ValueError, match="LogLikPlusPrior"):
        LikelihoodTemperingViaTarget().intermediate_target(target, state=0.4)


def test_via_target_terminal_state_recovers_base_distribution():
    """At beta=1, target_function and output_transform are identity."""
    target = _log_lik_plus_prior_target()
    intermediate = LikelihoodTemperingViaTarget().intermediate_target(target, state=1.0)
    x = jnp.asarray([0.5, -0.3])
    y_state = float(intermediate.target_single(x))
    y_base = float(target.target_single(x))
    assert y_state == pytest.approx(y_base, abs=1e-6)


# -------------------------------------------------------------------------
# Equivalence: both schemes produce numerically identical
# intermediate log-densities at the same input.
# -------------------------------------------------------------------------


def test_via_form_and_via_target_are_numerically_equivalent():
    """For a `LogLikPlusPrior` base, both schemes encode the same
    intermediate distribution at every state. The intermediate
    log-density at any (x, beta) should match between the two."""
    target = _log_lik_plus_prior_target()
    beta = 0.4
    via_form = LikelihoodTemperingViaForm().intermediate_target(target, state=beta)
    via_target = LikelihoodTemperingViaTarget().intermediate_target(target, state=beta)

    x = jnp.asarray([0.5, -0.3])
    out_form = float(via_form._unnormalized_log_prob(x))
    out_target = float(via_target._unnormalized_log_prob(x))
    assert out_form == pytest.approx(out_target, abs=1e-6)


def test_via_form_and_via_target_match_explicit_likelihood_tempering():
    """The shared mathematical content: log_p_t(x) = beta * log_lik(x) +
    log_prior(x)."""
    target = _log_lik_plus_prior_target()
    beta = 0.7
    x = jnp.asarray([0.5, -0.3])
    explicit = beta * float(_quadratic_log_lik(x)) + float(
        jnp.asarray(log_prob(target.prior, x))
    )

    via_form = LikelihoodTemperingViaForm().intermediate_target(target, state=beta)
    via_target = LikelihoodTemperingViaTarget().intermediate_target(target, state=beta)

    assert float(via_form._unnormalized_log_prob(x)) == pytest.approx(
        explicit, abs=1e-6
    )
    assert float(via_target._unnormalized_log_prob(x)) == pytest.approx(
        explicit, abs=1e-6
    )
