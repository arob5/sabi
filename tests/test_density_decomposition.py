"""Tests for ``DensityDecomposition`` and its concrete subclasses.

Coverage:
- ``DensityDecomposition`` is a ProbPipe ``Distribution`` and its
  ``_unnormalized_log_prob`` auto-derives from
  ``link(target_map(x)) + shift(x)``.
- Subclass shapes:
    - ``LogProbTermTarget`` with ``prior=None`` → full log-density.
    - ``LogProbTermTarget`` with ``prior=π`` → log-likelihood + prior.
    - ``LogProbTarget`` wrapping a ``TargetDistribution``.
    - ``GaussianForwardModelTarget``.
- ``pushforward`` closed-form (Gaussian-affine) and MC fallback paths.
- ``is_consistent_with`` strict / loose modes; propagates a TypeError
  when the target has no analytical density.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from probpipe import unnormalized_log_prob as pp_unnormalized_log_prob
from probpipe import log_prob as pp_log_prob
from probpipe.core._distribution_base import Distribution
from probpipe.distributions.continuous import Normal
from probpipe.distributions.multivariate import MultivariateNormal

from sabi.density_decomposition import (
    DensityDecomposition,
    GaussianForwardModelTarget,
    LogProbTarget,
    LogProbTermTarget,
    is_consistent_with,
)
from sabi.maps import Identity, LogProb, LogSquare

from tests._targets import (
    ConstantShiftedDecomposition,
    OpaqueTarget,
    QuadraticLogProbDecomposition,
    QuadraticTarget,
    box_support,
    box_uniform,
    quadratic_log_density,
)


# ---------------------------------------------------------------------------
# DensityDecomposition / LogProbTermTarget basics
# ---------------------------------------------------------------------------


def test_log_prob_term_target_no_shift_is_full_log_density():
    """``prior=None`` ⇒ ``shift = None`` ⇒ ``_unnormalized_log_prob == target_map``."""
    dd = QuadraticLogProbDecomposition()
    assert dd.shift is None
    assert isinstance(dd.link, Identity)

    x = jnp.asarray([0.5, -0.3])
    expected = float(quadratic_log_density(x))
    out = float(jnp.asarray(pp_unnormalized_log_prob(dd, x)))
    assert out == pytest.approx(expected, abs=1e-6)


def test_log_prob_term_target_with_prior_adds_log_prior():
    """``prior=π`` ⇒ ``shift = LogProb(π)`` ⇒ density is residual + log_prior."""
    prior = box_uniform()
    dd = QuadraticLogProbDecomposition(prior=prior)
    assert isinstance(dd.shift, LogProb)
    assert dd.shift.dist is prior

    x = jnp.asarray([0.5, -0.3])
    expected = float(quadratic_log_density(x)) + float(pp_log_prob(prior, x))
    out = float(jnp.asarray(pp_unnormalized_log_prob(dd, x)))
    assert out == pytest.approx(expected, abs=1e-6)


def test_density_decomposition_is_a_distribution():
    """``DensityDecomposition`` satisfies ProbPipe's Distribution interface."""
    from probpipe.core.protocols import SupportsUnnormalizedLogProb

    dd = QuadraticLogProbDecomposition()
    assert isinstance(dd, Distribution)
    assert isinstance(dd, SupportsUnnormalizedLogProb)
    assert dd.event_shape == (2,)
    assert dd.output_shape == ()


def test_decomposition_log_prob_vectorizes_via_probpipe_op():
    """Batched x via the ProbPipe op returns batched log-prob."""
    dd = QuadraticLogProbDecomposition()
    X = jnp.asarray([[0.0, 0.0], [0.5, -0.3], [1.0, 1.0]])
    out = jnp.asarray(pp_unnormalized_log_prob(dd, X))
    expected = jnp.asarray([float(quadratic_log_density(x)) for x in X])
    assert out.shape == (3,)
    assert jnp.allclose(out, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# LogProbTarget — wrap a TargetDistribution's analytical density
# ---------------------------------------------------------------------------


def test_log_prob_target_delegates_to_wrapped_target():
    target = QuadraticTarget()
    dd = LogProbTarget(target)

    assert dd.target is target
    assert isinstance(dd.link, Identity)
    assert dd.shift is None
    assert dd.event_shape == target.event_shape
    assert dd.support is target.support

    x = jnp.asarray([0.5, -0.3])
    expected = float(quadratic_log_density(x))
    out = float(jnp.asarray(pp_unnormalized_log_prob(dd, x)))
    assert out == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# GaussianForwardModelTarget
# ---------------------------------------------------------------------------


class _IdentityForwardDecomposition(GaussianForwardModelTarget):
    """Forward model is the identity ``f(x) = x``; obs and cov user-supplied."""

    @property
    def event_shape(self):
        return (2,)

    def target_map(self, x):
        return x


def test_gaussian_forward_model_density():
    """Density is `log N(obs | x, C) + log_prior(x)`."""
    prior = box_uniform()
    obs = jnp.asarray([0.0, 0.0])
    cov = jnp.eye(2)
    dd = _IdentityForwardDecomposition(
        name="gfm",
        support=box_support(),
        obs=obs,
        cov=cov,
        prior=prior,
    )
    assert dd.output_shape == (2,)

    x = jnp.asarray([0.5, -0.3])
    out = float(jnp.asarray(pp_unnormalized_log_prob(dd, x)))
    # Compare against an analytic reference: log N(obs | x, I) + log_prior(x)
    mvn = MultivariateNormal(loc=x, cov=cov, name="ref")
    expected = float(pp_log_prob(mvn, obs)) + float(pp_log_prob(prior, x))
    assert out == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# pushforward
# ---------------------------------------------------------------------------


def test_pushforward_closed_form_normal_through_identity_no_shift():
    """``link=Identity, shift=None`` × ``Normal`` → identity (input unchanged)."""
    dd = QuadraticLogProbDecomposition()
    x = jnp.asarray([1.0, 0.5])
    y_dist = Normal(loc=jnp.asarray(2.0), scale=jnp.asarray(0.5), name="y_dist")
    pushed = dd.pushforward(x, y_dist)
    # Identity dispatch returns the input distribution unchanged.
    assert pushed is y_dist


def test_pushforward_closed_form_with_logprob_shift():
    """``link=Identity, shift=LogProb(prior)`` × ``Normal`` → ``Normal``
    with ``loc`` shifted by ``log_prior(x)``."""
    prior = box_uniform()
    dd = QuadraticLogProbDecomposition(prior=prior)
    x = jnp.asarray([1.0, 0.5])
    y_dist = Normal(loc=jnp.asarray(2.0), scale=jnp.asarray(0.5), name="y_dist")
    pushed = dd.pushforward(x, y_dist)
    assert isinstance(pushed, Normal)
    expected_loc = 2.0 + float(pp_log_prob(prior, x))
    assert float(pushed.loc) == pytest.approx(expected_loc, abs=1e-6)
    assert float(pushed.scale) == pytest.approx(0.5, abs=1e-6)


def test_pushforward_mc_fallback_for_non_affine_link():
    """A non-affine link forces MC fallback (returns a non-Normal Distribution)."""

    class _LogSquareDecomp(LogProbTermTarget):
        @property
        def event_shape(self):
            return (2,)

        def target_map(self, x):
            return quadratic_log_density(x)

        @property
        def link(self):
            return LogSquare()

    dd = _LogSquareDecomp(name="logsq", support=box_support())
    x = jnp.asarray([0.0, 0.0])
    y_dist = Normal(loc=jnp.asarray(1.0), scale=jnp.asarray(0.5), name="y_dist")
    pushed = dd.pushforward(x, y_dist)
    assert isinstance(pushed, Distribution)
    assert not isinstance(pushed, Normal)


# ---------------------------------------------------------------------------
# is_consistent_with
# ---------------------------------------------------------------------------


def test_is_consistent_with_strict_passes_for_log_prob_target():
    target = QuadraticTarget()
    dd = LogProbTarget(target)
    x_test = jnp.asarray([[0.0, 0.0], [0.5, -0.3], [1.0, 1.0]])
    assert is_consistent_with(dd, target, x_test=x_test, strict=True)


def test_is_consistent_with_strict_rejects_constant_offset():
    target = QuadraticTarget()
    dd = ConstantShiftedDecomposition(offset=1.0)
    x_test = jnp.asarray([[0.0, 0.0], [0.5, -0.3]])
    assert not is_consistent_with(dd, target, x_test=x_test, strict=True)


def test_is_consistent_with_loose_accepts_constant_offset():
    target = QuadraticTarget()
    dd = ConstantShiftedDecomposition(offset=1.0)
    x_test = jnp.asarray([[0.0, 0.0], [0.5, -0.3], [1.0, 1.0]])
    assert is_consistent_with(dd, target, x_test=x_test, strict=False)


def test_is_consistent_with_loose_rejects_x_dependent_offset():
    """An x-dependent shift fails non-strict mode."""
    target = QuadraticTarget()
    normal_prior = MultivariateNormal(
        loc=jnp.zeros(2), cov=jnp.eye(2), name="normal_prior"
    )
    dd = QuadraticLogProbDecomposition(prior=normal_prior)
    x_test = jnp.asarray([[0.0, 0.0], [0.5, -0.3], [1.0, 1.0]])
    assert not is_consistent_with(dd, target, x_test=x_test, strict=False)


def test_is_consistent_with_propagates_error_for_opaque_target():
    """User inverse problems without analytical density: the ProbPipe op
    raises (since the target doesn't satisfy SupportsUnnormalizedLogProb).
    We let it propagate."""
    target = OpaqueTarget()
    dd = QuadraticLogProbDecomposition()
    x_test = jnp.asarray([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(TypeError, match="unnormalized_log_prob"):
        is_consistent_with(dd, target, x_test=x_test, strict=True)


def test_is_consistent_with_loose_requires_two_rows():
    target = QuadraticTarget()
    dd = LogProbTarget(target)
    with pytest.raises(ValueError, match="n >= 2"):
        is_consistent_with(
            dd, target, x_test=jnp.asarray([[0.0, 0.0]]), strict=False
        )


def test_is_consistent_with_rejects_rank_one_x_test():
    target = QuadraticTarget()
    dd = LogProbTarget(target)
    with pytest.raises(ValueError, match="at least rank-2"):
        is_consistent_with(dd, target, x_test=jnp.asarray([0.0, 0.0]))
