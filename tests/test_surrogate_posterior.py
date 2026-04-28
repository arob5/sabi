"""Tests for the v1.2 SurrogatePosterior hierarchy.

Coverage:
- Inheritance (SurrogatePosterior is a NumericRandomMeasure / RandomMeasure).
- Inner-support / inner-event-shape derived from the constructor args.
- Decoupling from Problem (constructed from math primitives only).
- Protocol opt-in matrix per subclass.
- Dirac equivalence: mean / expected_target on a WES return the inner empirical.
- GP-pushforward random_unnormalized_log_prob marginal shapes (Identity,
  LogLikPlusPrior); ForwardModel raises with a useful message.
- expected_target on the GP path returns a Distribution that satisfies
  SupportsUnnormalizedLogProb and SupportsSampling, and sampling delegates
  to ProbPipe condition_on (NUTS).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from probpipe import log_prob, mean, random_log_prob, random_unnormalized_log_prob, sample
from probpipe.core._distribution_base import Distribution
from probpipe.core._empirical import NumericEmpiricalDistribution
from probpipe.core._random_functions import RandomFunction
from probpipe.core._random_measures import NumericRandomMeasure, RandomMeasure
from probpipe.core.constraints import interval
from probpipe.core.protocols import (
    SupportsLogProb,
    SupportsMean,
    SupportsRandomLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
)
from probpipe.distributions.continuous import Normal, Uniform

from sabi.posterior import (
    GPPushforwardSurrogatePosterior,
    SurrogatePosterior,
    WeightedEmpiricalSurrogatePosterior,
    expected_target,
)
from sabi.problems.forms import ForwardModel, Identity, LogLikPlusPrior
from sabi.problems.gaussian2d import gaussian2d
from sabi.surrogates.gp import GPSurrogate


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


def _box_support(d: int = 2):
    return interval(jnp.full((d,), -5.0), jnp.full((d,), 5.0))


def _wes(n: int = 16, d: int = 2, seed: int = 0):
    """A Weighted-Empirical SP over a 2-D Gaussian-shaped log-density."""
    key = jax.random.key(seed)
    X = jax.random.uniform(key, shape=(n, d), minval=-3.0, maxval=3.0)
    Y = -0.5 * jnp.sum(X ** 2, axis=-1)  # already log-density values
    return WeightedEmpiricalSurrogatePosterior(
        X=X,
        Y=Y,
        support=_box_support(d),
        input_shape=(d,),
        log_density_form=Identity(),
        prior=None,
        name="wes_test",
    )


def _gp_sp(n: int = 20, d: int = 2, seed: int = 1, form=None, prior=None):
    """A GP-pushforward SP fit on a 2-D Gaussian log-density."""
    key = jax.random.key(seed)
    X = jax.random.uniform(key, shape=(n, d), minval=-3.0, maxval=3.0)
    Y = -0.5 * jnp.sum(X ** 2, axis=-1)
    surrogate = GPSurrogate().fit(X, Y)
    return GPPushforwardSurrogatePosterior(
        surrogate=surrogate,
        support=_box_support(d),
        input_shape=(d,),
        log_density_form=form if form is not None else Identity(),
        prior=prior,
        name="gp_sp_test",
    )


# -------------------------------------------------------------------------
# Inheritance and metadata
# -------------------------------------------------------------------------


def test_wes_is_numeric_random_measure():
    sp = _wes()
    assert isinstance(sp, SurrogatePosterior)
    assert isinstance(sp, NumericRandomMeasure)
    assert isinstance(sp, RandomMeasure)


def test_gp_sp_is_numeric_random_measure():
    sp = _gp_sp()
    assert isinstance(sp, SurrogatePosterior)
    assert isinstance(sp, NumericRandomMeasure)
    assert isinstance(sp, RandomMeasure)


def test_inner_support_and_event_shape_match_constructor_args():
    sp = _wes()
    assert sp.inner_event_shape == (2,)
    # inner_support is the same Constraint we passed in
    assert sp.inner_support is sp._support  # type: ignore[attr-defined]


def test_surrogate_posterior_requires_support():
    """Decoupled from Problem; takes math primitives directly. Missing support raises."""
    with pytest.raises(ValueError, match="support"):
        WeightedEmpiricalSurrogatePosterior(
            X=jnp.zeros((4, 2)),
            Y=jnp.zeros(4),
            support=None,  # type: ignore[arg-type]
            input_shape=(2,),
            log_density_form=Identity(),
        )


# -------------------------------------------------------------------------
# WeightedEmpirical: Dirac protocol surface
# -------------------------------------------------------------------------


def test_wes_satisfies_protocols():
    sp = _wes()
    assert isinstance(sp, SupportsSampling)
    assert isinstance(sp, SupportsMean)
    assert isinstance(sp, SupportsRandomLogProb)
    assert isinstance(sp, SupportsRandomUnnormalizedLogProb)


def test_wes_mean_returns_inner_empirical():
    sp = _wes()
    inner = mean(sp)
    assert isinstance(inner, NumericEmpiricalDistribution)
    assert inner is sp.inner_distribution


def test_wes_sample_returns_inner_distribution():
    sp = _wes()
    s = sample(sp, key=jax.random.key(0))
    # For sample_shape == (), a draw IS the (single) inner Distribution.
    assert isinstance(s, NumericEmpiricalDistribution)


def test_wes_random_log_prob_returns_random_function():
    sp = _wes()
    rf = random_log_prob(sp)
    assert isinstance(rf, RandomFunction)


def test_wes_random_log_prob_marginal_is_dirac_at_value():
    """The marginal at x is a Dirac at log_prob(empirical, x). Mean of a
    Dirac is its value, and SupportsMean lets us assert that."""
    sp = _wes()
    x = jnp.asarray([0.5, -0.3])
    marginal = random_log_prob(sp, x)
    expected = jnp.asarray(log_prob(sp.inner_distribution, x))
    assert isinstance(marginal, SupportsMean)
    assert float(mean(marginal)) == pytest.approx(float(expected), abs=1e-5)


def test_wes_random_unnormalized_log_prob_marginal_is_dirac():
    sp = _wes()
    x = jnp.asarray([0.5, -0.3])
    marginal = random_unnormalized_log_prob(sp, x)
    assert isinstance(marginal, SupportsMean)
    # For a normalized empirical, log_prob == unnormalized_log_prob; both work.
    expected = jnp.asarray(log_prob(sp.inner_distribution, x))
    assert float(mean(marginal)) == pytest.approx(float(expected), abs=1e-5)


# -------------------------------------------------------------------------
# GP-pushforward: random_unnormalized_log_prob shape and form-dispatch
# -------------------------------------------------------------------------


def test_gp_sp_random_unnormalized_log_prob_identity_returns_normal():
    sp = _gp_sp(form=Identity())
    x = jnp.asarray([0.4, -0.1])
    marginal = random_unnormalized_log_prob(sp, x)
    assert isinstance(marginal, Normal)
    # Mean equals surrogate's predictive mean at x.
    pred = sp.surrogate.predict(x[None])
    assert float(mean(marginal)) == pytest.approx(float(pred.mean[0]), abs=1e-5)


def test_gp_sp_random_unnormalized_log_prob_log_lik_plus_prior_shifts_mean():
    """LogLikPlusPrior shifts the marginal mean by joint log_prior(x);
    variance unchanged."""
    from sabi.problems.forms import _joint_log_prior

    prior = Uniform(
        low=jnp.full((2,), -5.0),
        high=jnp.full((2,), 5.0),
        name="prior_test",
    )
    sp = _gp_sp(form=LogLikPlusPrior(), prior=prior)
    x = jnp.asarray([0.4, -0.1])
    pred = sp.surrogate.predict(x[None])
    expected_mean = float(pred.mean[0]) + float(_joint_log_prior(prior, x))
    marginal = random_unnormalized_log_prob(sp, x)
    assert isinstance(marginal, Normal)
    assert float(mean(marginal)) == pytest.approx(expected_mean, abs=1e-5)


def test_gp_sp_random_unnormalized_log_prob_forward_model_raises():
    sp = _gp_sp(form=ForwardModel(log_lik_from_outputs=lambda x, y: -0.5 * jnp.sum(y ** 2)))
    x = jnp.asarray([0.0, 0.0])
    with pytest.raises(NotImplementedError, match="ForwardModel"):
        random_unnormalized_log_prob(sp, x)


def test_gp_sp_does_not_satisfy_supports_mean():
    """No closed-form expected posterior in v1.2 → SupportsMean is NOT satisfied."""
    sp = _gp_sp()
    assert not isinstance(sp, SupportsMean)
    with pytest.raises(TypeError, match="support"):
        mean(sp)


def test_gp_sp_does_not_satisfy_supports_random_log_prob():
    """Normalization intractable; only the unnormalized variant is exposed."""
    sp = _gp_sp()
    assert not isinstance(sp, SupportsRandomLogProb)
    assert isinstance(sp, SupportsRandomUnnormalizedLogProb)


# -------------------------------------------------------------------------
# expected_target estimator
# -------------------------------------------------------------------------


def test_expected_target_for_wes_is_inner_empirical():
    sp = _wes()
    et = expected_target(sp)
    # For Dirac SPs, expected_target collapses to mean(sp).
    assert isinstance(et, NumericEmpiricalDistribution)
    assert et is sp.inner_distribution
    assert et is mean(sp)


def test_expected_target_for_gp_sp_satisfies_unnormalized_log_prob_and_sampling():
    sp = _gp_sp()
    et = expected_target(sp)
    assert isinstance(et, Distribution)
    assert isinstance(et, SupportsUnnormalizedLogProb)
    assert isinstance(et, SupportsSampling)
    # Important: it's NOT a SupportsLogProb (the density is unnormalized).
    assert not isinstance(et, SupportsLogProb)


def test_expected_target_for_gp_sp_unnormalized_log_prob_matches_form():
    """`et._unnormalized_log_prob(x) == log_density_form(x, surrogate.predict(x).mean, prior)`."""
    sp = _gp_sp()
    et = expected_target(sp)
    x = jnp.asarray([0.3, -0.2])
    pred = sp.surrogate.predict(x[None])
    expected = float(sp.log_density_form(x, pred.mean[0], prior=sp.prior))
    assert float(et._unnormalized_log_prob(x)) == pytest.approx(expected, abs=1e-5)


def test_expected_target_for_gp_sp_sampling_runs_mcmc():
    """`sample(et)` should dispatch through condition_on → NUTS and return a
    chain of samples with the right shape. Slow-ish (full MCMC); kept tiny."""
    sp = _gp_sp(n=20)
    et = expected_target(
        sp,
        sampler_kwargs={"num_results": 100, "num_warmup": 50},
    )
    samples = sample(et, key=jax.random.key(0), sample_shape=(40,))
    arr = jnp.asarray(samples)
    assert arr.shape == (40, 2)


def test_expected_target_unsupported_sp_raises():
    """Unknown SurrogatePosterior subtype should raise TypeError."""

    class _DummySP(SurrogatePosterior):
        pass

    sp = _DummySP(
        support=_box_support(),
        input_shape=(2,),
        log_density_form=Identity(),
    )
    with pytest.raises(TypeError, match="expected_target"):
        expected_target(sp)
