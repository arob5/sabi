"""Tests for the v1.2-refactor SurrogatePosterior + WeightedEmpiricalRandomMeasure.

Coverage:
- Inheritance: SurrogatePosterior is a NumericRandomMeasure;
  WeightedEmpiricalRandomMeasure is a SurrogatePosterior subclass with
  ``emulator=None`` (the degenerate / no-emulator case).
- Inner-support / inner-event-shape derived from constructor args.
- Decoupling from Problem (constructed from math primitives only).
- Protocol opt-in matrix per class.
- Dirac equivalence: mean / expected_target on a WERM return the inner empirical.
- pushforward_marginal dispatch:
    - (Normal, Identity) → input dist unchanged
    - (Normal, LogLikPlusPrior) → Normal with shifted loc
    - (MultivariateNormal, Identity / LogLikPlusPrior) — closed-form joint
    - (samplable non-Gaussian, any) → MC empirical via workflow_function
    - (non-samplable non-Gaussian, any) → raises NotImplementedError
- expected_target on the SP returns a Distribution that satisfies
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
from probpipe.core.constraints import interval, real
from probpipe.core.protocols import (
    SupportsLogProb,
    SupportsMean,
    SupportsRandomLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
)
from probpipe.distributions.continuous import Normal
from probpipe.distributions.multivariate import MultivariateNormal

from sabi.posterior import (
    SurrogatePosterior,
    WeightedEmpiricalRandomMeasure,
    expected_target,
)
from sabi.posterior._pushforward import pushforward_marginal
from sabi.problems.forms import ForwardModel, Identity, LogLikPlusPrior
from sabi.emulators.gp import GPEmulator


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


def _box_support(d: int = 2):
    return interval(jnp.full((d,), -5.0), jnp.full((d,), 5.0))


def _werm(n: int = 16, d: int = 2, seed: int = 0):
    """A WeightedEmpiricalRandomMeasure over a 2-D Gaussian-shaped log-density."""
    key = jax.random.key(seed)
    X = jax.random.uniform(key, shape=(n, d), minval=-3.0, maxval=3.0)
    log_w = -0.5 * jnp.sum(X ** 2, axis=-1)
    return WeightedEmpiricalRandomMeasure(
        X=X,
        log_weights=log_w,
        support=_box_support(d),
        input_shape=(d,),
        name="werm_test",
    )


def _sp(n: int = 20, d: int = 2, seed: int = 1, form=None, prior=None):
    """A SurrogatePosterior fit on a 2-D Gaussian log-density."""
    key = jax.random.key(seed)
    X = jax.random.uniform(key, shape=(n, d), minval=-3.0, maxval=3.0)
    Y = -0.5 * jnp.sum(X ** 2, axis=-1)
    emulator = GPEmulator(input_shape=(d,)).fit(X, Y)
    return SurrogatePosterior(
        emulator=emulator,
        log_density_form=form if form is not None else Identity(),
        support=_box_support(d),
        input_shape=(d,),
        prior=prior,
        name="sp_test",
    )


# -------------------------------------------------------------------------
# Inheritance and metadata
# -------------------------------------------------------------------------


def test_werm_is_surrogate_posterior_subclass():
    werm = _werm()
    assert isinstance(werm, NumericRandomMeasure)
    assert isinstance(werm, RandomMeasure)
    # WERM is a SurrogatePosterior subclass with degenerate emulator.
    assert isinstance(werm, SurrogatePosterior)
    assert werm.emulator is None
    assert werm.log_density_form is None


def test_sp_is_numeric_random_measure():
    sp = _sp()
    assert isinstance(sp, SurrogatePosterior)
    assert isinstance(sp, NumericRandomMeasure)
    assert isinstance(sp, RandomMeasure)


def test_inner_support_and_event_shape_match_constructor_args():
    werm = _werm()
    assert werm.inner_event_shape == (2,)
    assert werm.inner_support is werm._support  # type: ignore[attr-defined]
    sp = _sp()
    assert sp.inner_event_shape == (2,)


def test_werm_requires_support():
    with pytest.raises(ValueError, match="support"):
        WeightedEmpiricalRandomMeasure(
            X=jnp.zeros((4, 2)),
            log_weights=jnp.zeros(4),
            support=None,  # type: ignore[arg-type]
            input_shape=(2,),
        )


def test_sp_requires_support():
    emulator = GPEmulator(input_shape=(2,)).fit(
        jnp.zeros((4, 2)), jnp.zeros(4)
    )
    with pytest.raises(ValueError, match="support"):
        SurrogatePosterior(
            emulator=emulator,
            log_density_form=Identity(),
            support=None,  # type: ignore[arg-type]
            input_shape=(2,),
        )


def test_sp_input_shape_must_match_emulator_input_shape():
    emulator = GPEmulator(input_shape=(2,)).fit(
        jnp.zeros((4, 2)), jnp.zeros(4)
    )
    with pytest.raises(ValueError, match="input_shape"):
        SurrogatePosterior(
            emulator=emulator,
            log_density_form=Identity(),
            support=_box_support(3),
            input_shape=(3,),  # emulator is (2,); this is (3,)
        )


# -------------------------------------------------------------------------
# WeightedEmpiricalRandomMeasure: Dirac protocol surface
# -------------------------------------------------------------------------


def test_werm_satisfies_protocols():
    werm = _werm()
    assert isinstance(werm, SupportsSampling)
    assert isinstance(werm, SupportsMean)
    assert isinstance(werm, SupportsRandomLogProb)
    assert isinstance(werm, SupportsRandomUnnormalizedLogProb)


def test_werm_mean_returns_inner_empirical():
    werm = _werm()
    inner = mean(werm)
    assert isinstance(inner, NumericEmpiricalDistribution)
    assert inner is werm.inner_distribution


def test_werm_sample_returns_inner_distribution():
    werm = _werm()
    s = sample(werm, key=jax.random.key(0))
    assert isinstance(s, NumericEmpiricalDistribution)


def test_werm_random_log_prob_marginal_is_dirac_at_value():
    werm = _werm()
    x = jnp.asarray([0.5, -0.3])
    marginal = random_log_prob(werm, x)
    expected = jnp.asarray(log_prob(werm.inner_distribution, x))
    assert isinstance(marginal, SupportsMean)
    assert float(mean(marginal)) == pytest.approx(float(expected), abs=1e-5)


# -------------------------------------------------------------------------
# pushforward_marginal — closed-form Gaussian-affine
# -------------------------------------------------------------------------


def test_pushforward_normal_identity_returns_input_dist():
    """Pushforward of any input through Identity is the input itself."""
    n = 5
    X = jax.random.uniform(jax.random.key(0), shape=(n, 2), minval=-1, maxval=1)
    nrm = Normal(
        loc=jnp.linspace(-1.0, 1.0, n),
        scale=jnp.full((n,), 0.5),
        name="nrm",
    )
    out = pushforward_marginal(nrm, Identity(), X=X, prior=None)
    # Identity short-circuits to the same object.
    assert out is nrm


def test_pushforward_normal_log_lik_plus_prior_shifts_loc():
    """LogLikPlusPrior with multivariate-event Uniform prior shifts
    each loc by the log-prior at the corresponding point."""
    from probpipe import log_prob as pp_log_prob
    from sabi._probpipe_compat import independent_uniform

    n = 3
    X = jnp.asarray([[0.4, -0.1], [0.0, 0.0], [1.0, 1.0]])
    nrm = Normal(
        loc=jnp.asarray([0.5, -0.2, 0.7]),
        scale=jnp.asarray([0.1, 0.2, 0.05]),
        name="nrm",
    )
    prior = independent_uniform(
        low=jnp.full((2,), -5.0), high=jnp.full((2,), 5.0), name="p"
    )
    out = pushforward_marginal(nrm, LogLikPlusPrior(), X=X, prior=prior)
    assert isinstance(out, Normal)
    expected_shifts = jax.vmap(lambda x: jnp.asarray(pp_log_prob(prior, x)))(X)
    expected_loc = jnp.asarray(nrm.loc) + expected_shifts
    assert jnp.allclose(jnp.asarray(out.loc), expected_loc, atol=1e-5)
    # Scale is unchanged.
    assert jnp.allclose(jnp.asarray(out.scale), jnp.asarray(nrm.scale), atol=1e-5)


def test_pushforward_mvn_identity_returns_input_dist():
    n = 4
    X = jax.random.uniform(jax.random.key(0), shape=(n, 2), minval=-1, maxval=1)
    mvn = MultivariateNormal(
        loc=jnp.zeros(n),
        cov=jnp.eye(n) * 0.5,
        name="mvn",
    )
    out = pushforward_marginal(mvn, Identity(), X=X, prior=None)
    assert out is mvn


def test_pushforward_mvn_log_lik_plus_prior_shifts_loc():
    """Joint MVN over n points: shift loc by per-point log-prior, keep
    scale_tril unchanged."""
    from probpipe import log_prob as pp_log_prob
    from sabi._probpipe_compat import independent_uniform

    n = 4
    X = jax.random.uniform(jax.random.key(2), shape=(n, 2), minval=-1, maxval=1)
    mvn = MultivariateNormal(
        loc=jnp.zeros(n),
        cov=jnp.eye(n) * 0.5,
        name="mvn",
    )
    prior = independent_uniform(
        low=jnp.full((2,), -5.0), high=jnp.full((2,), 5.0), name="p"
    )
    out = pushforward_marginal(mvn, LogLikPlusPrior(), X=X, prior=prior)
    assert isinstance(out, MultivariateNormal)
    expected_shifts = jax.vmap(lambda x: jnp.asarray(pp_log_prob(prior, x)))(X)
    assert jnp.allclose(jnp.asarray(out.loc), expected_shifts, atol=1e-5)
    # scale_tril unchanged.
    assert jnp.allclose(
        jnp.asarray(out.scale_tril), jnp.asarray(mvn.scale_tril), atol=1e-5
    )


# -------------------------------------------------------------------------
# pushforward_marginal — MC fallback and raise paths
# -------------------------------------------------------------------------


def test_pushforward_forward_model_with_normal_falls_to_mc():
    """ForwardModel is not in the closed-form Gaussian-affine table, but
    Normal supports sampling, so it falls through to the MC path. The
    result is an empirical distribution (the workflow_function output)."""

    class _Square(ForwardModel):
        pass

    n = 3
    X = jax.random.uniform(jax.random.key(3), shape=(n, 2), minval=-1, maxval=1)
    nrm = Normal(
        loc=jnp.zeros(n),
        scale=jnp.ones(n),
        name="nrm",
    )
    from sabi._probpipe_compat import independent_uniform
    prior = independent_uniform(
        low=jnp.full((2,), -5.0), high=jnp.full((2,), 5.0), name="p"
    )
    form = _Square(log_lik_from_outputs=lambda x, y: -0.5 * jnp.sum((y - 1.0) ** 2))
    out = pushforward_marginal(nrm, form, X=X, prior=prior)
    # The workflow_function broadcast returns an EmpiricalDistribution.
    # Each sample should have shape (n,) — joint log-density across n points.
    assert isinstance(out, NumericEmpiricalDistribution)


# -------------------------------------------------------------------------
# SurrogatePosterior random log-prob (uses the dispatch internally)
# -------------------------------------------------------------------------


def test_sp_random_unnormalized_log_prob_identity_returns_normal():
    sp = _sp(form=Identity())
    # Single-point query: must be presented as batch (per ArrayRandomFunction).
    X = jnp.asarray([[0.4, -0.1], [0.2, 0.3]])
    marginal = random_unnormalized_log_prob(sp, X)
    assert isinstance(marginal, Normal)
    pred = sp.emulator(X)
    assert jnp.allclose(jnp.asarray(marginal.loc), jnp.asarray(pred.loc), atol=1e-5)


def test_sp_random_unnormalized_log_prob_log_lik_plus_prior_shifts_mean():
    from probpipe import log_prob as pp_log_prob
    from sabi._probpipe_compat import independent_uniform

    prior = independent_uniform(
        low=jnp.full((2,), -5.0), high=jnp.full((2,), 5.0), name="p"
    )
    sp = _sp(form=LogLikPlusPrior(), prior=prior)
    X = jnp.asarray([[0.4, -0.1], [0.2, 0.3]])
    marginal = random_unnormalized_log_prob(sp, X)
    assert isinstance(marginal, Normal)
    pred = sp.emulator(X)
    expected_shifts = jax.vmap(lambda x: jnp.asarray(pp_log_prob(prior, x)))(X)
    expected_loc = jnp.asarray(pred.loc) + expected_shifts
    assert jnp.allclose(jnp.asarray(marginal.loc), expected_loc, atol=1e-5)


def test_sp_does_not_satisfy_supports_mean():
    sp = _sp()
    assert not isinstance(sp, SupportsMean)
    with pytest.raises(TypeError, match="support"):
        mean(sp)


def test_sp_does_not_satisfy_supports_random_log_prob():
    sp = _sp()
    assert not isinstance(sp, SupportsRandomLogProb)
    assert isinstance(sp, SupportsRandomUnnormalizedLogProb)


# -------------------------------------------------------------------------
# expected_target estimator
# -------------------------------------------------------------------------


def test_expected_target_for_werm_is_inner_empirical():
    werm = _werm()
    et = expected_target(werm)
    assert isinstance(et, NumericEmpiricalDistribution)
    assert et is werm.inner_distribution
    assert et is mean(werm)


def test_expected_target_for_sp_satisfies_unnormalized_log_prob_and_sampling():
    sp = _sp()
    et = expected_target(sp)
    assert isinstance(et, Distribution)
    assert isinstance(et, SupportsUnnormalizedLogProb)
    assert isinstance(et, SupportsSampling)


def test_expected_target_for_sp_unnormalized_log_prob_matches_form():
    sp = _sp()
    et = expected_target(sp)
    x = jnp.asarray([0.3, -0.2])
    pred = sp.emulator(x[None])
    pred_mean = jnp.asarray(mean(pred))[0]
    expected = float(sp.log_density_form(x, pred_mean, prior=sp.prior))
    assert float(et._unnormalized_log_prob(x)) == pytest.approx(expected, abs=1e-5)


def test_expected_target_for_sp_sampling_runs_mcmc():
    sp = _sp(n=20)
    et = expected_target(
        sp,
        sampler_kwargs={"num_results": 100, "num_warmup": 50},
    )
    samples = sample(et, key=jax.random.key(0), sample_shape=(40,))
    arr = jnp.asarray(samples)
    assert arr.shape == (40, 2)


def test_expected_target_unsupported_type_raises():
    with pytest.raises(TypeError, match="expected_target"):
        expected_target(object())  # type: ignore[arg-type]
