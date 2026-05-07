"""Tests for SurrogateDistribution / EmulatedDistribution / WeightedEmpiricalRandomMeasure.

Coverage:
- Sibling-structure invariants.
- ``EmulatedDistribution`` rejects ``None`` emulator / decomposition / support.
- Inner-support / inner-event-shape derived from constructor args.
- Decoupling from Problem (constructed from math primitives only).
- Protocol opt-in matrix per class.
- Dirac equivalence: ``mean`` / ``expected_target`` on a WERM return the inner empirical.
- ``EmulatedDistribution._random_unnormalized_log_prob`` routes through
  ``decomposition.pushforward(X, emulator(X))`` — closed-form
  Gaussian-affine for ``Identity / LogProb`` and MC for non-affine links.
- ``expected_target`` on an ``EmulatedDistribution`` satisfies
  ``SupportsUnnormalizedLogProb`` and ``SupportsSampling``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from probpipe import (
    log_prob,
    mean,
    random_log_prob,
    random_unnormalized_log_prob,
    sample,
)
from probpipe.core._distribution_base import Distribution
from probpipe.core._empirical import NumericEmpiricalDistribution
from probpipe.core._random_measures import NumericRandomMeasure, RandomMeasure
from probpipe.core.constraints import interval
from probpipe.core.protocols import (
    SupportsMean,
    SupportsRandomLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
)
from probpipe.distributions.continuous import Normal
from probpipe.distributions.multivariate import MultivariateNormal

from sabi._probpipe_compat import independent_uniform
from sabi.density_decomposition import DensityDecomposition, ScalarConstant
from sabi.maps import Identity as MapIdentity, LogProb, LogSquare
from sabi.surrogate import (
    EmulatedDistribution,
    SurrogateDistribution,
    WeightedEmpiricalRandomMeasure,
    expected_target,
)
from sabi.emulators import TinyGPEmulator


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


def _box_support(d: int = 2):
    return interval(jnp.full((d,), -5.0), jnp.full((d,), 5.0))


def _quad(x):
    return -0.5 * jnp.sum(x * x)


def _identity_decomposition() -> DensityDecomposition:
    return DensityDecomposition(
        target_single=_quad,
        output_shape=(),
        link=MapIdentity(),
    )


def _log_lik_plus_prior_decomposition(prior) -> DensityDecomposition:
    return DensityDecomposition(
        target_single=_quad,
        output_shape=(),
        link=MapIdentity(),
        shift=LogProb(prior),
    )


def _werm(n: int = 16, d: int = 2, seed: int = 0):
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


def _emulated(n: int = 20, d: int = 2, seed: int = 1, decomposition=None):
    key = jax.random.key(seed)
    X = jax.random.uniform(key, shape=(n, d), minval=-3.0, maxval=3.0)
    Y = -0.5 * jnp.sum(X ** 2, axis=-1)
    emulator = TinyGPEmulator(input_shape=(d,)).fit(X, Y)
    if decomposition is None:
        decomposition = _identity_decomposition()
    return EmulatedDistribution(
        emulator=emulator,
        decomposition=decomposition,
        support=_box_support(d),
        input_shape=(d,),
        name="emulated_test",
    )


# -------------------------------------------------------------------------
# Inheritance and metadata
# -------------------------------------------------------------------------


def test_sibling_structure_invariants():
    assert issubclass(EmulatedDistribution, SurrogateDistribution)
    assert issubclass(WeightedEmpiricalRandomMeasure, SurrogateDistribution)
    assert not issubclass(WeightedEmpiricalRandomMeasure, EmulatedDistribution)
    assert not issubclass(EmulatedDistribution, WeightedEmpiricalRandomMeasure)


def test_surrogate_distribution_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        SurrogateDistribution(name="bad")  # type: ignore[abstract]


def test_werm_is_random_measure():
    werm = _werm()
    assert isinstance(werm, NumericRandomMeasure)
    assert isinstance(werm, RandomMeasure)
    assert isinstance(werm, SurrogateDistribution)


def test_emulated_is_random_measure():
    emulated = _emulated()
    assert isinstance(emulated, EmulatedDistribution)
    assert isinstance(emulated, SurrogateDistribution)
    assert isinstance(emulated, NumericRandomMeasure)


def test_inner_support_and_event_shape_match_constructor_args():
    werm = _werm()
    assert werm.inner_event_shape == (2,)
    emulated = _emulated()
    assert emulated.inner_event_shape == (2,)


def test_werm_requires_support():
    with pytest.raises(ValueError, match="support"):
        WeightedEmpiricalRandomMeasure(
            X=jnp.zeros((4, 2)),
            log_weights=jnp.zeros(4),
            support=None,  # type: ignore[arg-type]
            input_shape=(2,),
        )


def test_emulated_distribution_requires_support():
    emulator = TinyGPEmulator(input_shape=(2,)).fit(
        jnp.zeros((4, 2)), jnp.zeros(4)
    )
    with pytest.raises(ValueError, match="support"):
        EmulatedDistribution(
            emulator=emulator,
            decomposition=_identity_decomposition(),
            support=None,  # type: ignore[arg-type]
            input_shape=(2,),
        )


def test_emulated_distribution_rejects_none_emulator():
    with pytest.raises(ValueError, match="non-None `emulator`"):
        EmulatedDistribution(
            emulator=None,  # type: ignore[arg-type]
            decomposition=_identity_decomposition(),
            support=_box_support(2),
            input_shape=(2,),
        )


def test_emulated_distribution_rejects_none_decomposition():
    emulator = TinyGPEmulator(input_shape=(2,)).fit(
        jnp.zeros((4, 2)), jnp.zeros(4)
    )
    with pytest.raises(ValueError, match="decomposition"):
        EmulatedDistribution(
            emulator=emulator,
            decomposition=None,  # type: ignore[arg-type]
            support=_box_support(2),
            input_shape=(2,),
        )


def test_emulated_distribution_input_shape_must_match_emulator_input_shape():
    emulator = TinyGPEmulator(input_shape=(2,)).fit(
        jnp.zeros((4, 2)), jnp.zeros(4)
    )
    with pytest.raises(ValueError, match="input_shape"):
        EmulatedDistribution(
            emulator=emulator,
            decomposition=_identity_decomposition(),
            support=_box_support(3),
            input_shape=(3,),
        )


# -------------------------------------------------------------------------
# WERM: Dirac protocol surface
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
# EmulatedDistribution._random_unnormalized_log_prob — closed-form path
# -------------------------------------------------------------------------


def test_emulated_random_unnormalized_log_prob_identity_returns_normal():
    """Identity link + Constant(0) shift: pushforward through a Normal
    emulator is a Normal with the same loc/scale (Affine(intercept=0))."""
    emulated = _emulated()
    X = jnp.asarray([[0.4, -0.1], [0.2, 0.3]])
    marginal = random_unnormalized_log_prob(emulated, X)
    assert isinstance(marginal, Normal)
    pred = emulated.emulator(X)
    assert jnp.allclose(jnp.asarray(marginal.loc), jnp.asarray(pred.loc), atol=1e-5)


def test_emulated_random_unnormalized_log_prob_log_lik_plus_prior_shifts_mean():
    """Identity link + LogProb(prior) shift: pushforward through a Normal
    is a Normal with loc shifted by ``log_prob(prior, x)`` per row."""
    prior = independent_uniform(
        low=jnp.full((2,), -5.0), high=jnp.full((2,), 5.0), name="p"
    )
    emulated = _emulated(decomposition=_log_lik_plus_prior_decomposition(prior))
    X = jnp.asarray([[0.4, -0.1], [0.2, 0.3]])
    marginal = random_unnormalized_log_prob(emulated, X)
    assert isinstance(marginal, Normal)
    pred = emulated.emulator(X)
    expected_shifts = jax.vmap(lambda x: jnp.asarray(log_prob(prior, x)))(X)
    expected_loc = jnp.asarray(pred.loc) + expected_shifts
    assert jnp.allclose(jnp.asarray(marginal.loc), expected_loc, atol=1e-5)


def test_emulated_random_unnormalized_log_prob_non_affine_link_falls_to_mc():
    """``LogSquare`` link is non-affine; the pushforward falls to MC and
    returns a non-Normal Distribution."""
    decomposition = DensityDecomposition(
        target_single=_quad,
        output_shape=(),
        link=LogSquare(),
        shift=ScalarConstant(jnp.asarray(0.0)),
    )
    emulated = _emulated(decomposition=decomposition)
    X = jnp.asarray([[0.4, -0.1], [0.2, 0.3]])
    marginal = random_unnormalized_log_prob(emulated, X)
    assert isinstance(marginal, Distribution)
    assert not isinstance(marginal, Normal)


def test_emulated_does_not_satisfy_supports_mean():
    emulated = _emulated()
    assert not isinstance(emulated, SupportsMean)


def test_emulated_satisfies_random_unnormalized_log_prob():
    emulated = _emulated()
    assert isinstance(emulated, SupportsRandomUnnormalizedLogProb)


# -------------------------------------------------------------------------
# expected_target estimator
# -------------------------------------------------------------------------


def test_expected_target_for_werm_is_inner_empirical():
    werm = _werm()
    et = expected_target(werm)
    assert isinstance(et, NumericEmpiricalDistribution)
    assert et is werm.inner_distribution
    assert et is mean(werm)


def test_expected_target_for_emulated_satisfies_unnormalized_log_prob_and_sampling():
    emulated = _emulated()
    et = expected_target(emulated)
    assert isinstance(et, Distribution)
    assert isinstance(et, SupportsUnnormalizedLogProb)
    assert isinstance(et, SupportsSampling)


def test_expected_target_for_emulated_unnormalized_log_prob_matches_decomposition():
    emulated = _emulated()
    et = expected_target(emulated)
    x = jnp.asarray([0.3, -0.2])
    pred = emulated.emulator(x[None])
    pred_mean = jnp.asarray(mean(pred))[0]
    expected = float(emulated.decomposition(x, pred_mean))
    assert float(et._unnormalized_log_prob(x)) == pytest.approx(expected, abs=1e-5)


def test_expected_target_for_emulated_sampling_runs_mcmc():
    emulated = _emulated(n=20)
    et = expected_target(
        emulated,
        sampler_kwargs={"num_results": 100, "num_warmup": 50},
    )
    samples = sample(et, key=jax.random.key(0), sample_shape=(40,))
    arr = jnp.asarray(samples)
    assert arr.shape == (40, 2)


def test_expected_target_unsupported_type_raises():
    with pytest.raises(TypeError, match="expected_target"):
        expected_target(object())  # type: ignore[arg-type]
