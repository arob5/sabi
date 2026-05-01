"""Unit tests for the DSP-GP primitives in `sabi.emulators.gpjax._dsp`.

Independent of any `Emulator` subclass: just exercises the priors,
the BoundedPositive paramax wrapper, and the MAP objective.

All tests skip cleanly when gpjax is not installed.
"""

from __future__ import annotations

import math

import pytest


def _enable_x64() -> None:
    import jax

    jax.config.update("jax_enable_x64", True)


# --- Priors ------------------------------------------------------------------


def test_dsp_lengthscale_prior_loc_scales_with_log_d():
    """``loc = √2 + 0.5·log(d)`` and ``scale = √3``, regardless of d."""
    pytest.importorskip("gpjax")
    _enable_x64()
    from sabi.emulators.gpjax._dsp import dsp_lengthscale_prior

    for d in (1, 5, 25, 100):
        prior = dsp_lengthscale_prior(d)
        expected_loc = math.sqrt(2.0) + 0.5 * math.log(d)
        assert math.isclose(float(prior.loc), expected_loc, rel_tol=1e-6)
        assert math.isclose(float(prior.scale), math.sqrt(3.0), rel_tol=1e-6)


def test_dsp_lengthscale_prior_rejects_nonpositive_d():
    pytest.importorskip("gpjax")
    from sabi.emulators.gpjax._dsp import dsp_lengthscale_prior

    with pytest.raises(ValueError, match=">= 1"):
        dsp_lengthscale_prior(0)


def test_dsp_noise_prior_is_lognormal_on_stddev():
    """Noise prior is LogNormal(-4, 1) on the *standard deviation*."""
    pytest.importorskip("gpjax")
    _enable_x64()
    from sabi.emulators.gpjax._dsp import dsp_noise_prior

    prior = dsp_noise_prior()
    assert math.isclose(float(prior.loc), -4.0, rel_tol=1e-6)
    assert math.isclose(float(prior.scale), 1.0, rel_tol=1e-6)


# --- BoundedPositive --------------------------------------------------------


def test_bounded_positive_unwraps_to_input_value():
    """``BoundedPositive(v, lower=L).unwrap()`` returns ``v`` to within fp slop."""
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp

    from sabi.emulators.gpjax._dsp import BoundedPositive

    bp = BoundedPositive(jnp.array(0.7), lower=2.5e-2)
    out = bp.unwrap()
    assert jnp.allclose(out, 0.7, atol=1e-6)


def test_bounded_positive_respects_floor():
    """For any unconstrained input, ``unwrap()`` is strictly above ``lower``."""
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp

    from sabi.emulators.gpjax._dsp import BoundedPositive

    bp = BoundedPositive(jnp.array(2.5e-2), lower=2.5e-2)
    # Even when initialized at the floor, unwrap should be > lower
    # (the constructor adds a tiny offset to avoid log(0)).
    out = float(bp.unwrap())
    assert out > 2.5e-2

    # And never falls below the floor under arbitrary unconstrained values.
    import equinox as eqx

    bp2 = eqx.tree_at(lambda b: b._unconstrained, bp, jnp.array(-50.0))
    out2 = float(bp2.unwrap())
    assert out2 >= 2.5e-2 - 1e-12  # softplus(-50) ≈ 0


def test_bounded_positive_unwraps_arrays_per_dim():
    """ARD-style: BoundedPositive over a vector unwraps elementwise."""
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp

    from sabi.emulators.gpjax._dsp import BoundedPositive

    values = jnp.array([0.1, 0.5, 1.7, 2.5e-2 + 1e-5])
    bp = BoundedPositive(values, lower=2.5e-2)
    out = bp.unwrap()
    assert out.shape == values.shape
    assert jnp.all(out > 2.5e-2)
    assert jnp.allclose(out, values, atol=1e-4)


def test_bounded_positive_works_with_paramax_unwrap_in_a_pytree():
    """A pytree containing BoundedPositive nodes unwraps cleanly via
    ``paramax.unwrap`` — this is what gpjax's ``fit_scipy`` does
    internally before calling the objective."""
    pytest.importorskip("gpjax")
    _enable_x64()
    import jax.numpy as jnp
    import paramax

    from sabi.emulators.gpjax._dsp import BoundedPositive

    tree = {
        "lengthscale": BoundedPositive(jnp.array([0.5, 1.0]), lower=2.5e-2),
        "noise": BoundedPositive(jnp.array(0.01), lower=1e-4),
        "static": "abc",
    }
    out = paramax.unwrap(tree)
    assert out["static"] == "abc"
    assert jnp.allclose(out["lengthscale"], jnp.array([0.5, 1.0]), atol=1e-5)
    assert jnp.allclose(out["noise"], jnp.array(0.01), atol=1e-5)


# --- MAP objective ----------------------------------------------------------


def test_dsp_map_objective_is_finite_on_a_small_fit():
    """Sanity check: the objective evaluates to a finite scalar on a
    small synthetic dataset and reproduces ``conjugate_mll +
    Σ log_priors``."""
    pytest.importorskip("gpjax")
    _enable_x64()
    import gpjax as gpx
    import jax.numpy as jnp
    import jax.random as jr

    from sabi.emulators.gpjax._dsp import (
        BoundedPositive,
        DSP_LENGTHSCALE_FLOOR,
        DSP_NOISE_FLOOR,
        dsp_lengthscale_prior,
        dsp_map_objective,
        dsp_noise_prior,
    )

    key = jr.key(0)
    n, d = 20, 3
    X = jr.uniform(key, (n, d))
    y = jnp.sin(X.sum(axis=-1, keepdims=True))
    data = gpx.Dataset(X=X, y=y)

    kernel = gpx.kernels.RBF(
        lengthscale=BoundedPositive(jnp.ones(d), lower=DSP_LENGTHSCALE_FLOOR),
    )
    meanf = gpx.mean_functions.Constant()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    # Gaussian likelihood re-wraps `obs_stddev` as NonNegativeReal in its
    # __init__ unless we swap it in post-construction with eqx.tree_at.
    lik = gpx.likelihoods.Gaussian(num_datapoints=data.n, obs_stddev=0.1)
    import equinox as eqx

    lik = eqx.tree_at(
        lambda l: l.obs_stddev,
        lik,
        BoundedPositive(jnp.array(0.1), lower=DSP_NOISE_FLOOR),
    )
    posterior = prior * lik

    import paramax

    unwrapped = paramax.unwrap(posterior)
    val = dsp_map_objective(unwrapped, data)
    assert jnp.isfinite(val)

    # Manual check: log(MAP) = mll + log_p(ls) + log_p(noise)
    mll = gpx.objectives.conjugate_mll(unwrapped, data)
    ls = unwrapped.prior.kernel.lengthscale
    noise = unwrapped.likelihood.obs_stddev
    expected = (
        mll
        + dsp_lengthscale_prior(d).log_prob(ls).sum()
        + dsp_noise_prior().log_prob(noise).sum()
    )
    assert jnp.allclose(val, expected, rtol=1e-6)
