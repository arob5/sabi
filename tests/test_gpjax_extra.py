"""Tests for the optional `gpjax` extra wiring.

Verifies:
- The `sabi.emulators.gpjax` package can be touched without gpjax
  installed (lazy `__getattr__` defers the gpjax import).
- When gpjax IS installed, importing `DSPGPEmulator` succeeds.
- The end-to-end gpjax fit/predict workflow we depend on still
  matches the gpjax 0.14 API (regression tripwire — if gpjax bumps
  break this, we want a loud test failure rather than silent
  fallthrough).

All tests in this module skip cleanly when gpjax is not installed.
"""

from __future__ import annotations

import pytest


def test_emulators_gpjax_package_importable_without_concrete_classes():
    """Touching `sabi.emulators.gpjax` should not require gpjax to be
    installed — the lazy ``__getattr__`` defers the import."""
    import sabi.emulators.gpjax  # noqa: F401  (presence check)


def test_emulators_gpjax_dir_lists_dspgp():
    """`__all__` advertises `DSPGPEmulator` regardless of whether
    gpjax is currently installed."""
    import sabi.emulators.gpjax as _gpjax_mod

    assert "DSPGPEmulator" in _gpjax_mod.__all__


def test_emulators_gpjax_unknown_attribute_raises_clean_error():
    """Accessing an unknown attribute gives a clear AttributeError
    listing the available exports."""
    import sabi.emulators.gpjax as _gpjax_mod

    with pytest.raises(AttributeError, match="DSPGPEmulator"):
        _ = _gpjax_mod.DoesNotExist  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# Below: tests that require gpjax actually installed.
# -----------------------------------------------------------------------------


def test_dspgp_module_importable_when_gpjax_present():
    """When gpjax is installed, the `dsp_gp` module imports without error."""
    pytest.importorskip("gpjax")
    from sabi.emulators.gpjax import dsp_gp  # noqa: F401


def test_gpjax_fit_predict_workflow_matches_expected_api():
    """Regression tripwire: confirm gpjax 0.14's fit_scipy + predict
    surface still behaves as documented. If gpjax bumps and breaks
    one of these calls, we get a loud failure here BEFORE the
    `DSPGPEmulator` integration tests start mis-skipping."""
    gpx = pytest.importorskip("gpjax")
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax.random as jr

    key = jr.key(0)
    n = 30
    X = jnp.linspace(0, 1, n).reshape(-1, 1)
    y = jnp.sin(2 * jnp.pi * X[:, 0:1]) + 0.05 * jr.normal(key, (n, 1))
    D = gpx.Dataset(X=X, y=y)

    kernel = gpx.kernels.RBF(lengthscale=jnp.ones(1))
    meanf = gpx.mean_functions.Constant()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n, obs_stddev=0.1)
    posterior = prior * likelihood

    opt_posterior, history = gpx.fit_scipy(
        model=posterior,
        objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
        train_data=D,
        verbose=False,
    )
    assert len(history) > 0

    X_test = jnp.linspace(0, 1, 5).reshape(-1, 1)
    latent = opt_posterior.predict(X_test, train_data=D)
    pred = opt_posterior.likelihood(latent)
    assert pred.mean.shape == (5,)
    assert pred.variance.shape == (5,)
