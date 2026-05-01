"""gpjax posterior construction.

Single helper:

- ``_build_dsp_posterior``: constructs a fresh ``ConjugatePosterior``
  with the DSP-recipe parameter wrappers (``BoundedPositive``-bounded
  lengthscale and noise stddev, fixed unit outputscale via
  ``paramax.NonTrainable``). The constructor pins ``posterior.jitter``
  and ``posterior.prior.jitter`` to the same user-supplied value so
  the optimized MLL and the predictive distribution share an identical
  numerical model — see the ``dsp_gp.py`` module docstring for the
  full discussion.

The output-space covariance scaling helper
(``_scale_cov_to_output_space``) lives in
:mod:`sabi.emulators.gp` since it's used by the shared
``GPEmulator.predict_covariance`` rather than only by the gpjax
backend.
"""

from __future__ import annotations

import equinox as eqx
import gpjax as gpx
import jax.numpy as jnp
from jax import Array

from sabi.emulators.gpjax._dsp import (
    DSP_LENGTHSCALE_FLOOR,
    DSP_NOISE_FLOOR,
    BoundedPositive,
)


def _build_dsp_posterior(
    *,
    d: int,
    n: int,
    kernel_name: str,
    init_lengthscale: Array,
    init_obs_stddev: Array,
    jitter: float,
):
    """Construct a fresh ConjugatePosterior with DSP-prior parameter wrappers.

    The kernel ``variance`` (outputscale) is held fixed at 1.0 by
    wrapping it in ``paramax.NonTrainable`` — this matches the DSP
    recipe of pinning outputscale=1.

    The posterior is constructed directly (rather than via
    ``prior * lik``) so that ``posterior.jitter`` and
    ``posterior.prior.jitter`` both take the user-supplied value.
    The ``prior * lik`` shortcut leaves ``posterior.jitter`` at the
    gpjax default (1e-6), independently of the prior's jitter — see
    the ``dsp_gp`` module docstring for the full discussion.
    """
    import paramax

    if kernel_name == "rbf":
        kernel_cls = gpx.kernels.RBF
    elif kernel_name == "matern52":
        kernel_cls = gpx.kernels.Matern52
    else:
        raise ValueError(
            f"DSPGPEmulator: kernel must be 'rbf' or 'matern52', got "
            f"{kernel_name!r}."
        )

    # Pin outputscale at 1.0: pass a `NonTrainable` wrapping a plain
    # array directly. The kernel's __init__ accepts any
    # AbstractUnwrappable for `variance` and stores it as-is. During
    # fit, `paramax.unwrap` applies stop_gradient to the inner array
    # so scipy can't move it.
    kernel = kernel_cls(
        lengthscale=BoundedPositive(init_lengthscale, lower=DSP_LENGTHSCALE_FLOOR),
        variance=paramax.NonTrainable(jnp.asarray(1.0, dtype=jnp.float64)),
        n_dims=d,
    )

    meanf = gpx.mean_functions.Constant()
    prior = gpx.gps.Prior(
        mean_function=meanf, kernel=kernel, jitter=jitter
    )
    lik = gpx.likelihoods.Gaussian(num_datapoints=n, obs_stddev=init_obs_stddev)
    # Likelihood __init__ re-wraps obs_stddev as NonNegativeReal; swap
    # it for our BoundedPositive after construction.
    lik = eqx.tree_at(
        lambda l: l.obs_stddev,
        lik,
        BoundedPositive(init_obs_stddev, lower=DSP_NOISE_FLOOR),
    )

    return gpx.gps.ConjugatePosterior(prior=prior, likelihood=lik, jitter=jitter)
