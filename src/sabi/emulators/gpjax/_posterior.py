"""gpjax posterior construction + output-space covariance scaling.

Two helpers used by ``DSPGPEmulator``:

- ``_build_dsp_posterior``: constructs a fresh ``ConjugatePosterior``
  with the DSP-recipe parameter wrappers (``BoundedPositive``-bounded
  lengthscale and noise stddev, fixed unit outputscale via
  ``paramax.NonTrainable``). The constructor pins ``posterior.jitter``
  and ``posterior.prior.jitter`` to the same user-supplied value so
  the optimized MLL and the predictive distribution share an identical
  numerical model — see the ``dsp_gp.py`` module docstring for the
  full discussion.
- ``_scale_cov_to_output_space``: brings a covariance from
  standardized-y space back to the original output space. Scalar
  output today; multi-output is flagged with a clear
  ``NotImplementedError`` and the per-mode Kronecker contract is
  spelled out so the helper is the single edit-site when multi-output
  ships.
"""

from __future__ import annotations

import equinox as eqx
import gpjax as gpx
import jax.numpy as jnp
from jax import Array

from sabi.emulators._scalers import ZScoreScaler
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


def _scale_cov_to_output_space(
    cov: Array, y_scaler: ZScoreScaler, *, output_shape: tuple[int, ...]
) -> Array:
    """Bring a covariance from standardized-y space back to the original
    output space, scaling by ``y_scaler.scale²``.

    Multi-output safety
    -------------------
    For the **scalar-output** case (``output_shape == ()``),
    ``y_scaler.scale`` is a 0-d array and the multiplication is a
    clean elementwise scalar broadcast against ``cov`` (shape
    ``(n, n)`` for joint inputs, or ``(n,)`` / ``(n, 1, 1)`` for
    marginals).

    For multi-output (``output_shape != ()``), the right thing depends
    on which cross-axes are joint:

    - ``joint_inputs=False, joint_outputs=True`` → cov shape
      ``(n, prod(out), prod(out))``: scaling is a Kronecker outer of
      ``scale ⊗ scale`` along the (prod(out), prod(out)) trailing
      block. Per-output diagonal becomes ``scale²``; off-diagonal
      cross-output scales become ``scale_i · scale_j``.
    - ``joint_inputs=True, joint_outputs=False`` → cov shape
      ``(*out, n, n)``: each output's (n, n) block scales by its own
      ``scale²``; outputs don't mix. So broadcasting ``scale²`` of
      shape ``out`` against the leading ``out`` axes is correct.
    - ``joint_inputs=True, joint_outputs=True`` → cov shape
      ``(n*prod(out), n*prod(out))``: needs the full Kronecker; a
      plain elementwise multiply is *wrong*.

    DSPGPEmulator is scalar-output in v1 (the constructor rejects
    non-empty ``output_shape``), so this helper currently asserts
    that invariant and uses the simple scalar broadcast. Multi-output
    support will need to fan out to the per-mode logic above. Keeping
    the helper centralized here so the change is a single-file edit
    when that lands.
    """
    if output_shape != ():
        # Defensive: even though DSPGPEmulator's __init__ rejects
        # non-empty output_shape today, this helper is the place
        # where a multi-output extension would need a careful
        # refactor. Fail loud here so a future "I'll just lift the
        # output_shape check" change doesn't silently miscalibrate
        # covariances.
        raise NotImplementedError(
            f"_scale_cov_to_output_space: multi-output (output_shape="
            f"{output_shape}) requires axis-aware Kronecker scaling; "
            f"see this helper's docstring for the per-mode contract."
        )
    return cov * (y_scaler.scale ** 2)
