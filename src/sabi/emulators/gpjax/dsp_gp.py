"""`DSPGPEmulator` — gpjax-backed dimension-scaled-prior GP emulator.

Implements the Hvarfner et al. (2024) recipe:

  *Vanilla Bayesian Optimization Performs Great in High Dimensions*,
  ICML 2024. https://arxiv.org/abs/2402.02229

Key idea: keep vanilla GP-BO, but tighten the kernel/likelihood priors
and **scale the lengthscale prior with the input dimension**:

- Lengthscale: ``LogNormal(loc = √2 + 0.5·log(d), scale = √3)`` (ARD)
- Outputscale: fixed at 1.0 (no `ScaleKernel` wrapper, kernel
  `variance` parameter is held constant via ``paramax.NonTrainable``).
- Noise: ``LogNormal(loc = -4.0, scale = 1.0)`` on the **standard
  deviation** — gpjax exposes ``obs_stddev`` rather than variance.
  Tracked separately from the GPyTorch reference, which puts the same
  prior on the variance parameter.
- Lengthscale floor: 2.5e-2; noise floor: 1e-4. Both initialized at
  the prior modes.

Inference: MAP — minimize ``-(conjugate_mll + Σ log_prior)`` via
``gpx.fit_scipy``. Optional multi-restart over random prior samples
(``n_starts``).

**Input/output scaling.** The DSP recipe is calibrated on inputs
normalized to ``[0, 1]^d`` and outputs standardized to zero-mean
unit-variance. ``DSPGPEmulator.fit(X, Y)`` applies these transforms
internally (min-max for X, z-score for Y) and stores the inverse
transforms for prediction time. Callers do NOT need to pre-scale.

Caveats:

- Scalar-output only (``output_shape=()``).
- ``predict_*`` methods all reuse a single Cholesky cache built at
  fit time (see :mod:`sabi.emulators.gpjax._cache`). Per-call marginal
  cost is O(n²·m + n·m); the joint-input case adds one ``K(Xt, Xt)``
  and a triangular solve. Refit returns a new instance with a fresh
  cache.
- Joint inputs and (trivially) joint outputs are supported via
  ``predict_covariance``; both class-level support flags are True.

Convention: ``predict_variance`` / ``predict_covariance`` return the
**latent** posterior (no observation noise). See the ``Emulator`` base
class for the rationale.

How gpjax handles jitter, and what we do
----------------------------------------

gpjax 0.14 stores **two independent** jitter values on a
``ConjugatePosterior`` and uses them in different places. Concretely:

- ``posterior.prior.jitter`` (the prior's jitter field) is used by:
    * ``gpjax.objectives.conjugate_mll`` for the training-side Sigma
      Cholesky, and
    * ``ConjugatePosterior.predict`` for the **test-side** covariance
      (added to ``K(Xt, Xt) - L⁻¹Kxt^T L⁻¹Kxt`` to keep it PSD).
- ``posterior.jitter`` (the posterior's own jitter field, default
  ``1e-6``) is used by:
    * ``ConjugatePosterior.predict`` for the **training-side** Sigma
      Cholesky.

These two values are independent at construction. The ``prior * lik``
shortcut (i.e., ``construct_posterior(prior, likelihood)``) creates
the posterior with its own default ``jitter=1e-6``; it does **not**
copy ``prior.jitter`` into the posterior. So passing
``Prior(jitter=X)`` alone leaves the posterior using a different
jitter at predict time than the MLL used at fit time — the optimized
hyperparameters are not the MAP of the model that
``posterior.predict`` evaluates, and a strict equivalence comparison
(cached vs naive predict path) catches the mismatch immediately.

We follow the gpjax convention of having both fields, but pin them
to the same user-supplied value.
:func:`sabi.emulators.gpjax._posterior._build_dsp_posterior`
constructs ``ConjugatePosterior`` directly with ``jitter=jitter``, so
both ``posterior.jitter`` and ``posterior.prior.jitter`` agree. The
training-side jitter in the cache is sourced from
``posterior.jitter`` (matching ``ConjugatePosterior.predict``); the
test-side jitter is sourced from ``posterior.prior.jitter`` (matching
the same gpjax method). With both fields aligned at construction,
the two paths are bit-equivalent — but the cache deliberately
mirrors the gpjax dispatch so any future divergence between the two
fields would surface as a test failure rather than a silent
numerical drift.
"""

from __future__ import annotations

import warnings
from typing import Any, Self

# Eager guard — fails fast with a helpful message if the optional
# `gpjax` extra isn't installed. Per-module rather than per-symbol so
# a single import line tells the user how to fix things.
try:
    import gpjax as _gpx  # noqa: F401  (presence check)
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    raise ImportError(
        "DSPGPEmulator requires the optional `gpjax` extra. Install with "
        "`pip install 'sabi[gpjax]'` or `uv sync --extra gpjax`."
    ) from e

import gpjax as gpx
import jax
import jax.numpy as jnp
from jax import Array

from sabi.emulators._scalers import MinMaxScaler, ZScoreScaler
from sabi.emulators.gp import GPEmulator
from sabi.emulators.gpjax._cache import _PredictCache
from sabi.emulators.gpjax._dsp import (
    DSP_LENGTHSCALE_FLOOR,
    DSP_NOISE_FLOOR,
    dsp_map_objective,
)
from sabi.emulators.gpjax._posterior import _build_dsp_posterior


__all__ = ["DSPGPEmulator"]


class DSPGPEmulator(GPEmulator):
    """Dimension-scaled-prior GP emulator (Hvarfner et al. 2024).

    Inherits the predict pipeline, ``condition_on``, and ``_replace``
    machinery from :class:`sabi.emulators.gp.GPEmulator`. This class
    only owns the DSP-specific bits: the kernel/prior construction
    and the multi-restart MAP fit.

    Constructor args:
        input_shape: ``(d,)`` — input dimensionality.
        output_shape: must be ``()`` (scalar output) — multi-output is
            a follow-up.
        name: optional emulator name (used for repr).
        kernel: ``"rbf"`` or ``"matern52"``. Default ``"rbf"`` matches
            Hvarfner et al. 2024.
        max_iters: max L-BFGS-B iterations for fit_scipy.
        jitter: prior jitter added to the gram diagonal for Cholesky
            stability (separate from the noise floor, which lives on
            ``obs_stddev``).
        verbose: passed through to ``gpx.fit_scipy``.
        n_starts: number of MAP optimization runs to launch. The first
            run uses the deterministic prior-mode init; subsequent
            runs sample lengthscale and noise from their priors and
            run the same optimizer. The result with the highest MAP
            objective is kept. Default ``1`` is equivalent to the
            single-fit behavior; increase when the LogNormal prior
            surface might trap L-BFGS-B at the floor on noisy data.
        restart_seed: PRNG seed used to draw the random restart inits.
            Determinism is preserved when ``n_starts == 1`` (no
            sampling happens) — this seed only matters when
            ``n_starts > 1``.
    """

    # Constructor takes ``kernel: str``; we store it as ``kernel_name``
    # to leave ``kernel`` available for any future "the actual kernel
    # object" attribute. ``GPEmulator._replace`` uses this map to
    # bridge constructor-arg ↔ attribute-name when reflecting on
    # ``__init__``.
    _replace_field_map = {"kernel": "kernel_name"}

    def __init__(
        self,
        *,
        input_shape: tuple[int, ...] = (2,),
        output_shape: tuple[int, ...] = (),
        name: str | None = None,
        kernel: str = "rbf",
        max_iters: int = 500,
        jitter: float = 1e-6,
        verbose: bool = False,
        n_starts: int = 1,
        restart_seed: int = 0,
        # Internal post-fit state (callers don't pass these).
        _x_scaler: MinMaxScaler | None = None,
        _y_scaler: ZScoreScaler | None = None,
        _opt_posterior: Any = None,
        _predict_cache: _PredictCache | None = None,
    ):
        if output_shape != ():
            raise ValueError(
                f"DSPGPEmulator is scalar-output only; "
                f"got output_shape={output_shape}."
            )
        if len(input_shape) != 1:
            raise ValueError(
                f"DSPGPEmulator expects input_shape=(d,); got {input_shape}."
            )
        if n_starts < 1:
            raise ValueError(
                f"DSPGPEmulator: n_starts must be >= 1, got {n_starts}."
            )
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            name=name or "DSPGPEmulator",
        )
        self.kernel_name = kernel
        self.max_iters = max_iters
        self.jitter = jitter
        self.verbose = verbose
        self.n_starts = n_starts
        self.restart_seed = restart_seed
        self._x_scaler = _x_scaler
        self._y_scaler = _y_scaler
        self._opt_posterior = _opt_posterior
        self._predict_cache = _predict_cache

    # --- fit ---------------------------------------------------------------

    def fit(self, X: Array, Y: Array) -> Self:
        """Fit the DSP-prior GP to ``(X, Y)`` via MAP optimization.

        With ``n_starts == 1`` (default), runs a single
        ``gpx.fit_scipy`` from the deterministic prior-mode init.
        With ``n_starts > 1``, the first run uses the prior-mode init
        and each subsequent run samples lengthscale + noise stddev
        from their respective priors. The optimized posterior with
        the highest MAP objective is kept.

        Args:
            X: shape ``(n, d)``. Internally rescaled to ``[0, 1]^d``
                via min-max scaling on the training set.
            Y: shape ``(n,)``. Internally standardized to zero-mean
                unit-variance.

        Returns:
            A new ``DSPGPEmulator`` carrying the best optimized
            posterior across all starts, plus the input/output
            scaling state needed for prediction.
        """
        # gpjax requires float64 throughout — its parameter wrappers
        # default to float64 internally and mismatched dtypes break
        # fit_scipy. Caller is expected to have x64 enabled.
        if not jax.config.jax_enable_x64:
            raise RuntimeError(
                "DSPGPEmulator requires JAX x64 mode. Enable with "
                "`jax.config.update('jax_enable_x64', True)` before fitting."
            )

        d = self.input_shape[0]
        if X.ndim != 2 or X.shape[1] != d:
            raise ValueError(
                f"DSPGPEmulator.fit: expected X.shape=(n, {d}), "
                f"got {tuple(X.shape)}."
            )
        if Y.shape != (X.shape[0],):
            raise ValueError(
                f"DSPGPEmulator.fit: expected Y.shape=({X.shape[0]},), "
                f"got {tuple(Y.shape)}."
            )

        x_scaler = MinMaxScaler.fit(X)
        y_scaler = ZScoreScaler.fit(Y)
        Xs = x_scaler.transform(X).astype(jnp.float64)
        Ys = y_scaler.transform(Y).astype(jnp.float64)

        data = gpx.Dataset(X=Xs, y=Ys.reshape(-1, 1))

        # Generate (n_starts) initialization tuples (ls_init, noise_init).
        inits = self._restart_inits(d)

        # Run each start, score by the MAP objective on the optimized
        # posterior, keep the best.
        import paramax

        best_posterior = None
        best_obj: float = -float("inf")
        for start_idx, (ls_init, noise_init) in enumerate(inits):
            posterior = _build_dsp_posterior(
                d=d,
                n=data.n,
                kernel_name=self.kernel_name,
                init_lengthscale=ls_init,
                init_obs_stddev=noise_init,
                jitter=self.jitter,
            )
            try:
                opt_posterior, _history = gpx.fit_scipy(
                    model=posterior,
                    objective=lambda p, dat: -dsp_map_objective(p, dat),
                    train_data=data,
                    max_iters=self.max_iters,
                    # Only chatter on the first start when verbose;
                    # otherwise the scipy progress bar floods stderr
                    # per restart.
                    verbose=self.verbose and best_posterior is None,
                )
            except (jnp.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
                # Bad random init → non-PSD Sigma at fit time. Skip
                # and let another start succeed; warn so the user
                # notices a flaky restart rather than silently
                # losing one.
                warnings.warn(
                    f"DSPGPEmulator.fit: restart {start_idx} skipped "
                    f"({type(exc).__name__}: {exc!s})",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            unwrapped = paramax.unwrap(opt_posterior)
            obj = float(dsp_map_objective(unwrapped, data))
            if obj > best_obj:
                best_obj = obj
                best_posterior = opt_posterior

        if best_posterior is None:
            raise RuntimeError(
                f"DSPGPEmulator.fit: all {self.n_starts} restart(s) failed "
                "to produce a usable posterior. Inspect the data for "
                "near-collinear inputs or try a larger jitter."
            )

        # Pre-solve the Cholesky + alpha vector once. predict_* will
        # reuse these instead of redoing them on every call.
        predict_cache = _PredictCache.build(best_posterior, Xs, Ys)

        return self._replace(
            _x_scaler=x_scaler,
            _y_scaler=y_scaler,
            _opt_posterior=best_posterior,
            _predict_cache=predict_cache,
        )

    def _restart_inits(self, d: int) -> list[tuple[Array, Array]]:
        """Yield the (lengthscale, noise_stddev) init tuples for each start.

        The first tuple is the deterministic prior-mode init — so a
        fit with ``n_starts=1`` is bit-equivalent to the
        pre-multistart behavior. Subsequent tuples are samples from
        the DSP priors, clamped to the floors. The PRNG is seeded
        from ``self.restart_seed``; a fixed seed makes restarts
        reproducible.
        """
        from sabi.emulators.gpjax._dsp import (
            dsp_lengthscale_prior,
            dsp_noise_prior,
        )

        # First start: deterministic prior-mode init (mode of
        # LogNormal(loc, scale) is exp(loc - scale^2)).
        ls_loc = jnp.sqrt(jnp.asarray(2.0)) + 0.5 * jnp.log(jnp.asarray(float(d)))
        ls_mode = jnp.exp(ls_loc - 3.0)  # scale = √3 → scale² = 3
        ls_floor = DSP_LENGTHSCALE_FLOOR + 1e-6
        noise_floor = DSP_NOISE_FLOOR + 1e-6
        ls_init0 = jnp.maximum(
            jnp.full((d,), ls_mode, dtype=jnp.float64),
            jnp.asarray(ls_floor, dtype=jnp.float64),
        )
        noise_init0 = jnp.maximum(
            jnp.asarray(jnp.exp(-5.0), dtype=jnp.float64),
            jnp.asarray(noise_floor, dtype=jnp.float64),
        )
        inits: list[tuple[Array, Array]] = [(ls_init0, noise_init0)]

        if self.n_starts == 1:
            return inits

        # Random restarts: per-start, draw d IID lengthscale samples
        # and a single noise-stddev sample from the priors. Floors
        # are applied so the optimizer's initial state is in the
        # bounded-positive region.
        import jax.random as jr

        ls_prior = dsp_lengthscale_prior(d)
        noise_prior = dsp_noise_prior()
        keys = jr.split(jr.key(self.restart_seed), self.n_starts - 1)
        for k in keys:
            k_ls, k_noise = jr.split(k)
            ls = ls_prior.sample(k_ls, sample_shape=(d,)).astype(jnp.float64)
            ls = jnp.maximum(ls, jnp.asarray(ls_floor, dtype=jnp.float64))
            noise = noise_prior.sample(k_noise).astype(jnp.float64)
            noise = jnp.maximum(noise, jnp.asarray(noise_floor, dtype=jnp.float64))
            inits.append((ls, noise))
        return inits

    @property
    def obs_noise_variance(self) -> Array | None:
        """MAP-fitted observation-noise variance (``obs_stddev²``).

        Sourced from the cache's ``noise_var`` field, which was
        populated at fit time from the unwrapped posterior's
        ``likelihood.obs_stddev``. Returns ``None`` if the emulator
        has not been fit yet — predicting on an unfit emulator
        already raises, but callers that introspect the property
        before fit get a graceful ``None``.
        """
        if self._predict_cache is None:
            return None
        return self._predict_cache.noise_var
