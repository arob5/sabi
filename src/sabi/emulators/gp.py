"""`GPEmulator` — shared abstract base for sabi's GP-style emulators.

Sits between :class:`Emulator` and the concrete backends
(:class:`sabi.emulators.tinygp.gp.TinyGPEmulator`,
:class:`sabi.emulators.gpjax.dsp_gp.DSPGPEmulator`). Carries the
state and behavior they have in common:

- Fields: ``_x_scaler``, ``_y_scaler``, and an opaque
  ``_predict_cache`` (anything matching :class:`PredictCacheProtocol`).
- Concrete predict pipeline: ``predict_mean``, ``predict_variance``,
  ``predict_covariance``. All routed through the cache and the
  scalers. ``_scale_cov_to_output_space`` lives on this module too;
  for now scalar-output only, with a documented multi-output
  Kronecker contract for the day that lands.
- Concrete ``condition_on(X_new, Y_new)`` that validates shapes,
  applies the frozen scalers, and delegates to
  ``cache.append_rows``. Subclasses don't need to reimplement.
- Concrete ``_replace(**overrides)`` helper that uses
  ``inspect.signature`` to enumerate constructor args. Subclasses
  whose constructor name differs from the attribute name (e.g.
  ``__init__(... kernel: str ...)`` storing ``self.kernel_name``)
  declare a small ``_replace_field_map`` class attribute to bridge
  the gap.

Subclasses provide:

- ``fit(X, Y)`` — backend-specific hyperparameter strategy.
- A cache type implementing :class:`PredictCacheProtocol`.

Class-level support flags (``supports_joint_inputs``,
``supports_joint_outputs``) default to ``True``; subclasses without
joint-mode support set them to ``False``.
"""

from __future__ import annotations

import inspect
from abc import abstractmethod
from typing import ClassVar, Protocol, Self

import jax.numpy as jnp
from jax import Array
from probpipe.distributions.gaussian_random_function import GaussianRandomFunction

from sabi.emulators._scalers import ZScoreScaler
from sabi.emulators.base import Emulator

__all__ = [
    "GPEmulator",
    "PredictCacheProtocol",
    "_scale_cov_to_output_space",
]


class PredictCacheProtocol(Protocol):
    """Protocol for the cache that backs a :class:`GPEmulator` subclass.

    Concrete cache classes are not required to inherit from this
    protocol; structural conformance is enough. Both backends'
    caches implement these methods on their own dataclasses.
    """

    Xs_train: Array
    Ys_train: Array

    def predict_latent(self, Xt: Array) -> tuple[Array, Array]:
        """Return ``(mean, var)`` of the latent posterior at ``Xt``.

        ``mean`` and ``var`` are each shape ``(m,)``. Variance is the
        marginal latent variance with prior jitter added — no
        observation noise.
        """
        ...

    def predict_latent_joint(self, Xt: Array) -> tuple[Array, Array]:
        """Return ``(mean, cov)`` for the joint latent posterior at ``Xt``.

        Cov is shape ``(m, m)``, symmetric, and PSD with prior jitter
        on the diagonal. Backends that don't support joint mode raise
        ``NotImplementedError`` here, and their class-level
        ``supports_joint_inputs`` should be ``False``.
        """
        ...

    def append_rows(self, Xs_new: Array, Ys_new: Array) -> "PredictCacheProtocol":
        """Return a new cache conditioned on ``(Xs_new, Ys_new)``.

        ``Xs_new`` and ``Ys_new`` are already in the
        scaler-transformed coordinates of the existing cache.
        Backends that don't support cheap-update conditioning raise
        ``NotImplementedError``.
        """
        ...


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

    For multi-output (``output_shape != ()``), the right thing
    depends on which cross-axes are joint:

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

    GPEmulator subclasses are scalar-output today, so this helper
    asserts that invariant and uses the simple scalar broadcast.
    Multi-output support will need to fan out to the per-mode logic
    above. Keeping the helper centralized here so the change is a
    single-file edit when that lands.
    """
    if output_shape != ():
        # Defensive: the constructors of all GPEmulator subclasses
        # reject non-empty output_shape today, but this helper is
        # the place where a multi-output extension would need a
        # careful refactor. Fail loud here so a future "I'll just
        # lift the output_shape check" change doesn't silently
        # miscalibrate covariances.
        raise NotImplementedError(
            f"_scale_cov_to_output_space: multi-output (output_shape="
            f"{output_shape}) requires axis-aware Kronecker scaling; "
            f"see this helper's docstring for the per-mode contract."
        )
    return cov * (y_scaler.scale ** 2)


class GPEmulator(Emulator, GaussianRandomFunction):
    """Abstract intermediate base for GP-style emulators.

    Concrete subclasses (:class:`TinyGPEmulator`,
    :class:`DSPGPEmulator`) provide:

    - ``fit(X, Y)`` — backend-specific hyperparameter strategy.
    - A cache populated by ``fit`` whose type satisfies
      :class:`PredictCacheProtocol`.

    The base provides the rest: predict pipeline, ``condition_on``
    (frozen-hyperparameter rank-update via ``cache.append_rows``),
    and ``_replace``.

    Subclasses with constructor-arg / attribute-name mismatches
    (e.g. ``__init__(*, kernel: str = "rbf")`` storing
    ``self.kernel_name``) declare a class-level
    ``_replace_field_map: dict[str, str]`` mapping constructor arg
    names to attribute names. The default empty mapping handles the
    common case where they match.
    """

    # Joint-input covariance and (trivially for scalar output)
    # joint-output mode. Subclasses without joint covariance support
    # set these to False and override predict_covariance to raise.
    supports_joint_inputs: bool = True
    supports_joint_outputs: bool = True

    # Constructor-arg → attribute-name overrides for ``_replace``.
    # Default empty: assumes constructor args and attribute names
    # match. Subclasses with a mismatch (e.g. ``kernel`` arg storing
    # ``self.kernel_name``) override by adding a class attribute:
    #
    #     _replace_field_map = {"kernel": "kernel_name"}
    _replace_field_map: ClassVar[dict[str, str]] = {}

    def _replace(self, **overrides) -> Self:
        """Return a copy of this emulator with the given fields replaced.

        Centralizes the "rebuild emulator with replacements" pattern
        used by ``fit``, ``condition_on``, and the dispatch
        handlers. Reflects on ``type(self).__init__``'s signature to
        enumerate constructor args; pulls current values from
        ``self`` (using ``_replace_field_map`` for any
        constructor-arg / attribute-name mismatches); and builds the
        replacement instance.
        """
        sig = inspect.signature(type(self).__init__)
        defaults: dict[str, object] = {}
        for arg_name in sig.parameters:
            if arg_name == "self":
                continue
            attr_name = self._replace_field_map.get(arg_name, arg_name)
            defaults[arg_name] = getattr(self, attr_name)
        defaults.update(overrides)
        return type(self)(**defaults)

    # --- predict pipeline --------------------------------------------------

    def predict_mean(self, X: Array) -> Array:
        """Posterior mean of the latent function at ``X``.

        ``X`` is in the original (raw) input space; the configured
        ``_x_scaler`` transforms it to the cache's coordinate
        system. Returned shape ``(n,)`` for scalar output.
        """
        self._require_fit()
        Xs = self._x_scaler.transform(X).astype(jnp.float64)
        mean_latent, _ = self._predict_cache.predict_latent(Xs)
        # Gaussian observation noise has zero mean, so latent and
        # observation predictive means coincide — we always return
        # the latent mean here, but the value is the same.
        return self._y_scaler.inverse_mean(mean_latent)

    def predict_variance(self, X: Array) -> Array:
        """Marginal posterior variance of the latent function at ``X``.

        Per the sabi `Emulator` convention, observation noise is
        **not** added to the diagonal — see
        :class:`Emulator` for the rationale and the
        ``obs_noise_variance`` property for the noise term.
        """
        self._require_fit()
        Xs = self._x_scaler.transform(X).astype(jnp.float64)
        _, latent_var = self._predict_cache.predict_latent(Xs)
        return self._y_scaler.inverse_var(jnp.maximum(latent_var, 0.0))

    def predict_covariance(
        self,
        X: Array,
        *,
        joint_inputs: bool = False,
        joint_outputs: bool = False,
    ) -> Array:
        """Posterior covariance of the latent function at ``X`` (no obs noise).

        Per the sabi `Emulator` convention, observation noise is not
        included on the diagonal — the diagonal of the returned
        matrix equals ``predict_variance(X)`` exactly.

        Shape contract for the scalar-output case:

        - ``joint_inputs=True`` (regardless of ``joint_outputs``):
          ``(n, n)``.
        - ``joint_inputs=False, joint_outputs=True``: ``(n, 1, 1)``.
        - ``joint_inputs=False, joint_outputs=False``: not
          implemented; callers should use ``predict_variance``.

        Subclasses that don't support joint-input covariance set
        ``supports_joint_inputs=False`` and override this method to
        raise.
        """
        self._require_fit()
        if not joint_inputs and not joint_outputs:
            raise NotImplementedError(
                "predict_covariance with joint_inputs=False, "
                "joint_outputs=False is not implemented; "
                "use predict_variance for the marginal case."
            )

        if not joint_inputs:
            # joint_outputs only: scalar output → reshape variance to (n, 1, 1).
            return self.predict_variance(X)[:, None, None]

        # joint_inputs=True (joint_outputs is trivial here): full (n, n).
        Xs = self._x_scaler.transform(X).astype(jnp.float64)
        _, latent_cov = self._predict_cache.predict_latent_joint(Xs)
        return _scale_cov_to_output_space(
            latent_cov, self._y_scaler, output_shape=self.output_shape
        )

    def _require_fit(self) -> None:
        if self._predict_cache is None:
            raise RuntimeError(
                f"{type(self).__name__} called before fit; conditioning "
                "state is unset."
            )

    # --- fit + condition_on ------------------------------------------------

    @abstractmethod
    def fit(self, X: Array, Y: Array) -> Self:
        """Backend-specific fit. See subclass docstrings for the contract."""

    def condition_on(self, X_new: Array, Y_new: Array) -> Self:
        """Append ``(X_new, Y_new)`` to the training set without refitting.

        Returns a new emulator that conditions on the augmented data
        with **frozen** hyperparameters and **frozen** input/output
        scalers. The cache's ``append_rows`` method does the
        backend-specific rank-update. Subclasses without cheap
        conditioning raise ``NotImplementedError`` from
        ``cache.append_rows`` itself.

        Crucially this does **not** re-run hyperparameter
        optimization; callers wanting refit semantics should call
        ``fit`` on the concatenated data instead. Loop code should
        prefer ``update_emulator(em, AppendRows(X_new, Y_new), ...)``,
        which dispatches through this method via the registered
        ``AppendRows`` handler.

        Args:
            X_new: ``(m, d)`` new training inputs in the original
                (raw) input space.
            Y_new: ``(m,)`` new training outputs in the original
                (raw) output space.
        """
        self._require_fit()
        d = self.input_shape[0]
        if X_new.ndim != 2 or X_new.shape[1] != d:
            raise ValueError(
                f"{type(self).__name__}.condition_on: expected X_new.shape="
                f"(m, {d}), got {tuple(X_new.shape)}."
            )
        if Y_new.shape != (X_new.shape[0],):
            raise ValueError(
                f"{type(self).__name__}.condition_on: expected Y_new.shape="
                f"({X_new.shape[0]},), got {tuple(Y_new.shape)}."
            )

        Xs_new = self._x_scaler.transform(X_new).astype(jnp.float64)
        Ys_new = self._y_scaler.transform(Y_new).astype(jnp.float64)
        new_cache = self._predict_cache.append_rows(Xs_new, Ys_new)
        return self._replace(_predict_cache=new_cache)
