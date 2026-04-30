r"""Likelihood-tempering schemes.

The two schemes here implement the same likelihood tempering family —
intermediate distribution :math:`\ell_t(x) = \log\pi_0(x) + \lambda_t
\log L(x)` for :math:`\lambda_t \in [0, 1]` — but place the
:math:`\lambda_t` factor at different points in the algorithm (see
``docs/tempering.md`` for the full case analysis):

- `LikelihoodTemperingViaForm`: emulator target :math:`f` is unchanged
  across rounds; the log-density form is rebuilt per round so the
  likelihood term is scaled by :math:`\lambda_t`. The emulator can be
  reused across all states with no refit (``is_invariant_target_function``
  returns True). Compatible with any base form: dispatches on form type
  (`LogLikPlusPrior`, `ForwardModel`, `Identity`) to scale the right
  part.
- `LikelihoodTemperingViaTarget`: emulator target :math:`f_t = \lambda_t
  \log L` varies with state; the form is unchanged across rounds. The
  emulator must be refit (or rescaled — see issue #4 for cheap-update
  dispatch) per round. Restricted to `LogLikPlusPrior` base forms (the
  case where :math:`f` is the log-likelihood directly).

Both schemes encode the same intermediate distribution; they differ
only in *where* the tempering enters the computation. Researchers may
prefer one over the other based on emulator-fit characteristics
(rescaling :math:`Y` may move into / out of well-conditioned regimes
for hyperparameter optimization).

The state PyTree is the inverse temperature :math:`\lambda \in [0, 1]`
(scalar). Pair with `FixedSchedule(states=(0.1, 0.5, 1.0))` or similar.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax import Array
from probpipe import log_prob
from probpipe.core._distribution_base import Distribution

from sabi.problems.forms import (
    ForwardModel,
    Identity,
    LogDensityForm,
    LogLikPlusPrior,
)
from sabi.problems.target_distribution import IntermediateTarget, TargetDistribution
from sabi.tempering.base import TemperingScheme


# ---------------------------------------------------------------------------
# Form-axis tempering: scheme + per-form-type tempered subclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LogLikPlusPriorTempered(LogDensityForm):
    r""":math:`\beta \cdot y + \log\pi_0(x)`. ``y`` is log-likelihood."""

    beta: Array

    def _call_single(self, x, y, *, prior=None):
        if prior is None:
            raise ValueError(
                "Tempered LogLikPlusPrior requires a non-None prior."
            )
        return self.beta * y + jnp.asarray(log_prob(prior, x))


@dataclass(frozen=True)
class _ForwardModelTempered(LogDensityForm):
    r""":math:`\beta \cdot \log L(x, y) + \log\pi_0(x)`."""

    beta: Array
    log_lik_from_outputs: Callable[[Array, Array], Array]

    def _call_single(self, x, y, *, prior=None):
        if prior is None:
            raise ValueError(
                "Tempered ForwardModel requires a non-None prior."
            )
        return self.beta * self.log_lik_from_outputs(x, y) + jnp.asarray(
            log_prob(prior, x)
        )


@dataclass(frozen=True)
class _IdentityTempered(LogDensityForm):
    r"""Geometric bridge: :math:`(1 - \beta) \log\pi_0(x) + \beta \cdot y`.

    The `Identity` base form treats :math:`y` as the full unnormalized
    log-posterior — so there's no built-in "likelihood vs. prior"
    decomposition to temper. The natural likelihood-tempering
    interpretation is the geometric bridge from prior to target,
    which requires the prior to be available. Errors if
    ``prior is None``.
    """

    beta: Array

    def _call_single(self, x, y, *, prior=None):
        if prior is None:
            raise ValueError(
                "Tempered Identity (geometric bridge) requires a non-None "
                "prior to define the bridge endpoints."
            )
        return (1.0 - self.beta) * jnp.asarray(log_prob(prior, x)) + self.beta * y


def _likelihood_tempered_form(
    base_form: LogDensityForm,
    beta: Array,
) -> LogDensityForm:
    """Dispatch on ``type(base_form)`` to produce the likelihood-tempered form."""
    if isinstance(base_form, LogLikPlusPrior):
        return _LogLikPlusPriorTempered(beta=beta)
    if isinstance(base_form, ForwardModel):
        return _ForwardModelTempered(
            beta=beta,
            log_lik_from_outputs=base_form.log_lik_from_outputs,
        )
    if isinstance(base_form, Identity):
        return _IdentityTempered(beta=beta)
    raise NotImplementedError(
        f"LikelihoodTemperingViaForm: no dispatch for "
        f"{type(base_form).__name__}. Supported forms: LogLikPlusPrior, "
        "ForwardModel, Identity."
    )


class LikelihoodTemperingViaForm(TemperingScheme):
    r"""Likelihood tempering with the emulator target unchanged.

    Per-round intermediate:

    .. math::

        f_{\text{state}} = f, \quad
        \phi_{\text{state}}(x, y, \pi_0) = \phi_t(\beta_t; x, y, \pi_0)

    The log-density form scales the likelihood term by :math:`\beta_t`;
    the emulator's training target is unchanged across rounds (so
    ``is_invariant_target_function`` returns True and the loop can
    reuse the same fitted emulator across states).

    Per-form-type math (dispatch internal):

    - `LogLikPlusPrior`:
      :math:`\beta_t \log L(x) + \log\pi_0(x)`.
    - `ForwardModel`:
      :math:`\beta_t \log L(x, g(x)) + \log\pi_0(x)`.
    - `Identity` (geometric bridge):
      :math:`(1 - \beta_t) \log\pi_0(x) + \beta_t \cdot y`.

    The state PyTree is the scalar :math:`\beta \in [0, 1]`.
    """

    def intermediate_target(
        self,
        base: TargetDistribution,
        state: Any,
    ) -> IntermediateTarget:
        beta = jnp.asarray(state)
        tempered_form = _likelihood_tempered_form(base.log_density_form, beta)
        return IntermediateTarget(
            name=base.name,
            input_shape=base.input_shape,
            output_shape=base.output_shape,
            target_function=base.target_function,  # f unchanged
            target_single=base.target_single,
            log_density_form=tempered_form,
            state=state,
            output_transform=_identity_output_transform,
            base_target_function=base.target_function,
            prior=base.prior,
        )

    def is_invariant_target_function(self, state_a: Any, state_b: Any) -> bool:
        return True  # emulator target is invariant under state changes

    def is_invariant_form(self, state_a: Any, state_b: Any) -> bool:
        return state_a == state_b


# ---------------------------------------------------------------------------
# Target-axis tempering
# ---------------------------------------------------------------------------


class LikelihoodTemperingViaTarget(TemperingScheme):
    r"""Likelihood tempering with the emulator target scaled by :math:`\beta`.

    Per-round intermediate:

    .. math::

        f_{\text{state}}(x) = \beta_t \cdot f(x), \quad
        \phi_{\text{state}}(x, y, \pi_0) = \phi(x, y, \pi_0)

    The form is unchanged; the emulator is fit on
    :math:`Y_t = \beta_t \cdot Y_{\text{raw}}` per round (or rescaled
    cheaply — issue #4). ``output_transform`` is :math:`(state, X,
    Y_{\text{raw}}) \mapsto state \cdot Y_{\text{raw}}`.

    Restricted to `LogLikPlusPrior` base forms — the case where the
    emulator's target :math:`f` is the log-likelihood directly. With
    `Identity` or `ForwardModel`, scaling :math:`f` by :math:`\beta`
    doesn't correspond to a likelihood-tempering interpretation
    (Identity already includes the prior; ForwardModel's :math:`f` is
    the model output, not the likelihood). Raises at
    ``intermediate_target`` time when applied to other form types.

    The state PyTree is the scalar :math:`\beta \in [0, 1]`.
    """

    def intermediate_target(
        self,
        base: TargetDistribution,
        state: Any,
    ) -> IntermediateTarget:
        if not isinstance(base.log_density_form, LogLikPlusPrior):
            raise ValueError(
                f"LikelihoodTemperingViaTarget requires the base form to be "
                f"`LogLikPlusPrior` (so f represents the log-likelihood "
                f"directly), got {type(base.log_density_form).__name__}. "
                "For tempering with `ForwardModel` / `Identity` forms, "
                "use `LikelihoodTemperingViaForm` instead."
            )
        beta = jnp.asarray(state)
        base_target_function = base.target_function
        base_target_single = base.target_single

        def tempered_target_function(X):
            return beta * base_target_function(X)

        def tempered_target_single(x):
            return beta * base_target_single(x)

        return IntermediateTarget(
            name=base.name,
            input_shape=base.input_shape,
            output_shape=base.output_shape,
            target_function=tempered_target_function,
            target_single=tempered_target_single,
            log_density_form=base.log_density_form,  # unchanged
            state=state,
            output_transform=_scale_output_transform,
            base_target_function=base_target_function,
            prior=base.prior,
        )

    def is_invariant_target_function(self, state_a: Any, state_b: Any) -> bool:
        return state_a == state_b

    def is_invariant_form(self, state_a: Any, state_b: Any) -> bool:
        return True  # form is invariant under state changes


# ---------------------------------------------------------------------------
# Output transforms
# ---------------------------------------------------------------------------


def _identity_output_transform(state: Any, X: Array, Y_raw: Array) -> Array:
    """For form-axis tempering: emulator target is unchanged."""
    return Y_raw


def _scale_output_transform(state: Any, X: Array, Y_raw: Array) -> Array:
    """For target-axis tempering: ``Y_train = state * Y_raw``."""
    return jnp.asarray(state) * Y_raw
