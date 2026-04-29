r"""Pushforward dispatch on ``(input_dist, log_density_form)``.

``pushforward_marginal(input_dist, form, X=X, prior=prior)`` is the
central entry point: given a ``Distribution[Array]`` describing a
surrogate's predictive at a batch of query points (univariate marginals,
joint over inputs, joint over outputs, or any combination per ProbPipe's
``ArrayRandomFunction`` shape table), and a ``LogDensityForm``, return a
new ``Distribution[Array]`` representing the marginal random log-density
at those query points.

For an input distribution :math:`F(x)` (the surrogate's predictive at
:math:`x`) and a form :math:`\Phi`, the marginal random log-density at
:math:`x` is

.. math::

    \tilde{p}(x) = \Phi(x, F(x); \text{prior}).

Closed-form Gaussian-affine case. For
:math:`F(x) \sim \mathcal{N}(\mu(x), \sigma^2(x))` and an affine form
:math:`\Phi(x, y; \cdot) = y + s(x)`, the pushforward is again Gaussian:

.. math::

    \tilde{p}(x) \sim \mathcal{N}\!\big(\mu(x) + s(x), \sigma^2(x)\big).

The two affine forms supported are :math:`\Phi = \mathrm{Identity}`
(with :math:`s(x) = 0`) and :math:`\Phi = \mathrm{LogLikPlusPrior}`
(with :math:`s(x) = \log p(x)`).

Dispatch (in order):

1. **Closed-form Gaussian-affine.** ``(Normal | MultivariateNormal,
   Identity | LogLikPlusPrior)`` — shift ``loc`` by :math:`s(X)`, leave
   ``scale`` / ``scale_tril`` unchanged.
2. **MC fallback.** ``isinstance(input_dist, SupportsSampling)`` — call
   the ``@workflow_function``-wrapped batched form ``_batch_form`` with
   ``ys=input_dist``. The Monte Carlo pushforward is *what ProbPipe's
   broadcasting does* when a workflow-wrapped function gets a
   Distribution in a non-Distribution-typed slot: samples
   ``ys ~ input_dist``, runs the wrapped function per sample (vmap when
   JAX-traceable, Python loop otherwise), and returns a
   ``NumericEmpiricalDistribution`` of the resulting joint log-density
   vectors. There's no separate "MC pushforward" abstraction in sabi —
   ``_batch_form`` is just a vectorized form that becomes a pushforward
   as a side effect of being broadcast.
3. **Otherwise raise.** Names the input-distribution type and form
   type; points to the partial-pushforward primitive in
   ``docs/probpipe_issues.md``.

The function is agnostic to whether the input distribution's
multivariate-ness comes from joint-inputs (``event_shape`` is the n
axis), joint-outputs (``event_shape`` is the output axis), or just
batched marginals (``batch_shape`` is the n axis). The closed-form case
adds the shift to ``loc`` regardless; the MC case lets ProbPipe
broadcasting handle the shape semantics natively.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from probpipe.core._distribution_base import Distribution
from probpipe.core.node import workflow_function
from probpipe.core.protocols import SupportsSampling
from probpipe.distributions.continuous import Normal
from probpipe.distributions.multivariate import MultivariateNormal

from sabi.problems.forms import (
    ForwardModel,
    Identity,
    LogDensityForm,
    LogLikPlusPrior,
    _joint_log_prior,
)


def pushforward_marginal(
    input_dist: Distribution,
    form: LogDensityForm,
    *,
    X: Array,
    prior: Distribution | None,
) -> Distribution:
    """Pushforward of `input_dist` (the surrogate's predictive at `X`)
    through `form`.

    Args:
        input_dist: surrogate's predictive distribution at `X`. May be a
            `Normal` (marginal mode, batch_shape=(n,)), a
            `MultivariateNormal` (joint mode, event_shape=(n,)), or any
            samplable `Distribution[Array]`.
        form: the `LogDensityForm` that defines the unnormalized
            log-posterior assembly.
        X: the query points, shape `(n,) + input_shape`. Used by forms
            that touch the prior (`LogLikPlusPrior`, `ForwardModel`) to
            evaluate the log-prior at each point.
        prior: optional prior distribution; required by forms that access
            it.

    Returns:
        A `Distribution[Array]` describing the pushforward at `X`.
    """
    # 1. Closed-form Gaussian-affine cases
    if isinstance(input_dist, (Normal, MultivariateNormal)):
        if isinstance(form, Identity):
            # log p̃(x; f) = f(x), so the marginal is the input itself.
            return input_dist
        if isinstance(form, LogLikPlusPrior):
            if prior is None:
                raise ValueError(
                    "pushforward_marginal: LogLikPlusPrior requires a "
                    "non-None prior."
                )
            # Shift the loc by the joint log-prior at each query point.
            shifts = jax.vmap(lambda x: _joint_log_prior(prior, x))(X)
            return _shift_gaussian_loc(input_dist, shifts)
        # Fall through to MC if it's some other form (e.g., ForwardModel).

    # 2. MC fallback for any samplable input. `_batch_form` is the form
    # vectorized over the n input axis, wrapped as a WorkflowFunction.
    # Passing `ys=input_dist` triggers ProbPipe's broadcasting: samples
    # are drawn from `input_dist`, the batched form is evaluated per
    # sample, and the result is a NumericEmpiricalDistribution. The MC
    # pushforward IS that broadcast — there's no sabi-defined "MC
    # pushforward" object.
    if isinstance(input_dist, SupportsSampling):
        return _batch_form(ys=input_dist, X=X, form=form, prior=prior)

    # 3. Otherwise raise
    raise NotImplementedError(
        f"pushforward_marginal: no dispatch path for input_dist of type "
        f"{type(input_dist).__name__} and form of type {type(form).__name__}. "
        "Closed-form is implemented for (Normal | MultivariateNormal, "
        "Identity | LogLikPlusPrior). Other combinations require an input "
        "distribution that supports sampling. See "
        "docs/probpipe_issues.md: 'Pushforward with partial information'."
    )


def _shift_gaussian_loc(input_dist: Distribution, shifts: Array) -> Distribution:
    """Add `shifts` to a Gaussian distribution's loc, keeping scale unchanged."""
    if isinstance(input_dist, Normal):
        return Normal(
            loc=input_dist.loc + shifts,
            scale=input_dist.scale,
            name=input_dist.name + "_shifted",
        )
    if isinstance(input_dist, MultivariateNormal):
        return MultivariateNormal(
            loc=input_dist.loc + shifts,
            scale_tril=input_dist.scale_tril,
            name=input_dist.name + "_shifted",
        )
    raise TypeError(
        f"_shift_gaussian_loc: not a Gaussian: {type(input_dist).__name__}"
    )


@workflow_function(n_broadcast_samples=64)
def _batch_form(
    *,
    ys: Array,
    X: Array,
    form: LogDensityForm,
    # Annotate as bare `Distribution` so WorkflowFunction's broadcast detector
    # classifies it as a Distribution-typed param and does NOT broadcast.
    # `Distribution | None` would not work — the Union type isn't a bare class
    # and the detector falls through to broadcasting. The actual runtime value
    # may still be `None`; the detector's `is_distribution_type(None)` check
    # short-circuits in that case.
    prior: Distribution,
) -> Array:
    """Vectorized log-density form: applies `form(x, y, prior=prior)`
    pointwise across the n input axis.

    The function itself just vmaps the form. The Monte Carlo pushforward
    arises when ProbPipe's `WorkflowFunction` broadcasting calls this
    function with `ys=input_dist` (a `Distribution` in a non-Distribution
    slot): samples are drawn from `input_dist`, this function is run per
    sample, and the result is a `NumericEmpiricalDistribution` of joint
    log-density vectors. The MC behavior is incidental machinery, not a
    property of the function.

    Args:
        ys: shape `(n,) + output_shape` — surrogate output values at the
            n query points. (When broadcast, each per-sample value passed
            to this function has this shape.)
        X: query points, shape `(n,) + input_shape`.
        form: the `LogDensityForm`.
        prior: prior distribution forwarded to forms that touch the prior;
            may be `None`. Annotated bare `Distribution` to keep
            WorkflowFunction from broadcasting on this slot.

    Returns:
        Shape `(n,)` — the joint log-density evaluated pointwise across
        the n input points for the given `ys`.
    """
    return jax.vmap(lambda x, y: form(x, y, prior=prior))(X, ys)
