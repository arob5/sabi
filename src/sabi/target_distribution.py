r"""`TargetDistribution` — an unnormalized target distribution.

A `TargetDistribution` is a ProbPipe `Distribution[Array]` representing
an unnormalized log-density on the parameter space. Its
`_unnormalized_log_prob` is the composition of a target function `f`
with a `LogDensityForm` `phi`:

.. math::

    \log p(x) = \phi(x, f(x); \text{prior}).

Every benchmark inference problem in sabi is fundamentally defined by
its `TargetDistribution`. `Problem` (in `sabi.problems.base`) bundles a
target distribution with benchmark-suite metadata (reference posterior,
name, etc.); the mathematical content lives here.

The same class represents the **base / final** target and (via
`IntermediateTarget`) intermediate targets a `TemperingScheme`
produces along the way — they have the same structure.

Role of ``prior``
------------------

The ``prior`` field is **required**. It plays two roles independent
of whether the prior also forms part of the target distribution:

1. **Defining the support of the parameter space.** ``support`` is
   not a separate field — it is exposed as a property delegating to
   ``prior.support``. The prior may have unbounded support (e.g., a
   `Normal` over R^d) when bounded support isn't desired.
2. **Sampling for initial design, candidate sets, and prior-sampling
   acquisitions.** The prior acts as the "design distribution"
   regardless of whether it's part of the target.

Whether the prior also enters the unnormalized target is the problem
builder's choice via ``log_density_form``: `LogLikPlusPrior` builds it
in (target = log_lik + log_prior); `Identity` doesn't. In Bayesian
settings the algorithmic ``prior`` may be a *truncated* version of
the modeling prior — e.g., a Gaussian Bayesian prior paired with a
uniform-box algorithmic prior used purely to bound sampling.

Shape contract
--------------

The ``prior`` is a multivariate-event Distribution: ``event_shape ==
input_shape``, ``batch_shape == ()``. ``log_prob(prior, x)`` for ``x``
of shape ``input_shape`` returns a scalar — exactly what the form
needs. For per-dim distributions (e.g., a per-dim
``Uniform(low_array, high_array)``), wrap with
`sabi._probpipe_compat.independent_uniform` to re-interpret the batch
dims as event dims.

Distribution interface
----------------------

- `SupportsUnnormalizedLogProb`: `_unnormalized_log_prob(x)` evaluates
  `phi(x, f(x), prior)`. Single-point ``value`` (shape ``input_shape``)
  is the primary contract; ProbPipe MCMC dispatches via this path.
  Batched input (shape ``(n,) + input_shape``) is also supported via
  shape detection. Calls `f` at fresh inputs — when the loop has
  cached `Y_raw`, it should use that directly rather than re-evaluating
  via this method.
- `condition_on(target_distribution, ...)` works directly: ProbPipe's
  registry auto-dispatches MCMC since the distribution satisfies
  `SupportsUnnormalizedLogProb`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from jax import Array
from probpipe.core._distribution_base import Distribution
from probpipe.core._numeric_record_distribution import NumericRecordDistribution
from probpipe.core.constraints import Constraint  # noqa: F401  (re-exported via support property)

from sabi.problems.forms import LogDensityForm


class TargetDistribution(NumericRecordDistribution):
    """Unnormalized target distribution: ``phi(x, f(x); prior)``.

    Args:
        name: ProbPipe distribution name.
        input_shape: shape of one parameter-space point.
        output_shape: shape of one target-function output.
        target_function: batched target ``(n,) + input_shape -> (n,) + output_shape``.
        target_single: single-point target ``input_shape -> output_shape``.
        log_density_form: composes ``(x, y, prior)`` into a scalar
            unnormalized log-density.
        prior: required `Distribution` over the parameter space. Must
            be multivariate-event (``event_shape == input_shape``).
            Defines the support and acts as the design distribution
            for sampling. May or may not also be part of the target
            distribution (depends on `log_density_form`). See module
            docstring for the full role description.
    """

    _sampling_cost: ClassVar[str] = "high"
    _preferred_orchestration: ClassVar[str | None] = None

    def __init__(
        self,
        *,
        name: str,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        target_function: Callable[[Array], Array],
        target_single: Callable[[Array], Array],
        log_density_form: LogDensityForm,
        prior: Distribution,
    ):
        if prior is None:
            raise ValueError(
                "TargetDistribution requires a non-None `prior`. The prior "
                "defines the support of the parameter space and acts as "
                "the design distribution. Use a Distribution with "
                "unbounded support (e.g., a Normal) if bounded support "
                "isn't needed."
            )
        self._input_shape = tuple(input_shape)
        self._output_shape = tuple(output_shape)
        self._target_function = target_function
        self._target_single = target_single
        self._log_density_form = log_density_form
        self._prior = prior
        super().__init__(name=name)

    # ------------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------------

    @classmethod
    def from_target_single(
        cls,
        *,
        target_single: Callable[[Array], Array],
        **kwargs: Any,
    ) -> TargetDistribution:
        """Build a `TargetDistribution` from a single-point ``target_single``.

        Wraps the input with `jax.vmap` to produce ``target_function``;
        most benchmarks have a natural single-point implementation and
        this helper avoids requiring callers to write the vmap by hand.
        """
        target_function = jax.vmap(target_single)
        return cls(
            target_function=target_function,
            target_single=target_single,
            **kwargs,
        )

    # ------------------------------------------------------------------------
    # Public accessors (read-only views of the underlying state)
    # ------------------------------------------------------------------------

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._output_shape

    @property
    def target_function(self) -> Callable[[Array], Array]:
        return self._target_function

    @property
    def target_single(self) -> Callable[[Array], Array]:
        return self._target_single

    @property
    def log_density_form(self) -> LogDensityForm:
        return self._log_density_form

    @property
    def prior(self) -> Distribution:
        return self._prior

    @property
    def support(self) -> Constraint:
        """Support of the parameter space, derived from ``prior.support``.

        ``prior`` is required, so ``support`` is always defined (may be
        unbounded — e.g., a Normal prior gives ``real`` support).
        """
        return self._prior.support

    # ------------------------------------------------------------------------
    # Distribution interface
    # ------------------------------------------------------------------------

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._input_shape

    def _unnormalized_log_prob(self, value: Array) -> Array:
        """Unnormalized log-density at ``value``.

        `value` may be a single point of shape ``input_shape`` (the
        primary contract; ProbPipe MCMC dispatches via this path) or a
        batch of shape ``(n,) + input_shape``. Detected by shape;
        single-point uses ``target_single`` + form's per-point hook,
        batched uses ``target_function`` + the form's batched call.
        """
        x = jnp.asarray(value)
        if x.shape == self._input_shape:
            y = self._target_single(x)
            # Use the form's per-point hook to avoid vmap overhead on a
            # single-point call.
            return self._log_density_form._call_single(x, y, prior=self._prior)
        # Batched (n,) + input_shape: use the form's public batched call.
        y = self._target_function(x)
        return self._log_density_form(x, y, prior=self._prior)


class IntermediateTarget(TargetDistribution):
    """A `TargetDistribution` that's one step of a tempering scheme.

    Identical structure to the base `TargetDistribution` (target function
    `f_state`, log-density form `phi_state`, prior, support), plus
    metadata that the algorithm loop uses to derive the emulator's
    training data efficiently:

    - ``state``: the tempering state that produced this intermediate.
    - ``output_transform``: a callable
      ``(state, X, Y_raw) -> Y_train`` that converts cached raw
      evaluations of the *base* target function ``f`` to training
      values for ``f_state``. The loop uses this instead of evaluating
      ``f_state`` directly when it has cached ``Y_raw``.
    - ``base_target_function``: the un-tempered base target ``f`` (for
      reference; the loop typically already has it via the base
      `TargetDistribution`).

    The relationship between ``target_function`` (which is ``f_state``)
    and ``output_transform`` is:

    .. code-block:: python

        f_state(x) == output_transform(state, x, base_target_function(x))

    For Case 1 (no tempering): ``output_transform`` is identity and
    ``f_state == f``. For Case 2 (likelihood tempering via target):
    ``output_transform(state, X, Y_raw) = state * Y_raw`` (state is
    the inverse temperature beta).

    See `docs/tempering.md` for the conceptual layering.
    """

    def __init__(
        self,
        *,
        name: str,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        target_function: Callable[[Array], Array],
        target_single: Callable[[Array], Array],
        log_density_form: LogDensityForm,
        state: Any,
        output_transform: Callable[[Any, Array, Array], Array],
        base_target_function: Callable[[Array], Array],
        prior: Distribution,
    ):
        self._state = state
        self._output_transform = output_transform
        self._base_target_function = base_target_function
        super().__init__(
            name=name,
            input_shape=input_shape,
            output_shape=output_shape,
            target_function=target_function,
            target_single=target_single,
            log_density_form=log_density_form,
            prior=prior,
        )

    @property
    def state(self) -> Any:
        return self._state

    @property
    def output_transform(self) -> Callable[[Any, Array, Array], Array]:
        return self._output_transform

    @property
    def base_target_function(self) -> Callable[[Array], Array]:
        return self._base_target_function
