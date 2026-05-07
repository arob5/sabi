r"""``pushforward(map, dist)`` — multiple-dispatch pushforward op.

Dispatches on ``(type(map), type(dist))``. Closed-form registrations
land per pair via :func:`pushforward.register`; an MC fallback covers
any ``(Map, SupportsSampling)`` not otherwise registered.

For an input distribution :math:`F` and a Map :math:`f`, the pushforward
is the distribution of :math:`f(Z)` where :math:`Z \sim F`. Closed-form
entries (per ``docs/link_functions.md`` §5.1):

- ``(Identity, *)`` — unchanged.
- ``(Constant(c), *)`` — :class:`Dirac(c) <sabi.maps._dirac.Dirac>`.
- ``(Affine, Normal)`` —
  :math:`\mathcal{N}(\mathrm{slope}\cdot\mu + \mathrm{intercept},\
  \lvert\mathrm{slope}\rvert\cdot\sigma)`.
- ``(Affine, MultivariateNormal)`` —
  :math:`\mathcal{N}(\mathrm{slope}\cdot\mu + \mathrm{intercept},\
  \lvert\mathrm{slope}\rvert\cdot L)` where :math:`L` is ``scale_tril``.
  (Scalar slope; sign carried as absolute value since the joint
  covariance is the same either way.)
- ``(Affine, NumericEmpiricalDistribution)`` — elementwise affine on
  stored samples. Required for the ``Compose`` recursion through MC
  intermediates (e.g. ``Affine ∘ LogSoftplus`` through an MVN).
- ``(Exp, Normal)`` — :math:`\mathrm{LogNormal}(\mu, \sigma)`.
- ``(Log, LogNormal)`` — :math:`\mathcal{N}(\mu, \sigma)`.
- ``(Compose, *)`` — recurse:
  ``pushforward(f, pushforward(g, dist))``.

Anything else with a samplable input falls to the MC path: samples are
drawn from the input distribution, the Map is applied per sample, and a
:class:`~probpipe.core._empirical.NumericEmpiricalDistribution` of
outputs is returned. This is the same machinery used by today's
:func:`sabi.surrogate._pushforward._batch_form`, generalised from
"form-specific MC" to "any Map".

The dispatch table walks both type MROs and returns the most-derived
match. New closed-form entries land via:

.. code-block:: python

    @pushforward.register(SomeMap, SomeDist)
    def _pf_some(map_: SomeMap, dist: SomeDist) -> Distribution:
        ...

The MC fallback's sample budget is fixed at ``n_broadcast_samples=64``,
matching the precedent in
:func:`sabi.surrogate._pushforward._batch_form`. Exposing it on the
``_mc_pushforward`` boundary so downstream callers can dial it is left
for the link-function / surrogate integration that follows this PR.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array
from probpipe.core._distribution_base import Distribution
from probpipe.core._empirical import NumericEmpiricalDistribution
from probpipe.core.node import workflow_function
from probpipe.core.protocols import SupportsSampling
from probpipe.distributions.continuous import LogNormal, Normal
from probpipe.distributions.multivariate import MultivariateNormal

from sabi.maps._affine import Affine, Constant, Identity
from sabi.maps._base import Compose, Map
from sabi.maps._dirac import Dirac
from sabi.maps._elementwise import Exp, Log

_Handler = Callable[[Map, Distribution], Distribution]


class _PushforwardDispatch:
    """Multiple-dispatch op keyed on ``(type(map), type(dist))``.

    Lookup walks ``type(map).__mro__ × type(dist).__mro__`` lexicographically
    (Map MRO outer, Distribution MRO inner) and returns the first
    registered handler. The Map MRO drives outer ordering because
    closed-form behavior is more strongly tied to the Map type than to
    the input distribution type.
    """

    def __init__(self) -> None:
        self._registry: dict[tuple[type, type], _Handler] = {}

    def register(
        self, map_cls: type[Map], dist_cls: type[Distribution]
    ) -> Callable[[_Handler], _Handler]:
        """Decorator: register a closed-form handler for the given
        ``(map_cls, dist_cls)`` pair.
        """
        def decorator(handler: _Handler) -> _Handler:
            self._registry[(map_cls, dist_cls)] = handler
            return handler

        return decorator

    def _lookup(self, map_cls: type, dist_cls: type) -> _Handler | None:
        for mc in map_cls.__mro__:
            for dc in dist_cls.__mro__:
                handler = self._registry.get((mc, dc))
                if handler is not None:
                    return handler
        return None

    def __call__(self, map_: Map, dist: Distribution) -> Distribution:
        """Dispatch to the registered closed-form handler, or fall back
        to MC for any ``(Map, SupportsSampling)`` input.

        Args:
            map_: the :class:`Map` to push samples through.
            dist: the input distribution. For closed-form paths, must
                match a registered ``(map_cls, dist_cls)`` pair; for the
                MC fallback, must implement ``SupportsSampling``.

        Returns:
            The pushforward distribution.
        """
        handler = self._lookup(type(map_), type(dist))
        if handler is not None:
            return handler(map_, dist)
        if isinstance(dist, SupportsSampling):
            return _mc_pushforward(map_=map_, dist=dist)
        raise NotImplementedError(
            f"pushforward: no dispatch path for map={type(map_).__name__} "
            f"and dist={type(dist).__name__}. Closed-form entries: see "
            f"sabi.maps._pushforward; MC fallback requires SupportsSampling."
        )


pushforward = _PushforwardDispatch()


# --------------------------------------------------------------------- #
# Closed-form registrations
# --------------------------------------------------------------------- #


@pushforward.register(Identity, Distribution)
def _pf_identity(map_: Identity, dist: Distribution) -> Distribution:  # noqa: ARG001
    return dist


@pushforward.register(Constant, Distribution)
def _pf_constant(map_: Constant, dist: Distribution) -> Distribution:  # noqa: ARG001
    return Dirac(map_.c)


@pushforward.register(Affine, Normal)
def _pf_affine_normal(map_: Affine, dist: Normal) -> Normal:
    return Normal(
        loc=map_.slope * dist.loc + map_.intercept,
        scale=jnp.abs(map_.slope) * dist.scale,
        name=dist.name,
    )


@pushforward.register(Affine, MultivariateNormal)
def _pf_affine_mvn(map_: Affine, dist: MultivariateNormal) -> MultivariateNormal:
    return MultivariateNormal(
        loc=map_.slope * dist.loc + map_.intercept,
        scale_tril=jnp.abs(map_.slope) * dist.scale_tril,
        name=dist.name,
    )


@pushforward.register(Affine, NumericEmpiricalDistribution)
def _pf_affine_empirical(
    map_: Affine, dist: NumericEmpiricalDistribution
) -> NumericEmpiricalDistribution:
    """Elementwise affine on stored samples; weights and event_shape unchanged."""
    return NumericEmpiricalDistribution(
        samples=map_.slope * dist.samples + map_.intercept,
        log_weights=dist.log_weights,
        name=dist.name,
    )


@pushforward.register(Exp, Normal)
def _pf_exp_normal(map_: Exp, dist: Normal) -> LogNormal:  # noqa: ARG001
    return LogNormal(loc=dist.loc, scale=dist.scale, name=dist.name)


@pushforward.register(Log, LogNormal)
def _pf_log_lognormal(map_: Log, dist: LogNormal) -> Normal:  # noqa: ARG001
    return Normal(loc=dist.loc, scale=dist.scale, name=dist.name)


@pushforward.register(Compose, Distribution)
def _pf_compose(map_: Compose, dist: Distribution) -> Distribution:
    """``pushforward(f @ g, dist) = pushforward(f, pushforward(g, dist))``."""
    return pushforward(map_.f, pushforward(map_.g, dist))


# --------------------------------------------------------------------- #
# MC fallback
# --------------------------------------------------------------------- #


@workflow_function(n_broadcast_samples=64)
def _apply_map(
    *,
    z: Array,
    map_: Map,
) -> Array:
    """Apply ``map_`` to ``z``.

    The Map's shape contract says ``z.shape == batch_shape +
    event_shape_in`` produces ``batch_shape + event_shape_out``, so
    no explicit ``jax.vmap`` is needed here — the per-sample shape
    that ProbPipe passes is whatever the input distribution's event
    sample looks like, which is already a valid Map input.

    The MC pushforward arises when ProbPipe's
    :class:`~probpipe.core.node.WorkflowFunction` broadcasting calls
    this with ``z=input_dist`` (a ``Distribution`` in a non-``Distribution``
    slot): samples are drawn from ``input_dist``, the function is run
    per sample, and the result is a
    :class:`~probpipe.core._empirical.NumericEmpiricalDistribution` of
    Map outputs.
    """
    return map_(z)


def _mc_pushforward(*, map_: Map, dist: Distribution) -> Distribution:
    """MC pushforward via ProbPipe's
    :class:`~probpipe.core.node.WorkflowFunction` broadcasting.

    Passing ``z=dist`` triggers per-sample evaluation of ``map_`` over
    samples drawn from ``dist``; the result is a
    :class:`~probpipe.core._empirical.NumericEmpiricalDistribution`.
    """
    return _apply_map(z=dist, map_=map_)
