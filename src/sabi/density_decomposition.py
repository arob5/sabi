r"""``DensityDecomposition`` and concrete subclasses.

A ``DensityDecomposition`` is a ProbPipe :class:`Distribution` that
encodes the algorithm-side decomposition of a target's unnormalized
log-density into

.. math::

    \log \tilde p(x) = \mathrm{link}\!\big(\mathrm{target\_map}(x)\big) + \mathrm{shift}(x).

The first term — ``link(target_map(x))`` — is the **log-prob residual**:
the contribution to the unnormalized log-density attributable to the
emulator's output ``y = target_map(x)``, after the ``link`` is applied.
The emulator's predictive distribution flows through this term via
:meth:`pushforward`. The second term — ``shift(x)`` — is the
deterministic ``x``-dependent additive contribution (typically
``LogProb(prior)``); it carries no emulator uncertainty.

There can be many ``DensityDecomposition`` instances for a single
``TargetDistribution`` — the choice of what to emulate is an
algorithmic decision, not a property of the target. See
``docs/density_decomposition.md`` for the design.

Class hierarchy
---------------

- :class:`DensityDecomposition` — abstract base. Subclasses define
  ``target_map`` and ``link`` (and may override ``shift``).
- :class:`LogProbTermTarget` — abstract subclass with
  ``link = Identity``. The emulator emits **one term** in a
  ``link(·) + shift(x)`` sum: the full unnormalized log-prob when
  ``prior=None`` (no shift), or one term plus the prior shift when
  ``prior=π``.
- :class:`LogProbTarget` — concrete leaf. Trivial decomposition that
  wraps a :class:`TargetDistribution`'s analytical
  ``_unnormalized_log_prob``: ``link = Identity``, ``shift = None``,
  and ``target_map`` delegates to the wrapped target. The canonical
  construction idiom for benchmarks.
- :class:`GaussianForwardModelTarget` — abstract subclass for
  ``π(x) · N(obs | f(x), C)``. ``link = GaussianLogLik(obs, cov)``,
  ``shift = LogProb(prior)``. Subclasses define the forward model
  ``f`` as ``target_map``.

Vectorization contract
----------------------

``DensityDecomposition`` is a ProbPipe ``NumericRecordDistribution`` —
it follows ProbPipe's vectorization contract for distributions. The
public surface is the ProbPipe ops (``unnormalized_log_prob``, etc.);
``target_map`` is also vectorized
(``batch_shape + input_shape -> batch_shape + output_shape``). See
``docs/notation.md`` for the contract.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

import jax.numpy as jnp
from jax import Array
from probpipe import unnormalized_log_prob as pp_unnormalized_log_prob
from probpipe.core._distribution_base import Distribution
from probpipe.core._numeric_record_distribution import NumericRecordDistribution
from probpipe.core.constraints import Constraint, real

from sabi.maps import (
    Affine,
    GaussianLogLik,
    Identity,
    LogProb,
    Map,
    pushforward as pushforward_op,
)

if TYPE_CHECKING:
    from sabi.target_distribution import TargetDistribution


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class DensityDecomposition(NumericRecordDistribution):
    """Abstract base: a ProbPipe distribution whose log-density decomposes.

    Subclasses must define :meth:`target_map` and the :attr:`link`
    property; :attr:`shift` defaults to ``None`` (no shift). The base
    auto-derives :meth:`_unnormalized_log_prob` as
    ``link(target_map(x)) + shift(x)``.

    Distribution semantics:

    - ``event_shape == input_shape`` (the parameter-space shape of one
      ``x``).
    - ``support`` is the :class:`Constraint` on ``x``.
    - ``output_shape`` is the shape of one ``y = target_map(x)``.
    - ``output_constraint`` is the :class:`Constraint` on ``y``
      (mostly metadata; default ``real``).

    Args:
        name: ProbPipe distribution name.
        input_shape: shape of one parameter-space point.
        support: ``Constraint`` on ``x``.
    """

    _sampling_cost: ClassVar[str] = "high"
    _preferred_orchestration: ClassVar[str | None] = None

    def __init__(self, *, name: str, input_shape: tuple[int, ...], support: Constraint):
        if support is None:
            raise ValueError(
                f"{type(self).__name__} requires a non-None `support` "
                "(`Constraint` on the parameter space)."
            )
        self._input_shape = tuple(input_shape)
        self._support = support
        super().__init__(name=name)

    # ------------------------------------------------------------------------
    # Distribution metadata
    # ------------------------------------------------------------------------

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shape

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._input_shape

    @property
    def support(self) -> Constraint:
        return self._support

    @property
    @abstractmethod
    def output_shape(self) -> tuple[int, ...]:
        """Shape of one ``y = target_map(x)``."""

    @property
    def output_constraint(self) -> Constraint:
        """Constraint on ``y``. Mostly metadata; default ``real``."""
        return real

    # ------------------------------------------------------------------------
    # Subclass hooks: target_map, link, shift
    # ------------------------------------------------------------------------

    @abstractmethod
    def target_map(self, x: Array) -> Array:
        """Vectorized: ``batch_shape + input_shape -> batch_shape + output_shape``.

        What the emulator approximates. Subclasses define this to
        return the emulator-target value at ``x`` (a log-density, a
        log-likelihood, a forward-model output, etc., depending on the
        decomposition shape).
        """

    @property
    @abstractmethod
    def link(self) -> Map:
        """The link :class:`Map`: ``y -> log-density-residual``."""

    @property
    def shift(self) -> Map | None:
        """The shift :class:`Map`: ``x -> additive shift``. ``None`` means no shift."""
        return None

    # ------------------------------------------------------------------------
    # Auto-derived: _unnormalized_log_prob, pushforward
    # ------------------------------------------------------------------------

    def _unnormalized_log_prob(self, x: Array) -> Array:
        r"""Unnormalized log-density at ``x``: ``link(target_map(x)) + shift(x)``.

        Decomposes as

        .. code-block:: text

            log p̃(x) = link(target_map(x)) + shift(x)
                       ╰─────────╮─────────╯   ╰────╮────╯
                       log-prob residual            deterministic shift

        The **log-prob residual** is the contribution to the
        unnormalized log-density attributable to the emulator's output
        ``y = target_map(x)``, after the ``link`` is applied. It is
        the term the emulator's predictive distribution flows through
        under :meth:`pushforward`. The **shift** is the deterministic
        ``x``-dependent additive term (typically ``LogProb(prior)``);
        it carries no emulator uncertainty.

        Subclasses implement the pieces (``target_map``, ``link``,
        ``shift``); the composition is fixed here.
        """
        log_prob_residual = self.link(self.target_map(x))
        if self.shift is None:
            return log_prob_residual
        return log_prob_residual + self.shift(x)

    def pushforward(self, x: Array, y_dist: Distribution) -> Distribution:
        r"""Push an emulator predictive ``y_dist`` at ``x`` through the decomposition.

        Constructs the per-``x`` map
        ``Affine(slope=1, intercept=shift(x)) @ link`` and dispatches
        through :func:`sabi.maps.pushforward`. When ``shift`` is
        ``None``, the shift composition collapses and we dispatch on
        ``link`` directly.

        Args:
            x: query point(s); single ``input_shape`` or batched
                ``(n,) + input_shape``. ``shift(x)`` provides the
                per-point intercept when ``shift`` is set.
            y_dist: emulator predictive at ``x`` (e.g. ``Normal`` with
                ``batch_shape=(n,)`` for marginal mode, or
                ``MultivariateNormal`` with ``event_shape=(n,)`` for
                joint mode).
        """
        if self.shift is None:
            return pushforward_op(self.link, y_dist)
        per_x_map = Affine(slope=jnp.asarray(1.0), intercept=self.shift(x)) @ self.link
        return pushforward_op(per_x_map, y_dist)


# ---------------------------------------------------------------------------
# LogProbTermTarget — link = Identity; emulator emits one term
# ---------------------------------------------------------------------------


class LogProbTermTarget(DensityDecomposition):
    """Emulator emits one term in a ``link(·) + shift(x)`` sum (``link = Identity``).

    With ``prior=None``, the emulator emits the full unnormalized
    log-prob (no shift). With ``prior=π``, the emulator emits one term
    (typically a log-likelihood) and ``shift = LogProb(π)`` adds the
    log-prior.

    Subclasses must define :meth:`target_map`. Override
    :attr:`output_shape` if the emulator output isn't scalar (default
    ``()``).

    Args:
        name: ProbPipe distribution name.
        input_shape: shape of one ``x``.
        support: ``Constraint`` on ``x``.
        prior: optional :class:`Distribution`; when set, ``shift =
            LogProb(prior)``. Default ``None`` (no shift).
    """

    def __init__(
        self,
        *,
        name: str,
        input_shape: tuple[int, ...],
        support: Constraint,
        prior: Distribution | None = None,
    ):
        self._prior = prior
        super().__init__(name=name, input_shape=input_shape, support=support)

    @property
    def prior(self) -> Distribution | None:
        return self._prior

    @property
    def link(self) -> Map:
        return Identity()

    @property
    def shift(self) -> Map | None:
        return LogProb(self._prior) if self._prior is not None else None

    @property
    def output_shape(self) -> tuple[int, ...]:
        return ()


# ---------------------------------------------------------------------------
# LogProbTarget — concrete: wrap a TargetDistribution's analytical density
# ---------------------------------------------------------------------------


class LogProbTarget(LogProbTermTarget):
    """Trivial decomposition: ``target_map`` = wrapped target's analytical density.

    The canonical idiom for benchmarks. Wraps a :class:`TargetDistribution`
    that exposes an analytical ``_unnormalized_log_prob`` and uses it as
    the emulator target. ``link = Identity``, ``shift = None``: the
    emulator approximates the full unnormalized log-prob directly.

    Args:
        target: the :class:`TargetDistribution` whose analytical
            ``_unnormalized_log_prob`` becomes ``target_map``.
        name: optional ProbPipe distribution name.
    """

    def __init__(self, target: "TargetDistribution", *, name: str | None = None):
        self._target = target
        super().__init__(
            name=name or f"log_prob_{target.name}",
            input_shape=target.input_shape,
            support=target.support,
            prior=None,
        )

    @property
    def target(self) -> "TargetDistribution":
        return self._target

    def target_map(self, x: Array) -> Array:
        # `pp_unnormalized_log_prob` returns a NumericRecord; jnp.asarray
        # strips the wrapper. See docs/probpipe_issues.md
        # "unnormalized_log_prob op returns wrapper, not bare array".
        return jnp.asarray(pp_unnormalized_log_prob(self._target, x))


# ---------------------------------------------------------------------------
# GaussianForwardModelTarget — link = GaussianLogLik; shift = LogProb(prior)
# ---------------------------------------------------------------------------


class GaussianForwardModelTarget(DensityDecomposition):
    r"""Density is :math:`\pi(x) \cdot \mathcal{N}(\mathrm{obs} \mid f(x), C)`.

    The emulator approximates the forward model ``f``; the link applies
    the Gaussian observation likelihood; the shift adds the log-prior.

    Subclasses define :meth:`target_map` (the forward model ``f``).

    Args:
        name: ProbPipe distribution name.
        input_shape: shape of one ``x``.
        support: ``Constraint`` on ``x``.
        obs: observation array, shape ``(d,)``.
        cov: ``d × d`` symmetric positive-definite covariance.
        prior: ``Distribution`` over the parameter space.
    """

    def __init__(
        self,
        *,
        name: str,
        input_shape: tuple[int, ...],
        support: Constraint,
        obs: Array,
        cov: Array,
        prior: Distribution,
    ):
        self._obs = jnp.asarray(obs)
        self._cov = jnp.asarray(cov)
        self._prior = prior
        super().__init__(name=name, input_shape=input_shape, support=support)

    @property
    def obs(self) -> Array:
        return self._obs

    @property
    def cov(self) -> Array:
        return self._cov

    @property
    def prior(self) -> Distribution:
        return self._prior

    @property
    def link(self) -> Map:
        return GaussianLogLik(obs=self._obs, cov=self._cov)

    @property
    def shift(self) -> Map:
        return LogProb(self._prior)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return tuple(self._obs.shape)


# ---------------------------------------------------------------------------
# Self-consistency check
# ---------------------------------------------------------------------------


def is_consistent_with(
    decomposition: DensityDecomposition,
    target: "TargetDistribution",
    *,
    x_test: Array,
    atol: float = 1e-6,
    strict: bool = True,
) -> bool:
    r"""Verify that ``decomposition`` reconstructs ``target``'s analytical density.

    Both ``decomposition`` and ``target`` are evaluated through
    ProbPipe's :func:`unnormalized_log_prob` op (vectorized; no manual
    ``vmap``). Comparison is pointwise (``strict=True``) or up to an
    additive constant (``strict=False``).

    Args:
        decomposition: candidate :class:`DensityDecomposition` to validate.
        target: a :class:`TargetDistribution` carrying an analytical
            ``_unnormalized_log_prob``. If the target does not implement
            an analytical density, ProbPipe's op raises ``TypeError``
            (``does not support unnormalized_log_prob``); the
            consistency check is undefined and the error propagates.
        x_test: shape ``(n,) + input_shape``. Requires ``n >= 2`` when
            ``strict=False``.
        atol: numerical tolerance.
        strict: when ``True`` (default), require pointwise equality
            within ``atol``. When ``False``, allow an additive
            constant — differences across rows must match within ``atol``.
    """
    x_test = jnp.asarray(x_test)
    if x_test.ndim < 2:
        raise ValueError(
            "is_consistent_with: x_test must be at least rank-2 "
            f"((n,) + input_shape); got rank {x_test.ndim} "
            f"(shape {tuple(x_test.shape)})."
        )
    if not strict and x_test.shape[0] < 2:
        raise ValueError(
            "is_consistent_with(strict=False): need n >= 2 rows in "
            f"x_test to compare differences; got n={x_test.shape[0]}."
        )

    target_log_prob = jnp.asarray(pp_unnormalized_log_prob(target, x_test))
    decomposition_log_prob = jnp.asarray(pp_unnormalized_log_prob(decomposition, x_test))

    if strict:
        return bool(
            jnp.all(
                jnp.abs(decomposition_log_prob - target_log_prob) <= atol
            )
        )
    diff = decomposition_log_prob - target_log_prob
    return bool(jnp.all(jnp.abs(diff - diff[0]) <= atol))
