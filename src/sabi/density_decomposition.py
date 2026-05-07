r"""``DensityDecomposition`` — emulator-side decomposition of an unnormalized log-density.

Encodes the relationship between what the emulator approximates
(``target_single``) and the unnormalized log target density. The
log-density at ``(x, y)`` is

.. math::

    \log \tilde p(x \mid y) = \mathrm{link}(y) + \mathrm{shift}(x).

The emulator emits ``y = target_single(x)``, the ``link`` map turns
``y`` into a log-density-residual, and the ``shift`` map adds an
``x``-dependent term (typically a log-prior). Both ``link`` and
``shift`` are :class:`sabi.maps.Map` instances; the
:func:`sabi.maps.pushforward` op handles emulator-predictive
pushforward through the composition.

There can be many ``DensityDecomposition`` instances for a single
``TargetDistribution``: the choice of what to emulate (full
log-density, log-likelihood, or forward-model output) is an
algorithmic decision, not a property of the target. See
``docs/density_decomposition.md`` for the design.

The four classmethod helpers cover the common construction patterns:

- :meth:`identity_from_target` — wrap a ``TargetDistribution``'s
  analytical ``_unnormalized_log_prob`` as the emulator target,
  ``link = Identity``, ``shift = Constant(0)``.
- :meth:`likelihood_with_prior` — emulator emits log-likelihood,
  shift adds the modeling-prior log-density.
- :meth:`forward_model` — emulator emits a forward-model output,
  link applies the observation likelihood, shift adds the
  modeling-prior log-density.

The free function :func:`is_consistent_with` regresses a
``DensityDecomposition`` against a ``TargetDistribution``'s analytical
density at a handful of test points — used by ``test_benchmarks`` to
guarantee benchmark factories produce internally-consistent
``(target, decomposition)`` pairs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array
from probpipe import unnormalized_log_prob as pp_unnormalized_log_prob
from probpipe.core._distribution_base import Distribution
from probpipe.core.constraints import Constraint, real

from sabi.maps import (
    Affine,
    Constant,
    GaussianLogLik,
    Identity,
    LogProb,
    Map,
    pushforward,
)
from sabi.maps._base import Map as _Map

if TYPE_CHECKING:
    from sabi.target_distribution import TargetDistribution


@dataclass(frozen=True)
class ScalarConstant(_Map):
    """Constant scalar shift that ignores the structure of its input.

    Distinct from :class:`sabi.maps.Constant`. ``Constant`` declares
    ``event_shape_in = ()`` and broadcasts its value across the *batch
    dims* of its input — useful for "broadcast a constant across a
    batch of scalars." For ``DensityDecomposition.shift``, the
    intended semantics is "ignore a vector ``x`` of shape
    ``input_shape`` and return a single scalar shift" (so that
    ``link(y) + shift(x)`` produces a scalar log-density per the math
    in ``docs/density_decomposition.md`` §3.2). ``ScalarConstant``
    implements that: for any input rank, it returns a scalar (or a
    per-batch-element scalar) of value ``c``.

    Used as the default ``DensityDecomposition.shift`` (with ``c=0.0``).
    Users who want broadcast-Constant semantics should pass
    ``shift=Constant(...)`` explicitly; users who want a scalar shift
    other than zero can pass ``shift=ScalarConstant(c=value)``.
    """

    c: Array = field(default_factory=lambda: jnp.asarray(0.0))
    event_shape_in: tuple[int, ...] = ()
    event_shape_out: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "c", jnp.asarray(self.c))

    def __call__(self, z: Array) -> Array:
        z_arr = jnp.asarray(z)
        c = jnp.asarray(self.c)
        # Single-point: z.shape == input_shape — return c (scalar).
        # Batched: z.shape == (n,) + input_shape — return (n,) filled with c.
        if z_arr.ndim <= 1:
            return c
        return jnp.broadcast_to(c, (z_arr.shape[0],))


@dataclass(frozen=True)
class DensityDecomposition:
    r"""Emulator-side decomposition of an unnormalized log-density.

    The unnormalized log-density at ``(x, y)`` is
    :math:`\log \tilde p = \mathrm{link}(y) + \mathrm{shift}(x)`. The
    emulator approximates ``target_single``; ``target_map`` is the
    batched view, derived once via ``jax.vmap``.

    Args:
        target_single: single-point target callable
            ``input_shape -> output_shape``. What the emulator fits.
        output_shape: shape of one ``y``.
        link: :class:`Map` ``y -> log-density-residual``. ``Identity`` for
            log-density / log-likelihood emulation; ``GaussianLogLik`` /
            ``LogSoftplus`` / ``LogSquare`` for non-default links.
        shift: :class:`Map` ``x -> additive shift``. Defaults to
            ``Constant(0.0)``. ``LogProb(modeling_prior)`` adds a
            modeling-prior log-density.
        constraint: :class:`Constraint` on the emulator output ``y``.
            Mostly metadata — acquisitions and pushforward primitives may
            read it; the decomposition itself does not enforce it.

    Notes:
        ``target_map`` is derived once from ``target_single`` via
        ``jax.vmap`` and cached on the instance via ``object.__setattr__``
        (since the dataclass is frozen). Subclasses that want a
        vectorised batched implementation can pass a custom
        ``target_single`` whose JAX-traceability already covers the
        batched case.
    """

    target_single: Callable[[Array], Array]
    output_shape: tuple[int, ...]
    link: Map
    shift: Map = field(default_factory=ScalarConstant)
    constraint: Constraint = field(default_factory=lambda: real)

    def __post_init__(self) -> None:
        # Cache the vmapped batched view. Frozen dataclass — bypass via
        # object.__setattr__.
        object.__setattr__(self, "_target_map", jax.vmap(self.target_single))

    @property
    def target_map(self) -> Callable[[Array], Array]:
        """Batched view of ``target_single``: ``(n,) + input_shape -> (n,) + output_shape``.

        Derived once at construction via ``jax.vmap``.
        """
        return self._target_map  # type: ignore[attr-defined]

    def __call__(self, x: Array, y: Array) -> Array:
        r"""Deterministic log-density at ``(x, y)``: ``link(y) + shift(x)``.

        Both single-point (``x.shape == input_shape``,
        ``y.shape == output_shape``) and batched
        (``x.shape == (n,) + input_shape``,
        ``y.shape == (n,) + output_shape``) inputs are supported via
        broadcasting; ``Map`` semantics handle the batch dim.
        """
        return self.link(y) + self.shift(x)

    def density_at(self, x: Array) -> Array:
        """Compose ``target_single`` with ``link`` / ``shift``: log-density at ``x``.

        Equivalent to ``self(x, self.target_single(x))`` in single-point
        mode. For batched ``x`` of shape ``(n,) + input_shape`` use
        ``self(x, self.target_map(x))`` directly — this method routes
        through the single-point callable.
        """
        return self(x, self.target_single(x))

    def pushforward(self, x: Array, y_dist: Distribution) -> Distribution:
        r"""Push an emulator predictive ``y_dist`` at point(s) ``x`` to a log-density distribution.

        Constructs the per-``x`` map
        ``Affine(slope=1.0, intercept=shift(x)) @ link`` and dispatches
        through :func:`sabi.maps.pushforward`. The emulator's predictive
        distribution over ``y`` becomes the predictive distribution over
        ``link(y) + shift(x)``.

        Args:
            x: query point(s); single point ``input_shape`` or batched
                ``(n,) + input_shape``. ``shift(x)`` provides the
                per-point intercept.
            y_dist: emulator predictive at ``x`` (e.g. ``Normal`` with
                ``batch_shape=(n,)`` for marginal mode, or
                ``MultivariateNormal`` with ``event_shape=(n,)`` for
                joint mode).

        Returns:
            A ``Distribution`` over the log-density-residual at ``x``
            shifted by ``shift(x)``. Closed-form for
            ``(Affine, Normal | MultivariateNormal)``; MC fallback for
            non-affine links via ``Compose`` recursion.
        """
        per_x_map = Affine(slope=jnp.asarray(1.0), intercept=self.shift(x)) @ self.link
        return pushforward(per_x_map, y_dist)

    # ------------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------------

    @classmethod
    def identity_from_target(
        cls,
        target: "TargetDistribution",
        *,
        output_shape: tuple[int, ...] = (),
    ) -> "DensityDecomposition":
        """Trivial decomposition: emulate the target's analytical log-density.

        ``target_single`` is the target's analytical
        ``_unnormalized_log_prob`` (via ProbPipe's
        :func:`probpipe.unnormalized_log_prob` op);
        ``link = Identity``, ``shift = _ZeroShift()``. Requires
        ``target`` to implement ``_unnormalized_log_prob`` analytically.

        Args:
            target: a ``TargetDistribution`` carrying an analytical
                ``_unnormalized_log_prob``.
            output_shape: shape of one emulator output. Default ``()``
                (the analytical density is scalar).
        """
        def target_single(x: Array) -> Array:
            return jnp.asarray(pp_unnormalized_log_prob(target, x))

        return cls(
            target_single=target_single,
            output_shape=output_shape,
            link=Identity(),
        )

    @classmethod
    def likelihood_with_prior(
        cls,
        log_likelihood: Callable[[Array], Array],
        *,
        output_shape: tuple[int, ...],
        modeling_prior: Distribution,
        link: Map | None = None,
    ) -> "DensityDecomposition":
        r"""Emulator emits ``log L(x)``; shift adds the modeling prior.

        :math:`\log \tilde p(x) = \mathrm{link}(\log L(x)) + \log \pi_0(x)`.
        Default ``link = Identity()`` corresponds to the exp-link
        assumption :math:`\tilde p \propto \exp(\log L + \log \pi_0)`.

        Args:
            log_likelihood: single-point ``input_shape -> ()`` callable
                emitting log-likelihood.
            output_shape: shape of one emulator output. Typically ``()``.
            modeling_prior: ``Distribution`` over the parameter space.
                The shift becomes ``LogProb(modeling_prior)``.
            link: optional non-default link. Defaults to
                ``Identity()`` (exp link in log-space).
        """
        return cls(
            target_single=log_likelihood,
            output_shape=output_shape,
            link=link if link is not None else Identity(),
            shift=LogProb(modeling_prior),
        )

    @classmethod
    def forward_model(
        cls,
        forward_model: Callable[[Array], Array],
        *,
        output_shape: tuple[int, ...],
        log_lik_from_outputs: Map,
        modeling_prior: Distribution,
    ) -> "DensityDecomposition":
        r"""Emulator emits a forward-model output; ``link`` applies a likelihood.

        :math:`\log \tilde p(x) = \mathrm{log\_lik\_from\_outputs}(\mathrm{forward\_model}(x))
        + \log \pi_0(x)`. ``log_lik_from_outputs`` is a :class:`Map` —
        for example ``GaussianLogLik(obs, cov)`` — whose
        ``event_shape_in`` matches ``output_shape``.

        Args:
            forward_model: single-point callable emitting a forward-model
                output (e.g. simulated observation).
            output_shape: shape of one forward-model output.
            log_lik_from_outputs: Map from the forward-model output to
                a scalar log-likelihood.
            modeling_prior: ``Distribution`` over the parameter space.
        """
        return cls(
            target_single=forward_model,
            output_shape=output_shape,
            link=log_lik_from_outputs,
            shift=LogProb(modeling_prior),
        )

    @classmethod
    def gaussian_forward_model(
        cls,
        forward_model: Callable[[Array], Array],
        *,
        output_shape: tuple[int, ...],
        obs: Array,
        cov: Array,
        modeling_prior: Distribution,
    ) -> "DensityDecomposition":
        r"""Convenience: forward model with a Gaussian observation likelihood.

        :math:`\log \tilde p(x) = \log \mathcal{N}(\mathrm{obs} \mid g(x), C)
        + \log \pi_0(x)`. Equivalent to :meth:`forward_model` with
        ``log_lik_from_outputs = GaussianLogLik(obs, cov)``.
        """
        return cls.forward_model(
            forward_model=forward_model,
            output_shape=output_shape,
            log_lik_from_outputs=GaussianLogLik(obs=obs, cov=cov),
            modeling_prior=modeling_prior,
        )


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

    Computes ``decomposition.density_at(x_i)`` and
    ``target._unnormalized_log_prob(x_i)`` at each row of ``x_test`` and
    compares them. ``strict=True`` (the default) requires pointwise
    equality within ``atol``; ``strict=False`` allows an additive
    constant — only the differences between rows must match.

    Returns ``True`` trivially when ``target._unnormalized_log_prob``
    raises ``NotImplementedError`` (user inverse problems with no
    analytical density).

    Args:
        decomposition: candidate decomposition to validate.
        target: a ``TargetDistribution`` (analytical
            ``_unnormalized_log_prob`` if available).
        x_test: shape ``(n,) + input_shape``. Requires ``n >= 2`` when
            ``strict=False``.
        atol: numerical tolerance for the comparison.
        strict: when ``True`` (default), require pointwise equality
            within ``atol``:
            ``decomposition.density_at(x) ≈ target._unnormalized_log_prob(x)``.
            When ``False``, allow an additive constant — differences
            across rows must match within ``atol``. Requires ``n >= 2``.
    """
    x_test = jnp.asarray(x_test)
    if x_test.ndim < 2:
        raise ValueError(
            "is_consistent_with: x_test must be at least rank-2 "
            "((n,) + input_shape); got rank "
            f"{x_test.ndim} (shape {tuple(x_test.shape)})."
        )
    if not strict and x_test.shape[0] < 2:
        raise ValueError(
            "is_consistent_with(strict=False): need n >= 2 rows in "
            "x_test to compare differences across rows; got "
            f"n={x_test.shape[0]}."
        )

    try:
        target_log_density = jax.vmap(target._unnormalized_log_prob)(x_test)
    except NotImplementedError:
        # User inverse problem with no analytical density — trivially
        # consistent.
        return True

    decomposition_log_density = jax.vmap(decomposition.density_at)(x_test)
    target_log_density = jnp.asarray(target_log_density)
    decomposition_log_density = jnp.asarray(decomposition_log_density)

    if strict:
        return bool(
            jnp.all(
                jnp.abs(decomposition_log_density - target_log_density) <= atol
            )
        )
    # Up-to-constant: subtract row-0, compare differences.
    diff = decomposition_log_density - target_log_density
    return bool(jnp.all(jnp.abs(diff - diff[0]) <= atol))
