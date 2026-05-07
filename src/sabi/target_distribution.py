r"""`TargetDistribution` — math-only identity of an unnormalized target distribution.

After the ``DensityDecomposition`` split (issue #65),
``TargetDistribution`` carries only the mathematical identity of the
target: a name, a parameter-space shape, an explicit support, and
optionally an analytical ``_unnormalized_log_prob``. The
algorithmically-relevant pieces — what the emulator approximates
(``target_single``) and how its output composes into log-density (``link``,
``shift``) — live on a separate :class:`sabi.density_decomposition.DensityDecomposition`
on the algorithm.

A single ``TargetDistribution`` can be paired with many decompositions:
the choice of what to emulate (full log-density vs. log-likelihood vs.
forward-model output) is an algorithmic decision, not a property of
the target. See ``docs/density_decomposition.md`` for the design.

Two cases for ``_unnormalized_log_prob``:

- **Benchmark targets**: pass a callable ``unnormalized_log_prob`` to
  the constructor. The target will satisfy
  ``SupportsUnnormalizedLogProb`` and is directly consumable by
  ProbPipe ops (e.g., ``condition_on(target)`` for NUTS).
- **User inverse problems**: omit ``unnormalized_log_prob``. Calling
  ``_unnormalized_log_prob`` raises ``NotImplementedError`` with a
  pointer to the algorithm's ``DensityDecomposition``. The MCMC-relevant
  random log-density boundary moves to
  ``EmulatedDistribution._random_unnormalized_log_prob``, which composes
  the decomposition through the emulator predictive.

The ``IntermediateTarget`` subclass mirrors the same shape, plus a
``state`` and ``output_transform`` describing how to derive
``Y_train`` for the intermediate's tempering state from cached
``Y_raw``. Intermediate targets do not carry ``target_single`` /
``log_density_form`` / ``prior`` — those are read from the bridging
scheme + the algorithm's ``DensityDecomposition``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

import jax.numpy as jnp
from jax import Array
from probpipe.core._numeric_record_distribution import NumericRecordDistribution
from probpipe.core.constraints import Constraint

if TYPE_CHECKING:
    # Avoid the circular import sabi.target_distribution → sabi.tempering →
    # sabi.tempering.base → sabi.target_distribution. With `from __future__
    # import annotations`, the annotation is a string at runtime.
    from sabi.tempering.output_transform import OutputTransform


class TargetDistribution(NumericRecordDistribution):
    """Math-only identity of an unnormalized target distribution.

    Args:
        name: ProbPipe distribution name.
        input_shape: shape of one parameter-space point.
        support: ``Constraint`` over the parameter space.
        unnormalized_log_prob: optional analytical density callable
            ``input_shape -> ()``. When provided, the target satisfies
            ``SupportsUnnormalizedLogProb`` directly. When omitted,
            ``_unnormalized_log_prob`` raises ``NotImplementedError``;
            the algorithm interacts with the target via the
            ``DensityDecomposition`` instead.
    """

    _sampling_cost: ClassVar[str] = "high"
    _preferred_orchestration: ClassVar[str | None] = None

    def __init__(
        self,
        *,
        name: str,
        input_shape: tuple[int, ...],
        support: Constraint,
        unnormalized_log_prob: Callable[[Array], Array] | None = None,
    ):
        if support is None:
            raise ValueError(
                "TargetDistribution requires a non-None `support` "
                "(`Constraint` over the parameter space)."
            )
        self._input_shape = tuple(input_shape)
        self._support = support
        self._analytical_unnormalized_log_prob = unnormalized_log_prob
        super().__init__(name=name)

    # ------------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------------

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shape

    @property
    def support(self) -> Constraint:
        return self._support

    # ------------------------------------------------------------------------
    # Distribution interface
    # ------------------------------------------------------------------------

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._input_shape

    def _unnormalized_log_prob(self, value: Array) -> Array:
        """Analytical unnormalized log-density at ``value``.

        Routes through the user-supplied ``unnormalized_log_prob``
        callable (single-point ``input_shape -> ()``). Both single-point
        (``value.shape == input_shape``) and batched
        (``value.shape == (n,) + input_shape``) inputs are accepted —
        dispatched by ``ndim``.

        Raises:
            NotImplementedError: when no analytical density was supplied
                at construction. User inverse problems should interact
                with the target via ``algorithm.density_decomposition``
                rather than via this method.
            ValueError: when ``value.ndim`` matches neither the
                single-point rank ``len(input_shape)`` nor the batched
                rank ``1 + len(input_shape)``.
        """
        if self._analytical_unnormalized_log_prob is None:
            raise NotImplementedError(
                f"{type(self).__name__} {self.name!r} has no analytical "
                "`_unnormalized_log_prob`. The MCMC-relevant random "
                "log-density boundary lives on the algorithm's "
                "`DensityDecomposition` "
                "(`EmulatedDistribution._random_unnormalized_log_prob`); "
                "do not call this method directly. To attach an "
                "analytical density (e.g. for a benchmark), pass "
                "`unnormalized_log_prob=...` to the constructor."
            )
        x = jnp.asarray(value)
        ndim_single = len(self._input_shape)
        if x.ndim == ndim_single:
            if x.shape != self._input_shape:
                raise ValueError(
                    f"_unnormalized_log_prob: single-point input expects "
                    f"shape {self._input_shape}, got {tuple(x.shape)}."
                )
            return jnp.asarray(self._analytical_unnormalized_log_prob(x))
        if x.ndim == ndim_single + 1:
            if x.shape[1:] != self._input_shape:
                raise ValueError(
                    f"_unnormalized_log_prob: batched input expects shape "
                    f"(n,) + {self._input_shape}, got {tuple(x.shape)}."
                )
            import jax

            return jax.vmap(self._analytical_unnormalized_log_prob)(x)
        raise ValueError(
            f"_unnormalized_log_prob: expected ndim {ndim_single} (single "
            f"point) or {ndim_single + 1} (batched), got ndim={x.ndim} "
            f"(shape={tuple(x.shape)})."
        )


class IntermediateTarget(TargetDistribution):
    """A ``TargetDistribution`` produced by a tempering scheme at one state.

    Carries the same math identity (``name``, ``input_shape``,
    ``support``, optional ``_unnormalized_log_prob``) as the base, plus
    metadata the loop uses to derive the emulator's training data
    efficiently:

    - ``state``: the tempering state that produced this intermediate.
    - ``output_transform``: an `OutputTransform` value object describing
      how to derive ``Y_train`` for the intermediate from cached raw
      evaluations ``Y_raw`` of the base target. Carries both
      ``apply(state, X, Y_raw) -> Y_train`` and
      ``diff(state_a, state_b) -> EmulatorUpdate | None`` for the
      cheap-update fast path; see ``sabi.emulators.dispatch``. Callable
      on instances via ``__call__``.

    Per the ``DensityDecomposition`` split, intermediate targets no
    longer carry ``target_single`` / ``log_density_form`` / ``prior``
    fields — those quantities live on a per-state effective
    ``DensityDecomposition`` produced by
    ``TemperingScheme.intermediate_decomposition``.

    See ``docs/tempering.md`` for the conceptual layering.
    """

    def __init__(
        self,
        *,
        name: str,
        input_shape: tuple[int, ...],
        support: Constraint,
        state: Any,
        output_transform: "OutputTransform",
        unnormalized_log_prob: Callable[[Array], Array] | None = None,
    ):
        self._state = state
        self._output_transform = output_transform
        super().__init__(
            name=name,
            input_shape=input_shape,
            support=support,
            unnormalized_log_prob=unnormalized_log_prob,
        )

    @property
    def state(self) -> Any:
        return self._state

    @property
    def output_transform(self) -> "OutputTransform":
        return self._output_transform
