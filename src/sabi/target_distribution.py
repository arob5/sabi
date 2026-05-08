r"""``TargetDistribution`` — math-only identity of an unnormalized target.

A ``TargetDistribution`` carries the mathematical identity of the
target — name, parameter-space shape, and support — with no
algorithm-side details (which live on a :class:`sabi.density_decomposition.DensityDecomposition`
on the ``Algorithm``).

The class is **abstract** — subclasses with an analytical
unnormalized log-density implement ``_unnormalized_log_prob`` per the
ProbPipe protocol. For benchmark targets, the subclass provides the
analytical density directly. For user inverse problems where no
analytical density is available, the user simply does not subclass-
with-density: the algorithm interacts with the target only via the
``DensityDecomposition``, and the absence of ``_unnormalized_log_prob``
naturally means ``isinstance(target, SupportsUnnormalizedLogProb)``
returns False.

The ``IntermediateTarget`` subclass is metadata-only: it carries the
``state`` and ``output_transform`` produced by a tempering scheme, with
no analytical density of its own. The per-state effective
:class:`DensityDecomposition` lives separately, produced by
``TemperingScheme.intermediate_decomposition``.

Vectorization
-------------

``TargetDistribution`` is a ProbPipe ``NumericRecordDistribution`` and
follows ProbPipe's vectorization contract for distributions: external
callers go through the ProbPipe ops (``unnormalized_log_prob(target,
X)`` etc.), which broadcast over the leading batch axes; subclasses'
``_unnormalized_log_prob(x)`` hooks see a single event per call. See
``docs/notation.md`` for the contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from probpipe.core._numeric_record_distribution import NumericRecordDistribution
from probpipe.core.constraints import Constraint

if TYPE_CHECKING:
    from sabi.tempering.output_transform import OutputTransform


class TargetDistribution(NumericRecordDistribution):
    """Math-only identity of an unnormalized target distribution.

    Subclasses with an analytical density implement
    ``_unnormalized_log_prob(x)`` per the ProbPipe protocol; the
    constructor stores ``name``, ``input_shape``, and ``support``.

    Args:
        name: ProbPipe distribution name.
        input_shape: shape of one parameter-space point.
        support: ``Constraint`` over the parameter space.
    """

    _sampling_cost: ClassVar[str] = "high"
    _preferred_orchestration: ClassVar[str | None] = None

    def __init__(
        self,
        *,
        name: str,
        input_shape: tuple[int, ...],
        support: Constraint,
    ):
        if support is None:
            raise ValueError(
                f"{type(self).__name__} requires a non-None `support` "
                "(`Constraint` over the parameter space)."
            )
        self._input_shape = tuple(input_shape)
        self._support = support
        super().__init__(name=name)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shape

    @property
    def support(self) -> Constraint:
        return self._support

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._input_shape

    # `_unnormalized_log_prob` is intentionally NOT defined here.
    # Subclasses with an analytical density override it; subclasses
    # without (user inverse problems) leave it undefined and
    # `isinstance(target, SupportsUnnormalizedLogProb)` returns False.


class IntermediateTarget(TargetDistribution):
    """A ``TargetDistribution`` produced by a tempering scheme at one state.

    Carries the same math identity as the base, plus per-state metadata:

    - ``state``: the tempering state PyTree.
    - ``output_transform``: an :class:`OutputTransform` describing how
      to derive ``Y_train`` for the intermediate from cached raw
      evaluations ``Y_raw``. Used by the loop's emulator-update path.

    Intermediate targets are metadata-only — they do not carry an
    analytical density. The per-state effective
    :class:`DensityDecomposition` lives separately, produced by
    ``TemperingScheme.intermediate_decomposition``. See
    ``docs/tempering.md``.
    """

    def __init__(
        self,
        *,
        name: str,
        input_shape: tuple[int, ...],
        support: Constraint,
        state: Any,
        output_transform: "OutputTransform",
    ):
        self._state = state
        self._output_transform = output_transform
        super().__init__(name=name, input_shape=input_shape, support=support)

    @property
    def state(self) -> Any:
        return self._state

    @property
    def output_transform(self) -> "OutputTransform":
        return self._output_transform
