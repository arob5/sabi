"""Free-function deterministic posterior estimators.

Each estimator is a function `(surrogate_distribution) -> Distribution[Array]`
that returns a concrete deterministic posterior approximation. Type
dispatch on the runtime type matches ProbPipe's op-dispatch style.

For Dirac surrogate posteriors (`WeightedEmpiricalRandomMeasure`), all
sensible deterministic estimators coincide and reduce to the underlying
empirical — there's nothing random to estimate.

Currently shipped:

- `expected_target(surrogate_distribution)` — plug the surrogate's
  predictive mean into the decomposition. The expectation of the
  target map under the surrogate distribution. Biased in the
  emulator-pushforward case (the plug-in is not the same as the
  unbiased expected posterior `mean(rm)`).

The unbiased expected posterior is exposed via
`mean(surrogate_distribution)` (handled by ProbPipe's `mean` op via
`SupportsMean`). For the Dirac case this returns the inner empirical;
for the emulator-pushforward case there is no general `SupportsMean`
implementation today and `mean(emulated_distribution)` raises.

Future estimators — see `docs/probpipe_issues.md` for the
partial-pushforward primitive that would generalize their construction.
"""

from __future__ import annotations

from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from jax import Array
from probpipe import condition_on, mean, sample
from probpipe.core._distribution_base import Distribution
from probpipe.core._numeric_record_distribution import NumericRecordDistribution
from probpipe.core.constraints import Constraint

from sabi.density_decomposition import DensityDecomposition
from sabi.emulators.base import Emulator
from sabi.surrogate.surrogate_distribution import (
    EmulatedDistribution,
    SurrogateDistribution,
)
from sabi.surrogate.weighted_empirical import WeightedEmpiricalRandomMeasure


def expected_target(
    surrogate_distribution: SurrogateDistribution,
    *,
    sampler: str | None = None,
    sampler_kwargs: dict[str, Any] | None = None,
    name: str | None = None,
) -> Distribution:
    """Return the deterministic posterior obtained by plugging the
    surrogate's predictive mean into the decomposition.

    Type dispatch on the runtime type of ``surrogate_distribution``:

    - :class:`WeightedEmpiricalRandomMeasure` (Dirac): returns the
      inner empirical (which IS the deterministic target). ``sampler``
      / ``sampler_kwargs`` are ignored.
    - :class:`EmulatedDistribution`: returns an
      ``_ExpectedTargetDistribution`` whose ``_unnormalized_log_prob``
      is the decomposition composed with the emulator's predictive
      mean. Sampling delegates to ProbPipe ``condition_on`` (auto-dispatched
      MCMC). ``sampler`` selects a specific method (e.g.
      ``"tfp_nuts"``); ``sampler_kwargs`` forwards arguments like
      ``num_results``, ``num_warmup``.
    """
    if isinstance(surrogate_distribution, WeightedEmpiricalRandomMeasure):
        return surrogate_distribution.inner_distribution
    if isinstance(surrogate_distribution, EmulatedDistribution):
        return _ExpectedTargetDistribution(
            emulator=surrogate_distribution.emulator,
            decomposition=surrogate_distribution.decomposition,
            input_shape=surrogate_distribution.inner_event_shape,
            support=surrogate_distribution.inner_support,
            sampler=sampler,
            sampler_kwargs=sampler_kwargs,
            name=name or f"expected_target_{surrogate_distribution.name}",
        )
    raise TypeError(
        f"expected_target: unsupported posterior type "
        f"{type(surrogate_distribution).__name__}."
    )


class _ExpectedTargetDistribution(NumericRecordDistribution):
    """The deterministic posterior obtained by plugging the emulator's
    predictive mean into the decomposition.

    ``_unnormalized_log_prob(x) = decomposition(x, emulator_mean(x))``.

    Sampling delegates to ProbPipe ``condition_on(self)``; the registry
    auto-selects an MCMC method (typically ``tfp_nuts``) since this
    distribution satisfies ``SupportsUnnormalizedLogProb``.
    """

    _sampling_cost: ClassVar[str] = "high"
    _preferred_orchestration: ClassVar[str | None] = None

    def __init__(
        self,
        emulator: Emulator,
        decomposition: DensityDecomposition,
        *,
        input_shape: tuple[int, ...],
        support: Constraint,
        sampler: str | None = None,
        sampler_kwargs: dict[str, Any] | None = None,
        name: str | None = None,
    ):
        self._emulator = emulator
        self._decomposition = decomposition
        self._input_shape = tuple(input_shape)
        self._support_value = support
        self._sampler = sampler
        self._sampler_kwargs = dict(sampler_kwargs) if sampler_kwargs else {}
        super().__init__(name=name or "expected_target")

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._input_shape

    @property
    def support(self) -> Constraint:
        return self._support_value

    def _unnormalized_log_prob(self, value: Array) -> Array:
        """Plug-in posterior log-density at ``value``.

        Detected by ``ndim``: single-point at rank ``len(input_shape)``,
        batched at rank ``1 + len(input_shape)``. Avoids the shape-
        equality misidentification trap (``(n, k)`` with ``n == k``
        looks like a single point under shape-equality).
        """
        x = jnp.asarray(value)
        ndim_single = len(self._input_shape)
        if x.ndim == ndim_single:
            if x.shape != self._input_shape:
                raise ValueError(
                    f"_unnormalized_log_prob: single-point input expects "
                    f"shape {self._input_shape}, got {tuple(x.shape)}."
                )
            pred = self._emulator(x[None])
            pred_mean = jnp.asarray(mean(pred))
            return self._decomposition(x, pred_mean[0])
        if x.ndim == ndim_single + 1:
            if x.shape[1:] != self._input_shape:
                raise ValueError(
                    f"_unnormalized_log_prob: batched input expects shape "
                    f"(n,) + {self._input_shape}, got {tuple(x.shape)}."
                )
            pred = self._emulator(x)
            pred_mean = jnp.asarray(mean(pred))
            # decomposition.__call__ broadcasts via Map semantics; vmap
            # to be explicit over the leading n axis.
            return jax.vmap(self._decomposition)(x, pred_mean)
        raise ValueError(
            f"_unnormalized_log_prob: expected ndim {ndim_single} (single "
            f"point) or {ndim_single + 1} (batched), got ndim={x.ndim} "
            f"(shape={tuple(x.shape)})."
        )

    def _sample(self, key, sample_shape: tuple[int, ...] = ()) -> Array:
        """Draw posterior samples via ProbPipe ``condition_on``.

        Note:
            This method **cannot** be called inside ``jax.jit`` /
            ``jax.vmap`` / ``jax.grad``. The ``int(...).item()`` cast
            below is required because ProbPipe's
            ``condition_on(..., random_seed=int)`` accepts only a
            Python ``int``, not a JAX key. Calling ``.item()`` on a
            traced array raises ``ConcretizationTypeError``. The cast
            can be dropped once ProbPipe's MCMC dispatch grows a
            JAX-key-aware ``random_seed`` argument; tracked in
            ``docs/probpipe_issues.md``.
        """
        kwargs = dict(self._sampler_kwargs)
        if self._sampler is not None:
            kwargs["method"] = self._sampler
        kwargs.setdefault(
            "random_seed",
            int(jax.random.randint(key, (), 0, 2**31 - 1).item()),
        )
        approx = condition_on(self, **kwargs)
        return jnp.asarray(sample(approx, key=key, sample_shape=sample_shape))
