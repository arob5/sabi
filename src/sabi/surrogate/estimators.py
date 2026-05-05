"""Free-function deterministic posterior estimators.

Each estimator is a function `(sp) -> Distribution[Array]` that returns a
concrete deterministic posterior approximation. Type dispatch on `sp`
matches ProbPipe's op-dispatch style.

For Dirac surrogate posteriors (`WeightedEmpiricalRandomMeasure`), all
sensible deterministic estimators coincide and reduce to the underlying
empirical — there's nothing random to estimate.

v1.2 ships:

- `expected_target(sp)` — plug the surrogate's predictive mean into the
  log-density form. Renamed from "plug-in mean" because it's the
  expectation of the target map under the surrogate distribution. Biased
  in the GP-pushforward case (the plug-in is not the same as the
  unbiased expected posterior `mean(rm)`).

The unbiased expected posterior is exposed via `mean(sp)` (handled by
ProbPipe's `mean` op via `SupportsMean`). For Dirac SPs this returns the
inner empirical; for the GP-pushforward case there is no general
`SupportsMean` implementation in v1.2 and `mean(gp_sp)` raises.

Future estimators (NOT in v1.2) — see `docs/probpipe_issues.md` for the
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

from sabi.surrogate.surrogate_distribution import SurrogateDistribution
from sabi.surrogate.weighted_empirical import WeightedEmpiricalRandomMeasure
from sabi.problems.forms import LogDensityForm
from sabi.emulators.base import Emulator


def expected_target(
    sp,
    *,
    sampler: str | None = None,
    sampler_kwargs: dict[str, Any] | None = None,
    name: str | None = None,
) -> Distribution:
    """Return the deterministic posterior obtained by plugging the
    surrogate's predictive mean into the log-density form.

    Type dispatch on `sp`:

    - `WeightedEmpiricalRandomMeasure` (Dirac): returns the inner
      empirical (which IS the deterministic target). `sampler` /
      `sampler_kwargs` are ignored.
    - `SurrogateDistribution`: returns an `_ExpectedTargetDistribution`
      whose `_unnormalized_log_prob` is the form composed with the
      surrogate's predictive mean. Sampling delegates to ProbPipe
      `condition_on` (auto-dispatched MCMC). `sampler` selects a
      specific method (e.g. `"tfp_nuts"`); `sampler_kwargs` forwards
      arguments like `num_results`, `num_warmup`.
    """
    if isinstance(sp, WeightedEmpiricalRandomMeasure):
        return sp.inner_distribution
    if isinstance(sp, SurrogateDistribution):
        return _ExpectedTargetDistribution(
            emulator=sp.emulator,
            log_density_form=sp.log_density_form,
            prior=sp.prior,
            input_shape=sp.inner_event_shape,
            support=sp.inner_support,
            sampler=sampler,
            sampler_kwargs=sampler_kwargs,
            name=name or f"expected_target_{sp.name}",
        )
    raise TypeError(
        f"expected_target: unsupported posterior type {type(sp).__name__}."
    )


class _ExpectedTargetDistribution(NumericRecordDistribution):
    """The deterministic posterior obtained by plugging the emulator's
    predictive mean into the log-density form.

    `_unnormalized_log_prob(x) = log_density_form(x, emulator_mean(x), prior=prior)`.

    Sampling delegates to ProbPipe `condition_on(self)`; the registry
    auto-selects an MCMC method (typically `tfp_nuts`) since this
    distribution satisfies `SupportsUnnormalizedLogProb`.
    """

    _sampling_cost: ClassVar[str] = "high"
    _preferred_orchestration: ClassVar[str | None] = None

    def __init__(
        self,
        emulator: Emulator,
        log_density_form: LogDensityForm,
        prior: Distribution | None,
        *,
        input_shape: tuple[int, ...],
        support: Constraint,
        sampler: str | None = None,
        sampler_kwargs: dict[str, Any] | None = None,
        name: str | None = None,
    ):
        self._emulator = emulator
        self._form = log_density_form
        self._prior = prior
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
            # Form's per-point hook; avoids vmap overhead on a single x.
            return self._form._call_single(x, pred_mean[0], prior=self._prior)
        if x.ndim == ndim_single + 1:
            if x.shape[1:] != self._input_shape:
                raise ValueError(
                    f"_unnormalized_log_prob: batched input expects shape "
                    f"(n,) + {self._input_shape}, got {tuple(x.shape)}."
                )
            pred = self._emulator(x)
            pred_mean = jnp.asarray(mean(pred))
            return self._form(x, pred_mean, prior=self._prior)
        raise ValueError(
            f"_unnormalized_log_prob: expected ndim {ndim_single} (single "
            f"point) or {ndim_single + 1} (batched), got ndim={x.ndim} "
            f"(shape={tuple(x.shape)})."
        )

    def _sample(self, key, sample_shape: tuple[int, ...] = ()) -> Array:
        kwargs = dict(self._sampler_kwargs)
        if self._sampler is not None:
            kwargs["method"] = self._sampler
        # condition_on accepts a `random_seed` int; derive deterministically
        # from the JAX key so callers see reproducible draws.
        kwargs.setdefault(
            "random_seed",
            int(jax.random.randint(key, (), 0, 2**31 - 1).item()),
        )
        approx = condition_on(self, **kwargs)
        return jnp.asarray(sample(approx, key=key, sample_shape=sample_shape))
