"""Tests for `Metric` implementations and the protocol declaration."""

import jax
import jax.numpy as jnp
from dataclasses import dataclass, replace
from typing import ClassVar

from probpipe import sample
from probpipe.core._empirical import NumericEmpiricalDistribution
from probpipe.core.protocols import SupportsLogProb, SupportsSampling

from sabi.metrics import (
    Metric,
    MetricContext,
    MetricTarget,
    MissingProtocolError,
    ReferenceMMD,
)
from sabi.problems.gaussian import gaussian2d


def _problem_no_ref():
    p = gaussian2d()
    return replace(p, reference_distribution=None)


def _ctx_for(estimate, problem) -> MetricContext:
    """Build a minimal MetricContext sufficient for ReferenceMMD-style tests.

    `surrogate_distribution` and design data aren't read by
    `ReferenceMMD`; pass placeholders.
    """
    n = 1
    X = jnp.zeros((n,) + problem.input_shape)
    Y = jnp.zeros((n,) + problem.output_shape)
    return MetricContext(
        estimate=estimate,
        surrogate_distribution=None,  # type: ignore[arg-type]
        problem=problem,
        X=X,
        Y_raw=Y,
        Y_train=Y,
        tempering_state=None,
        round_idx=0,
        metric_target=MetricTarget.CURRENT,
    )


def test_reference_mmd_returns_empty_when_no_reference():
    posterior = NumericEmpiricalDistribution(
        samples=jax.random.normal(jax.random.key(0), shape=(64, 2)),
        name="posterior",
    )
    problem = _problem_no_ref()
    out = ReferenceMMD()(_ctx_for(posterior, problem), key=jax.random.key(0))
    assert out == {}


def test_reference_mmd_low_for_samples_from_reference():
    """Posterior samples drawn from the *same* reference should yield small MMD."""
    problem = gaussian2d()
    samples = jnp.asarray(
        sample(
            problem.reference_distribution,
            key=jax.random.key(7),
            sample_shape=(2048,),
        )
    )
    posterior = NumericEmpiricalDistribution(samples=samples, name="posterior")
    out = ReferenceMMD()(_ctx_for(posterior, problem), key=jax.random.key(0))
    assert set(out.keys()) == {"mmd", "mmd2"}
    assert out["mmd"] >= 0.0
    assert out["mmd2"] < 0.05


def test_reference_mmd_detects_shifted_samples():
    problem = gaussian2d()
    samples = jnp.asarray(
        sample(
            problem.reference_distribution, key=jax.random.key(7), sample_shape=(2048,)
        )
    ) + 3.0
    posterior = NumericEmpiricalDistribution(samples=samples, name="posterior")
    out = ReferenceMMD()(_ctx_for(posterior, problem), key=jax.random.key(0))
    assert out["mmd2"] > 0.1
    assert out["mmd"] > 0.0


def test_reference_mmd_declares_supports_sampling():
    """Static check: ReferenceMMD's `requires` includes SupportsSampling."""
    assert SupportsSampling in ReferenceMMD.requires


def test_reference_mmd_declares_keys():
    """Static check: ReferenceMMD declares the keys it produces so the
    loop can validate collisions upfront."""
    assert ReferenceMMD.keys == ("mmd", "mmd2")


@dataclass(frozen=True)
class _FakeMetricNeedingLogProb(Metric):
    """A metric that requires SupportsLogProb — used to test the loop's
    protocol-mismatch raising behaviour."""

    requires: ClassVar[tuple[type, ...]] = (SupportsLogProb,)
    keys: ClassVar[tuple[str, ...]] = ("x",)

    def __call__(self, ctx, *, key):
        return {"x": 1.0}


def test_metric_protocol_mismatch_detected_by_isinstance():
    """The loop's check is `isinstance(estimate, p) for p in metric.requires`.
    Verify that a SupportsSampling-only Distribution fails the SupportsLogProb
    check (so the loop will raise)."""
    posterior = NumericEmpiricalDistribution(
        samples=jax.random.normal(jax.random.key(0), shape=(32, 2)),
        name="posterior",
    )
    metric = _FakeMetricNeedingLogProb()
    # NumericEmpiricalDistribution: samplable but no log-prob
    assert isinstance(posterior, SupportsSampling)
    assert not isinstance(posterior, SupportsLogProb)
    # The loop helper does the check; here we just confirm the protocol semantics
    # that the helper relies on.
    _ = metric  # silence unused
    _ = MissingProtocolError  # imported for downstream use
