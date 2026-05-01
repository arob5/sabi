"""Tests for `PosteriorMetric` implementations and the protocol declaration."""

import jax
import jax.numpy as jnp
import pytest
from dataclasses import dataclass, replace
from probpipe import sample
from probpipe.core._distribution_base import Distribution
from probpipe.core._empirical import NumericEmpiricalDistribution
from probpipe.core.protocols import SupportsLogProb, SupportsSampling

from sabi.metrics.base import MissingProtocolError, PosteriorMetric
from sabi.metrics.posterior_mmd import ReferenceMMD
from sabi.problems.gaussian import gaussian2d


def _problem_no_ref():
    p = gaussian2d()
    return replace(p, reference_distribution=None)


def test_reference_mmd_returns_empty_when_no_reference():
    posterior = NumericEmpiricalDistribution(
        samples=jax.random.normal(jax.random.key(0), shape=(64, 2)),
        name="posterior",
    )
    out = ReferenceMMD()(posterior, _problem_no_ref(), key=jax.random.key(0))
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
    out = ReferenceMMD()(posterior, problem, key=jax.random.key(0))
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
    out = ReferenceMMD()(posterior, problem, key=jax.random.key(0))
    assert out["mmd2"] > 0.1
    assert out["mmd"] > 0.0


def test_reference_mmd_declares_supports_sampling():
    """Static check: ReferenceMMD's `requires` includes SupportsSampling."""
    assert SupportsSampling in ReferenceMMD.requires


@dataclass(frozen=True)
class _FakeMetricNeedingLogProb(PosteriorMetric):
    """A metric that requires SupportsLogProb — used to test the loop's
    protocol-mismatch raising behaviour."""

    requires = (SupportsLogProb,)

    def __call__(self, posterior, problem, *, key):
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
