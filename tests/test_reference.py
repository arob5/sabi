"""Tests for sabi.reference: IO round-trip + cache load/regenerate logic."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from probpipe.core._empirical import NumericEmpiricalDistribution
from probpipe.core._numeric_record_distribution import NumericRecordDistribution
from probpipe.core.constraints import Constraint

from sabi._probpipe_compat import independent_uniform
from sabi.reference import cache as cache_module
from sabi.reference.cache import load_or_generate_reference_samples
from sabi.reference.io import (
    read_metadata_json,
    read_samples_parquet,
    write_metadata_json,
    write_samples_parquet,
)


# ---------------------------------------------------------------------------
# Test target classes (subclasses with vectorized analytical density).
# ---------------------------------------------------------------------------


def _box_support(d: int = 2) -> Constraint:
    return independent_uniform(
        low=jnp.full(d, -5.0), high=jnp.full(d, 5.0), name=f"box_{d}d"
    ).support


class _StandardNormalTarget(NumericRecordDistribution):
    """Standard ``d``-D normal log-density. Vectorized over the
    trailing event axis."""

    def __init__(self, *, d: int, name: str):
        self._d = d
        self._support = _box_support(d)
        super().__init__(name=name)

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self._d,)

    @property
    def support(self) -> Constraint:
        return self._support

    def _unnormalized_log_prob(self, theta):
        return -0.5 * jnp.sum(theta * theta, axis=-1)


class _BadTarget(NumericRecordDistribution):
    """Target whose `_unnormalized_log_prob` raises — used in cache-hit
    tests to confirm NUTS doesn't run."""

    def __init__(self, *, d: int = 2):
        self._d = d
        self._support = _box_support(d)
        super().__init__(name="bad")

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self._d,)

    @property
    def support(self) -> Constraint:
        return self._support

    def _unnormalized_log_prob(self, theta):
        raise AssertionError("NUTS must not run on a cache hit.")


def _make_2d_target() -> NumericRecordDistribution:
    return _StandardNormalTarget(d=2, name="std_2d_normal")


def _make_1d_target() -> NumericRecordDistribution:
    return _StandardNormalTarget(d=1, name="std_1d_normal")


# ---------------------------------------------------------------------------
# IO round-trips
# ---------------------------------------------------------------------------


def test_samples_parquet_roundtrip(tmp_path):
    samples = jax.random.normal(jax.random.key(0), shape=(50, 3))
    path = tmp_path / "samples.parquet"
    write_samples_parquet(samples, path)
    loaded = read_samples_parquet(path)
    assert loaded.shape == samples.shape
    assert jnp.allclose(jnp.asarray(loaded), samples, atol=1e-6)


def test_metadata_json_roundtrip(tmp_path):
    md = {
        "problem_name": "test",
        "diagnostics": {"max_rhat": 1.01, "min_ess": 500.0, "num_divergences": 0},
        "input_shape": [2],
        "n_samples": 100,
    }
    path = tmp_path / "metadata.json"
    write_metadata_json(md, path)
    loaded = read_metadata_json(path)
    assert loaded == md


# ---------------------------------------------------------------------------
# Cache load / regenerate
# ---------------------------------------------------------------------------


def test_cache_hit_skips_nuts(tmp_path):
    cache_dir = tmp_path / "refs"
    problem_dir = cache_dir / "fake_problem"
    problem_dir.mkdir(parents=True)
    fname = "k1_nuts_n10_w5_c1_s0"
    samples = jax.random.normal(jax.random.key(7), shape=(10, 2))
    write_samples_parquet(samples, problem_dir / f"{fname}.parquet")
    write_metadata_json({"problem_name": "fake_problem"}, problem_dir / f"{fname}.json")

    target = _BadTarget(d=2)

    ref = load_or_generate_reference_samples(
        problem_name="fake_problem",
        cache_key="k1",
        target=target,
        num_results=10,
        num_warmup=5,
        num_chains=1,
        random_seed=0,
        cache_dir=cache_dir,
    )
    assert isinstance(ref, NumericEmpiricalDistribution)
    assert jnp.allclose(jnp.asarray(ref.samples), samples, atol=1e-6)


def test_cache_keys_disambiguate_by_params(tmp_path):
    cache_dir = tmp_path / "refs"
    problem_dir = cache_dir / "fake_problem"
    problem_dir.mkdir(parents=True)
    samples_a = jnp.zeros((4, 2)) + 1.0
    samples_b = jnp.zeros((4, 2)) + 2.0
    fname_a = "kA_nuts_n4_w2_c1_s0"
    fname_b = "kB_nuts_n4_w2_c1_s0"
    write_samples_parquet(samples_a, problem_dir / f"{fname_a}.parquet")
    write_metadata_json({"k": "A"}, problem_dir / f"{fname_a}.json")
    write_samples_parquet(samples_b, problem_dir / f"{fname_b}.parquet")
    write_metadata_json({"k": "B"}, problem_dir / f"{fname_b}.json")

    target = _BadTarget(d=2)

    ref_a = load_or_generate_reference_samples(
        problem_name="fake_problem",
        cache_key="kA",
        target=target,
        num_results=4, num_warmup=2, num_chains=1, random_seed=0,
        cache_dir=cache_dir,
    )
    ref_b = load_or_generate_reference_samples(
        problem_name="fake_problem",
        cache_key="kB",
        target=target,
        num_results=4, num_warmup=2, num_chains=1, random_seed=0,
        cache_dir=cache_dir,
    )
    assert jnp.allclose(jnp.asarray(ref_a.samples), samples_a)
    assert jnp.allclose(jnp.asarray(ref_b.samples), samples_b)


def test_quality_threshold_failure_raises(tmp_path):
    cache_dir = tmp_path / "refs"

    with pytest.raises(ValueError, match="min ESS"):
        load_or_generate_reference_samples(
            problem_name="bad_quality",
            cache_key="k",
            target=_make_2d_target(),
            num_results=20, num_warmup=10, num_chains=1, random_seed=0,
            cache_dir=cache_dir,
            quality_thresholds={"min_ess": 1e9},
        )

    assert not (cache_dir / "bad_quality").exists() or not any(
        p.is_file() for p in (cache_dir / "bad_quality").iterdir()
    )


def test_regenerate_forces_fresh_nuts_run(tmp_path, monkeypatch):
    cache_dir = tmp_path / "refs"
    common_kwargs = dict(
        problem_name="tiny_gaussian",
        cache_key="k",
        target=_make_1d_target(),
        num_results=50, num_warmup=50, num_chains=1, random_seed=0,
        cache_dir=cache_dir,
        quality_thresholds={
            "max_rhat": 1e3,
            "min_ess": 1.0,
            "max_divergence_rate": 1.0,
        },
    )

    ref_first = load_or_generate_reference_samples(**common_kwargs)
    assert isinstance(ref_first, NumericEmpiricalDistribution)
    artifact_dir = cache_dir / "tiny_gaussian"
    assert artifact_dir.exists()
    assert any(p.suffix == ".parquet" for p in artifact_dir.iterdir())

    call_count = {"n": 0}
    real_generate = cache_module.generate_via_nuts

    def counting_generate(*args, **kwargs):
        call_count["n"] += 1
        return real_generate(*args, **kwargs)

    monkeypatch.setattr(cache_module, "generate_via_nuts", counting_generate)

    ref_second = load_or_generate_reference_samples(
        regenerate=True, **common_kwargs
    )

    assert call_count["n"] == 1
    assert isinstance(ref_second, NumericEmpiricalDistribution)
    samples = jnp.asarray(ref_second.samples)
    assert samples.shape == (50, 1)
    assert jnp.all(jnp.isfinite(samples))
    assert jnp.all(samples >= -5.0)
    assert jnp.all(samples <= 5.0)
    assert jnp.any(jnp.all(samples == samples[0], axis=-1))
