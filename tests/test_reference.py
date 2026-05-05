"""Tests for sabi.reference: IO round-trip + cache load/regenerate logic.

Covers:
- Parquet write/read round-trip preserves shapes and values.
- Metadata JSON round-trip.
- `load_or_generate_reference_samples` cache hit (existing artifact loads
  without invoking NUTS).
- `regenerate=True` forces fresh generation even when a cached artifact
  exists.
- Quality threshold gating raises a clear error on bad NUTS runs.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import pytest
from probpipe.core._empirical import NumericEmpiricalDistribution

from sabi._probpipe_compat import independent_uniform
from sabi.problems.forms import Identity
from sabi.reference import cache as cache_module
from sabi.reference.cache import load_or_generate_reference_samples
from sabi.reference.io import (
    read_metadata_json,
    read_samples_parquet,
    write_metadata_json,
    write_samples_parquet,
)


def _scalar_normal_target(theta):
    """Standard 2-D normal log-density (used for cheap NUTS smoke tests)."""
    return -0.5 * jnp.sum(theta ** 2)


def _tiny_normal_target(theta):
    """Standard 1-D normal log-density (used for the cheapest possible NUTS run)."""
    return -0.5 * jnp.sum(theta ** 2)


def _box_prior():
    """A 2-D multivariate-event uniform on [-5, 5]^2 — used wherever
    the cache layer wants a prior (sometimes a no-op cache hit; pass
    a valid prior anyway since `prior` is now required)."""
    return independent_uniform(
        low=jnp.full(2, -5.0), high=jnp.full(2, 5.0), name="box_prior"
    )


def _tiny_box_prior():
    """A 1-D multivariate-event uniform on [-5, 5] for the cheapest NUTS test."""
    return independent_uniform(
        low=jnp.full(1, -5.0), high=jnp.full(1, 5.0), name="tiny_box_prior"
    )


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


def test_cache_hit_skips_nuts(tmp_path):
    """If the artifact files exist, load from disk without running NUTS.

    We construct the artifact manually (bypassing NUTS) and confirm
    `load_or_generate_reference_samples` returns those exact samples.
    """
    cache_dir = tmp_path / "refs"
    problem_dir = cache_dir / "fake_problem"
    problem_dir.mkdir(parents=True)
    # Match the filename construction in cache._artifact_path /  _sampler_tag.
    fname = "k1_nuts_n10_w5_c1_s0"
    samples = jax.random.normal(jax.random.key(7), shape=(10, 2))
    write_samples_parquet(samples, problem_dir / f"{fname}.parquet")
    write_metadata_json({"problem_name": "fake_problem"}, problem_dir / f"{fname}.json")

    # Pass deliberately-broken target_map — if it gets called, the test fails.
    def bad_target(theta):
        raise AssertionError("NUTS must not run on a cache hit.")

    ref = load_or_generate_reference_samples(
        problem_name="fake_problem",
        cache_key="k1",
        target_map=bad_target,
        log_density_form=Identity(),
        prior=_box_prior(),
        input_shape=(2,),
        num_results=10,
        num_warmup=5,
        num_chains=1,
        random_seed=0,
        cache_dir=cache_dir,
    )
    assert isinstance(ref, NumericEmpiricalDistribution)
    assert jnp.allclose(jnp.asarray(ref.samples), samples, atol=1e-6)


def test_cache_keys_disambiguate_by_params(tmp_path):
    """Different `cache_key` → different artifact path; one cached entry
    doesn't shadow the other."""
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

    def _bad(theta):
        raise AssertionError("Should be a cache hit.")

    ref_a = load_or_generate_reference_samples(
        problem_name="fake_problem",
        cache_key="kA",
        target_map=_bad,
        log_density_form=Identity(),
        prior=_box_prior(),
        input_shape=(2,),
        num_results=4,
        num_warmup=2,
        num_chains=1,
        random_seed=0,
        cache_dir=cache_dir,
    )
    ref_b = load_or_generate_reference_samples(
        problem_name="fake_problem",
        cache_key="kB",
        target_map=_bad,
        log_density_form=Identity(),
        prior=_box_prior(),
        input_shape=(2,),
        num_results=4,
        num_warmup=2,
        num_chains=1,
        random_seed=0,
        cache_dir=cache_dir,
    )
    assert jnp.allclose(jnp.asarray(ref_a.samples), samples_a)
    assert jnp.allclose(jnp.asarray(ref_b.samples), samples_b)


def test_quality_threshold_failure_raises(tmp_path):
    """Setting an absurdly high `min_ess` threshold must cause regeneration
    to raise rather than silently saving a bad artifact."""
    cache_dir = tmp_path / "refs"

    with pytest.raises(ValueError, match="min ESS"):
        load_or_generate_reference_samples(
            problem_name="bad_quality",
            cache_key="k",
            target_map=_scalar_normal_target,
            log_density_form=Identity(),
            prior=_box_prior(),
            input_shape=(2,),
            num_results=20,
            num_warmup=10,
            num_chains=1,
            random_seed=0,
            cache_dir=cache_dir,
            quality_thresholds={"min_ess": 1e9},
        )

    # No artifact should have been written.
    assert not (cache_dir / "bad_quality").exists() or not any(
        p.is_file() for p in (cache_dir / "bad_quality").iterdir()
    )


def test_regenerate_forces_fresh_nuts_run(tmp_path, monkeypatch):
    """`regenerate=True` must invoke NUTS even when a cached artifact exists.

    Strategy: do a real NUTS run on a tiny 1-D Gaussian to populate the
    cache, then call again with `regenerate=True` while wrapping
    `generate_via_nuts` in a counting proxy. The second call must hit the
    proxy exactly once, and must return a well-formed
    `NumericEmpiricalDistribution` over post-warmup draws.

    The NUTS budget is intentionally tiny (50 results, 50 warmup, 1 chain)
    so the test runs in a few seconds. R-hat / ESS thresholds are loosened
    to match — this isn't a quality test.
    """
    cache_dir = tmp_path / "refs"
    common_kwargs = dict(
        problem_name="tiny_gaussian",
        cache_key="k",
        target_map=_tiny_normal_target,
        log_density_form=Identity(),
        prior=_tiny_box_prior(),
        input_shape=(1,),
        num_results=50,
        num_warmup=50,
        num_chains=1,
        random_seed=0,
        cache_dir=cache_dir,
        quality_thresholds={
            "max_rhat": 1e3,
            "min_ess": 1.0,
            "max_divergence_rate": 1.0,
        },
    )

    # 1. Populate the cache with a real NUTS run.
    ref_first = load_or_generate_reference_samples(**common_kwargs)
    assert isinstance(ref_first, NumericEmpiricalDistribution)
    # Cache-on-disk sanity check: artifact files must exist now.
    artifact_dir = cache_dir / "tiny_gaussian"
    assert artifact_dir.exists()
    assert any(p.suffix == ".parquet" for p in artifact_dir.iterdir())

    # 2. Wrap generate_via_nuts in a counting proxy and call again with
    #    `regenerate=True`. The proxy must fire exactly once.
    call_count = {"n": 0}
    real_generate = cache_module.generate_via_nuts

    def counting_generate(*args, **kwargs):
        call_count["n"] += 1
        return real_generate(*args, **kwargs)

    monkeypatch.setattr(cache_module, "generate_via_nuts", counting_generate)

    ref_second = load_or_generate_reference_samples(
        regenerate=True, **common_kwargs
    )

    assert call_count["n"] == 1, (
        "regenerate=True must invoke generate_via_nuts exactly once even "
        "when an artifact is already cached on disk."
    )

    # 3. Verify the regenerated distribution is well-formed.
    assert isinstance(ref_second, NumericEmpiricalDistribution)
    samples = jnp.asarray(ref_second.samples)
    # 1 chain * 50 post-warmup draws -> (50, 1).
    assert samples.shape == (50, 1)
    # Every sample is finite and lives in the prior's support box.
    assert jnp.all(jnp.isfinite(samples))
    assert jnp.all(samples >= -5.0)
    assert jnp.all(samples <= 5.0)
    # The first sample of `ref_second.samples` should be in `ref_second`'s
    # own sample set (membership sanity).
    assert jnp.any(jnp.all(samples == samples[0], axis=-1))
