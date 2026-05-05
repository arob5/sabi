import dataclasses

import jax
import jax.numpy as jnp
import pytest
from probpipe import sample
from probpipe.core._distribution_base import Distribution
from probpipe.core._empirical import NumericEmpiricalDistribution
from probpipe.core.constraints import Constraint
from probpipe.core.protocols import SupportsSampling
from probpipe.distributions.multivariate import MultivariateNormal

from sabi.problems.banana import banana
from sabi.problems.base import BenchmarkProblem, Problem
from sabi.problems.benchmarks import gaussian_2d
from sabi.problems.gaussian import gaussian


def test_gaussian_2d_has_expected_shapes_and_types():
    problem = gaussian_2d()
    assert problem.input_shape == (2,)
    assert problem.output_shape == ()
    assert isinstance(problem.prior, Distribution)
    assert isinstance(problem.support, Constraint)
    assert isinstance(problem.reference_distribution, MultivariateNormal)


def test_gaussian_2d_log_prob_integrates_to_one():
    problem = gaussian_2d()
    xs = jnp.linspace(-6.0, 6.0, 300)
    ys = jnp.linspace(-6.0, 6.0, 300)
    grid = jnp.stack(jnp.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    log_probs = problem.target_map(grid)
    dx = float((xs[1] - xs[0]) * (ys[1] - ys[0]))
    total = float(jnp.sum(jnp.exp(log_probs))) * dx
    assert total == pytest.approx(1.0, abs=1e-3)


def test_gaussian_reference_distribution_samples_match_custom_moments():
    """`reference_distribution` is the analytic MVN; sampling from it should
    reproduce the requested mean and cov when constructed via `gaussian(d=2, ...)`."""
    problem = gaussian(d=2, mean=(1.0, -0.5), cov=((2.0, 0.3), (0.3, 1.5)))
    samples = jnp.asarray(
        sample(
            problem.reference_distribution,
            key=jax.random.key(0),
            sample_shape=(4096,),
        )
    )
    emp_mean = jnp.mean(samples, axis=0)
    emp_cov = jnp.cov(samples, rowvar=False)
    assert jnp.allclose(emp_mean, jnp.asarray([1.0, -0.5]), atol=0.1)
    assert jnp.allclose(emp_cov, jnp.asarray([[2.0, 0.3], [0.3, 1.5]]), atol=0.2)


def test_gaussian_d_default_recovers_2d_shape():
    """`gaussian()` defaults to d=2; isotropic identity covariance."""
    p = gaussian()
    assert p.input_shape == (2,)
    assert isinstance(p.reference_distribution, MultivariateNormal)


def test_gaussian_higher_d_shapes_and_log_prob():
    """`gaussian(d=5)` exposes a 5-D Problem; log-density agrees with the
    underlying MultivariateNormal for the same x."""
    p = gaussian(d=5)
    assert p.input_shape == (5,)
    x = jnp.zeros(5)
    # At x=0 with mean=0, cov=I_5, log p = -0.5 * 5 * log(2π).
    expected = -0.5 * 5.0 * float(jnp.log(2 * jnp.pi))
    assert float(p.log_posterior(x)) == pytest.approx(expected, abs=1e-6)


def test_gaussian_rejects_d_less_than_1():
    with pytest.raises(ValueError, match="d must be"):
        gaussian(d=0)


def test_gaussian_rejects_mean_shape_mismatch():
    with pytest.raises(ValueError, match="mean must have shape"):
        gaussian(d=3, mean=(0.0, 0.0))  # only 2 entries for d=3


def test_gaussian_rejects_cov_shape_mismatch():
    with pytest.raises(ValueError, match="cov must be"):
        gaussian(d=3, cov=((1.0, 0.0), (0.0, 1.0)))  # 2x2 for d=3


def test_gaussian_rejects_non_pd_cov():
    with pytest.raises(ValueError, match="positive-definite"):
        # 2x2 matrix with negative determinant.
        gaussian(d=2, cov=((1.0, 2.0), (2.0, 1.0)))


def test_gaussian_rejects_invalid_bounds_radius():
    with pytest.raises(ValueError, match="bounds_radius"):
        gaussian(bounds_radius=0.0)


def test_gaussian_2d_benchmark_preserves_correlated_default():
    """`gaussian_2d()` (the validated `BenchmarkProblem` factory) ships
    a fixed correlated covariance; sampling from its analytic reference
    should reproduce those moments."""
    problem = gaussian_2d()
    samples = jnp.asarray(
        sample(
            problem.reference_distribution,
            key=jax.random.key(0),
            sample_shape=(4096,),
        )
    )
    emp_cov = jnp.cov(samples, rowvar=False)
    assert jnp.allclose(emp_cov, jnp.asarray([[1.0, 0.5], [0.5, 1.0]]), atol=0.1)


def test_banana_has_expected_shapes_and_types():
    problem = banana()
    assert problem.input_shape == (2,)
    assert problem.output_shape == ()
    assert isinstance(problem.prior, Distribution)
    assert isinstance(problem.support, Constraint)
    assert isinstance(problem.reference_distribution, NumericEmpiricalDistribution)
    assert isinstance(problem.reference_distribution, SupportsSampling)


def test_banana_log_prob_integrates_to_one():
    problem = banana(a=1.0, b=4.0)
    xs = jnp.linspace(-4.0, 4.0, 300)
    ys = jnp.linspace(-10.0, 4.0, 300)
    grid = jnp.stack(jnp.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    log_probs = problem.target_map(grid)
    dx = float((xs[1] - xs[0]) * (ys[1] - ys[0]))
    total = float(jnp.sum(jnp.exp(log_probs))) * dx
    assert total == pytest.approx(1.0, abs=5e-3)


def test_banana_d_default_recovers_2d():
    """`banana()` with no args must equal `banana(d=2)` — back-compat."""
    p_default = banana()
    p_explicit = banana(d=2)
    assert p_default.input_shape == p_explicit.input_shape == (2,)
    # The log-density at the same point must agree (both use Identity form).
    x = jnp.asarray([0.3, -1.7])
    assert float(p_default.log_posterior(x)) == pytest.approx(
        float(p_explicit.log_posterior(x)), abs=1e-12
    )


def test_banana_higher_d_shapes():
    """`banana(d=10)` exposes a 10-D Problem."""
    problem = banana(d=10)
    assert problem.input_shape == (10,)
    assert problem.output_shape == ()
    assert isinstance(problem.reference_distribution, NumericEmpiricalDistribution)


def test_banana_higher_d_log_prob_factored_marginals():
    r"""For d ≥ 3 the i ≥ 3 dims are independent N(0, c²); fixing the
    first two and varying x_3 should reproduce the Gaussian quadratic
    in x_3."""
    a, b, c = 1.0, 4.0, 1.5
    problem = banana(d=4, a=a, b=b, c=c)
    base = jnp.asarray([0.5, -1.0, 0.0, 0.0])
    perturbed = jnp.asarray([0.5, -1.0, 0.7, -0.4])
    diff = float(problem.log_posterior(perturbed) - problem.log_posterior(base))
    expected = -0.5 / (c * c) * float(0.7 ** 2 + 0.4 ** 2)
    assert diff == pytest.approx(expected, abs=1e-10)


def test_banana_higher_d_reference_marginals():
    """Reference samples reproduce the analytic per-dim marginals:
    var(x_1) ≈ a², z = x_2 + x_1² - a² has var ≈ 1/b, var(x_i) ≈ c² for i ≥ 3.
    """
    a, b, c = 1.0, 4.0, 1.5
    problem = banana(d=5, a=a, b=b, c=c, n_reference_samples=8000)
    samples = jnp.asarray(
        sample(
            problem.reference_distribution,
            key=jax.random.key(1),
            sample_shape=(),  # reference is already a sample bank; () returns it as-is
        )
    )
    if samples.ndim == 1:
        # Some empirical implementations return a single-row draw for () shape;
        # fall back to the stored samples directly.
        samples = jnp.asarray(problem.reference_distribution.samples)
    x1 = samples[:, 0]
    x2 = samples[:, 1]
    z = x2 + x1 ** 2 - a ** 2
    assert float(jnp.var(x1)) == pytest.approx(a ** 2, abs=0.1)
    assert float(jnp.var(z)) == pytest.approx(1.0 / b, abs=0.05)
    for i in range(2, 5):
        assert float(jnp.mean(samples[:, i])) == pytest.approx(0.0, abs=0.1)
        assert float(jnp.var(samples[:, i])) == pytest.approx(c ** 2, abs=0.2)


def test_banana_rejects_d_less_than_2():
    with pytest.raises(ValueError, match="d must be"):
        banana(d=1)


def test_banana_rejects_invalid_scales():
    with pytest.raises(ValueError):
        banana(a=0.0)
    with pytest.raises(ValueError):
        banana(b=-1.0)
    with pytest.raises(ValueError):
        banana(d=3, c=0.0)


def test_banana_custom_bounds_length_validated():
    with pytest.raises(ValueError, match="bounds must have length"):
        banana(d=3, bounds=((-1.0, 1.0), (-1.0, 1.0)))


def test_benchmark_problem_requires_reference_distribution():
    """Constructing a BenchmarkProblem without a reference must fail —
    the validated tier exists exactly to guarantee one is present."""
    base = banana()
    with pytest.raises(ValueError, match="reference_distribution"):
        BenchmarkProblem(
            target_distribution=base.target_distribution,
            reference_distribution=None,
            name="missing_ref",
        )


def test_benchmark_problem_requires_name():
    """The benchmark's name is its identity; empty names are rejected."""
    base = banana()
    with pytest.raises(ValueError, match="name"):
        BenchmarkProblem(
            target_distribution=base.target_distribution,
            reference_distribution=base.reference_distribution,
            name="",
        )


def test_benchmark_problem_is_frozen():
    """Frozen dataclass: instances cannot be mutated after construction."""
    base = banana()
    bp = BenchmarkProblem(
        target_distribution=base.target_distribution,
        reference_distribution=base.reference_distribution,
        name="banana_2d",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        bp.name = "renamed"  # type: ignore[misc]


def test_benchmark_problem_isinstance_of_problem():
    """A BenchmarkProblem IS-A Problem — every Problem-consuming caller
    accepts it without changes (loop, build.py, metrics, ...)."""
    base = banana()
    bp = BenchmarkProblem(
        target_distribution=base.target_distribution,
        reference_distribution=base.reference_distribution,
        name="banana_2d",
    )
    assert isinstance(bp, Problem)
    assert bp.input_shape == base.input_shape
    assert bp.artifact_version == "v1"


def test_banana_reference_samples_satisfy_constraint():
    """Under z = x₂ + x₁² - a², z ~ N(0, 1/b)."""
    a, b = 1.0, 4.0
    problem = banana(a=a, b=b)
    samples = jnp.asarray(
        sample(
            problem.reference_distribution,
            key=jax.random.key(0),
            sample_shape=(4096,),
        )
    )
    x1 = samples[:, 0]
    x2 = samples[:, 1]
    z = x2 + x1 ** 2 - a ** 2
    assert float(jnp.mean(z)) == pytest.approx(0.0, abs=0.05)
    assert float(jnp.var(z)) == pytest.approx(1.0 / b, abs=0.05)
    assert float(jnp.var(x1)) == pytest.approx(a ** 2, abs=0.1)
