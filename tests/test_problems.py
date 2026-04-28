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
from sabi.problems.gaussian2d import gaussian2d


def test_gaussian2d_has_expected_shapes_and_types():
    problem = gaussian2d()
    assert problem.input_shape == (2,)
    assert problem.output_shape == ()
    assert isinstance(problem.prior, Distribution)
    assert isinstance(problem.support, Constraint)
    assert isinstance(problem.reference_distribution, MultivariateNormal)


def test_gaussian2d_log_prob_integrates_to_one():
    problem = gaussian2d()
    xs = jnp.linspace(-6.0, 6.0, 300)
    ys = jnp.linspace(-6.0, 6.0, 300)
    grid = jnp.stack(jnp.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    log_probs = jax.vmap(problem.target_function)(grid)
    dx = float((xs[1] - xs[0]) * (ys[1] - ys[0]))
    total = float(jnp.sum(jnp.exp(log_probs))) * dx
    assert total == pytest.approx(1.0, abs=1e-3)


def test_gaussian2d_reference_distribution_samples_match_moments():
    """`reference_distribution` is the analytic MVN; sampling from it should
    reproduce the requested mean and cov."""
    problem = gaussian2d(mean=(1.0, -0.5), cov=((2.0, 0.3), (0.3, 1.5)))
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
    log_probs = jax.vmap(problem.target_function)(grid)
    dx = float((xs[1] - xs[0]) * (ys[1] - ys[0]))
    total = float(jnp.sum(jnp.exp(log_probs))) * dx
    assert total == pytest.approx(1.0, abs=5e-3)


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
