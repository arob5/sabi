import jax.numpy as jnp
import pytest
from probpipe.distributions.continuous import Normal, Uniform

from sabi.problems.forms import ForwardModel, Identity, LogLikPlusPrior


def test_identity_passes_y_through():
    form = Identity()
    x = jnp.asarray([0.3])
    y = jnp.asarray(-1.5)
    assert float(form(x, y)) == -1.5


def test_log_lik_plus_prior_adds_prior():
    """Standard normal prior with x=2 → log_prior(2) = -0.5*4 - 0.5*log(2π)."""
    prior = Normal(loc=jnp.asarray(0.0), scale=jnp.asarray(1.0), name="p")
    form = LogLikPlusPrior()
    x = jnp.asarray(2.0)  # scalar; matches Normal's event_shape == ()
    y = jnp.asarray(1.5)
    expected = 1.5 + (-0.5 * 4.0 - 0.5 * float(jnp.log(2 * jnp.pi)))
    assert float(form(x, y, prior=prior)) == pytest.approx(expected, abs=1e-5)


def test_log_lik_plus_prior_errors_without_prior():
    form = LogLikPlusPrior()
    with pytest.raises(ValueError, match="prior"):
        form(jnp.asarray([0.0]), jnp.asarray(0.0), prior=None)


def test_forward_model_combines_likelihood_and_prior():
    """Uniform prior on [-2, 2] (uniform density 1/4 → log = -log(4)).
    Likelihood is -0.5 * (y - 1)^2 with y=2.0 → -0.5."""
    prior = Uniform(low=jnp.asarray(-2.0), high=jnp.asarray(2.0), name="p")
    form = ForwardModel(log_lik_from_outputs=lambda x, y: -0.5 * jnp.sum((y - 1.0) ** 2))
    x = jnp.asarray(1.0)
    y = jnp.asarray([2.0])
    expected = -0.5 + (-jnp.log(4.0))
    assert float(form(x, y, prior=prior)) == pytest.approx(float(expected), abs=1e-5)
