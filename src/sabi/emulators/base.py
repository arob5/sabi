"""`Emulator` — fittable predictive model for some target map.

`Emulator` inherits from ProbPipe's `ArrayRandomFunction`: it IS a random
function and gets the full shape contract for free
(`__call__(X, joint_inputs, joint_outputs) -> Distribution`, `input_shape`,
`output_shape`, `_validate_X`, etc.). The only addition is the abstract
`fit(X, Y) -> Self`, which signals the algorithmic role: an emulator is
typically constructed by conditioning a prior random function on training
evaluations of the target.

Concrete Gaussian emulators inherit from both `Emulator` and
`GaussianRandomFunction` (diamond inheritance over `ArrayRandomFunction`,
resolved by Python's C3 MRO). See `sabi.emulators.tinygp.gp.GPEmulator`.

Forward-look (post-v1.2): ProbPipe's `condition_on(prior_rf, X=X, y=Y)` is
the natural way to build a posterior random function from training data.
The `fit` method here is a v1.x bridge that captures the same idea without
requiring sabi to wire ProbPipe's full conditioning machinery yet.

Naming convention: in sabi, "emulator" is reserved specifically for the
predictive model fit to observations of the target function. The broader
word "surrogate" denotes any approximate quantity replacing its exact
analog (hence `SurrogatePosterior` for the surrogate of the true
posterior).

Latent vs. observation predictive
---------------------------------

Sabi emulators report the **latent** posterior. Concretely:

- ``predict_mean(X)``: posterior mean of the latent function ``f(X)``.
- ``predict_variance(X)``: marginal posterior variance of ``f(X)``,
  with **no observation noise added to the diagonal**.
- ``predict_covariance(X, joint_inputs=True)``: full posterior
  covariance of ``f(X)``, with no observation noise added.

Rationale: sabi targets sequential surrogate-based Bayesian inference
on (in v1) deterministic targets. The GP's noise term is primarily a
Cholesky-stability regularizer rather than a model of real measurement
noise. Acquisition functions (EI, etc.) and pushforward-based posterior
estimators want the *latent* uncertainty — the uncertainty over the
true function value at unobserved inputs — not the broader observation-
predictive uncertainty.

Callers that want the observation-predictive distribution can build it
explicitly: ``Var[y* | data] = predict_variance(X) + sigma_n²``, where
``sigma_n²`` is whatever observation-noise model the application has.
Sabi emulators don't expose an ``obs_stddev`` accessor in v1; backends
that have one (e.g. ``DSPGPEmulator``'s gpjax-fitted ``obs_stddev``)
keep it as internal implementation detail.

When v2's noisy-target work lands and the latent vs. observation
distinction becomes user-facing, this module will grow a separate
``predict_obs_*`` family rather than overloading the existing
``predict_*`` methods.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Self

from jax import Array
from probpipe.core._random_functions import ArrayRandomFunction


class Emulator(ArrayRandomFunction):
    """A fittable predictive model used as an emulator for a target map.

    Subclasses implement `fit(X, Y) -> Self` and either inherit the
    `predict` / `__call__` machinery from a sibling class
    (e.g. `GaussianRandomFunction`) or override `predict` directly.
    """

    @abstractmethod
    def fit(self, X: Array, Y: Array) -> Self:
        """Condition the emulator on training evaluations of the target.

        Args:
            X: training inputs, shape `(n,) + input_shape`.
            Y: training outputs, shape `(n,) + output_shape`.

        Returns:
            A new (or self-mutated) `Emulator` carrying the conditioned state.
        """
