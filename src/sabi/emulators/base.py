"""`Emulator` — fittable predictive model for some target map.

`Emulator` IS a ProbPipe `ArrayRandomFunction` and inherits the full
shape contract. The only addition is the abstract `fit(X, Y) -> Self`.
Concrete Gaussian emulators (`TinyGPEmulator`, `DSPGPEmulator`) inherit
from both `Emulator` and `GaussianRandomFunction`. See ``docs/design.md``
§4.3 for the architectural context and ``docs/notation.md`` for the
"emulator" vs. "surrogate" naming convention.

Latent vs. observation predictive
---------------------------------

Sabi emulators report the **latent** posterior — `predict_*` methods
do NOT add observation noise to the diagonal. The GP's noise term in
the deterministic-target setting is primarily a Cholesky-stability
regularizer; acquisitions (EI, etc.) and pushforward-based estimators
want the latent uncertainty.

Callers that want the observation-predictive distribution build it
explicitly: ``Var[y* | data] = predict_variance(X) + obs_noise_variance``.
Backends with a fitted noise term expose it via the
``obs_noise_variance`` property; backends without one return ``None``.
When noisy targets become user-facing, this module will grow a separate
``predict_obs_*`` family rather than overloading the existing methods.
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

    @property
    def obs_noise_variance(self) -> Array | None:
        """Observation-noise variance, or ``None`` if not modeled.

        Sabi's ``predict_*`` family returns the **latent** posterior
        — variance/covariance of the latent function ``f(x*)``, with
        no observation noise added on the diagonal. Callers that want
        the observation-predictive variance build it explicitly:

        .. code-block:: python

            obs_var = em.predict_variance(X)
            sigma2 = em.obs_noise_variance
            if sigma2 is not None:
                obs_var = obs_var + sigma2

        Backends that fit an observation-noise term (e.g.
        ``TinyGPEmulator``'s constructor ``noise``,
        ``DSPGPEmulator``'s MAP-fitted ``obs_stddev``) override this
        property; the base returns ``None``.

        Returns:
            Scalar JAX array (variance, not stddev), or ``None`` if
            the emulator does not model observation noise.
        """
        return None
