"""`PosteriorMetric` protocol — distributions in, named-scalars out.

A metric is a callable taking a `Distribution[Array]` (the current posterior
estimate), the `Problem`, and a JAX PRNG `key`, and returning a dict of named
scalars. The dict lets related quantities stay bundled (e.g. MMD² and its
square root, forward+reverse KL from a single sample pair) without forcing
the caller to invoke multiple metrics for one computation.

**Protocol declaration.** Each metric class declares which ProbPipe
`Supports*` protocols it requires from the posterior estimate via the
`requires` class attribute. The loop introspects this and **raises** a
clear `MissingProtocolError` when the estimate doesn't satisfy a metric's
needs (rather than silently skipping). Example: a metric that needs samples
declares `requires = (SupportsSampling,)`; one that needs a closed-form
log-density declares `requires = (SupportsLogProb,)`.

Keys returned in the dict are merged into the per-round log row; namespace
your keys when a collision is possible across metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from jax import Array
from probpipe.core._distribution_base import Distribution

from sabi.problems.base import Problem


class PosteriorMetric(ABC):
    #: Tuple of ProbPipe `Supports*` protocol classes required from the
    #: posterior estimate. The loop verifies each and raises
    #: `MissingProtocolError` if any are not satisfied.
    requires: ClassVar[tuple[type, ...]] = ()

    @abstractmethod
    def __call__(
        self,
        posterior: Distribution,
        problem: Problem,
        *,
        key: Array,
    ) -> dict[str, float]:
        """Compute named scalar metrics for the current posterior estimate.

        Args:
            posterior: a ProbPipe `Distribution[Array]` over `problem.input_shape`.
                Satisfies every protocol in `self.requires`.
            problem: provides reference distribution, prior, etc. as needed.
            key: JAX PRNG key for any sampling the metric does internally.

        Returns:
            Dict mapping metric name → scalar.
        """


class MissingProtocolError(TypeError):
    """Raised when a `PosteriorMetric`'s required protocols aren't satisfied
    by the posterior estimate."""
