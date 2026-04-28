"""`Surrogate` — fittable predictive process for some target map.

`Surrogate` inherits from ProbPipe's `ArrayRandomFunction`: it IS a random
function and gets the full shape contract for free
(`__call__(X, joint_inputs, joint_outputs) -> Distribution`, `input_shape`,
`output_shape`, `_validate_X`, etc.). The only addition is the abstract
`fit(X, Y) -> Self`, which signals the algorithmic role: a surrogate is
typically constructed by conditioning a prior random function on training
evaluations of the target.

Concrete Gaussian surrogates inherit from both `Surrogate` and
`GaussianRandomFunction` (diamond inheritance over `ArrayRandomFunction`,
resolved by Python's C3 MRO). See `sabi.surrogates.gp.GPSurrogate`.

Forward-look (post-v1.2): ProbPipe's `condition_on(prior_rf, X=X, y=Y)` is
the natural way to build a posterior random function from training data.
The `fit` method here is a v1.x bridge that captures the same idea without
requiring sabi to wire ProbPipe's full conditioning machinery yet.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Self

from jax import Array
from probpipe.core._random_functions import ArrayRandomFunction


class Surrogate(ArrayRandomFunction):
    """A fittable predictive process used as a surrogate for a target map.

    Subclasses implement `fit(X, Y) -> Self` and either inherit the
    `predict` / `__call__` machinery from a sibling class
    (e.g. `GaussianRandomFunction`) or override `predict` directly.
    """

    @abstractmethod
    def fit(self, X: Array, Y: Array) -> Self:
        """Condition the surrogate on training evaluations of the target.

        Args:
            X: training inputs, shape `(n,) + input_shape`.
            Y: training outputs, shape `(n,) + output_shape`.

        Returns:
            A new (or self-mutated) `Surrogate` carrying the conditioned state.
        """
