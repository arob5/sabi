"""Shared test config and fixtures."""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Enable x64 so analytic references and Cholesky-cache equivalence tests
# have the precision they assume.
jax.config.update("jax_enable_x64", True)


def make_acquisition_state(
    *,
    problem=None,
    n: int = 20,
    seed: int = 0,
):
    """Build an `AcquisitionState` for tests of acquisitions / optimizers / imputers.

    Defaults to the validated `gaussian_2d()` benchmark with a fitted
    `TinyGPEmulator` and an `EmulatedDistribution` wrapping a
    ``LogProbTarget(...)``. Design points
    are drawn from a uniform-over-support distribution (the loop's
    default `initial_design_distribution`).

    Args:
        problem: optional `Problem`; defaults to `gaussian_2d()`.
        n: number of design points.
        seed: PRNG seed for the initial-design draw.

    Returns:
        `AcquisitionState` populated from the fitted emulator-backed SP.
    """
    from probpipe import sample as pp_sample
    from sabi._probpipe_compat import independent_uniform
    from sabi.acquisitions.base import AcquisitionState
    from sabi.acquisitions.random import DistributionSampling
    from sabi.algorithms.algorithm import Algorithm
    from sabi.density_decomposition import DensityDecomposition, LogProbTarget
    from sabi.emulators import TinyGPEmulator
    from sabi.problems.benchmarks import gaussian_2d
    from sabi.surrogate.surrogate_distribution import EmulatedDistribution

    if problem is None:
        problem = gaussian_2d()
    target = problem.target_distribution
    decomposition = LogProbTarget(target)
    # Uniform over the target's box support — the algorithm's default
    # initial-design distribution under the post-#65 layout.
    box = target.support
    initial_design = independent_uniform(
        low=jnp.asarray(box.low),
        high=jnp.asarray(box.high),
        name=f"{problem.name}_test_design",
    )
    X = jnp.asarray(pp_sample(initial_design, key=jax.random.key(seed), sample_shape=(n,)))
    Y = decomposition.target_map(X)
    emulator = TinyGPEmulator(input_shape=target.event_shape).fit(X, Y)
    surrogate_distribution = EmulatedDistribution(
        emulator=emulator,
        decomposition=decomposition,
        support=target.support,
    )
    algorithm = Algorithm(
        emulator_factory=lambda: TinyGPEmulator(input_shape=target.event_shape),
        acquisition=DistributionSampling(),
        density_decomposition=decomposition,
        initial_design_distribution=initial_design,
        x_support=target.support,
    )
    return AcquisitionState(
        problem=problem,
        algorithm=algorithm,
        surrogate_distribution=surrogate_distribution,
        X=X,
        Y_raw=Y,
        Y_train=Y,
        x_support=target.support,
    )
