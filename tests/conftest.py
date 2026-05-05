"""Shared test config and fixtures."""

from __future__ import annotations

import jax

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
    `TinyGPEmulator` and a `SurrogateDistribution` wrapping it. Design
    points come from `PriorSampler` — the same path used by acquisition
    candidate sets, so the GP is trained on the same support the
    acquisitions will explore.

    Replaces the three near-identical `_state(...)` helpers that lived
    in `test_acquisitions.py`, `test_optim.py`, and `test_fantasize.py`.

    Args:
        problem: optional `Problem`; defaults to `gaussian_2d()`.
        n: number of design points.
        seed: PRNG seed for the initial-design draw.

    Returns:
        `AcquisitionState` populated from the fitted SP / design.
    """
    from sabi.acquisitions.base import AcquisitionState
    from sabi.emulators import TinyGPEmulator
    from sabi.problems.benchmarks import gaussian_2d
    from sabi.sampling import PriorSampler
    from sabi.surrogate.surrogate_distribution import SurrogateDistribution

    if problem is None:
        problem = gaussian_2d()
    X = PriorSampler().sample(problem, jax.random.key(seed), n)
    Y = problem.target_map(X)
    emulator = TinyGPEmulator(input_shape=problem.input_shape).fit(X, Y)
    sp = SurrogateDistribution(
        emulator=emulator,
        log_density_form=problem.log_density_form,
        support=problem.support,
        input_shape=problem.input_shape,
        prior=problem.prior,
    )
    return AcquisitionState(
        problem=problem,
        surrogate_distribution=sp,
        X=X,
        Y_raw=Y,
        Y_train=Y,
    )
