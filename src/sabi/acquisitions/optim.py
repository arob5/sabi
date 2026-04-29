"""Pointwise optimizers for `PointwiseScoredAcquisition`.

Three concrete optimizers ship in v1.4:

- `CandidateSetOptimizer` — random candidates from the design distribution,
  scored vmapped, top-q returned. The v1.x EI behavior; cheap and the
  default for backwards compatibility.
- `ContinuousMultiStartOptimizer` — random candidates, score-filter to
  top-`n_starts`, BFGS each in unconstrained reparameterization space,
  return top-q distinct local maxima. Gradient-based; substantially more
  accurate on smooth scores when the surrogate is well-conditioned.
- `GreedyMultiPointOptimizer` — for `q > 1` with in-batch diversity. Wraps
  an inner optimizer (any of the above with `q=1`); after each pick,
  hallucinates an observation at the pending point via a pluggable
  `FantasyImputer` and refits the surrogate before the next pick.

For non-trivial supports (anything other than `interval(low, high)`), the
continuous optimizer raises with a pointer to the `Constraint` → bijector
gap in `docs/probpipe_issues.md`. v1.4's benchmarks all have box supports.

Multi-start currently uses a Python loop over BFGS solves. JAX-vmap of
`optimistix.minimise` should work in principle and is the natural future
optimization (TODO marker on the relevant block).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array
from probpipe.core.constraints import Constraint, _Interval

from sabi.acquisitions.base import AcquisitionState
from sabi.acquisitions.fantasize import FantasyImputer, KrigingBeliever
from sabi.initial_designs.base import sample_initial

if TYPE_CHECKING:
    from sabi.acquisitions.base import PointwiseScoredAcquisition


# ---------------------------------------------------------------------------
# Optimizer base
# ---------------------------------------------------------------------------


class PointwiseOptimizer(ABC):
    """Optimizes a pointwise score over the problem's design space."""

    @abstractmethod
    def optimize(
        self,
        acq: "PointwiseScoredAcquisition",
        state: AcquisitionState,
        q: int,
        key: Array,
    ) -> Array:
        """Return `(q,) + state.problem.input_shape`."""


# ---------------------------------------------------------------------------
# Candidate-set: random candidates → top-q. v1.x EI's default.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateSetOptimizer(PointwiseOptimizer):
    """Random candidates from `problem.prior`, score each, return top-q.

    Cheap; gradient-free; matches v1.x EI behavior. Quality is governed
    by `n_candidates` and the prior's coverage of the high-score regions.
    """

    n_candidates: int = 1024

    def optimize(self, acq, state, q, key):
        key_cand, _ = jax.random.split(key)
        candidates = sample_initial(state.problem, key_cand, self.n_candidates)
        scores = jax.vmap(lambda x: acq.score(x, state))(candidates)
        top_idx = jnp.argsort(-scores)[:q]
        return candidates[top_idx]


# ---------------------------------------------------------------------------
# Continuous multi-start: BFGS in reparameterized space, top-q maxima.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContinuousMultiStartOptimizer(PointwiseOptimizer):
    """Multi-start BFGS in unconstrained reparameterization space.

    Pipeline:

    1. Sample `n_seeding_candidates` from `problem.prior`.
    2. Score each (vmap); take top-`n_starts` as BFGS init points.
    3. Reparameterize each start to unconstrained space (sigmoid for
       `interval(low, high)` supports).
    4. Run optimistix BFGS on the *negative* score in unconstrained space
       (Python loop over starts; vmap is a future optimization).
    5. Map the optimized points back to support space, re-score, return
       top-q.

    Args:
        n_starts: number of BFGS seeds (default 16).
        n_seeding_candidates: candidates drawn before score-filtering to
            seeds (default 256). Should be >= `n_starts`.
        bfgs_max_steps: max steps per BFGS solve.
        bfgs_rtol / bfgs_atol: convergence tolerances.
    """

    n_starts: int = 16
    n_seeding_candidates: int = 256
    bfgs_max_steps: int = 50
    bfgs_rtol: float = 1e-5
    bfgs_atol: float = 1e-5

    def optimize(self, acq, state, q, key):
        if self.n_seeding_candidates < self.n_starts:
            raise ValueError(
                f"n_seeding_candidates ({self.n_seeding_candidates}) must be "
                f">= n_starts ({self.n_starts})."
            )
        bijector = _make_bijector(state.problem.support)

        # 1-2: seed selection
        key_seed, _ = jax.random.split(key)
        seed_candidates = sample_initial(
            state.problem, key_seed, self.n_seeding_candidates
        )
        seed_scores = jax.vmap(lambda x: acq.score(x, state))(seed_candidates)
        top_seed_idx = jnp.argsort(-seed_scores)[: self.n_starts]
        starts = seed_candidates[top_seed_idx]

        # 3: reparameterize starts
        u_starts = jax.vmap(bijector.inverse)(starts)

        # 4: BFGS each start in unconstrained space
        def neg_score_unconstrained(u: Array, _args=None) -> Array:
            x = bijector.forward(u)
            return -acq.score(x, state)

        solver = optx.BFGS(rtol=self.bfgs_rtol, atol=self.bfgs_atol)
        # TODO(vmap-multistart): replace this Python loop with
        # `jax.vmap(optimistix.minimise, ...)` once we've validated it
        # works with our solver settings. The per-call BFGS body is
        # already JIT-compiled by optimistix.
        u_opt_list: list[Array] = []
        for u_init in u_starts:
            try:
                sol = optx.minimise(
                    neg_score_unconstrained,
                    solver,
                    u_init,
                    args=None,
                    max_steps=self.bfgs_max_steps,
                    throw=False,
                )
                u_opt = sol.value
            except Exception:
                u_opt = u_init
            u_opt = jnp.where(jnp.isfinite(u_opt), u_opt, u_init)
            u_opt_list.append(u_opt)
        u_opt_stack = jnp.stack(u_opt_list)

        # 5: forward-map and re-score
        x_opt = jax.vmap(bijector.forward)(u_opt_stack)
        opt_scores = jax.vmap(lambda x: acq.score(x, state))(x_opt)
        # Replace any non-finite scores with -inf so they sort to the bottom.
        opt_scores = jnp.where(jnp.isfinite(opt_scores), opt_scores, -jnp.inf)
        top_q_idx = jnp.argsort(-opt_scores)[:q]
        return x_opt[top_q_idx]


# ---------------------------------------------------------------------------
# Greedy multi-point: pick → fantasize → pick → ... for q > 1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GreedyMultiPointOptimizer(PointwiseOptimizer):
    """For `q > 1`: iteratively pick one point at a time, hallucinating
    observations at each pick to drive in-batch diversity on the next pick.

    The inner optimizer picks each individual point (typically with `q=1`).
    The fantasy imputer decides what y-value the hallucinated point gets;
    `KrigingBeliever` (default) uses the surrogate's predictive mean —
    standard for greedy BO.

    Args:
        inner: per-point optimizer (e.g., `CandidateSetOptimizer()`).
        imputer: how to hallucinate `y` at pending points.
    """

    inner: PointwiseOptimizer
    imputer: FantasyImputer = field(default_factory=KrigingBeliever)

    def optimize(self, acq, state, q, key):
        if q < 1:
            raise ValueError(f"q must be ≥ 1, got {q}.")
        keys = jax.random.split(key, q)
        x_picks: list[Array] = []
        cur_state = state
        for i in range(q):
            x_i = self.inner.optimize(acq, cur_state, 1, keys[i])[0]
            x_picks.append(x_i)
            if i < q - 1:
                # Hallucinate y at the pending picks (use the *original* state
                # so the imputer always sees real Y values, not previous
                # hallucinations).
                x_pending = jnp.stack(x_picks)
                y_pending = self.imputer.impute(x_pending, state)
                new_X = jnp.concatenate([state.X, x_pending], axis=0)
                new_Y = jnp.concatenate([state.Y, y_pending], axis=0)
                new_surrogate = state.surrogate.fit(new_X, new_Y)
                cur_state = replace(
                    state,
                    surrogate=new_surrogate,
                    X=new_X,
                    Y=new_Y,
                )
        return jnp.stack(x_picks)


# ---------------------------------------------------------------------------
# Constraint → bijector for unconstrained reparameterization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _IntervalSigmoidBijector:
    """Element-wise sigmoid mapping unconstrained ℝ ↔ `interval(low, high)`.

    Forward: `x = low + (high - low) * sigmoid(u)`.
    Inverse: `u = log((x - low) / (high - x))` — the logit of the
        normalized coordinate.

    Both work element-wise on any input shape; the rank of `low` / `high`
    matches `problem.input_shape`.
    """

    low: Array
    high: Array

    def forward(self, u: Array) -> Array:
        return self.low + (self.high - self.low) * jax.nn.sigmoid(u)

    def inverse(self, x: Array) -> Array:
        # Clip so logit doesn't blow up at the boundary.
        eps = 1e-6
        norm = (x - self.low) / (self.high - self.low)
        norm = jnp.clip(norm, eps, 1.0 - eps)
        return jnp.log(norm) - jnp.log1p(-norm)


def _make_bijector(constraint: Constraint) -> _IntervalSigmoidBijector:
    """Build an unconstrained-↔-support bijector from a `Constraint`.

    v1.4 supports only `interval(low, high)` constraints. For other
    constraint types, raises `NotImplementedError` with a pointer to the
    `Constraint → bijector` ProbPipe gap.
    """
    if isinstance(constraint, _Interval):
        low = jnp.asarray(constraint.low)
        high = jnp.asarray(constraint.high)
        return _IntervalSigmoidBijector(low=low, high=high)
    raise NotImplementedError(
        f"ContinuousMultiStartOptimizer: no reparameterization for "
        f"Constraint of type {type(constraint).__name__}. v1.4 supports "
        f"only `interval(low, high)` supports. See "
        f"docs/probpipe_issues.md: 'No Constraint → TFP bijector registry'."
    )
