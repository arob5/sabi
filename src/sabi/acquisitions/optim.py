r"""Pointwise optimizers for `PointwiseScoredAcquisition`.

Three concrete optimizers ship today:

- ``CandidateSetOptimizer`` — random candidates from
  ``state.algorithm.initial_design_distribution`` (or the optimizer's
  ``candidate_distribution`` override), scored via
  ``acq.score(X, state)``, top-q returned. Cheap and the default.
- ``ContinuousMultiStartOptimizer`` — random seeds, score-filter to
  top-`n_starts`, BFGS each in unconstrained reparameterization space
  (TFP bijector dispatched on ``state.x_support``), return top-q
  distinct local maxima. Gradient-based.
- ``GreedyMultiPointOptimizer`` — for ``q > 1`` with in-batch
  diversity via fantasy imputation.

Per-role distribution fields after the ``DensityDecomposition`` split
(issue #65):

- ``CandidateSetOptimizer.candidate_distribution`` — ``Distribution``
  whose samples seed the candidate set. Defaults to
  ``state.algorithm.initial_design_distribution`` when ``None``.
- ``ContinuousMultiStartOptimizer.seed_distribution`` — ``Distribution``
  whose samples seed BFGS. Same default.

For non-trivial supports (anything other than ``interval(low, high)``),
``ContinuousMultiStartOptimizer`` raises with a pointer to the
``Constraint`` → bijector gap in ``docs/probpipe_issues.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array
from probpipe import sample as pp_sample
from probpipe.core._distribution_base import Distribution
from probpipe.core.constraints import Constraint, _Interval
from tensorflow_probability.substrates.jax.bijectors import Bijector, Sigmoid

from sabi.acquisitions.base import AcquisitionState
from sabi.acquisitions.fantasize import FantasyImputer, KrigingBeliever
from sabi.surrogate.surrogate_distribution import EmulatedDistribution

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
        """Return ``(q,) + state.problem.target_distribution.input_shape``."""


def _resolve_optimizer_distribution(
    explicit: Distribution | None,
    state: AcquisitionState,
    *,
    field_name: str,
    optimizer_class: str,
) -> Distribution:
    """Resolve a per-optimizer distribution field, falling back to the algorithm's default."""
    if explicit is not None:
        return explicit
    dist = state.algorithm.initial_design_distribution
    if dist is None:
        raise ValueError(
            f"{optimizer_class}: `{field_name}` is None and "
            "`state.algorithm.initial_design_distribution` is not set. "
            f"Pass `{optimizer_class}({field_name}=...)` explicitly, "
            "or set `Algorithm.initial_design_distribution` "
            "(or `x_support`)."
        )
    return dist


# ---------------------------------------------------------------------------
# Candidate-set: random candidates → top-q. The default optimizer.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateSetOptimizer(PointwiseOptimizer):
    r"""Random candidates from ``candidate_distribution``, score each, return top-q.

    Cheap; gradient-free. Quality is governed by ``n_candidates`` and
    the candidate distribution's coverage of the high-score regions.

    Args:
        n_candidates: number of candidates scored each call.
        candidate_distribution: ``Distribution`` for the candidate set.
            Defaults to ``state.algorithm.initial_design_distribution``
            at run time when ``None``.
    """

    n_candidates: int = 1024
    candidate_distribution: Distribution | None = None

    def optimize(self, acq, state, q, key):
        if q > self.n_candidates:
            raise ValueError(
                f"q ({q}) exceeds n_candidates ({self.n_candidates}); "
                f"cannot return more picks than candidates scored. Increase "
                f"`n_candidates` (or decrease `q`)."
            )
        key_cand, _ = jax.random.split(key)
        dist = _resolve_optimizer_distribution(
            self.candidate_distribution,
            state,
            field_name="candidate_distribution",
            optimizer_class=type(self).__name__,
        )
        candidates = jnp.asarray(
            pp_sample(dist, key=key_cand, sample_shape=(self.n_candidates,))
        )
        scores = acq.score(candidates, state)
        top_idx = jnp.argsort(-scores)[:q]
        return candidates[top_idx]


# ---------------------------------------------------------------------------
# Continuous multi-start: BFGS in reparameterized space, top-q maxima.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContinuousMultiStartOptimizer(PointwiseOptimizer):
    r"""Multi-start BFGS in unconstrained reparameterization space.

    Args:
        n_starts: number of BFGS seeds.
        n_seeding_candidates: candidates drawn before score-filtering to
            seeds. Must be :math:`\ge` ``n_starts``.
        bfgs_max_steps: max steps per BFGS solve.
        bfgs_rtol / bfgs_atol: convergence tolerances.
        seed_distribution: ``Distribution`` for the seeding candidate
            set. Defaults to
            ``state.algorithm.initial_design_distribution`` at run time
            when ``None``.
    """

    n_starts: int = 16
    n_seeding_candidates: int = 256
    bfgs_max_steps: int = 50
    bfgs_rtol: float = 1e-5
    bfgs_atol: float = 1e-5
    seed_distribution: Distribution | None = None

    def optimize(self, acq, state, q, key):
        if q > self.n_starts:
            raise ValueError(
                f"q ({q}) exceeds n_starts ({self.n_starts}); cannot return "
                f"more picks than BFGS seeds. Increase `n_starts` (or "
                f"decrease `q`)."
            )
        if self.n_seeding_candidates < self.n_starts:
            raise ValueError(
                f"n_seeding_candidates ({self.n_seeding_candidates}) must be "
                f">= n_starts ({self.n_starts})."
            )
        bijector = _make_bijector(state.x_support)

        # 1-2: seed selection
        key_seed, _ = jax.random.split(key)
        dist = _resolve_optimizer_distribution(
            self.seed_distribution,
            state,
            field_name="seed_distribution",
            optimizer_class=type(self).__name__,
        )
        seed_candidates = jnp.asarray(
            pp_sample(dist, key=key_seed, sample_shape=(self.n_seeding_candidates,))
        )
        seed_scores = acq.score(seed_candidates, state)
        top_seed_idx = jnp.argsort(-seed_scores)[: self.n_starts]
        starts = seed_candidates[top_seed_idx]

        # 3: reparameterize to unconstrained space
        u_starts = jax.vmap(bijector.inverse)(starts)

        # 4: BFGS each start, minimizing -score in unconstrained space.
        def neg_score_unconstrained(u: Array, _args=None) -> Array:
            x = bijector.forward(u)
            return -acq.score(x[None], state)[0]

        solver = optx.BFGS(rtol=self.bfgs_rtol, atol=self.bfgs_atol)
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
            except (jnp.linalg.LinAlgError, FloatingPointError):
                u_opt = u_init
            u_opt = jnp.where(jnp.isfinite(u_opt), u_opt, u_init)
            u_opt_list.append(u_opt)
        u_opt_stack = jnp.stack(u_opt_list)

        # 5: forward-map and re-score
        x_opt = jax.vmap(bijector.forward)(u_opt_stack)
        opt_scores = acq.score(x_opt, state)
        opt_scores = jnp.where(jnp.isfinite(opt_scores), opt_scores, -jnp.inf)
        top_q_idx = jnp.argsort(-opt_scores)[:q]
        return x_opt[top_q_idx]


# ---------------------------------------------------------------------------
# Greedy multi-point: pick → fantasize → pick → ... for q > 1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GreedyMultiPointOptimizer(PointwiseOptimizer):
    r"""For ``q > 1``: iteratively pick one point at a time, hallucinating
    observations at each pick to drive in-batch diversity on the next.
    """

    inner: PointwiseOptimizer
    imputer: FantasyImputer = field(default_factory=KrigingBeliever)

    def optimize(self, acq, state, q, key):
        if q < 1:
            raise ValueError(f"q must be >= 1, got {q}.")
        keys = jax.random.split(key, q)
        x_picks: list[Array] = []
        cur_state = state
        for i in range(q):
            x_i = self.inner.optimize(acq, cur_state, 1, keys[i])[0]
            x_picks.append(x_i)
            if i < q - 1:
                x_pending = jnp.stack(x_picks)
                y_pending = self.imputer.impute(x_pending, state)
                new_X = jnp.concatenate([state.X, x_pending], axis=0)
                new_Y_train = jnp.concatenate([state.Y_train, y_pending], axis=0)
                current = state.surrogate_distribution
                if not isinstance(current, EmulatedDistribution):
                    raise ValueError(
                        f"GreedyMultiPointOptimizer requires an emulator-backed "
                        f"`EmulatedDistribution`; got "
                        f"{type(current).__name__}."
                    )
                new_emulator = current.emulator.fit(new_X, new_Y_train)
                new_surrogate_distribution = EmulatedDistribution(
                    emulator=new_emulator,
                    decomposition=current.decomposition,
                    support=current.inner_support,
                    input_shape=current.inner_event_shape,
                    name=current.name,
                )
                cur_state = replace(
                    state,
                    surrogate_distribution=new_surrogate_distribution,
                    X=new_X,
                    Y_train=new_Y_train,
                )
        return jnp.stack(x_picks)


# ---------------------------------------------------------------------------
# Constraint → TFP bijector for unconstrained reparameterization
# ---------------------------------------------------------------------------


def _make_bijector(constraint: Constraint) -> Bijector:
    """Return a TFP `Bijector` mapping unconstrained ℝ ↔ `constraint`'s support."""
    if isinstance(constraint, _Interval):
        return Sigmoid(
            low=jnp.asarray(constraint.low),
            high=jnp.asarray(constraint.high),
        )
    raise NotImplementedError(
        f"ContinuousMultiStartOptimizer: no bijector dispatch for "
        f"Constraint of type {type(constraint).__name__}. Only "
        f"`interval(low, high)` supports are implemented. See "
        f"docs/probpipe_issues.md: 'Constraint -> Bijector mapping'."
    )
