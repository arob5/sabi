r"""Pointwise optimizers for `PointwiseScoredAcquisition`.

Three concrete optimizers ship today:

- ``CandidateSetOptimizer`` — random candidates from the design
  distribution, scored via ``acq.score(X, state)``, top-q returned.
  Cheap and the default.
- ``ContinuousMultiStartOptimizer`` — random candidates, score-filter
  to top-`n_starts`, BFGS each in unconstrained reparameterization
  space (TFP bijector dispatched on the support `Constraint`), return
  top-q distinct local maxima. Gradient-based; substantially more
  accurate on smooth scores when the emulator is well-conditioned.
- ``GreedyMultiPointOptimizer`` — for `q > 1` with in-batch diversity.
  Wraps an inner optimizer (any of the above with `q=1`); after each
  pick, hallucinates an observation at the pending point via a
  pluggable ``FantasyImputer`` and refits the emulator before the
  next pick.

For non-trivial supports (anything other than ``interval(low, high)``),
``ContinuousMultiStartOptimizer`` raises with a pointer to the
``Constraint`` → bijector gap in ``docs/probpipe_issues.md``. All
shipped benchmarks have box supports.

Multi-start currently uses a Python loop over BFGS solves
(``# TODO(vmap-multistart)``). JAX-vmap of ``optimistix.minimise``
should work in principle and is the natural future optimization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array
from probpipe.core.constraints import Constraint, _Interval
from tensorflow_probability.substrates.jax.bijectors import Bijector, Sigmoid

from sabi.acquisitions.base import AcquisitionState
from sabi.acquisitions.fantasize import FantasyImputer, KrigingBeliever
from sabi.sampling import BatchSampler, PriorSampler
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
        """Return ``(q,) + state.problem.input_shape``."""


# ---------------------------------------------------------------------------
# Candidate-set: random candidates → top-q. The default optimizer.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateSetOptimizer(PointwiseOptimizer):
    r"""Random candidates from a `BatchSampler`, score each, return top-q.

    Cheap; gradient-free. Quality is governed by ``n_candidates`` and
    the sampler's coverage of the high-score regions. Concretely: draw
    :math:`X \sim \text{sampler}^{n_{\text{candidates}}}`, evaluate
    :math:`s = \text{acq.score}(X, \text{state})`, return the rows of
    :math:`X` corresponding to the top-q entries of :math:`s`.

    Args:
        n_candidates: number of candidates scored each call.
        candidate_sampler: `BatchSampler` for the candidate set. Default
            is `PriorSampler` (samples from ``problem.prior``).
    """

    n_candidates: int = 1024
    candidate_sampler: BatchSampler = field(default_factory=PriorSampler)

    def optimize(self, acq, state, q, key):
        if q > self.n_candidates:
            raise ValueError(
                f"q ({q}) exceeds n_candidates ({self.n_candidates}); "
                f"cannot return more picks than candidates scored. Increase "
                f"`n_candidates` (or decrease `q`)."
            )
        key_cand, _ = jax.random.split(key)
        candidates = self.candidate_sampler.sample(
            state.problem, key_cand, self.n_candidates
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

    Pipeline:

    1. Sample ``n_seeding_candidates`` from ``seed_sampler``.
    2. Score each via ``acq.score``; take top-``n_starts`` as BFGS init points.
    3. Reparameterize each start to unconstrained space via the bijector
       :math:`b: \mathbb{R}^d \to \text{support}` (sigmoid for
       ``interval(low, high)`` supports).
    4. Run optimistix BFGS minimizing :math:`-\text{acq.score}(b(u))` per
       seed (Python loop; vmap is a future optimization).
    5. Map optimized :math:`u^*` back to support space, re-score, return
       top-q.

    Args:
        n_starts: number of BFGS seeds.
        n_seeding_candidates: candidates drawn before score-filtering to
            seeds. Must be :math:`\ge` ``n_starts``.
        bfgs_max_steps: max steps per BFGS solve.
        bfgs_rtol / bfgs_atol: convergence tolerances.
        seed_sampler: `BatchSampler` for the seeding candidate set.
            Default is `PriorSampler` (samples from ``problem.prior``).
    """

    n_starts: int = 16
    n_seeding_candidates: int = 256
    bfgs_max_steps: int = 50
    bfgs_rtol: float = 1e-5
    bfgs_atol: float = 1e-5
    seed_sampler: BatchSampler = field(default_factory=PriorSampler)

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
        bijector = _make_bijector(state.problem.support)

        # 1-2: seed selection
        key_seed, _ = jax.random.split(key)
        seed_candidates = self.seed_sampler.sample(
            state.problem, key_seed, self.n_seeding_candidates
        )
        seed_scores = acq.score(seed_candidates, state)
        top_seed_idx = jnp.argsort(-seed_scores)[: self.n_starts]
        starts = seed_candidates[top_seed_idx]

        # 3: reparameterize to unconstrained space
        u_starts = jax.vmap(bijector.inverse)(starts)

        # 4: BFGS each start, minimizing -score in unconstrained space.
        def neg_score_unconstrained(u: Array, _args=None) -> Array:
            x = bijector.forward(u)
            # Single-point evaluation routed through the public batched
            # API; works whether the acquisition implements
            # `_score_single` or overrides `score` directly.
            return -acq.score(x[None], state)[0]

        solver = optx.BFGS(rtol=self.bfgs_rtol, atol=self.bfgs_atol)
        # TODO(vmap-multistart): replace this Python loop with
        # `jax.vmap(optimistix.minimise, ...)` once we've validated it
        # works with our solver settings.
        u_opt_list: list[Array] = []
        for u_init in u_starts:
            # ``throw=False`` makes optimistix report convergence
            # failures via ``sol.result`` rather than raising — so
            # this ``except`` only triggers on genuine numeric blow-
            # ups inside the solver / score function. Narrow to the
            # known failure modes (Cholesky / linear-solve breakdown
            # under ill-conditioned Hessians; NaN propagation under
            # x64) so an unrelated bug — e.g. a future shape
            # mismatch in the score function — surfaces as a real
            # error rather than being silently swallowed.
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

    At step :math:`i = 1, \dots, q`:

    1. :math:`x_i = \text{inner.optimize}(\text{acq, } S_{i-1}, q=1)`,
       where :math:`S_0` is the input state and :math:`S_{i}` is the
       state with the previous picks added as fantasies.
    2. :math:`y_{1:i} = \text{imputer.impute}(x_{1:i}, S_0)` —
       imputation always uses the **original** state's data so the
       imputer sees real observations.
    3. :math:`S_i` = ``replace(S_0, X=X_0 ∪ x_{1:i}, Y=Y_0 ∪ y_{1:i},
       emulator=S_0.emulator.fit(X_i, Y_i))``.

    Args:
        inner: per-point optimizer (e.g., ``CandidateSetOptimizer()``).
        imputer: how to hallucinate ``y`` at pending points; default
            ``KrigingBeliever``.
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
                # Hallucinate y at all pending picks (use the *original*
                # state so imputers see real Y_train values, not
                # previous hallucinations).
                x_pending = jnp.stack(x_picks)
                y_pending = self.imputer.impute(x_pending, state)
                new_X = jnp.concatenate([state.X, x_pending], axis=0)
                # Imputer hallucinates training-scale values; extend
                # Y_train. Y_raw is left as the original (we don't
                # have raw evaluations at the hallucinated points; the
                # emulator only consumes Y_train anyway).
                new_Y_train = jnp.concatenate([state.Y_train, y_pending], axis=0)
                # Refit the emulator on the augmented training design.
                # Build a fresh `EmulatedDistribution` so downstream
                # score calls see the new emulator.
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
                    log_density_form=current.log_density_form,
                    support=current.inner_support,
                    input_shape=current.inner_event_shape,
                    prior=current.prior,
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
    """Return a TFP `Bijector` mapping unconstrained ℝ ↔ `constraint`'s support.

    Currently supports only ``interval(low, high)`` constraints; the
    dispatch here delegates to TFP's :class:`Sigmoid(low, high)`. For
    other constraint types, raises ``NotImplementedError`` with a
    pointer to the ``Constraint`` → bijector ProbPipe gap in
    ``docs/probpipe_issues.md`` (the gap is the inverse Constraint →
    Bijector mapping ProbPipe doesn't yet ship; the bijector itself
    comes from TFP).
    """
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
