# `TargetDistribution` / `DensityDecomposition` split

**Status:** implemented (issue #65)
**Issue:** [#65](https://github.com/arob5/sabi/issues/65)
**Last updated:** 2026-05-07

> Two coupled refactors of `TargetDistribution` and the algorithm-side
> "prior":
>
> 1. **Math vs. emulation split.** `TargetDistribution` carries only the
>    mathematical identity of the target distribution (support,
>    optionally an analytical unnormalized log-density). The
>    algorithmically-relevant pieces — what the emulator approximates
>    (`target_single`) and how its output composes into log-density —
>    move into a new `DensityDecomposition` class.
> 2. **Per-role distribution fields.** The single `prior` field on
>    `TargetDistribution`, which today fills four distinct roles
>    (modeling-prior add-on, initial-design distribution, candidate-set
>    sampler default, parameter-space support definer), splits into
>    per-role fields on `Algorithm` and per-component fields on the
>    optimizers.
>
> `DensityDecomposition` is a single parametric class — no subclass
> hierarchy. It absorbs and supersedes [#59](https://github.com/arob5/sabi/pull/59)'s
> `DensityForm`: same `(link: Map, shift: Map)` shape, plus
> `target_single` and `output_shape`. PR #59's `Map` infrastructure and
> pushforward dispatch are unchanged; this proposal renames + extends
> the form class only.
>
> The `PriorSampler` / `BatchSampler` wrapper layer is removed in favor
> of plain ProbPipe `Distribution` + `pp_sample` op.
>
> This is a **design-only** document. Implementation lands across
> follow-up issues — see [§7](#7-phasing-and-follow-up-issues).
>
> Sabi's API is not yet stable; this proposal does not preserve
> back-compat with current names or signatures.

## 1. Motivation

`TargetDistribution` today does three jobs:

1. **Mathematical identity** — defines the parameter space (via
   `prior.support`), names the target, holds `input_shape`.
2. **What the emulator fits** — `target_single` and `output_shape`.
3. **How to compose target-map output into the log target density** —
   `log_density_form`.

Jobs 2 and 3 are tightly coupled (a form's signature dictates what
`target_single` must return), but neither is part of "what the target
distribution *is* mathematically." A given `TargetDistribution` can be
approximated via multiple choices of (target map, link). Those are
algorithmic decisions, not properties of the target. Putting them on
`TargetDistribution` forces benchmarks to bake one decomposition in,
even though the same target can legitimately be paired with many.

Separately, the `prior` field on `TargetDistribution` is conflating two
genuinely different concepts:

- **The modeling prior** in a Bayesian inverse problem — part of the
  unnormalized log-density when the emulator fits log-likelihood.
  Belongs to the link/shift composition.
- **The algorithmic "initial distribution"** — used for initial design,
  candidate sets in pointwise optimizers, and search-region definition
  for acquisitions. Belongs to the algorithm.

The existing `TargetDistribution` docstring even acknowledges the
conflation: "In Bayesian settings the algorithmic `prior` may be a
*truncated* version of the modeling prior." That admits one field is
doing two jobs.

This refactor splits both conflations cleanly. `TargetDistribution`
shrinks to a math-identity object. `DensityDecomposition` (a flat
parametric class — no hierarchy) carries the algorithm-facing
emulation choice. Per-role distribution fields land where they
naturally belong — initial design on `Algorithm`, candidate sets and
seeds on the optimizers, modeling-prior add-on on the
`DensityDecomposition`'s `shift`.

## 2. Current state map

| Site | What it does |
|------|--------------|
| `src/sabi/target_distribution.py` (deleted in #65) | Was the `TargetDistribution(name, input_shape, output_shape, target_single, log_density_form, prior)` wrapper — six fields, three jobs. Replaced by direct subclassing of ProbPipe's `NumericRecordDistribution` for benchmark targets, and `DensityDecomposition` for the algorithm-side bundle. |
| `src/sabi/problems/forms.py` (deleted in #65) | Was the `LogDensityForm` family — three subclasses for one abstraction. Replaced by `DensityDecomposition` plus the `Map` ABC. |
| [`src/sabi/problems/base.py`](../src/sabi/problems/base.py) | `Problem` (post-#61): pure identity wrapper. Unchanged by this refactor. |
| `src/sabi/sampling.py` (deleted in #65) | Was the `BatchSampler` ABC + `PriorSampler` wrapper. Removed; importers route through `probpipe.sample` directly. |
| [`src/sabi/acquisitions/random.py`](../src/sabi/acquisitions/random.py) | `PriorSampling` — wraps a `BatchSampler`. Today's only sampling-style acquisition. |
| [`src/sabi/acquisitions/optim.py`](../src/sabi/acquisitions/optim.py) | `CandidateSetOptimizer.candidate_sampler: BatchSampler` and `ContinuousMultiStartOptimizer.seed_sampler: BatchSampler` — both default to `PriorSampler`. The bijector for unconstrained reparameterization reads `state.problem.target_distribution.support`. |
| [`src/sabi/algorithms/loop.py`](../src/sabi/algorithms/loop.py) | Calls `target.target_map(X)` for design evaluations (line 320, 542). Reads `support`, `input_shape`, `prior` from `target_distribution` to assemble `SurrogateDistribution`. |
| `src/sabi/tempering/` | Intermediate targets carry `target_single` / `target_map` / `prior` / `density_form` through the bridging scheme. After the refactor, intermediates compose `Map`s on a `DensityDecomposition` instead. |
| `src/sabi/surrogate/` | `EmulatedDistribution` reads `density_form` and `prior` to evaluate surrogate density. After the refactor, reads them off the `DensityDecomposition` from the algorithm. |

The split is forced by the shape of the data: jobs 1, 2, 3 of
`TargetDistribution` answer different questions for different consumers,
and the `prior` field's four roles fan out to four different sites.

## 3. Proposal

### 3.1 The target: any `NumericRecordDistribution`

The original draft proposed a sabi-side `TargetDistribution` class as
a thin wrapper for ``(name, input_shape, support)`` plus the analytical
density. The implementation drops that wrapper entirely:
**``Problem.target_distribution`` is just a ProbPipe
`NumericRecordDistribution`**, with no sabi-side class in between. The
ProbPipe base already provides ``event_shape``, ``support``, ``name``,
and the ``_unnormalized_log_prob`` protocol — exactly what the wrapper
was duplicating.

For benchmarks, the subclass provides the analytical density directly:

```python
from probpipe.core._numeric_record_distribution import NumericRecordDistribution

class BananaTarget(NumericRecordDistribution):
    @property
    def event_shape(self): return (self._d,)

    @property
    def support(self): return self._support

    def _unnormalized_log_prob(self, x):
        # Vectorized: x.shape == batch_shape + (d,) -> batch_shape
        ...
```

For user inverse problems where no analytical density is available,
the subclass simply does not override `_unnormalized_log_prob`. The
absence of the override means ``isinstance(target,
SupportsUnnormalizedLogProb)`` returns False and ProbPipe ops
naturally complain — no hand-rolled `NotImplementedError` needed.

Dropped fields, relative to the v0.x layout: ``prior``,
``target_single``, ``target_map``, ``output_shape``,
``log_density_form``. ``support`` becomes a first-class field on the
target's subclass (no longer derived from ``prior.support``). The
``target_distribution`` field on ``Problem`` remains; its type widens
from a sabi-specific class to ``NumericRecordDistribution``.

The MCMC-relevant random log-density boundary is
``EmulatedDistribution._random_unnormalized_log_prob``, which calls
``decomposition.pushforward(X, emulator_predictive)``. See
[link_functions.md §7](link_functions.md#7-probpipe-boundary--log-density-at-the-mcmc-seam)
for the full boundary discussion.

### 3.2 `DensityDecomposition`

`DensityDecomposition` is an **abstract** ProbPipe
`NumericRecordDistribution`. Subclasses define `target_map` (what the
emulator approximates) and `link` (the `Map` from emulator output to
log-density residual), and may override `shift` (default `None`).
The base auto-derives `_unnormalized_log_prob`:

```python
class DensityDecomposition(NumericRecordDistribution, ABC):
    def __init__(self, *, name, input_shape, support):
        ...

    @abstractmethod
    def target_map(self, x: Array) -> Array:
        """Vectorized: batch_shape + input_shape -> batch_shape + output_shape."""

    @property
    @abstractmethod
    def link(self) -> Map: ...

    @property
    def shift(self) -> Map | None:
        return None

    @property
    @abstractmethod
    def output_shape(self) -> tuple[int, ...]: ...

    @property
    def output_constraint(self) -> Constraint:
        return real    # constraint on y; subclasses override

    def _unnormalized_log_prob(self, x):
        log_prob_residual = self.link(self.target_map(x))
        if self.shift is None:
            return log_prob_residual
        return log_prob_residual + self.shift(x)

    def pushforward(self, x, y_dist) -> Distribution:
        if self.shift is None:
            return pushforward(self.link, y_dist)
        per_x_map = Affine(slope=1.0, intercept=self.shift(x)) @ self.link
        return pushforward(per_x_map, y_dist)
```

The unnormalized log-density at `x` decomposes as
`link(target_map(x)) + shift(x)`. The first term — the **log-prob
residual** — is the contribution attributable to the emulator's
output `y = target_map(x)`, after the link is applied. It is the
term the emulator's predictive distribution flows through under
`pushforward`. The shift is the deterministic x-dependent additive
term (typically `LogProb(prior)`); it carries no emulator
uncertainty.

`DensityDecomposition` is a Distribution: `support` is the
constraint on `x` (parameter-space); `output_shape` is the shape of
one `y`; `output_constraint` is the constraint on `y` (mostly
metadata; default `real`). External callers go through the ProbPipe
op (`unnormalized_log_prob(decomposition, X)`), which handles
batching per the standard contract. There is no separate `__call__(x, y)`
or `density_at(x)` method on the base class — the unnormalized
log-density at `x` flows through the standard
`unnormalized_log_prob` op like any other Distribution.

There can be many `DensityDecomposition` instances for a single
`TargetDistribution` — the choice of what to emulate (full
log-density vs. log-likelihood vs. forward-model output) is an
algorithmic decision, not a property of the target. The concrete
subclasses below cover the canonical patterns:

| Pattern | Subclass | `link` | `shift` |
|---------|----------|--------|---------|
| Full log-density | `LogProbTermTarget(prior=None)` | `Identity()` | `None` |
| Likelihood + prior | `LogProbTermTarget(prior=π)` | `Identity()` | `LogProb(π)` |
| Forward model with Gaussian likelihood | `GaussianForwardModelTarget(obs, cov, prior=π)` | `GaussianLogLik(obs, cov)` | `LogProb(π)` |
| `LogProbTarget(target)` | (concrete leaf of `LogProbTermTarget`) | `Identity()` | `None` |

The `Map` types (`Identity`, `LogProb`, `GaussianLogLik`, `Affine`,
…) and the `pushforward(Map, Distribution)` dispatch are exactly
the abstractions from [#64](https://github.com/arob5/sabi/pull/72)
(the Map ABC + pushforward landed earlier in the link-functions
phasing). This refactor uses them; it does not modify them.

#### 3.2.1 Concrete subclasses

The hierarchy is:

- `DensityDecomposition` (abstract base; subclasses define
  `target_map`, `link`, optionally override `shift`).
- `LogProbTermTarget(DensityDecomposition, ABC)` — fixes
  `link = Identity`. Optional `prior` field (when set,
  `shift = LogProb(prior)`; else `shift = None`). Subclasses define
  `target_map`. Used when the emulator emits one term in a
  `link(·) + shift(x)` sum: the full log-prob (`prior=None`) or one
  term + the prior shift (`prior=π`).
- `LogProbTarget(LogProbTermTarget)` — concrete leaf. Wraps a
  `TargetDistribution`'s analytical density: `target_map` delegates
  to the target's `unnormalized_log_prob` op. The canonical idiom
  for benchmarks: `LogProbTarget(problem.target_distribution)`.
- `GaussianForwardModelTarget(DensityDecomposition, ABC)` — fixes
  `link = GaussianLogLik(obs, cov)`, `shift = LogProb(prior)`.
  Subclasses define `target_map` (the forward model `f`).

Each benchmark module (`problems/banana.py`, etc.) ships a
`NumericRecordDistribution` subclass with the analytical density
(e.g., `BananaTarget`). The canonical emulator strategy —
"emulate the full unnormalized log-density" — is constructed by
wrapping that target with `LogProbTarget(problem.target_distribution)`
rather than with a per-benchmark decomposition class. Users adding
new strategies subclass `DensityDecomposition` directly. The yaml
runner config selects which subclass to instantiate via a
`density_decomposition: { kind: <name> }` block; the registry
plumbing currently supports `kind: identity_from_target` (constructs
`LogProbTarget(problem.target_distribution)`) and extends as more
shapes ship.

The previous draft proposed four classmethod helpers
(`identity_from_target`, `likelihood_with_prior`, `forward_model`,
`gaussian_forward_model`). The implementation uses subclasses
instead — cleaner OO, scales better when more decomposition shapes
land, and aligns with how every ProbPipe `Distribution` is defined.


#### 3.2.2 Self-consistency

When the user pairs a `DensityDecomposition` with a benchmark
`TargetDistribution` (which carries an analytical
`_unnormalized_log_prob`), there's a real question of whether the two
agree.

```python
def is_consistent_with(
    decomposition: DensityDecomposition,
    target: TargetDistribution,
    *,
    x_test: Array,
    atol: float = 1e-6,
    strict: bool = True,
) -> bool:
    """Verify that `decomposition` reconstructs `target`'s analytical
    unnormalized log-density at `x_test`.

    Args:
        decomposition: candidate decomposition paired with `target`.
        target: a `TargetDistribution` with analytical
            `_unnormalized_log_prob`. Returns `True` trivially when
            `target` does not implement an analytical density (i.e.,
            user inverse problems).
        x_test: shape `(n,) + input_shape`. Requires `n >= 2` when
            `strict=False`.
        atol: numerical tolerance for the comparison.
        strict: when `True` (default), require pointwise equality
            within `atol`:
            ``decomposition.density_at(x) ≈ target._unnormalized_log_prob(x)``.
            When `False`, allow an additive constant — only differences
            across rows must match within `atol`. Requires `n >= 2`.

    The strict default exists because some downstream paths
    (metrics that compare emulator predictions against the target's
    analytical log-density) rely on exact correspondence, not match
    up-to-constant. Pass `strict=False` only when the relevant
    consumers are constant-invariant.
    """
```

``is_consistent_with`` is exercised directly in
`tests/test_density_decomposition.py` (strict and loose modes against
custom decomposition pairs). The per-benchmark sweep — running
``is_consistent_with`` against every benchmark factory's analytical
target — is deferred to a follow-up: every benchmark currently uses
the canonical `LogProbTarget(problem.target_distribution)` pairing
(checked by construction, since `LogProbTarget` delegates to the
target's analytical density), so the regression sweep adds value
only once benchmarks ship custom `DensityDecomposition` subclasses.
Strict-default by design — benchmark factories must produce pairs
whose analytical evaluations match exactly.

### 3.3 `Algorithm` additions

```python
class Algorithm:
    # ... existing fields (emulator_factory, acquisition, schedule, ...)
    density_decomposition: DensityDecomposition | None = None
    initial_design_distribution: Distribution | None = None
    x_support: Constraint | None = None
```

Run-time defaults (resolved when `run(...)` is called):

| Field | `None` default behavior |
|---|---|
| `x_support` | Falls back to `target_distribution.support` |
| `initial_design_distribution` | Falls back to `Uniform(x_support)` if `x_support` is bounded; otherwise raises with a clear pointer to both fields |
| `density_decomposition` | Required by every algorithm currently in sabi; raises if `None`. Typed as `Optional` as future-proofing for any algorithm that wouldn't need one. |

`x_support` and `target.support` will usually coincide; the field
exists so they *can* differ. The motivating case: the user wants the
acquisition's search region restricted to a subset of the parameter
space (e.g., concentrate the initial design on a sub-region while
still permitting the target to be defined on the full
`target.support`).

#### 3.3.1 The four "prior" roles, after the split

The current `prior` field served four roles. Each lands somewhere
specific:

| Role | New home |
|------|----------|
| Modeling-prior add-on (Bayesian density assembly) | `DensityDecomposition.shift` (e.g., `LogProb(modeling_prior)`) |
| Initial design distribution | `Algorithm.initial_design_distribution` |
| Candidate-set sampler (pointwise optimizers) | `CandidateSetOptimizer.candidate_distribution` (per-optimizer field) |
| BFGS seed sampler (continuous optimizer) | `ContinuousMultiStartOptimizer.seed_distribution` (per-optimizer field) |
| Parameter-space support definer | `TargetDistribution.support` (math) + `Algorithm.x_support` (algorithmic search region, default-equal) |

In Bayesian inverse-problem benchmarks, the user can choose to pass
the same `Distribution` to multiple of these slots (e.g., the
modeling prior can serve as both `LogProb(modeling_prior)` in the
shift and as the `initial_design_distribution`). That's an explicit
choice, not a hidden coupling.

### 3.4 Per-optimizer fields

```python
@dataclass(frozen=True)
class CandidateSetOptimizer(PointwiseOptimizer):
    n_candidates: int = 1024
    candidate_distribution: Distribution | None = None  # NEW

@dataclass(frozen=True)
class ContinuousMultiStartOptimizer(PointwiseOptimizer):
    n_starts: int = 16
    n_seeding_candidates: int = 256
    bfgs_max_steps: int = 50
    bfgs_rtol: float = 1e-5
    bfgs_atol: float = 1e-5
    seed_distribution: Distribution | None = None       # NEW
```

Both default to `Algorithm.initial_design_distribution` when `None`,
read off the `AcquisitionState` at run time. Per-component
independence (each optimizer can use a different sampler) plus a
sensible default.

### 3.5 Acquisition strategies

The `Acquisition` ABC remains as-is — already maximally flexible
(`select_batch(state, q, key) -> Array`). Custom strategies
(clustering, Stein thinning, mixture-with-current-estimate, etc.)
subclass directly.

Today's `PriorSampling` is renamed `DistributionSampling` and
generalized:

```python
@dataclass(frozen=True)
class DistributionSampling(Acquisition):
    """Sample `q` points from a Distribution.

    `distribution_from_state` is called per round and returns the
    Distribution to sample from — typically the algorithm's
    `initial_design_distribution`, the current surrogate, or a
    mixture. For complex strategies (clustering, Stein thinning, …)
    implement `Acquisition` directly.
    """
    distribution_from_state: Callable[[AcquisitionState], Distribution] = (
        lambda state: state.algorithm.initial_design_distribution
    )

    def select_batch(self, state, q, key):
        dist = self.distribution_from_state(state)
        return jnp.asarray(pp_sample(dist, key=key, sample_shape=(q,)))
```

The default factory recovers today's `PriorSampling` semantics under
the new field names. Common compositions (mixtures, current-estimate
sampling) are expressed by passing different distributions; complex
strategies bypass this helper and implement `Acquisition` directly.

### 3.6 Sampling

`src/sabi/sampling.py` is **deleted**. `PriorSampler` and
`BatchSampler` go away. Every site that needs sampling calls
`probpipe.sample(distribution, key=key, sample_shape=(n,))` directly.
Sites that require sampling enforce `SupportsSampling` via runtime
protocol check at construction.

### 3.7 Benchmarks

Benchmark factories return only the `Problem` (mathematical identity).
The decomposition is an algorithmic choice and is constructed
separately by the user:

```python
problem = banana_2d()
# Problem(target_distribution=TargetDistribution(...,
#                                                support=…,
#                                                _unnormalized_log_prob=…),
#         reference_distribution=…,
#         name="banana_2d")

decomposition = DensityDecomposition.identity_from_target(
    problem.target_distribution
)
algorithm = Algorithm(
    density_decomposition=decomposition,
    ...,
)
run(problem, algorithm, key)
```

`identity_from_target` is the recommended idiom for benchmarks. It
wraps the target's analytical `_unnormalized_log_prob` (via ProbPipe's
`unnormalized_log_prob` op) into a `target_single` callable, with
`link = Identity` and `shift = Constant(0.0)`. Users who want a
non-trivial decomposition (e.g., emulate log-likelihood instead of
log-density) call `DensityDecomposition.likelihood_with_prior(...)` or
construct `DensityDecomposition(...)` directly.

**The benchmark module does not expose a standalone
`banana_log_density(d, a, b)` factory.** The math identity lives on
the `TargetDistribution` (via its analytical
`_unnormalized_log_prob`); the recommended way to consume it is
`DensityDecomposition.identity_from_target(...)`. No duplicate
parameterization between benchmark factory and density factory.

## 4. Coordination with #59

PR #59 v0.3 introduces a single parametric `DensityForm` carrying
`(link: Map, shift: Map)`. This proposal extends that class with
`target_single` and `output_shape`, and renames it to
`DensityDecomposition`.

### 4.1 Interaction matrix

| #59 piece | This PR | Interaction |
|---|---|---|
| Single `DensityForm` class with `(link, shift, constraint)` | Renamed to `DensityDecomposition` and extended with `(target_single, output_shape)` | This PR supersedes the name and shape; the substantive design (parametric form via `Map`s, link + shift split) is preserved verbatim. |
| `Map` ABC + concrete `Map` subclasses (`Identity`, `Affine`, `Constant`, `Exp`, `Log`, `Softplus`, `LogSoftplus`, `Square`, `LogSquare`, `GaussianLogLik`, `LogProb`, `Compose`) | Untouched | Used directly by `DensityDecomposition.link` and `DensityDecomposition.shift`. No changes. |
| `pushforward(Map, Distribution)` multi-dispatch op | Untouched | Used inside `DensityDecomposition.pushforward`. The per-x-`Map` construction (`Affine(intercept=shift(x)) @ link`) is identical. |
| `DensityForm.__call__(x, y)` and `DensityForm.pushforward(x, y_dist)` | Hosted on `DensityDecomposition` instead | Mechanical move. |
| `DensityForm.constraint` field | Becomes `DensityDecomposition.constraint` | Field rehost. |
| `TemperingScheme` → `BridgingScheme` rename + bridges as `Map` compositions | Compatible | Bridges return a new `DensityDecomposition` by composing `Map`s on the base's `link` / `shift`; `target_single` and `output_shape` pass through unchanged. The bridge entry point is `intermediate_decomposition(base, β)` (renamed from #59 v0.3's `intermediate_form`). See [#59 v0.4 §8.3](https://github.com/arob5/sabi/blob/docs/issue-55-link-functions-design/docs/link_functions.md#83-likelihoodbridgeviaform-link-agnostic-bridging-via-map-composition). |
| `LikelihoodBridgeViaTargetRescale` raises on non-exp link | Untouched | Same constraint — link must be `Identity` (representing the exp link in log-space) for the `Y_train = β·Y_raw` optimization to be valid. The check is `isinstance(base.link, Identity)`, raising `NotImplementedError` with a pointer to `LikelihoodBridgeViaForm` for any other link. See [#59 v0.4 §8.5](https://github.com/arob5/sabi/blob/docs/issue-55-link-functions-design/docs/link_functions.md#85-likelihoodbridgeviatargetrescale--exp-link-only-optimisation). |
| `pushforward_marginal` dispatch site (currently in `_pushforward.py`) | Reads `link`/`shift` off the `DensityDecomposition` instead of off the `DensityForm` | Field rehost. |
| Phasing (#59 v0.4 §9): 1 design doc, 2 Map + pushforward infra, 3 concrete Map subclasses (largely subsumed by phase 2), 4 acquisition layered access, 5 bridging rename, 6 softplus link end-to-end, 7 square link end-to-end, 8 bridging tutorial, 9 link-aware acquisition audit | Recommended landing order: #59 phase 2 → this PR's implementation → #59 phases 3+. Phase 3 in #59 v0.4 reduces to "concrete `Map` subclasses for non-default links," since the rename + extension to `DensityDecomposition` happens in this PR. | Coordinated re-sequencing already reflected in #59 v0.4. |

### 4.2 Recommended landing order

1. **#59 phase 1** (design doc — already this PR's sibling).
2. **#59 phase 2** (`Map` + pushforward infrastructure, no behavior
   change). Lands the abstractions both PRs depend on.
3. **This PR's implementation.** Renames `DensityForm` →
   `DensityDecomposition`, adds `target_single` and `output_shape`,
   restructures `TargetDistribution`, splits the prior roles, removes
   `sampling.py`. Rebases mechanically on top of #59 phase 2.
4. **#59 phases 3+.** `DensityForm` is already renamed by this PR, so
   #59 phase 3 ("rename + link field on Identity / LikelihoodWithPrior")
   reduces to wiring up the additional concrete `Map` subclasses
   (`Softplus`, `Square`, etc.) into the existing
   `DensityDecomposition` shape.

If #59's owner prefers to land phases 2 and 3 together as a single PR
(unifying the form class and adding the link infrastructure in one
commit), this proposal can rebase below or above that PR depending on
sequencing. Either order works; the design contributions are
orthogonal beyond the class rehost.

## 5. Migration footprint

Roughly ~40 source/test files. Bigger than [#61](https://github.com/arob5/sabi/pull/61).

### `src/sabi/`

- **`target_distribution.py`** — drop `prior`, `target_single`,
  `target_map`, `output_shape`, `log_density_form` from
  `TargetDistribution`. Add `support: Constraint` as explicit field.
  `_unnormalized_log_prob` becomes either an analytical implementation
  (for benchmarks) or `NotImplementedError`. `IntermediateTarget`
  follows the same shape, plus `state` and `output_transform` as
  before. **`IntermediateTarget` no longer carries `target_single` /
  `target_map` / `density_form` / `prior`** — those are read from the
  bridging scheme + the algorithm's `DensityDecomposition`.
- **`problems/forms.py` → `density_decomposition.py`** — module rename
  (likely under `src/sabi/` directly, not `problems/`). The
  `LogDensityForm` class hierarchy is removed; `DensityDecomposition`
  is the single replacement, with `target_single`, `output_shape`,
  `link`, `shift`, `constraint` fields and the four classmethod
  helpers from §3.2.1. (Coordinate with #59 on the `Map` import path.)
- **`algorithms/algorithm.py`** — add `density_decomposition`,
  `initial_design_distribution`, `x_support` fields with `None`
  defaults. Add a resolver that fills defaults from the `Problem` /
  `TargetDistribution` at run time.
- **`algorithms/loop.py`** — every
  `target_distribution.target_map(X)` / `target_single(X)` call
  switches to `algorithm.density_decomposition.target_map(X)`.
  `_build_surrogate_distribution` reads `support`, `input_shape` from
  `target_distribution` and decomposition info from
  `algorithm.density_decomposition`. ~10 call-site updates.
- **`tempering/`** — bridging schemes operate on the algorithm's
  `DensityDecomposition` to produce a per-state effective
  decomposition (`Map` composition on `link` / `shift`). After #59's
  `tempering` → `bridging` rename, the dispatch is "compose this
  bridge's `Map`s onto the decomposition's `link` / `shift`."
- **`surrogate/surrogate_distribution.py`,
  `surrogate/estimators.py`** — `EmulatedDistribution` reads
  decomposition info from `algorithm.density_decomposition` (via the
  factory args), not from the target. Surrogate density evaluation
  uses `decomposition.pushforward(x, emulator(x))`.
- **`acquisitions/random.py`** — rename `PriorSampling` →
  `DistributionSampling`. Default `distribution_from_state` to
  `lambda state: state.algorithm.initial_design_distribution`.
- **`acquisitions/optim.py`** — `CandidateSetOptimizer` and
  `ContinuousMultiStartOptimizer` gain `candidate_distribution` /
  `seed_distribution` fields. `_make_bijector` reads from
  `state.algorithm.x_support` (not `target.support`) for the
  acquisition's reparameterization region.
- **`acquisitions/base.py`** — `AcquisitionState` exposes
  `algorithm: Algorithm` (or at least the relevant subset:
  `initial_design_distribution`, `x_support`,
  `density_decomposition`). Today it has `problem` only.
- **`sampling.py`** — **deleted**. Importers switch to
  `probpipe.sample`.
- **`runner/build.py`** — Hydra-builder updates: emit a
  `DensityDecomposition` from config alongside the `Algorithm`.
- **`problems/banana.py`, `problems/gaussian.py`,
  `problems/neals_funnel.py`, `problems/benchmarks.py`** — each
  benchmark factory returns only a `Problem` with a
  `TargetDistribution` whose `_unnormalized_log_prob` is implemented
  analytically. No standalone target-function factories.
- **`problems/base.py`** — `Problem` shape unchanged from
  [#61](https://github.com/arob5/sabi/pull/61). Updated docstring to
  clarify that `target_distribution` may have a
  `NotImplementedError` `_unnormalized_log_prob` for user inverse
  problems.

### `tests/`

- Every test that constructed `TargetDistribution(…, target_single=…,
  log_density_form=…, prior=…)` switches to constructing both a
  `TargetDistribution(…, support=…)` and a separate
  `DensityDecomposition(target_single=…, output_shape=…, link=…,
  shift=…)`.
- Every test that read `problem.target_distribution.target_map` /
  `.target_single` / `.density_form` / `.prior` switches to reading
  from the algorithm's `DensityDecomposition`.
- New
  `tests/test_benchmarks.py::test_density_decomposition_consistency`
  loops over every benchmark factory's
  `(TargetDistribution, default_decomposition)` pair and asserts
  `is_consistent_with(strict=True)` at a handful of test points.
- `test_target_distribution.py` shrinks substantially —
  `TargetDistribution` is now mostly a passive metadata holder.
- New `test_density_decomposition.py` covers the
  `DensityDecomposition` field contracts, the four classmethod
  helpers, `is_consistent_with` (both `strict=True` and
  `strict=False` modes), and the `pushforward` dispatch through the
  decomposition.

### `docs/`

- `notation.md`, `design.md` — update prose where it described
  `target_map` / `prior` as living on the target. Document the new
  layering.
- `getting_started.ipynb` — re-execute against the new
  `Algorithm(density_decomposition=…)` shape.
- `link_functions.md` (#59's design doc) — update cross-references;
  the examples land on `DensityDecomposition` instances.
- `tempering.md` — align with whatever bridging name lands.

## 6. Open questions and decisions deferred

1. **Field name `density_decomposition` on `Algorithm`.** Verbose;
   could be just `decomposition`. Verbose form preserved here for
   clarity; trim during implementation if it reads heavily.
2. **`Problem` carrying nullable analytical density.** Today's
   `Problem.target_distribution` is always non-None (set by #61). After
   this refactor, `target_distribution._unnormalized_log_prob` may
   raise `NotImplementedError` for user inverse problems. Whether this
   materializes as a strict ABC distinction (`AnalyticalTargetDistribution`
   vs. `OpaqueTargetDistribution`) or as a single class with a
   protocol-checked method is deferred to implementation. The
   user-facing surface is the same either way.
3. **Module placement.** `density_decomposition.py` likely lives at
   `src/sabi/density_decomposition.py` (top-level alongside
   `target_distribution.py`). It could equally live under
   `src/sabi/algorithms/`, since it's an algorithmic-side object.
   Defer to taste.

## 7. Phasing and follow-up issues

This proposal lands as a single implementation PR (call it #YY when
filed). The internal phasing within that PR's commit history:

1. Land `DensityDecomposition` (extends #59's `DensityForm` with
   `target_single` / `output_shape`; rename module).
2. Trim `TargetDistribution` (drop the moved-out fields; add
   `support`).
3. Add `Algorithm` fields and run-time defaults.
4. Migrate the loop, surrogate, acquisitions, tempering, and
   benchmarks to the new layout.
5. Delete `sampling.py`, rename / generalize `PriorSampling`.
6. Update tests and docs.
7. Add the consistency-check regression test.

Each step is a coherent commit; the full migration is one PR because
the steps are tightly coupled at the type level (a partial migration
won't typecheck or run).

If the PR turns out larger than reviewable in one pass, split (1) +
(2) into a "DensityDecomposition lands; TargetDistribution shrinks"
PR and stack (3)–(7) on top. Decide at implementation time.
