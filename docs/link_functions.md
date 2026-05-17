# Form abstraction refactor — `DensityDecomposition`, `Map`, and bridging

**Status:** draft v0.4 — pre-implementation
**Issue:** [#55](https://github.com/arob5/sabi/issues/55)
**Last updated:** 2026-05-07

> Two coupled refactors, motivated by issue #55:
>
> 1. **Form unification.** Collapse today's `LogDensityForm` hierarchy
>    (`Identity`, `LogLikPlusPrior`, `ForwardModel`) into a single
>    parametric host class with `(link: Map, shift: Map, constraint)`
>    fields. Specific forms — including alternative link functions like
>    softplus and square — become `Map` compositions, not subclasses.
> 2. **Bridging rename and reshape.** `TemperingScheme` becomes
>    `BridgingScheme`. Likelihood tempering is one bridge family among
>    several future ones; bridges operate on the host class by composing
>    `Map`s on its `link` and `shift`.
>
> A `pushforward(Map, Distribution) → Distribution` multi-dispatch op
> handles both deterministic evaluation and pushing emulator predictives
> through forms. The `Map` abstraction is intentionally aligned with what
> ProbPipe is converging on; sabi's local op becomes a re-export when
> ProbPipe lands.
>
> **Coordination with [PR #63](https://github.com/arob5/sabi/pull/63).** The
> host class is named `DensityDecomposition` and lives at top-level
> `src/sabi/density_decomposition.py`, per the `TargetDistribution` /
> `DensityDecomposition` split that PR #63 introduces. PR #63 carries the
> rename and extends the class with `target_single` + `output_shape`
> fields; this PR (#59) contributes the `(link, shift)` `Map` shape,
> the `Map` ABC + concrete maps, the `pushforward(Map, Distribution)`
> dispatch, and the bridging restructure. The two PRs land
> sequentially — recommended order in [§9](#9-phasing-and-follow-up-issues).
>
> This is a **design-only** PR: no source code changes. Implementation is
> split across follow-up issues (see [§9](#9-phasing-and-follow-up-issues)).
>
> Sabi's API is not yet stable; this proposal does not preserve back-compat
> with current names or signatures.

## 1. Motivation

The current `LogDensityForm` hierarchy treats three instances of the same
abstraction as separate classes:

- `Identity._call_single(x, y) = y` — the emulator emits log-density; the
  form is the identity function.
- `LogLikPlusPrior._call_single(x, y) = y + log_prior(x)` — the emulator
  emits log-likelihood; the form adds the log-prior.
- `ForwardModel._call_single(x, y) = log_lik_from_outputs(x, y) +
  log_prior(x)` — the emulator emits a forward-model output; the form
  applies a user-supplied likelihood and adds the log-prior.

All three are instances of "the emulator emits $y$ at $x$, and a function
$\Phi(x, y) \mapsto \log p(x)$ assembles the log-density." `ForwardModel`
in particular is *already* a general "link function" implementation — its
`log_lik_from_outputs` callback can be any mapping from emulator output to
log-likelihood, including a Gaussian density
$\log\mathcal{N}(\mathrm{obs} \mid f(x), C)$, a Student-$t$ density, or
anything else — but the abstraction's name advertises it as forward-model
specific. Adding alternative link functions to `LogLikPlusPrior` (issue
#55's original framing) means either patching the class with a `link`
field or adding new sibling classes; both fight the underlying structure.

The structural fix is to collapse the hierarchy. Define a single host
class (`DensityDecomposition`, per [PR #63](https://github.com/arob5/sabi/pull/63))
that combines:

- a `Map` `link: y \mapsto z` (the emulator-output side; ranges from the
  identity to a Gaussian log-likelihood),
- a `Map` `shift: x \mapsto z'` (the deterministic x-dependent additive
  term, typically a log-prior closed in at construction),
- a `Constraint` on $y$,
- (per PR #63) `target_single` and `output_shape`, the description of
  what the emulator approximates.

Specific forms become specific `Map` choices. Alternative link functions
are concrete `Map` subclasses (`Softplus`, `LogSoftplus`, `Square`, …)
that slot in. Forward-model emulation is a `Map` choice
(`GaussianLogLik`). Tempering is `Map` composition on the host's
`link`. The downstream pushforward / estimator / acquisition machinery
dispatches on `Map` types, not on form subclasses.

The exponential link is no longer privileged — under this design, the
default form for log-density emulation has `link=Identity()`;
log-likelihood-plus-prior under exp link has `link=Identity(),
shift=LogProb(modeling_prior)`; under softplus link, `link=LogSoftplus(),
shift=LogProb(modeling_prior)`. The codepath is the same in all three
cases.

## 2. Current state map

The form hierarchy and its consumers:

| Site | What it does |
|------|--------------|
| `src/sabi/problems/forms.py` (was) | `LogDensityForm` family — `Identity`, `LogLikPlusPrior`, `ForwardModel`. Three classes for one abstraction. **Deleted** in [#65](https://github.com/arob5/sabi/pull/73): replaced by `sabi.density_decomposition.DensityDecomposition` (single parametric class with `(link, shift, target_single, output_shape, constraint)`). |
| `src/sabi/surrogate/_pushforward.py:79` (was) | `pushforward_marginal` dispatched on `(input_dist, form_subclass)` for closed-form Gaussian-affine. **Deleted** in [#65](https://github.com/arob5/sabi/pull/73): subsumed by [`sabi.maps.pushforward`](../src/sabi/maps/_pushforward.py) (multi-dispatch on `(Map, Distribution)` from PR #64). |
| `src/sabi/surrogate/_pushforward.py:144` (was) | `_shift_gaussian_loc` — bespoke handling of "shift `loc` by per-point log-prior, leave `scale_tril` alone." Generalised to `(Affine, Normal | MultivariateNormal)` registrations in [`sabi.maps`](../src/sabi/maps/_pushforward.py). |
| [`src/sabi/surrogate/estimators.py:129`](../src/sabi/surrogate/estimators.py) | `_ExpectedTargetDistribution._unnormalized_log_prob` plug-in mean. The bias note at line 15 is a symptom of the existing exp-link composition. |
| [`src/sabi/surrogate/weighted_empirical.py:64`](../src/sabi/surrogate/weighted_empirical.py) | `WeightedEmpiricalRandomMeasure` log-weights. Form output is converted externally; class is link-agnostic. |
| `src/sabi/target_distribution.py:56` (deleted in #65) | Was `TargetDistribution._unnormalized_log_prob` — ProbPipe boundary; consumes log-density. Replaced by direct ProbPipe `NumericRecordDistribution` subclassing on the benchmark side. |
| [`src/sabi/tempering/likelihood.py`](../src/sabi/tempering/likelihood.py) | `_LogLikPlusPriorTempered`, `_ForwardModelTempered`, `_IdentityTempered` — three subclasses for one β-scale rewrite. |

Two things stand out:

- The pushforward / tempering layers branch on form *subclass*. Adding a
  new link function multiplies this branching. A single
  `DensityDecomposition` with composable `Map`s collapses the branching
  into per-Map dispatch.
- `_shift_gaussian_loc` is a one-off Gaussian-affine pushforward primitive.
  Generalising to `pushforward(Affine, Normal | MVN)` makes it the first
  entry in a registry that handles arbitrary `Map`s.

## 3. The `Map` abstraction

A `Map` is a single-variable measurable function with a declared shape
contract that participates in pushforward dispatch.

```python
class Map(ABC):
    """Single-variable callable participating in pushforward dispatch."""
    event_shape_in:  tuple[int, ...]
    event_shape_out: tuple[int, ...]

    def __call__(self, z: Array) -> Array: ...

    def __matmul__(self, other: Map) -> Map:
        """Composition: (f @ g)(z) = f(g(z))."""
        return Compose(self, other)
```

### 3.1 Shape contract

Inspired by TFP bijectors but stripped — sabi's Maps don't require
bijectivity, so we don't carry `inverse_min_event_ndims` or
`log_det_jacobian` slots.

- `event_shape_in`: shape of one input event. For most form-related Maps,
  this is `()` (scalar emulator output) or `output_shape` (vector
  emulator output, e.g. `(d,)` for a forward-model output of dimension
  $d$).
- `event_shape_out`: shape of one output event. For form `link`s that
  produce log-density-residual, this is `()` (scalar). For `shift`s,
  also `()`.
- **Broadcasting.** Maps broadcast over leading batch axes; given input
  of shape `batch_shape + event_shape_in`, the output has shape
  `batch_shape + event_shape_out`. Same convention as `ArrayRandomFunction`
  in [`docs/notation.md`](notation.md).
- **Composition.** `f @ g` requires `f.event_shape_in == g.event_shape_out`.
  Result: `event_shape_in = g.event_shape_in`,
  `event_shape_out = f.event_shape_out`.

### 3.2 Concrete Maps

The first cut. Each Map is a small dataclass.

| Map | $f(z)$ | event shapes | Notes |
|-----|--------|--------------|-------|
| `Identity` | $z$ | `() → ()` | Closed-form pushforward through any distribution. |
| `Constant(c)` | $c$ (ignores $z$) | `(any) → c.shape` | Used as default `shift`. |
| `Affine(slope, intercept)` | $\mathrm{slope} \cdot z + \mathrm{intercept}$ | `() → ()` | Closed-form pushforward through Gaussians. |
| `Compose(f, g)` | $f(g(z))$ | depends | Built via `f @ g`. |
| `Exp` | $e^z$ | `() → ()` | Closed form for `(Exp, Normal) → LogNormal`. |
| `Log` | $\log z$ | `() → ()` | Closed form for `(Log, LogNormal) → Normal`. |
| `Softplus` | $\log(1 + e^z)$ | `() → ()` | MC fallback for `(Softplus, Normal)`. |
| `LogSoftplus` | $\log\log(1 + e^z)$ (numerically stable) | `() → ()` | MC fallback. |
| `Square` | $z^2$ | `() → ()` | Closed form for `(Square, Normal) →` non-central χ². |
| `LogSquare` | $2\log\lvert z\rvert$ | `() → ()` | MC fallback. |
| `GaussianLogLik(obs, cov)` | $\log\mathcal{N}(\mathrm{obs} \mid z, \mathrm{cov})$ | `(d,) → ()` | Used by forward-model forms. |
| `LogProb(dist)` | $\log p_\mathrm{dist}(z)$ | `dist.event_shape → ()` | Used as `shift` for prior addition. |

Note that `Constant`, `LogProb`, and `GaussianLogLik` close over their
parameters at construction; the `__call__` is single-argument.

### 3.3 ProbPipe alignment

The `Map` ABC, the shape contract, and the per-`Map` pushforward
dispatch all match what ProbPipe is converging on. Bijectors will
land in ProbPipe as a `Map` subclass that adds `inverse` and
`log_det_jacobian`; sabi's `Map` ABC is the strict superclass.

When ProbPipe's `Map` lands, sabi's becomes a re-export and our concrete
maps probably migrate to ProbPipe entirely. The `pushforward(Map,
Distribution)` op also moves up. The migration is rename-level, not
refactor-level, and is the explicit motivation for the design choices in
this section.

## 4. The `DensityDecomposition` host class

The host class is `DensityDecomposition`, defined fully in
[PR #63](https://github.com/arob5/sabi/pull/63)'s
`docs/density_decomposition.md`. It carries the `link` / `shift` / `constraint`
fields contributed by this PR, plus `target_single` and `output_shape`
contributed by PR #63 (the description of what the emulator approximates).
The combined shape:

```python
@dataclass(frozen=True)
class DensityDecomposition:
    target_single: Callable[[Array], Array]   # x -> y    (PR #63)
    output_shape: tuple[int, ...]              # shape of one y    (PR #63)
    link:  Map                                  # y -> log-density-residual    (this PR)
    shift: Map = Constant(0.0)                  # x -> additive shift    (this PR)
    constraint: Constraint = NoConstraint       # constraint on y    (this PR)

    def __call__(self, x: Array, y: Array) -> Array:
        """Deterministic evaluation: log-density at (x, y)."""
        return self.link(y) + self.shift(x)

    def density_at(self, x: Array) -> Array:
        """Compose target_single with link/shift: log-density at x."""
        return self(x, self.target_single(x))

    def pushforward(self, x: Array, y: Distribution) -> Distribution:
        """Push the emulator's predictive `y` at point(s) `x` to log-density."""
        per_x_map = Affine(slope=1.0, intercept=self.shift(x)) @ self.link
        return pushforward(per_x_map, y)
```

The decomposition's log-density is $\Phi(x, y) = \mathrm{link}(y) +
\mathrm{shift}(x)$ — a single-variable `link` (in $y$) plus a
deterministic additive shift (in $x$). All x-dependence enters through
the shift. The modeling prior, when present, closes into the `shift`
`Map` at construction time (typically `shift = LogProb(modeling_prior)`)
— it is not pulled from a `target.prior` field at call time. PR #63
drops the `prior` field from `TargetDistribution` for exactly this
reason.

### 4.1 Specific decompositions

The current `LogDensityForm` subclasses become construction idioms.
The ProbPipe-style examples below show the `link` / `shift` choices
this PR introduces; PR #63's helper classmethods (`identity_from_target`,
`likelihood_with_prior`, `forward_model`) wrap the common patterns and
are the recommended user-facing entry points.

```python
# Today's Identity (log-density emulator):
DensityDecomposition(target_single=..., output_shape=(), link=Identity())

# Today's LogLikPlusPrior (log-likelihood + prior, exp link):
DensityDecomposition(target_single=..., output_shape=(),
                     link=Identity(), shift=LogProb(modeling_prior))

# Today's ForwardModel with Gaussian likelihood:
DensityDecomposition(target_single=..., output_shape=(d,),
                     link=GaussianLogLik(obs, cov), shift=LogProb(modeling_prior))

# New: log-likelihood + prior, softplus link (issue #55's motivation):
DensityDecomposition(target_single=..., output_shape=(),
                     link=LogSoftplus(), shift=LogProb(modeling_prior))

# New: log-likelihood + prior, square link:
DensityDecomposition(target_single=..., output_shape=(),
                     link=LogSquare(), shift=LogProb(modeling_prior),
                     constraint=NonNegative())
```

No subclasses; the variation is entirely in the `Map`s.

### 4.2 What the emulator emits, in each case

It is worth being explicit about this, since the answer changes with
`link` and was the source of confusion in earlier drafts of this doc:

| Decomposition | What the emulator's $y$ represents |
|---------------|-------------------------------------|
| `link=Identity()`, `shift=Constant(0)` | log-density (full, modeling prior absorbed into `target_single`). |
| `link=Identity()`, `shift=LogProb(modeling_prior)` | log-likelihood. |
| `link=LogSoftplus()`, `shift=LogProb(modeling_prior)` | $\mathrm{softplus}^{-1}(\mathrm{lik})$ — neither log nor density. The whole point of softplus is that this quantity has a more compact range than $\log\mathrm{lik}$ in the high-density region. |
| `link=LogSquare()`, `shift=LogProb(modeling_prior)` | $\sqrt{\mathrm{lik}}$ (with sign). |
| `link=GaussianLogLik(obs, C)`, `shift=LogProb(modeling_prior)` | A forward-model output (e.g. simulated observations). The likelihood is computed from $y$ via the Gaussian density. |

`link=Identity()` with a `LogProb` shift is the common case:
log-likelihood emulation, exp link, modeling prior added. Choosing a
non-identity `link` is opting into a different emulated quantity,
deliberately.

### 4.3 Constraints

`Constraint` on $y$ is mostly metadata. Concrete constraints:

- `NoConstraint` — default; `link` accepts any real $y$.
- `NonNegative` — used by `Square`/`LogSquare` to pin sign convention.
- `Positive` — for any link whose `log_link` has a singularity at zero.

Acquisitions and pushforward primitives can read the constraint to
validate inputs or pin sign conventions; the `DensityDecomposition`
itself does not enforce it (no runtime checks on $y$).

## 5. Pushforward dispatch

`pushforward(map, dist) → dist` is multiple-dispatch on `(type(map),
type(dist))`. Closed-form registrations land per pair; an MC fallback
covers everything else.

### 5.1 Dispatch table

For the typical sabi cases:

| Map | `Normal(batch_shape=(n,))` | `MVN(event_shape=(n,))` |
|-----|---------------------------|--------------------------|
| `Identity` | unchanged | unchanged |
| `Constant(c)` | `Dirac(c)` | `Dirac(c)` |
| `Affine(slope, intercept)` | closed: `Normal(slope·loc + intercept, |slope|·scale)` | closed: `MVN(slope·loc + intercept, |slope|·scale_tril)` |
| `Compose(f, g)` | `pushforward(f, pushforward(g, dist))` | same, via recursion |
| `Exp` | closed: `LogNormal(loc, scale)` | MC |
| `Log` | (only registered for `LogNormal`) | MC |
| `Square` | non-central χ² (closed) | MC (no clean joint closed form) |
| `Softplus`, `LogSoftplus`, `LogSquare`, `GaussianLogLik`, ... | MC | MC |

### 5.2 Closed-form Gaussian-affine path

The common case in sabi today is `link=Identity()` with
`shift=LogProb(modeling_prior)`. The decomposition's pushforward
composes `Affine(slope=1, intercept=shift(x)) @ Identity =
Affine(slope=1, intercept=shift(x))`. Pushing a `Normal` or `MVN`
through this `Affine` shifts `loc` by `shift(x)` and leaves `scale` /
`scale_tril` unchanged — equivalent to today's `_shift_gaussian_loc`,
but as a generic Affine pushforward rather than form-specific code.

### 5.3 MC fallback for nonlinear and joint cases

For nonlinear element-wise Maps (e.g. `LogSoftplus`) or any
`(Map, MultivariateNormal)` pair without a closed-form entry, the
fallback samples from the input distribution, applies the Map per
sample, and returns a `NumericEmpiricalDistribution`. The joint
correlation structure (when applicable) is carried in the sample
correlations. The `@workflow_function(n_broadcast_samples=64)`
machinery now lives in [`sabi.maps._pushforward._mc_pushforward`](../src/sabi/maps/_pushforward.py)
(generalised from the original form-specific `_batch_form` to "any
Map" — and the form-specific dispatch module was removed in #65).

### 5.4 Worked example: joint MVN through a softplus-link decomposition

Setting: GP emulator at $n$ query points, joint mode (returns
`MVN(loc=(n,), scale_tril=(n, n))`); decomposition is
`DensityDecomposition(target_single=..., output_shape=(),
link=LogSoftplus(), shift=LogProb(modeling_prior))`.

```python
emulator_pred = emulator(X)                                  # MVN(event_shape=(n,))
log_density_pred = decomposition.pushforward(X, emulator_pred)
```

Internally:

1. `decomposition.pushforward` constructs `per_x_map = Affine(slope=1, intercept=shift(X)) @ LogSoftplus()`.
   `shift(X)` has shape `(n,)`; `Affine.intercept` has shape `(n,)`.
2. `pushforward(per_x_map, emulator_pred)` recurses through `Compose`:
   first `pushforward(LogSoftplus(), MVN)` — no closed form, MC fallback
   produces samples of shape `(S, n)`.
3. Then `pushforward(Affine(slope=1, intercept=(n,)), empirical_dist)` —
   shifts each sample's $i$-th component by `intercept[i]`. Closed form on
   `NumericEmpiricalDistribution` (just elementwise add).
4. Result: `NumericEmpiricalDistribution` of joint log-density samples,
   shape `(S, n)`, with joint structure preserved through the Affine
   shift.

For the same decomposition under joint MVN with `link=Identity()` (the
common case), the entire path is closed form: step 2 collapses to
identity, step 3 is `pushforward(Affine, MVN)` → `MVN`, no MC needed.

### 5.5 Point-set agnostic

The decomposition's `pushforward(x, y)` doesn't care about the choice
of $X$ — it operates on whatever `Distribution` the emulator produces.
Per-point marginal mode (`Normal` with `batch_shape=(n,)`), joint mode
(`MVN` with `event_shape=(n,)`), heteroskedastic Gaussian, mixture,
Student-$t$: all handled as long as the dispatch table has entries for
the relevant `(Map, Distribution)` pair, otherwise MC.

## 6. Acquisition customisation axes

Many acquisitions admit two orthogonal customisation axes. The framework
should make both consistent and discoverable.

### 6.1 Axis 1 — which intermediate distribution to target

Existing axis. `AcquisitionTarget` selects the bridging-state at which the
acquisition's surrogate is built:

- `CURRENT` — current round's intermediate (default).
- `NEXT` — next round's intermediate (one-step look-ahead, SMC-flavour).
- `TERMINAL` — final intermediate (always optimise toward the final target).

This is unchanged by the form refactor — it sits at the algorithm level
and is orthogonal to which layer is queried.

### 6.2 Axis 2: which predictive layer to target

New axis, surfaced by the `Map`-dispatch design. The three layers map to
ProbPipe primitives that already exist (or compose trivially from ones that
do), so no new sabi-side methods are needed:

| Layer | Op | What it returns |
|-------|-----|-----------------|
| Emulator (raw $y$) | `surr_dist.emulator(X)` | `Distribution[Array]` over $y$ — the emulator predictive at $X$. |
| Unnormalised log-density | `random_unnormalized_log_prob(surr_dist, X)` | `Distribution[Array]` over $\log\tilde p(X)$ — what `EmulatedDistribution` exposes via `_random_unnormalized_log_prob`. |
| Unnormalised density | `pushforward(Exp(), random_unnormalized_log_prob(surr_dist, X))` | `Distribution[Array]` over $\tilde p(X)$. |

`EmulatedDistribution` exposes `emulator` as a property (so callers can
write `surr_dist.emulator(X)` to get the raw predictive) and implements
`_random_unnormalized_log_prob() -> RandomFunction` (per ProbPipe's
`SupportsRandomUnnormalizedLogProb`); the `RandomFunction` returned
pushes the emulator predictive through `form.pushforward` under the
hood. Both the deterministic and random log-density paths converge here:
`unnormalized_log_prob(target_dist, x)` and
`random_unnormalized_log_prob(surr_dist, X)` are the non-random and
random analogues of the same quantity.

The density layer falls out of `Exp` map composition: `pushforward(Exp(),
log_density_dist)` is the (random) unnormalised density. ProbPipe could
later add a `random_unnormalized_prob` op as the analogue of its existing
deterministic `unnormalized_prob` (which is `exp(unnormalized_log_prob)`);
sabi's pushforward composition gives the same result without waiting on
that op.

### 6.3 Acquisition example

```python
@dataclass
class MaxVariance(Acquisition):
    target_intermediate: AcquisitionTarget = AcquisitionTarget.CURRENT
    target_layer: PredictiveLayer = PredictiveLayer.LOG_DENSITY

    def acquire(self, surr_dist: SurrogateDistribution, X_candidate) -> Array:
        match self.target_layer:
            case PredictiveLayer.EMULATOR:
                pred = surr_dist.emulator(X_candidate)
            case PredictiveLayer.LOG_DENSITY:
                pred = random_unnormalized_log_prob(surr_dist, X_candidate)
            case PredictiveLayer.DENSITY:
                pred = pushforward(Exp(), random_unnormalized_log_prob(surr_dist, X_candidate))
        return variance(pred)
```

`PredictiveLayer` is a small enum:

```python
class PredictiveLayer(Enum):
    EMULATOR     = "emulator"
    LOG_DENSITY  = "log_density"
    DENSITY      = "density"
```

Acquisitions that don't vary along axis 2 (e.g. log-EI, where the
log-density layer is fixed by definition) simply omit the parameter.

The two-axis structure should be documented prominently in
`docs/acquisitions.md` (does not yet exist; would land alongside this
work) and demonstrated end-to-end in the bridging tutorial notebook
(see [§9](#9-phasing-and-follow-up-issues)).

## 7. ProbPipe boundary — log-density at the MCMC seam

Today, sabi exposes `TargetDistribution._unnormalized_log_prob(x) →
log-density` to ProbPipe's `SupportsUnnormalizedLogProb` protocol.
After [PR #63](https://github.com/arob5/sabi/pull/63),
`TargetDistribution` may implement `_unnormalized_log_prob`
analytically (for benchmarks) or raise `NotImplementedError` (for user
inverse problems). The MCMC-relevant random log-density boundary moves
to `EmulatedDistribution._random_unnormalized_log_prob`, which composes
the decomposition: `decomposition.link(y) + decomposition.shift(x)`,
pushed through the emulator predictive. ProbPipe MCMC sees the same
shape it does today.

For `Map`s whose `log_link` is numerically singular at certain inputs
(e.g. `LogSquare` at $z = 0$), the `Map` author handles stability —
not the boundary. The constraint metadata on `DensityDecomposition` can
express "$y$ should be non-negative" / "$y$ should be away from zero"
so acquisitions can avoid problematic regions.

A second protocol returning density-scale (`SupportsUnnormalizedDensity`)
could be added later if and when ProbPipe lands one; sabi does not need
it to ship the refactor.

## 8. Bridging

(Renamed from "tempering". The mathematical concept is *bridging* — a
parameterised path between an initial and a target distribution;
likelihood tempering is one specific bridge family.)

### 8.1 What's wrong with the current naming

The current `TemperingScheme` is two abstractions in one:

1. The mathematical *family of intermediate distributions* — defined by
   $(\pi_0, \pi_\mathrm{target}, \beta)$ for likelihood tempering, but the
   concept generalises to other parametrisations (mixture, data, ...).
2. The algorithmic *strategy for realising the intermediate at each round* —
   either by rebuilding the form (`ViaForm`) or by rescaling the emulator
   target (`ViaTarget`).

Conflating them means new bridge families inherit a `ViaForm`/`ViaTarget`
distinction whether or not it makes sense for them, and link-compatibility
dispatch has nowhere clean to live. Renaming and reshaping in the same
pass.

(Note: under the post-#63 layout, the `DensityDecomposition` lives on
`Algorithm`, not on `Problem` or `IntermediateTarget`. Bridges produce a
new `DensityDecomposition` per state; the bridging scheme reads the base
decomposition off the algorithm and composes `Map`s on its `link` /
`shift`.)

### 8.2 The new shape

```python
class BridgingScheme(ABC):
    """Family of intermediate distributions parameterised by a state."""
    initial: TargetDistribution
    target:  TargetDistribution

    def intermediate(self, state: Any) -> IntermediateTarget: ...
    def is_invariant_target_map(self, state_a, state_b) -> bool: ...
    def is_invariant_form(self, state_a, state_b) -> bool: ...
```

Subclasses encode bridge families plus implementation strategies:

```python
class LikelihoodBridgeViaForm(BridgingScheme):
    """π_β = π_init · (π_target / π_init)^β. β factor enters via the form."""

class LikelihoodBridgeViaTargetRescale(BridgingScheme):
    """Same intermediate density family. β factor enters via Y_train rescaling.
    Restricted to forms with link = Identity (the current ExpLink case)."""

class GeometricBridge(BridgingScheme):
    """π_β = π_init^(1-β) · π_target^β. For forms with shift = Constant(0)."""
```

(The `sabi.tempering` → `sabi.bridging` rename is a search-and-replace
across `src/sabi/tempering/` → `src/sabi/bridging/` plus updating
[`docs/tempering.md`](tempering.md) → `docs/bridging.md`. Material churn
but mechanical.)

### 8.3 `LikelihoodBridgeViaForm`: link-agnostic bridging via Map composition

```python
class LikelihoodBridgeViaForm(BridgingScheme):
    initial: TargetDistribution

    def intermediate_decomposition(
        self, base: DensityDecomposition, beta: float
    ) -> DensityDecomposition:
        # The base decomposition represents log_target = base.link(y) + base.shift(x),
        # with shift typically = LogProb(modeling_prior). Tempering raises the
        # likelihood part to β: π_init · (π_target / π_init)^β = link(y)^β · π_init(x).
        # In log-space: β · base.link(y) + base.shift(x).
        return DensityDecomposition(
            target_single=base.target_single,                     # unchanged
            output_shape=base.output_shape,                       # unchanged
            link=Affine(slope=beta, intercept=0.0) @ base.link,   # scale link by β
            shift=base.shift,                                     # unchanged (= log p_init)
            constraint=base.constraint,
        )
```

Note that the β-scaling is just a `Map` composition — nothing about it
depends on `base.link`. This works under any link, including the new
`LogSoftplus` and `LogSquare` cases. The current
`_LogLikPlusPriorTempered` and `_ForwardModelTempered` collapse into
this single rule. `target_single` and `output_shape` are passed through
unchanged: bridging modifies how the emulator's output composes into
log-density, not what the emulator approximates.

### 8.4 `GeometricBridge` — for decompositions with shift = Constant(0)

The current `_IdentityTempered` has `Identity` base form (no prior in
the decomposition) and produces $(1-\beta) \log\pi_0(x) + \beta \cdot
y$. This is the geometric bridge:

```python
class GeometricBridge(BridgingScheme):
    initial: TargetDistribution

    def intermediate_decomposition(
        self, base: DensityDecomposition, beta: float
    ) -> DensityDecomposition:
        # base.shift is typically Constant(0); base.link encodes the full
        # target log-density. New decomposition interpolates between log p_init
        # and base via β.
        return DensityDecomposition(
            target_single=base.target_single,
            output_shape=base.output_shape,
            link=Affine(slope=beta, intercept=0.0) @ base.link,
            shift=Affine(slope=(1 - beta), intercept=0.0) @ LogProb(self.initial),
            constraint=base.constraint,
        )
```

The two bridges share "scale link by β" and differ only in shift
handling: `LikelihoodBridgeViaForm` keeps the existing shift (because
that shift *is* $\log\pi_\mathrm{init}$, not a separate term to bridge);
`GeometricBridge` constructs a new shift $(1-\beta)\log\pi_\mathrm{init}$.

### 8.5 `LikelihoodBridgeViaTargetRescale` — exp-link-only optimisation

Today's `LikelihoodTemperingViaTarget` rescales `Y_train = β · Y_raw`.
This is mathematically equivalent to via-form bridging *only* when the
emulator's $y$ is log-likelihood, i.e. when `link == Identity()`. For
non-`Identity` links, scaling $y$ by $\beta$ doesn't correspond to
raising the link's output to power $\beta$.

```python
class LikelihoodBridgeViaTargetRescale(BridgingScheme):
    initial: TargetDistribution

    def intermediate_target(self, base: DensityDecomposition, beta: float):
        if not isinstance(base.link, Identity):
            raise NotImplementedError(
                f"LikelihoodBridgeViaTargetRescale requires base.link "
                f"to be Identity (so y is log-likelihood and scaling y by β "
                f"is equivalent to scaling the link by β). "
                f"Got link={type(base.link).__name__}. "
                f"Use LikelihoodBridgeViaForm for non-Identity links."
            )
        # ... existing target-rescale logic.
```

Forward-model decompositions (`link=GaussianLogLik(obs, C)`) are also
not `Identity`, so the same error fires. No special forward-model
branch is needed.

### 8.6 Future bridge families

`LikelihoodBridgeViaForm` and `GeometricBridge` are the first two concrete
bridges. Future families slot in as additional `BridgingScheme` subclasses
without affecting `Map` or `DensityDecomposition`:

- `MixtureBridge`: $\pi_\beta = (1-\beta)\pi_0 + \beta\pi_\mathrm{target}$.
  Algebraically additive in densities, not multiplicative; pushforward
  composes differently. New `BridgingScheme` subclass.
- `DataBridge`: $\pi_\beta(x) \propto \pi_0(x) \cdot \prod_{i \in S_\beta}
  L_i(x)$ for a sequence of data subsets. State PyTree is a subset
  index, not a scalar.

Neither requires changes to `Map` or `DensityDecomposition`. The bridge
abstraction is link-agnostic in its mathematical content; link-compatibility
checks live per-strategy as in §8.5.

## 9. Phasing and follow-up issues

This doc is the deliverable for **Phase 1** of the issue. Subsequent
phases land in their own PRs / issues. The phasing is now coordinated
with [PR #63](https://github.com/arob5/sabi/pull/63) — that PR's
implementation lands between this PR's Phase 2 and Phase 3, performing
the rename + extension of the host class. Phases 3+ in this PR proceed
against the post-#63 codebase.

1. **Phase 1 — design doc** (this PR). No code changes.
2. **Phase 2 — `Map` + `pushforward` infrastructure** (separate
   issue/PR). Add the `Map` ABC; concrete maps (`Identity`, `Constant`,
   `Affine`, `Compose`, `Exp`, `Log`, `Softplus`, `LogSoftplus`, `Square`,
   `LogSquare`, `LogProb`, `GaussianLogLik`); the `pushforward(Map,
   Distribution)` multiple-dispatch op with closed-form registrations
   for the entries in [§5.1](#51-dispatch-table) and an MC fallback
   that generalises today's `_batch_form`. No form / surrogate /
   bridging changes yet — this phase ships the primitives both this PR
   and PR #63 depend on.
3. **PR #63's implementation** (sibling PR, lands between this PR's
   phases 2 and 3). Renames `LogDensityForm` family →
   `DensityDecomposition` (single class with `target_single`,
   `output_shape`, `link`, `shift`, `constraint`), moves it to
   top-level `src/sabi/density_decomposition.py`, restructures
   `TargetDistribution` to drop its `prior` / `target_single` /
   `output_shape` / `log_density_form` fields, splits the prior roles
   across `Algorithm` / per-optimizer fields, and removes
   `sampling.py`. See PR #63's design doc for the full migration
   footprint.
4. **Phase 3 (this PR, post-#63) — concrete `Map` subclasses for
   non-default links** (separate issue/PR, possibly merged with Phase
   2). After PR #63 lands, the `DensityDecomposition` shape is in
   place; this phase ensures the `Softplus` / `LogSoftplus` / `Square`
   / `LogSquare` `Map` subclasses are wired in with their pushforward
   dispatch entries. (Largely subsumed by Phase 2 if the `Map` infra
   ships all concrete subclasses together. Kept as a placeholder
   phase in case shipping happens incrementally.)
5. **Phase 4 — acquisition layered access** (separate issue/PR).
   `EmulatedDistribution` already exposes `emulator` as a property and
   implements `_random_unnormalized_log_prob` (via the
   `pushforward_marginal` path that becomes
   `decomposition.pushforward` after PR #63). Phase 4's work is
   acquisition-side: introduce the `PredictiveLayer` enum and add
   `target_layer` parameter to acquisitions that benefit (likely just
   `MaxVariance` for now), with the dispatch wired to
   `surr_dist.emulator(X)` /
   `random_unnormalized_log_prob(surr_dist, X)` /
   `pushforward(Exp(), random_unnormalized_log_prob(surr_dist, X))`
   per [§6.2](#62-axis-2-which-predictive-layer-to-target). No new
   methods on `SurrogateDistribution`.
6. **Phase 5 — bridging rename + collapse** (separate issue/PR).
   `sabi.tempering` → `sabi.bridging`; `TemperingScheme` →
   `BridgingScheme`; per-form-type `_*Tempered` classes collapse into
   `LikelihoodBridgeViaForm.intermediate_decomposition` returning a
   single `DensityDecomposition` (per
   [§8.3](#83-likelihoodbridgeviaform-link-agnostic-bridging-via-map-composition));
   `_IdentityTempered` becomes `GeometricBridge`;
   `LikelihoodTemperingViaTarget` becomes
   `LikelihoodBridgeViaTargetRescale` raising on non-`Identity` link.
7. **Phase 6 — softplus link end-to-end** (separate issue/PR). Verify
   `Softplus` and `LogSoftplus` numerical stability in `log_link`;
   register pushforward dispatch entries (closed-form where applicable,
   MC fallback elsewhere); end-to-end test on a synthetic 2D posterior
   comparing GP fit quality and posterior recovery against the
   `Identity` link baseline.
8. **Phase 7 — square link end-to-end** (separate issue/PR). Verify
   `Square` and `LogSquare` numerical stability; pin sign convention
   via `NonNegative` constraint and any required output transform;
   register non-central χ² closed-form pushforward; benchmark
   validation.
9. **Phase 8 — bridging tutorial notebook** (separate issue/PR). New
   notebook `docs/tutorials/bridging.ipynb` (or similar) walking through
   several worked cases: untempered, `LikelihoodBridgeViaForm` under
   `Identity` link, `LikelihoodBridgeViaForm` under `LogSoftplus` link,
   `GeometricBridge`, `LikelihoodBridgeViaTargetRescale` (and its
   failure mode under non-`Identity` link with the informative error).
   Covers the documentation requirement; the notebook is the primary
   artefact for "bridging × emulators × acquisitions interaction"
   understanding.
10. **Phase 9 — link-aware acquisition audit** (separate issue/PR).
    Audit acquisitions for tail-sensitivity introduced by exp link;
    introduce link-aware variants of EI / VBMC-style acquisitions where
    the closed-form depends on the link choice.

Phases 6 and 7 can land in either order. Phase 8 lands incrementally
as 6/7 enable each scenario.

## 10. Open questions

1. **Multimodal sign convention for `Square` link.** How is $y$'s sign
   pinned in the multimodal case? Most natural: assume $y \ge 0$
   everywhere via a prior or an output transform, but this gives back
   the non-negativity constraint we were trying to avoid. Defer to
   Phase 7.
2. **Per-`Problem` vs per-run link choice.** Resolved by [PR #63](https://github.com/arob5/sabi/pull/63):
   the `DensityDecomposition` lives on `Algorithm`, not `Problem`. The
   user constructs a decomposition and pairs it with a benchmark at
   algorithm-construction time; ablations are handled by passing a
   different decomposition to the runner.
3. **Heteroskedastic forward models.** Decompositions where the
   likelihood depends on $x$ multiplicatively (e.g. $\log\mathcal{N}(\mathrm{obs}
   \mid f(x), C(x))$ with $x$-dependent noise covariance) violate the
   "additive x-shift only" constraint. The shape of
   `DensityDecomposition` doesn't support this directly. Users with
   this case can subclass `DensityDecomposition` and override
   `__call__` / `pushforward`; no abstraction in the base needs to
   bend. Worth a paragraph in the user-facing docs, not a feature in v1.
4. **Default `link` and `shift` arguments.** PR #63 defaults
   `shift=Constant(0.0)` on `DensityDecomposition` and requires explicit
   `link`. That matches the lean here.
5. **Bijector unification.** When ProbPipe's bijectors land as a `Map`
   subclass, can sabi's `Map` ABC be the same? Likely yes; revisit when
   ProbPipe ships.

## 11. Composition with adjacent issues

- **[PR #63](https://github.com/arob5/sabi/pull/63) — `TargetDistribution`
  / `DensityDecomposition` split.** Sibling design PR. Hosts the
  `DensityDecomposition` class this PR's `link` / `shift` machinery
  plugs into; recommended landing order is in
  [§9](#9-phasing-and-follow-up-issues).
- **#3 — input/output transforms.** Output-side warps (e.g. an affine
  standardisation of `Y_train`) sit *between* the emulator and the
  decomposition in the layer chain. A learned warp composed with
  `link=Identity()` can recover softplus-like behaviour, but the
  abstractions are conceptually distinct — the decomposition's link is
  a fixed mathematical relationship, the warp is a learned change of
  variables.
- **#50 — noisy targets.** Both touch the pushforward dispatch. The
  noise enters by perturbing the emulator's predictive at observed
  points; the decomposition's `link` is downstream of that. The
  closed-form pushforward registrations in [§5.1](#51-dispatch-table)
  must compose with whatever noise model #50 lands. Designing the
  decomposition abstraction first means the noise work has fewer
  link-specific decisions to make.
- **#4 — `update_emulator` cheap-path dispatch.** The emulator-update
  optimisation interacts with `LikelihoodBridgeViaTargetRescale`: when
  only $Y_\mathrm{train}$ is rescaled, a cheap update is possible.
  With link-compatibility error dispatch (only `Identity` link
  supports the rescale), the cheap-update path remains a clean
  optimisation.

## 12. Style references

- [`docs/design.md`](design.md) — overall sabi architecture.
- [`docs/tempering.md`](tempering.md) — current tempering design (will
  be rewritten as `docs/bridging.md` in Phase 5; this doc and that one
  should be kept in sync after Phase 5).
- [`docs/probpipe_random_measure_proposal.md`](probpipe_random_measure_proposal.md)
  — proposal-style design doc that informed this one.
- [`docs/notation.md`](notation.md) — shape conventions; the `Map`
  shape contract in [§3.1](#31-shape-contract) follows the same
  per-event / batch-broadcasting conventions.
