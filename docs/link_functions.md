# Link functions and bridging — design proposal

**Status:** draft v0.2 — pre-implementation
**Issue:** [#55](https://github.com/arob5/sabi/issues/55)
**Last updated:** 2026-05-06

> Proposal for two coupled refactors:
>
> 1. Generalise sabi's emulated quantity beyond log-density. Today every
>    `LogDensityForm` returns an unnormalised log-posterior, with `density ∝
>    exp(form_output)` applied implicitly downstream. We replace this with an
>    explicit **link function** abstraction: `density ∝ link(g(x))` for `link
>    ∈ {exp, softplus, square, …}`.
> 2. Reframe `TemperingScheme` as `BridgingScheme` — the operative concept is a
>    *family of intermediate distributions between an initial and target
>    distribution*, of which likelihood tempering is one instance. Several
>    redundant abstractions collapse under this framing.
>
> The two refactors are coupled because the bridge's interaction with non-exp
> links surfaces dispatch decisions that didn't exist before (e.g.,
> "via-target-rescale" tempering only works under exp link). Doing them in one
> design pass avoids re-litigating the form/link API when the bridging
> rewrite lands.
>
> This is a **design-only** PR: no source code changes. Implementation is split
> across follow-up issues (see [§9](#9-phasing-and-follow-up-issues)).
>
> Sabi's API is not yet stable; this proposal does not preserve back-compat
> with current names or signatures.

## 1. Motivation

The exponential link is the default not because it is best, but because it is
implicit. Every form in
[`src/sabi/problems/forms.py`](../src/sabi/problems/forms.py) returns a value
$g(x)$ that is interpreted as $\log p(x)$, and the consumer (ProbPipe MCMC,
`Weights.log_weights` softmax) applies $\exp$ to recover an unnormalised
density. Sabi never calls `exp` itself — but the assumption is everywhere.

The exp link has well-known instabilities:

- **Tail blow-up.** Small perturbations in the emulator's mean translate to
  multiplicative blow-up in density when $\log p$ is large. A GP uncertainty
  band of $\pm 2$ around $\log p = 30$ is densities differing by $e^4 \approx
  55\times$.
- **Pushforward variance.** Pushing a Gaussian through `exp` gives a log-normal
  whose mean and variance are dominated by the upper tail of the Gaussian. The
  bias note in
  [`src/sabi/surrogate/estimators.py:15`](../src/sabi/surrogate/estimators.py)
  is a direct symptom: the plug-in mean estimator is biased because of exactly
  this tail-domination.
- **Acquisition behaviour.** Acquisitions that integrate the surrogate against
  a posterior weight inherit the same tail sensitivity.
- **GP fitting.** Log-densities span huge dynamic range — often hundreds of
  nats. A stationary-kernel GP struggles to fit a function ranging over
  $[-300, 0]$ with sharp falloff. Output-scaling helps with the symptom; the
  underlying issue (`exp` couples small fitting errors to large density errors)
  remains.

Alternative links (softplus, square) preserve non-negativity of density while
giving the emulated quantity a more compact range. The motivation for the
abstraction is thus narrow: **let the user choose what quantity the emulator
fits, with `log-likelihood` (the current default, exp link) as one option
among several**.

## 2. Current state map

The implicit-exp assumption is baked into the following sites. Any link
abstraction must intercept all of them.

| Site | What it does | Why it's exp-bound |
|------|--------------|---------------------|
| [`src/sabi/problems/forms.py`](../src/sabi/problems/forms.py) | `LogDensityForm` family — `Identity`, `LogLikPlusPrior`, `ForwardModel` | Return value is documented as "unnormalised log-posterior". |
| [`src/sabi/surrogate/_pushforward.py:79`](../src/sabi/surrogate/_pushforward.py) | `pushforward_marginal` Gaussian-affine closed form | Closed form preserves Gaussianity by shifting `loc`; only valid because the form is *additive* in log-density space. |
| [`src/sabi/surrogate/estimators.py:129`](../src/sabi/surrogate/estimators.py) | `_ExpectedTargetDistribution._unnormalized_log_prob` plug-in mean | Plugs emulator's predictive mean into form, treats result as log-density. |
| [`src/sabi/surrogate/weighted_empirical.py:64`](../src/sabi/surrogate/weighted_empirical.py) | `WeightedEmpiricalRandomMeasure` log-weights | Weights consumed via softmax in ProbPipe `Weights` — assumes log-space input. |
| [`src/sabi/target_distribution.py:56`](../src/sabi/target_distribution.py) | `TargetDistribution._unnormalized_log_prob` | ProbPipe `SupportsUnnormalizedLogProb` boundary — MCMC expects log-density. |
| [`src/sabi/tempering/likelihood.py`](../src/sabi/tempering/likelihood.py) | `_LogLikPlusPriorTempered`, `_ForwardModelTempered`, `_IdentityTempered` | All do `β·log_likelihood_term`; tempering composes additively in log-space (see [§8](#8-bridging)). |

The `WeightedEmpiricalRandomMeasure` site is link-agnostic in practice: weights
are computed *outside* the class via the form, so the class only sees the
already-converted log-weights. The other five sites all directly compose with
form output.

## 3. The `Link` abstraction

A `Link` is the function $\ell$ that maps the emulator's output $g(x)$ to an
unnormalised likelihood (or density, depending on the form). It carries two
methods plus an inverse:

```python
class Link(ABC):
    def link(self, g: Array) -> Array: ...        # g → unnormalised lik/density
    def log_link(self, g: Array) -> Array: ...    # g → log unnormalised lik/density
                                                  # (numerically stable; not log(link(g)))
    def inv_link(self, d: Array) -> Array: ...    # unnormalised lik/density → g
                                                  # (used at construction / reference targets)
```

The four candidate concrete links:

### 3.1 `ExpLink` — the status quo, in disguise

$$\ell(g) = \exp(g),\quad \log\ell(g) = g,\quad \ell^{-1}(d) = \log(d).$$

The fact that `log_link` is the identity is exactly why the current codebase
doesn't have to call any link function: the emulator emits $g(x) = \log
\mathrm{lik}(x)$ directly, and the form returns log-density without applying
any transformation.

**The most important sentence in this whole document:** under `ExpLink`, *$g$
is log-likelihood* (or log-posterior, depending on the form). Choosing
`ExpLink` is equivalent to "the emulator fits log-likelihood / log-posterior",
which is the common case and the current behaviour.

### 3.2 `SoftplusLink`

$$\ell(g) = \log(1 + e^g),\quad \log\ell(g) = \log\log(1 + e^g),\quad
\ell^{-1}(d) = \log(e^d - 1).$$

Range: $(0, \infty)$. Behaves like $\exp(g)$ for $g \ll 0$ (preserving
log-likelihood tail-decay) and like $g$ for $g \gg 0$ (much milder dynamic
range). Smooth and differentiable everywhere; non-negative without any
constraint on $g$.

The emulator now fits $g(x) = \mathrm{softplus}^{-1}(\mathrm{lik}(x))$, *not*
log-likelihood. In the high-density region, $g$ has a compact range; in the
tails, $g \approx \log\mathrm{lik}$. Tighter GP fits in the high-density
region are the qualitative argument for softplus.

Pushforward of $\mathcal{N}(\mu, \sigma^2)$: no closed form. Stable
approximations: Gauss–Hermite quadrature for low-order moments; Monte Carlo
via the existing MC fallback in `pushforward_marginal`.

### 3.3 `SquareLink`

$$\ell(g) = g^2,\quad \log\ell(g) = 2\log|g|,\quad \ell^{-1}(d) = \sqrt{d}.$$

Range: $[0, \infty)$. Differentiable, but $\log\ell$ has a singularity at
$g = 0$. Has a clean Hellinger / $L^2$-geometry interpretation and connects
to the warped-GP literature for non-negative quantities; also appears in
Bayesian-quadrature work (Osborne et al.).

The emulator fits $g(x) = \sqrt{\mathrm{lik}(x)}$ (with sign).

Pushforward of $\mathcal{N}(\mu, \sigma^2)$ through $g^2$ is a scaled
non-central chi-squared with one degree of freedom and noncentrality
parameter $\lambda = (\mu/\sigma)^2$ — closed-form moments and density.

**Sign-convention drawback:** density is invariant under $g \mapsto -g$. The
emulator's $g$ is identifiable only up to a global sign per disconnected
support component. Practical implementations pin a sign convention (e.g.
$g \ge 0$ via a prior or an output transform that absorbs the sign).
Multimodal posteriors with disconnected high-density regions are a known
failure mode worth flagging.

### 3.4 `SigmoidLink` (for density ratios)

$$\ell(g) = \sigma(g) = \frac{1}{1 + e^{-g}}.$$

Range $(0, 1)$. Less natural for unnormalised densities (which are unbounded
above), but fits when sabi is used to emulate a *density ratio* — a
likelihood relative to a reference, or a posterior relative to the prior —
where the natural range is bounded. Listed for completeness; the design
must not preclude it but does not need to ship it in the first cut.

### 3.5 Identity-with-positivity (ruled out)

$\ell(g) = \max(g, 0)$ is non-differentiable at zero, which breaks
GP-machinery composition. Smooth approximations *are* softplus and square.
Not pursued.

## 4. The `DensityForm` abstraction

`LogDensityForm` is renamed to `DensityForm` and its base contract reduces
to:

```python
class DensityForm(ABC):
    def _call_single(self, x, y, *, prior) -> Array:
        """Return the unnormalised log-density at x given emulator output y."""
```

This is the abstraction in your first requirement: *"an underlying emulator
and a function that transforms the emulator target to obtain (log)
unnormalised density."* Every form is "emulator + transformation function";
the link is one factorisation of the transformation, not a universal field.

Three concrete subclasses, with renaming:

### 4.1 `Identity`

```python
@dataclass(frozen=True)
class Identity(DensityForm):
    """Emulator emits g(x); density(x) ∝ link(g(x)). No prior involvement."""
    link: Link

    def _call_single(self, x, y, *, prior=None) -> Array:
        return self.link.log_link(y)
```

Under `ExpLink`: $g(x) = \log p(x)$, the form returns $y$ unchanged. Same as
today's `Identity`.

Under `SoftplusLink`: emulator fits $g(x) = \mathrm{softplus}^{-1}(p(x))$;
form returns $\log\log(1 + e^y)$.

### 4.2 `LikelihoodWithPrior`

(Renamed from `LogLikPlusPrior`.)

```python
@dataclass(frozen=True)
class LikelihoodWithPrior(DensityForm):
    """Emulator emits g(x); likelihood(x) ∝ link(g(x)); density = link(g(x)) · prior(x)."""
    link: Link

    def _call_single(self, x, y, *, prior) -> Array:
        return self.link.log_link(y) + log_prob(prior, x)
```

Under `ExpLink`: $g(x) = \log L(x)$, form returns $y + \log\pi_0(x)$. Same as
today's `LogLikPlusPrior`.

Under `SoftplusLink`: emulator fits $g(x) = \mathrm{softplus}^{-1}(L(x))$;
form returns $\log\log(1 + e^y) + \log\pi_0(x)$.

### 4.3 `ForwardModel` (link-free by design)

```python
@dataclass(frozen=True)
class ForwardModel(DensityForm):
    """Emulator emits forward-model output y; user-supplied likelihood does the rest."""
    log_lik_from_outputs: Callable[[Array, Array], Array]

    def _call_single(self, x, y, *, prior) -> Array:
        return self.log_lik_from_outputs(x, y) + log_prob(prior, x)
```

`ForwardModel` does not carry a `Link`. The reason: the emulator's output is a
forward-model quantity (e.g. simulated observations), not a density-related
quantity. The user-supplied `log_lik_from_outputs` already absorbs whatever
likelihood model they want to apply. The link concept describes *"how the
emulator's output relates to density"* — and for forward-model emulation,
that mapping lives entirely inside the user's likelihood code.

This is not an awkward exception: `ForwardModel` is still a `DensityForm`
(satisfies the same `_call_single` contract) and integrates with downstream
machinery identically. It just doesn't expose a link as a separate axis.

(For users who want to fit a non-log likelihood — e.g. emulate
$\sqrt{\mathrm{lik}}$ via `ForwardModel` — they write
`log_lik_from_outputs(x, y) = 2 * log(abs(y))`, doing the link inside their
code. This is mathematically equivalent to a `LikelihoodWithPrior(SquareLink)`
where the emulator fits $\sqrt{L}$. The two paths are alternatives, not in
conflict.)

## 5. Map-based pushforward dispatch

The current `pushforward_marginal(input_dist, form, X, prior)` collapses
several distinct transformations into one function. To support both
non-exp links *and* the acquisition flexibility your second requirement asks
for, we factor the transformations into separate `Map` objects and reify
pushforward as multiple-dispatch on `(Map, Distribution)`.

This is also the abstraction ProbPipe is converging on for its
transport-map / pushforward op. By building it locally now, the eventual
swap to ProbPipe's primitive is a class-rename, not a refactor.

### 5.1 The `Map` abstraction

```python
class Map(ABC):
    """A measurable function. Pushforwards dispatch on (Map, Distribution) types."""
    def __call__(self, x: Array) -> Array: ...
```

Concrete maps for the link / form machinery, named for their mathematical
content (not their role in sabi):

| Map | $f(x)$ | Used for |
|-----|--------|----------|
| `Identity` | $x$ | `ExpLink.log_link` (the no-op case); identity pushforward |
| `Log` | $\log x$ | `ExpLink.inv_link`; numerically risky in tails |
| `Softplus` | $\log(1 + e^x)$ | `SoftplusLink.link` |
| `LogSoftplus` | $\log\log(1 + e^x)$ | `SoftplusLink.log_link` (numerically stable form) |
| `Square` | $x^2$ | `SquareLink.link` |
| `LogSquare` | $2\log|x|$ | `SquareLink.log_link` (numerically stable form) |
| `Sigmoid`, `LogSigmoid` | $\sigma(x)$, $\log\sigma(x)$ | `SigmoidLink` family |
| `Affine(a, b)` | $a \cdot x + b$ | scaling + shift; covers "shift loc by log-prior" |

Maps compose via `Compose(f, g)(x) = f(g(x))`. A link's `log_link` method
returns the appropriate `Map`; the link's `link` method likewise returns a
density-scale `Map`.

The "add log-prior to a log-density" operation is *not* a separate map type —
at fixed query point $x$, it's `Affine(a=1, b=log_prior(x))`. Across a batch
of query points, it's a per-row `Affine` shift, which is just elementwise
addition of a `Distribution` with a constant `Array`. ProbPipe will likely
have a primitive for this (distribution + array → distribution); sabi's
local stand-in is the existing `_shift_gaussian_loc` in
[`src/sabi/surrogate/_pushforward.py:144`](../src/sabi/surrogate/_pushforward.py).

### 5.2 The `pushforward` op

```python
def pushforward(map: Map, dist: Distribution) -> Distribution:
    """Multiple-dispatch: closed form for known (Map, Distribution) pairs; MC fallback."""
```

Closed-form registrations land per (Map type, Distribution type):

| Map | Distribution | Pushforward result |
|-----|-------------|--------------------|
| `Identity` | any | the same distribution |
| `Affine(a, b)` | `Normal(loc, scale)` | `Normal(a·loc + b, |a|·scale)` |
| `Affine(a, b)` | `MultivariateNormal(loc, scale_tril)` | `MultivariateNormal(a·loc + b, |a|·scale_tril)` |
| `Square` | `Normal(loc, scale)` | scaled non-central chi-squared, df=1 |
| `Log` | `Normal(loc, scale)` | log-normal |
| `Softplus`, `LogSoftplus` | `Normal` | no closed form — falls through to MC |
| `Compose(f, g)` | any | `pushforward(f, pushforward(g, dist))` |
| (any) | `SupportsSampling` | MC fallback via the existing `_batch_form` machinery |

The MC fallback is the existing `@workflow_function`-wrapped batched-form
pattern, generalised: any callable wrapped with `@workflow_function` becomes
a pushforward when handed a `Distribution` in a non-Distribution-typed slot.
Sabi already does this for `_batch_form`; the `Map` abstraction just gives
it a name.

### 5.3 What this gets us

The composition for "marginal random log-density at $X$" decomposes
explicitly:

```python
emulator_predictive = emulator(X)                           # Distribution[Array]
log_lik_predictive  = pushforward(form.link.log_link_map(),
                                  emulator_predictive)      # Distribution[Array]
log_density_predictive = log_lik_predictive + log_prior(X)  # Distribution[Array] + Array
```

Each line is a separate, named operation. Acquisitions that want to query
intermediate layers (next subsection) get them by composing different maps.

## 6. Acquisition customisation axes

Many acquisitions admit two orthogonal customisation axes. The framework
should make both consistent and discoverable.

### 6.1 Axis 1 — which intermediate distribution to target

Existing axis. `AcquisitionTarget` selects the bridging-state at which the
acquisition's surrogate is built:

- `CURRENT` — current round's intermediate (default).
- `NEXT` — next round's intermediate (one-step look-ahead, SMC-flavour).
- `TERMINAL` — final intermediate (always optimise toward the final target).

This is unchanged by the link / bridging refactor — it sits at the
algorithm level and is orthogonal to which layer is queried.

### 6.2 Axis 2 — which predictive layer to target

New axis, surfaced by the `Map`-dispatch design. A `SurrogateDistribution`
(post-rename) exposes layered predictives:

```python
class SurrogateDistribution:
    def emulator_predictive_at(self, X) -> Distribution: ...     # raw emulator
    def pre_link_predictive_at(self, X) -> Distribution: ...     # = emulator (alias for clarity)
    def log_density_predictive_at(self, X) -> Distribution: ...  # current pushforward_marginal
    def density_predictive_at(self, X) -> Distribution: ...      # = pushforward(link.link_map, emulator)
```

(Aliasing `emulator_predictive_at` and `pre_link_predictive_at` makes both
mental models accessible; the values are the same `Distribution`.)

Acquisitions that vary on this axis carry it as a parameter:

```python
@dataclass
class MaxVariance(Acquisition):
    target_intermediate: AcquisitionTarget = AcquisitionTarget.CURRENT  # axis 1
    target_layer: PredictiveLayer = PredictiveLayer.LOG_DENSITY         # axis 2

    def acquire(self, sp: SurrogateDistribution, X_candidate: Array) -> Array:
        match self.target_layer:
            case PredictiveLayer.EMULATOR:    pred = sp.emulator_predictive_at(X_candidate)
            case PredictiveLayer.LOG_DENSITY: pred = sp.log_density_predictive_at(X_candidate)
            case PredictiveLayer.DENSITY:     pred = sp.density_predictive_at(X_candidate)
        return variance(pred)
```

Acquisitions that *don't* vary along axis 2 (e.g. log-EI, where the
log-density layer is fixed by definition) simply don't take the parameter.

### 6.3 Naming

```python
class PredictiveLayer(Enum):
    EMULATOR     = "emulator"      # raw emulator predictive
    LOG_DENSITY  = "log_density"   # post-form, in log space
    DENSITY      = "density"       # post-link, in density space
```

The two-axis structure should be documented prominently in
`docs/acquisitions.md` (does not yet exist; would land alongside this work),
and demonstrated end-to-end in the tutorial-notebook (see
[§9](#9-phasing-and-follow-up-issues)).

## 7. ProbPipe boundary — log-density at the MCMC seam

Today, sabi exposes `TargetDistribution._unnormalized_log_prob(x) →
log-density` to ProbPipe's `SupportsUnnormalizedLogProb` protocol. This is
the only point where MCMC sees the form's output.

Under the link abstraction, the boundary is unchanged: `_unnormalized_log_prob`
returns log-density. Inside, the form composes
`link.log_link(y) + log_prob(prior, x)` and returns the result. ProbPipe MCMC
sees a log-density just like it does today.

For some links (e.g. `square` near $g \approx 0$), $\log\ell(g) = 2\log|g|$ is
numerically singular. The form must implement `log_link` carefully — but
this is the link author's responsibility, not the boundary's.

A second protocol (`SupportsUnnormalizedDensity` returning density-scale)
could be added later if and when ProbPipe lands one. Sabi does not need it
to ship link functions.

## 8. Bridging

(Renamed from "tempering". The mathematical concept is *bridging* — a
parameterised path between an initial and a target distribution; tempering
is one specific bridge family.)

This section is intentionally long. Bridging is where the link choice
interacts most subtly with existing machinery, and your second set of
requirements asks for a clean abstraction that supports multiple bridge
families and surfaces incompatible combinations as explicit errors.

### 8.1 What's wrong with the current naming

The current `TemperingScheme` is two abstractions in one:

1. The mathematical *family of intermediate distributions* — defined by
   $(\pi_0, \pi_\text{target}, \beta)$ for likelihood tempering, but the
   concept generalises to other parametrisations.
2. The algorithmic *strategy for realising the intermediate at each round* —
   either by rebuilding the form (`ViaForm`) or by rescaling the emulator
   target (`ViaTarget`).

Conflating them means new bridge families inherit a `ViaForm`/`ViaTarget`
distinction whether or not it makes sense for them, and link-compatibility
dispatch (where some strategies only work under exp link) has nowhere clean
to live.

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

A `BridgingScheme` is defined purely in terms of (initial, target, state).
Subclasses encode different bridge families *and* implementation strategies
together, but with consistent naming that surfaces both axes:

```python
class LikelihoodBridgeViaForm(BridgingScheme):
    """Likelihood bridge: π_β = π₀ · L^β. β factor enters via the form."""
    # Works under any link. Form-axis tempering.

class LikelihoodBridgeViaTargetRescale(BridgingScheme):
    """Likelihood bridge: π_β = π₀ · L^β. β factor enters via Y_train rescaling."""
    # Works ONLY under ExpLink. Target-axis tempering.
    # Raises at intermediate_target() time if the base form's link is not ExpLink.
```

(The `Tempering` → `Bridging` rename is a search-and-replace across
`src/sabi/tempering/` → `src/sabi/bridging/`, plus updating
[`docs/tempering.md`](tempering.md) → `docs/bridging.md` with parallel
content. Material churn but mechanical.)

### 8.3 Why the via-target-rescale strategy needs ExpLink

The mathematical family of likelihood bridges is

$$\pi_\beta(x) \;\propto\; \pi_0(x) \cdot L(x)^\beta.$$

The existing two strategies are equivalent under exp link because of the
identity $\log L^\beta = \beta \log L$ — scaling the *log-likelihood* by
$\beta$ is the same as raising the *likelihood* to power $\beta$.

Under non-exp links, the emulator's output $g$ is no longer log-likelihood.
For `SquareLink` ($g = \sqrt{L}$), raising likelihood to $\beta$ requires
$g \mapsto g^\beta$, *not* $g \mapsto \beta g$. For `SoftplusLink`, there
is no closed-form rescaling at all.

So:

- **`LikelihoodBridgeViaForm` works under any link.** The form's
  `_call_single` is rebuilt per state to compute
  $\beta \cdot \log\ell(g(x)) + \log\pi_0(x)$. This is link-aware *only*
  in that it uses the link's `log_link` method, which already exists. The
  $\beta$ factor multiplies the log-likelihood (not the emulator output);
  this is well-defined for every link.
- **`LikelihoodBridgeViaTargetRescale` only works under `ExpLink`.** The
  rescaling $Y_\text{train} = \beta \cdot Y_\text{raw}$ is mathematically
  equivalent to via-form *only* when $g$ is log-likelihood, i.e. under
  exp link. Other links would need link-specific rescaling — which we could
  add per-link later, but the obvious cases (square, softplus) don't have
  cheap closed-form rescaling, so the via-target-rescale optimisation
  is genuinely exp-link-specific.

### 8.4 Error dispatch

Per your third requirement, incompatible combinations raise informative
errors at construction or scheme-application time. Concretely:

```python
class LikelihoodBridgeViaTargetRescale(BridgingScheme):
    def intermediate(self, beta: float) -> IntermediateTarget:
        link = self.target.density_form.link if hasattr(self.target.density_form, "link") else None
        if not isinstance(link, ExpLink):
            raise NotImplementedError(
                f"LikelihoodBridgeViaTargetRescale requires the target form's link "
                f"to be ExpLink, because Y_train = β·Y_raw is mathematically equivalent "
                f"to via-form bridging only when g is log-likelihood. "
                f"Got link={type(link).__name__}. "
                f"Use LikelihoodBridgeViaForm (works under any link), or, if you want "
                f"target-rescale optimisation under your link, file an issue describing "
                f"the link-specific rescaling."
            )
        # ... existing target-rescale logic
```

`ForwardModel` doesn't expose a `link` attribute (it has none), so the
isinstance check above fails the way we want — `ForwardModel` is
incompatible with `LikelihoodBridgeViaTargetRescale` and the error message
should say so.

### 8.5 Form-by-form bridge math

Per-form details for `LikelihoodBridgeViaForm`, with explicit reference to
the link:

| Base form | Tempered density (any link) | Form output at state $\beta$ |
|-----------|------------------------------|--------------------------------|
| `Identity(link)` | $\pi_0(x)^{1-\beta} \cdot \ell(g(x))^\beta$ — geometric bridge | $(1-\beta) \log\pi_0(x) + \beta \log\ell(g(x))$ |
| `LikelihoodWithPrior(link)` | $\pi_0(x) \cdot \ell(g(x))^\beta$ | $\log\pi_0(x) + \beta \log\ell(g(x))$ |
| `ForwardModel(log_lik_from_outputs)` | $\pi_0(x) \cdot \exp(\beta \cdot \mathrm{log\_lik\_from\_outputs}(x, y))$ | $\log\pi_0(x) + \beta \cdot \mathrm{log\_lik\_from\_outputs}(x, y)$ |

The `Identity` case is the geometric bridge, identical to today's
`_IdentityTempered`. The `LikelihoodWithPrior` case generalises today's
`_LogLikPlusPriorTempered` to any link by routing through `link.log_link`
instead of treating $g$ as log-likelihood directly. The `ForwardModel`
case is unchanged from today's `_ForwardModelTempered` (no link involved).

The three classes collapse into a single `_LikelihoodBridgedForm` plus a
small per-form-type dispatch (or a `match` statement on the base form's
type — sabi already uses `isinstance` dispatch in the bridging module).
The total LoC is smaller than the current three-class implementation.

### 8.6 Future bridge families

`LikelihoodBridgeViaForm` is the first concrete bridge. Future families
slot in alongside it without affecting the link abstraction:

- `MixtureBridge`: $\pi_\beta = (1-\beta) \pi_0 + \beta \pi_\text{target}$.
  Algebraically different (additive in densities, not multiplicative);
  pushforward composes differently. Adds a new `BridgingScheme` subclass
  with its own `intermediate` method.
- `DataBridge`: $\pi_\beta(x) \propto \pi_0(x) \cdot \prod_{i \in S_\beta}
  L_i(x)$ for some sequence of data subsets $S_\beta$. Adds a new
  subclass; state PyTree is a subset index, not a scalar.

Neither requires any change to `Link` or `DensityForm`. The bridge
abstraction is link-agnostic in its mathematical content; link-compatibility
checks live per-strategy.

## 9. Phasing and follow-up issues

This doc is the deliverable for **Phase 1** of the issue. Subsequent phases
land in their own PRs / issues. The phasing acknowledges that this is now a
materially larger refactor than the original issue scoped.

1. **Phase 1 — design doc** (this PR). No code changes.
2. **Phase 2 — `Link` + `Map` infrastructure** (separate issue/PR). Add
   `Link` ABC and `ExpLink` concrete subclass; add `Map` ABC, concrete
   maps (`Identity`, `Affine`, `Log`, etc.), `pushforward(Map,
   Distribution)` multiple-dispatch op with closed-form registrations
   and MC fallback. No form / surrogate / bridging changes yet — this
   phase ships the primitives.
3. **Phase 3 — `DensityForm` + form rename** (separate issue/PR).
   `LogDensityForm` → `DensityForm`; `LogLikPlusPrior` →
   `LikelihoodWithPrior`; add `link: Link` field to `Identity` and
   `LikelihoodWithPrior`; rewrite `_call_single` bodies in terms of
   `link.log_link`. `ForwardModel` unchanged. Behaviour-preserving for
   all existing tests under `ExpLink`; benchmarks unchanged.
4. **Phase 4 — surrogate layered predictive access** (separate
   issue/PR). Add `SurrogateDistribution.{emulator,log_density,density}_predictive_at`
   methods, all built on the Phase 2 pushforward op. Audit current
   acquisitions for layer-access opportunities; introduce
   `PredictiveLayer` enum and add `target_layer` parameter to acquisitions
   that benefit (likely just `MaxVariance` for now).
5. **Phase 5 — bridging rename + collapse** (separate issue/PR).
   `sabi.tempering` → `sabi.bridging`; `TemperingScheme` →
   `BridgingScheme`; collapse the three `_*Tempered` form classes into
   one `_LikelihoodBridgedForm` with per-base-form dispatch;
   `LikelihoodTemperingViaForm` → `LikelihoodBridgeViaForm`;
   `LikelihoodTemperingViaTarget` → `LikelihoodBridgeViaTargetRescale`,
   raising on non-exp link.
6. **Phase 6 — `SoftplusLink`** (separate issue/PR). Concrete link with
   numerically-stable `log_link` and `inv_link`; `LogSoftplus` map and
   Gauss–Hermite or MC pushforward primitive; end-to-end test on a
   synthetic 2D posterior comparing GP fit quality and posterior recovery
   against `ExpLink`.
7. **Phase 7 — `SquareLink`** (separate issue/PR). Concrete link with
   sign-convention pinning; `LogSquare` and `Square` maps; non-central
   chi-squared closed-form pushforward; benchmark validation.
8. **Phase 8 — bridging tutorial notebook** (separate issue/PR). A new
   notebook in `docs/` (or `docs/tutorials/`) that explores the bridging
   abstraction across several worked cases: untempered loop,
   `LikelihoodBridgeViaForm` under `ExpLink`, `LikelihoodBridgeViaForm`
   under `SoftplusLink`, `LikelihoodBridgeViaTargetRescale` under
   `ExpLink`, the failure mode of `LikelihoodBridgeViaTargetRescale +
   SoftplusLink` (showing the error). Covers your documentation
   requirement; the notebook is the primary artefact for
   "bridging × emulators × acquisitions interaction" understanding.
9. **Phase 9 — link-aware acquisition audit** (separate issue/PR). Audit
   acquisitions for tail-sensitivity introduced by exp link; introduce
   link-aware variants of EI / VBMC-style acquisitions where the
   closed-form depends on the link choice.

Phases 2 and 3 are tightly coupled but worth separating: Phase 2's
`Link` + `Map` infrastructure can land and be tested in isolation
(unit-test the maps, the pushforward dispatch); Phase 3 wires them into
the existing form hierarchy without surprises. Phases 6 and 7 can land
in either order. Phase 8 is the documentation artefact; consider
landing pieces of it incrementally as Phases 6/7 enable each scenario.

## 10. Open questions

1. **Multimodal sign convention for square link.** How is $g$'s sign pinned
   in the multimodal case? Most natural: assume $g \ge 0$ everywhere via a
   prior or a transform, but this gives back the non-negativity constraint
   we were trying to avoid. Defer to the square-link implementation PR
   (Phase 7).
2. **Per-`Problem` vs per-run link choice.** Is the link a property of the
   `Problem` (the user picks it once) or a per-run config (the runner
   sweeps it for ablation)? Default: `Problem` carries it (via the form's
   `link` field), but the runner can override for ablation by constructing
   a copy of the problem with a different form.
3. **Inferred / learned links.** Out of scope; composes with #3 (input/output
   transforms). Listed as a longer-term direction.
4. **`ForwardModel` link variant.** Should there be a `ForwardModel` variant
   that exposes a `Link`, where the user supplies $\mathrm{lik\_from\_outputs}
   (x, y) \to \mathrm{pre\text{-}link\ likelihood}$ instead of
   $\log L$? Current proposal: no — the user can do this inside their
   existing `log_lik_from_outputs` callback. But if a use case emerges,
   adding `ForwardModelWithLink` is additive.
5. **`pre_link_predictive_at` vs `emulator_predictive_at` aliasing.** The
   two names describe the same value (the emulator's predictive). Aliasing
   is for clarity — does it instead introduce confusion? Could ship with
   just `emulator_predictive_at` and let the docstring explain that it's
   also "the pre-link distribution".

## 11. Composition with adjacent issues

- **#3 — input/output transforms.** Output-side warps (e.g. an affine
  standardisation of `Y_train`) sit *between* the emulator and the link in
  the layer chain. The link is "what the emulator's pre-warp output means
  as a density"; the output transform is "what the emulator's training
  input is, given a pre-link target value." A learned warp composed with
  an exp link can recover softplus-like behaviour, but the abstractions
  are conceptually distinct — the link is a fixed mathematical
  relationship, the warp is a learned change of variables.
- **#50 — noisy targets.** Both touch the pushforward dispatch. The noise
  enters by perturbing the emulator's predictive at observed points; the
  link is downstream of that. The closed-form pushforward registrations in
  [§5.2](#52-the-pushforward-op) must compose with whatever noise model #50
  lands. Designing the link abstraction first means the noise work has
  fewer link-specific decisions to make.
- **#4 — `update_emulator` cheap-path dispatch.** The emulator-update
  optimisation interacts with `LikelihoodBridgeViaTargetRescale`: when only
  $Y_\text{train}$ is rescaled, a cheap update is possible. With
  link-compatibility error dispatch (only ExpLink supports the rescale),
  the cheap-update path remains a clean optimisation.

## 12. Style references

- [`docs/design.md`](design.md) — overall sabi architecture.
- [`docs/tempering.md`](tempering.md) — current tempering design (will be
  rewritten as `docs/bridging.md` in Phase 5; this doc and that one
  should be kept in sync after Phase 5).
- [`docs/probpipe_random_measure_proposal.md`](probpipe_random_measure_proposal.md)
  — proposal-style design doc that informed this one.
