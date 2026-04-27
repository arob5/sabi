# `RandomMeasure` — design proposal for ProbPipe

> **Pasteable handoff** from a sabi planning session to a fresh Claude Code
> session in `/Users/andrewroberts/Desktop/git-repos/prob-pipe`. The goal of
> the receiving session is to design and implement a `RandomMeasure`
> abstraction in ProbPipe — a "distribution-valued probability distribution"
> — that sabi will then build on top of for its `SurrogatePosterior`
> hierarchy in v1.2.
>
> This document lays out the conceptual goal, a starting-point shape, and an
> explicit list of open questions. **The questions are intentionally
> unanswered** — they're what the receiving session is expected to settle
> with the help of current ProbPipe context, code, and tests.

## ⚠️ Receiving session: required reading first

Before designing or writing any code, **establish complete and current
context on ProbPipe**. The proposal below was drafted from a sabi-side
survey and may already be subtly inconsistent with current ProbPipe
conventions. Your job is to align this proposal with how ProbPipe is
actually structured today, not to import it wholesale.

Specifically, before drafting anything:

1. Read `probpipe/__init__.py` and walk the directory under `probpipe/core/`
   to understand the hierarchy of `Distribution`, `NumericRecordDistribution`,
   `EmpiricalDistribution`, `RecordDistribution`, `DistributionArray`, etc.
2. Read the protocol declarations (`probpipe/core/protocols.py` or wherever
   the `Supports*` types live) to understand the current style of declaring
   optional capabilities.
3. Determine the current state of `RandomFunction` — is there a class? a
   `dev/random-function` branch with WIP? merged into main? — and let that
   answer drive whether `_random_log_prob` lands in this PR.
4. Read `probpipe/_weights.py` (`Weights`), `probpipe/distributions/multivariate.py`,
   `probpipe/distributions/continuous.py`, and `probpipe/distributions/transformed.py`
   to understand parameter-handling and dtype conventions.
5. Read `probpipe/core/_distribution_array.py` (or wherever `DistributionArray`
   lives) and understand how ProbPipe currently represents shape-indexed
   collections of distributions. **Directly relevant** to the
   batch-of-random-measures question.
6. Look at recent commits (since 2026-04-01) touching `probpipe/core/` for the
   active rework of the distribution / record / array hierarchy.

**Avoid duplication.** If ProbPipe already has machinery that solves part of
this problem — particularly anything related to shape-indexed collections,
support metadata, or batched sampling — reuse it instead of inventing
parallel infrastructure. Several of the open questions below explicitly hinge
on what ProbPipe already has versus what would need to be added; answering
those well requires the context you'll build by doing the reading above.

## What is a `RandomMeasure`?

A **distribution-valued random variable**. Formally, for some sample space
`T`, a random measure `M` is a probability distribution over Π(T) — the
space of probability distributions on `T`. A draw `D ~ M` is a
`Distribution[T]`.

Three operations of interest, in increasing optionality:

1. **Sample** — draw one inner distribution: `M ↦ (key ↦ Distribution[T])`.
2. **Mean / expected distribution** — the marginalization
   `D̄(A) = ∫ D(A) dM(D)` for measurable `A ⊆ T`. Itself a `Distribution[T]`
   — the "mean of a random measure".
3. **Random log-density** — the random function `x ↦ log D(x)` whose
   marginal at any `x` is a `Distribution[Array]` (because `log D(x)` is a
   random scalar, varying in `D`). Same idea for unnormalized log-density.
   Naturally a `RandomFunction[T, Array]`.

## Why ProbPipe needs it

**Sabi context.** Sabi is a benchmark framework for sequential
surrogate-based Bayesian inference. The central object is a
`SurrogatePosterior`: given a stochastic surrogate (e.g. a GP) approximating
an expensive log-density, the induced posterior is a *random* probability
distribution — different surrogate function draws give different
posteriors. This is exactly a random measure. Sabi v1.2 wants
`SurrogatePosterior` to be a proper subclass of `RandomMeasure[Array]`,
with concrete subclasses for:

- A "weighted-empirical" baseline (Dirac random measure, no real surrogate
  uncertainty).
- A GP-pushforward random measure (proper random measure with non-trivial
  `_random_log_prob`).

**Beyond sabi.** Random measures appear in:
- Bayesian nonparametrics (Dirichlet-process measures, Pitman–Yor, etc.).
- Particle methods (the empirical measure of an SMC particle ensemble is a
  natural Dirac random measure).
- Anywhere distributions themselves are objects of inference (Bayesian model
  averaging, stochastic block models with distribution-valued parameters).

ProbPipe currently has no abstraction here; the local `SurrogatePosterior`
in sabi is a stand-in until ProbPipe lands a first-class concept.

## Proposed shape (starting point)

These are the **starting-point** shapes; the receiving session is expected
to refine them per ProbPipe conventions.

### Inheritance

```python
class RandomMeasure[T](Distribution[Distribution[T]]):
    ...
```

A `RandomMeasure[T]` IS-A `Distribution` whose element type is itself a
`Distribution[T]`. Implications:

- `sample(rm, key)` returns a `Distribution[T]` via the existing
  `SupportsSampling` protocol (subject to the open questions on shape /
  batching).
- `RandomMeasure` does **not** inherit from `NumericRecordDistribution` — it
  is parallel to it under the same `Distribution` root.

The receiving session should verify that ProbPipe's `Distribution` base
makes this generic-parameter pattern sensible — particularly whether
`Distribution[Distribution[T]]` is well-formed in current ProbPipe, or
whether a different way of expressing "element type is itself a
Distribution" is preferred.

### Sampling

`RandomMeasure[T]` implements `SupportsSampling`. A draw is a
`Distribution[T]`. Whether `sample(rm, key, sample_shape=...)` accepts
non-trivial `sample_shape` is the **batch-of-distributions question** below.

### Expected distribution ("mean")

The canonical random-measure mean is `D̄ = ∫ D dM(D)`, a `Distribution[T]`.
This collides with ProbPipe's existing `SupportsMean` if the latter is
committed to returning an array — see decision (1).

### Random log-density (OPTIONAL)

```python
class SupportsRandomLogProb(Protocol):
    def _random_log_prob(self) -> RandomFunction[T, Array]: ...

class SupportsRandomUnnormalizedLogProb(Protocol):
    def _random_unnormalized_log_prob(self) -> RandomFunction[T, Array]: ...
```

Mirrors `SupportsLogProb` / `SupportsUnnormalizedLogProb` with `random_`
prefix and `RandomFunction` return. Whether this lands in this PR depends
on `RandomFunction`'s availability — see decision (2).

### Inner-distribution metadata

A `RandomMeasure[T]` carries metadata about the inner distributions: their
support, their event shape (when applicable), maybe their dtype. This is
the **central design question** — see the next section.

## ⚠️ Open question: shape and support semantics for distribution-valued distributions

The **single trickiest aspect** of `RandomMeasure`. Worth dedicated thought
before any code lands.

For a regular ProbPipe `Distribution[T]`:

- `support: Constraint` describes the support of `T`-valued samples.
- `event_shape: tuple[int, ...]` describes the shape of one sample (when
  `T` is array-like, e.g. on `NumericRecordDistribution`).
- `batch_shape: tuple[int, ...]` describes a batch of independent
  distributions sharing parameters/shape, that together produce a batched
  array sample.

For a `RandomMeasure[T] = Distribution[Distribution[T]]`, **two layers of
"shape" and "support" exist simultaneously**:

| layer | meaning | analogue in current ProbPipe |
|---|---|---|
| outer support | support of the random measure itself — i.e., the space of `Distribution[T]`s in its support | `Distribution.support` |
| inner support | the support of `T` — what every inner distribution's `support` is | (no current analogue) |
| outer event "shape" | the "shape" of a single random-measure draw — but a draw is a `Distribution[T]`, not a tensor | `Distribution.event_shape` |
| inner event shape | the `event_shape` of the inner distributions (only meaningful when `T` is array-like) | `NumericRecordDistribution.event_shape` |
| outer batch shape | a batch of independent random measures | `NumericRecordDistribution.batch_shape` |
| inner batch shape | conceivably each inner distribution itself is batched | (no current analogue, or `DistributionArray`) |

Subquestions:

- **Does the outer support / outer event shape have any usable content?**
  The outer support is "all distributions over `T` with such-and-such inner
  support" — hard to describe with a `Constraint` object. Outer event shape
  is even murkier — a `Distribution` is not a tensor and has no scalar
  shape. One plausible direction: outer support and outer event shape are
  trivial / not exposed on `RandomMeasure`. The receiving session should
  confirm or reject.
- **How is inner support exposed?** As an `inner_support: Constraint`
  property? As a class attribute set by subclasses? As a generic-type bound
  elaborated by ProbPipe's existing dispatch machinery? The mirror question
  for `inner_event_shape`. Pick the form that aligns with how
  `Distribution.support` and `event_shape` are currently expressed in
  ProbPipe.
- **Outer batch — how is "a batch of independent random measures"
  expressed?** Expanded into the next major question below.
- **Is "inner batch" a useful concept?** Probably not for sabi-shaped use
  cases (each inner draw is a single distribution), but ProbPipe may have
  other use cases that warrant it.

Establishing **crisp answers and clear documentation of the layering** is
the most important deliverable of this PR alongside the class itself.
Cleanly separating "outer" from "inner" shape semantics will keep
`RandomMeasure` from becoming a confusing two-headed beast every time
someone reads about its `event_shape` or `support`.

## ⚠️ Open question: batch of random measures

ProbPipe currently represents shape-indexed collections of distributions
in (at least) two ways — verify against current ProbPipe state:

1. `NumericRecordDistribution.batch_shape`: a batched array distribution.
   One `_sample` call returns a batched array.
2. `DistributionArray`: a flat tuple of components addressed by a
   multi-dimensional batch shape. Each component is itself a scalar
   distribution.

For a "batch of random measures", neither is an obvious fit:

- `NumericRecordDistribution`-style batching produces array samples — but a
  `RandomMeasure` sample is a `Distribution`, which has no array dtype.
- `DistributionArray`-style would give a tuple of inner `RandomMeasure`s,
  each producing a `Distribution[T]` on sample — closer, but unclear how it
  composes with `inner_support` etc.

Possible directions (not exhaustive):

- (A) `RandomMeasure` does not support `batch_shape` directly; users wrap an
  explicit `DistributionArray[RandomMeasure[T]]` for batches. Simple but
  possibly lossy when inner `T` is array-like and a richer representation
  would be more natural.
- (B) `RandomMeasure` has its own `batch_shape` semantics that conform to
  ProbPipe's batch standards but produce a sequence / `DistributionArray` of
  inner distributions on sample, rather than a batched array.
- (C) ProbPipe extends its batch infrastructure to support
  distribution-valued samples generically (batched samples are themselves
  a distribution-of-distributions or a `DistributionArray`-shaped object).
  Larger ProbPipe change but cleanest in the long run.

**This is also a moment to take stock of whether ProbPipe's batch-handling
primitives are themselves underdeveloped.** Building `RandomMeasure` may
surface gaps the receiving session should flag — and possibly address in
this PR or a follow-up. The receiving session has license to propose batch
infrastructure improvements if `RandomMeasure` requires them; if those are
larger than this PR can absorb, file them as ProbPipe issues and decide
which of (A) / (B) / (C) is the v1 answer.

## Open question: `Numeric` specialization

When `T` is array-like (the common sabi case — `T = Array`), more becomes
tractable:

- `inner_event_shape` is well-defined (it's the `event_shape` of the inner
  distributions).
- `sample(rm, key, sample_shape)` could meaningfully accept a numeric
  `sample_shape` and return a batched representation — e.g., a
  `DistributionArray[Distribution[Array]]` of shape `sample_shape`, or some
  richer "batched random distribution" object.
- The expected distribution `mean(rm)` is a `Distribution[Array]` with
  `event_shape == inner_event_shape`.

Should ProbPipe expose a `NumericRandomMeasure[Array](RandomMeasure[Array])`
parallel to `NumericRecordDistribution` that adds these niceties? Or stay
generic and let subclasses opt into numeric-specific behaviour?

For the non-array case (e.g., `T` is itself a structured / record type, or
`T` is something exotic), the answer to "what does sampling with a
non-trivial `sample_shape` mean?" depends heavily on what ProbPipe's
batch-of-distributions story looks like — see the previous question. The
two questions are tangled and should be worked through together.

## Open question: interaction with ProbPipe broadcasting

ProbPipe has broadcasting infrastructure (e.g. `BroadcastDistribution`,
`WorkflowFunction.n_broadcast_samples`) that lets workflows run
sample-shape-aware computations against ProbPipe distributions. Two
subquestions:

- **Should `RandomMeasure` participate in broadcasting at all?**
  Conceptually, "broadcast a function over samples from a random measure"
  means: draw a batch of inner distributions, apply the function to each.
  The result is a batch of whatever-the-function-returns. Fits the
  broadcasting story conceptually, but requires the
  batch-of-distributions question to be resolved first.
- **If yes, what's the contract?** Does `WorkflowFunction` see a single
  `RandomMeasure` and produce a batched output? Or is broadcasting
  fundamentally about array-shaped outputs and `RandomMeasure` is
  opt-out?

The receiving session should review ProbPipe's current broadcasting
semantics and decide whether `RandomMeasure` slots in cleanly, requires
extension, or stays out of the broadcasting path entirely.

## Other design decisions

These are open questions from the original sabi planning session, retained
verbatim for the receiving session to settle.

1. **`SupportsMean` generalization vs. separate protocol.** Should
   ProbPipe's `SupportsMean` become parameterized by element type `T` (so
   `Distribution[T]._mean -> T`, giving `_mean -> Distribution[T]` on
   `RandomMeasure[T]`), or should `RandomMeasure` introduce a separate
   `SupportsExpectedDistribution` protocol returning `Distribution[T]`? The
   first is cleaner if `SupportsMean` is not already array-bound; the
   second is less invasive.

2. **`RandomFunction` availability.** Is `RandomFunction` first-class in
   current ProbPipe? If yes, `_random_log_prob` returns it directly. If no,
   the choices are: (a) wait for `RandomFunction` and skip this protocol,
   (b) ship `RandomMeasure` without `_random_log_prob` and add it later,
   (c) define a minimal `RandomFunction` placeholder in this PR.

3. **`inner_support` form.** Instance property (matches `Distribution.support`)?
   Class attribute? Generic-type bound? Pick the form that aligns with how
   `support` already works on `Distribution`.

4. **`inner_event_shape` placement.** Required for all `RandomMeasure[T]`,
   or only meaningful when `T` is array-like and lives on a
   `NumericRandomMeasure`-style subclass?

5. **`_sample` signature compatibility.** ProbPipe's existing `_sample` may
   commit to array-shaped `T` in its typing or return-shape conventions.
   Verify, and adjust the `RandomMeasure._sample` signature accordingly.

6. **Outer `event_shape` on `RandomMeasure`.** Confirm this should be
   empty / absent — the random measure's "events" are distributions, which
   don't have a numeric shape. Inner event shape is what users will reach
   for.

7. **Naming.** Sticking with `RandomMeasure`. Alternatives considered
   (`DistributionRandomVariable`, `MeasureValuedDistribution`) — the
   standard math term aligns with the `RandomFunction` precedent.

## Forward compatibility

These don't need to land in this PR, but the abstraction shouldn't
preclude them.

- **`condition_on(rm, observation) → RandomMeasure`.** Updating a random
  measure on new evidence (e.g., a new surrogate evaluation in sabi).
  Should compose via the existing `condition_on` op machinery.
- **`MixtureDistribution` connection.** A finite-support `RandomMeasure` —
  a discrete distribution over a finite set of `Distribution[T]`s — is
  exactly a mixture distribution. Worth a note in the docstring; a future
  `MixtureDistribution[T]` might inherit from `RandomMeasure[T]` or be a
  related construct.

## Test plan

The receiving session should accompany the PR with at least:

1. **Construction + protocol declaration.** A simple synthetic
   `RandomMeasure` subclass (e.g. wrapping a `DistributionArray[Normal]`
   into a uniform-mixture-over-normals random measure). Verify `_sample`
   returns a `Distribution[Array]`, `_mean` (when implemented) returns the
   marginalized mixture, etc.
2. **Inner support / event shape.** Construct random measures with
   different inner supports and confirm the chosen `inner_support` /
   `inner_event_shape` machinery reflects them.
3. **Optional protocol opt-in.** Confirm a `RandomMeasure` that doesn't
   implement `_random_log_prob` reports the absence cleanly via the
   existing `Supports*` protocol-check mechanism.
4. **Forward compatibility smoke tests.** No need to land `condition_on`
   here, but verify there are no obvious type / dispatch obstacles.

If `RandomFunction` is in scope: tests for `_random_log_prob` returning a
valid `RandomFunction` whose marginals are `Distribution[Array]`s.

## Coordination

Branch off `main`. No dependency on other open PRs (PRs #145 and #146
merged 2026-04-27).

After this PR lands, a sabi `SurrogatePosterior` v1.2 implementation will
follow — a `RandomMeasure[Array]` subclass with two concrete subtypes
(Dirac at an empirical measure, GP-pushforward). **Sabi work shouldn't
drive this PR's scope.** If there are sabi-specific shape needs that
don't fit `RandomMeasure` cleanly, surface them as separate ProbPipe
issues rather than bending `RandomMeasure` to fit sabi.
