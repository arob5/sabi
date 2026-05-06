# Sabi at a glance

Sabi is a **test framework for sequential, surrogate-based Bayesian
inference algorithms** — not an inference library you'd run on your
own model. You bring an algorithm (an emulator + an acquisition rule +
a tempering schedule + …); sabi runs it on a benchmark with a known
reference posterior and tells you how well it did. If you're trying to
fit your own posterior, look elsewhere. If you want to compare or
ablate inference algorithms on ground-truth problems, you're in the
right place.

This page is the structural map. Read it once before drilling into
[`notation`](notation.md) (conventions) and
[`run_walkthrough`](run_walkthrough.md) (what happens when you call
`run`).

## The five load-bearing abstractions

Every component in sabi is one of five things. Each is swappable —
that's how ablations are expressed.

### `Problem`

What gets inferred. Carries the parameter space (`prior`), the target
map `f`, the unnormalized log-density form (likelihood +
prior, identity, forward model + noise, …), and a reference
distribution used to score the run. Canonical concrete:
`BenchmarkProblem`, returned by factories like `banana_2d()`. See
[`api/sabi/problems`](api/sabi/sabi.problems.md).

### `Emulator`

A predictive model fit to evaluations of the *target map* `f`. Takes
`(X, Y_train)`, returns a distribution over `f`. Canonical concrete:
`TinyGPEmulator` (zero-optimization GP, the Getting Started default);
`DSPGPEmulator` is the recommended choice when `d ≥ 5`. See
[`emulators`](emulators.md) and
[`api/sabi/emulators`](api/sabi/sabi.emulators.md).

### `Acquisition`

Picks the next `q` points to evaluate. Reads a
`SurrogateDistribution` plus the design history; returns the next
batch. Canonical concrete: `ExpectedImprovement`. The hierarchy
splits: `PointwiseScoredAcquisition` factors batch selection through
a per-point score so a multi-start optimizer can drive it. See
[`api/sabi/acquisitions`](api/sabi/sabi.acquisitions.md).

### `SurrogateDistribution`

A `RandomMeasure` over the *target distribution* — the
emulator-driven approximation that the acquisition consumes and that
metrics score. Canonical concrete: `EmulatedDistribution`
(emulator + log-density form pushed forward through a noise model).
A `WeightedEmpiricalRandomMeasure` baseline is provided for
emulator-free comparisons. See
[`api/sabi/surrogate`](api/sabi/sabi.surrogate.md).

### Posterior estimators and metrics

A **posterior estimator** is a deterministic projection from
`SurrogateDistribution` to a single `Distribution` — the loop's
"answer" for the round. Canonical concrete: `expected_target`
(plug-in: predictive mean of `f` into the log-density form).

A **metric** scores either the surrogate or the estimate against the
reference. Canonical concrete: `MMD` (kernel two-sample). Metrics are
scheduled — they don't have to fire every round. See
[`scheduled_metrics`](scheduled_metrics.md) and
[`api/sabi/metrics`](api/sabi/sabi.metrics.md).

## Emulator vs. surrogate

These two words look interchangeable. They aren't — and the
distinction is the most important conceptual move in sabi.

- **Emulator** — a predictive model over the *target map* `f` (a
  function over the parameter space). Tempering-agnostic: it just
  consumes `(X, Y_train)`.
- **Surrogate** — any approximate quantity replacing its exact
  analog. The `SurrogateDistribution` is the surrogate of the *target
  distribution*. The "surrogate posterior" of BO literature is the
  same concept. The emulator is *one piece* of how a
  `SurrogateDistribution` is constructed (emulator + form via
  pushforward), but the surrogate could in principle be built without
  an emulator at all (the `WeightedEmpiricalRandomMeasure` baseline
  does exactly this).

Mnemonic: **emulator → function (`f`); surrogate → distribution (`π`)**.

## `Algorithm` is the composition object

`Algorithm` is a frozen dataclass bundling every component the loop
needs to run end to end:

```python
algorithm = Algorithm(
    emulator_factory=lambda: TinyGPEmulator(input_shape=(2,)),
    acquisition=ExpectedImprovement(),
    n_initial=8,
    n_rounds=8,
    q=1,
    initial_sampler=PriorSampler(),     # default
    tempering_scheme=NoTempering(),     # default
    schedule=UntemperedSchedule(),      # default
    metrics=(MMD(...),),
)

result = run(problem, algorithm, key)   # → RunResult
```

The fields aren't a config — they're *components*. Ablating an
algorithm means swapping a field. A single `Problem` can be paired
with many `Algorithm`s; that's what sabi exists to make easy. The
loop body itself is in
[`src/sabi/algorithms/loop.py`](../src/sabi/algorithms/loop.py); the
[walkthrough](run_walkthrough.md) explains what it does.

## Tempering in one paragraph

A `TemperingScheme` builds a sequence of *intermediate* target
distributions that bridge a tractable starting point to the actual
target. The schedule says which intermediate each round operates
against. You need this when the target is hard enough that fitting an
emulator on it from cold start is infeasible — typical examples are
peaky likelihoods or high-dimensional posteriors. Default is
`NoTempering` + `UntemperedSchedule` — the loop just runs at the base
target every round. See [`tempering`](tempering.md) for the case
analysis (which scheme picks which axis, when to prefer each).

## How the parts fit together

```
                       ┌──────────────┐
                       │   Problem    │
                       │              │
                       │  prior       │
                       │  target_map  │
                       │  reference   │
                       └──────┬───────┘
                              │
   ┌──────────────────────────▼──────────────────────────┐
   │                       Algorithm                     │
   │                                                     │
   │   emulator_factory   ──→  Emulator                  │
   │   acquisition        ──→  Acquisition               │
   │   initial_sampler    ──→  X  (round 0)              │
   │   tempering_scheme   ──→  IntermediateTarget        │
   │   metrics            ──→  scored each round + final │
   └──────────────────────────┬──────────────────────────┘
                              │
                          run(...)
                              │
                              ▼
                          RunResult
                          (X, Y_raw, Y_train,
                           emulator, per_round_metrics,
                           final_estimate, final_metrics)
```

Inside a single round, the data flows:

```
state ──output_transform──> Y_train             (re-derived from cached Y_raw)
(X, Y_train) ──fit──> Emulator
Emulator ──pushforward──> SurrogateDistribution
SurrogateDistribution ──select_batch──> x_new
x_new ──target_map──> y_new_raw ──append──> Y_raw
```

The walkthrough page traces this round by round.

## Where to next

- [`notation`](notation.md) — shape conventions (`X`, `Y`, `n`, `d`,
  `p`, `q`), `x` vs `θ`, and the public-batched / private-single-point
  split that every method follows.
- [`run_walkthrough`](run_walkthrough.md) — what `run(problem,
  algorithm, key)` actually does, helper by helper, with file:line
  links into [`loop.py`](../src/sabi/algorithms/loop.py).
- [`design`](design.md) — the long-form design record (525 lines,
  contributor-oriented). Source of truth for the architectural
  decisions summarized above.
