# Tempering in sabi

This document explains how tempering works in sabi. We start from the
pure mathematical setup (no emulator), then introduce the emulator
layer, then walk through the common cases that exercise different
parts of the machinery.

The notation here mirrors the in-code naming: `target_function` /
`f`, `log_density_form` / `phi`, `tempering_state` / `state` (often
just `t` indexing the round, or `beta` / `lambda` for likelihood
tempering). For shape conventions, see [`notation.md`](notation.md).

## 1. Conceptual layering

```
Problem
  └── target_distribution: TargetDistribution     ← pure math
        ├── target_function (batched f)
        ├── target_single (single-point f)
        ├── log_density_form (phi)
        ├── prior, support
        └── (Distribution interface: _unnormalized_log_prob = phi(x, f(x), prior))

TemperingScheme    ← family of intermediate targets indexed by state
  └── intermediate_target(base, state) → IntermediateTarget
                                            ├── (TargetDistribution fields)
                                            ├── state
                                            ├── output_transform(state, X, Y_raw) → Y_train
                                            └── base_target_function (un-tempered f)

TemperingSchedule  ← state sequence: schedule.next(round_idx) → (state, final)

Emulator           ← predictive model fit to (X, Y_train)
SurrogatePosterior ← Emulator + log_density_form (the SP at a particular state)

Algorithm          ← composition: emulator_factory + tempering_scheme +
                     schedule + acquisition + acquisition_target
```

Three orthogonal pieces of machinery interact:

1. **`TemperingSchedule`** produces *states*. A state is an opaque
   PyTree (a scalar `beta` for likelihood tempering, a subset index
   for data tempering, anything for exotic schemes).
2. **`TemperingScheme`** consumes states to produce
   `IntermediateTarget`s. Each intermediate is a fully-resolved target
   distribution at a particular state — a `TargetDistribution`
   subclass with state and `output_transform` metadata.
3. **`AcquisitionTarget`** picks *which* state the acquisition's
   `SurrogatePosterior` is built at: `CURRENT` (the round's state),
   `NEXT` (the next round's state), or `TERMINAL` (the schedule's
   final state).

The `Emulator` is tempering-agnostic — it just fits to whatever
`(X, Y_train)` the loop hands it. The state-dependence enters via
how `Y_train` is derived (the scheme's `output_transform`).

## 2. The two axes

A `TemperingScheme` can vary the intermediate distribution along two
axes independently:

| Axis | What varies with state | Mechanism |
|--|--|--|
| **Target** | The emulator's training target `f_state` | `output_transform(state, X, Y_raw)` is non-trivial |
| **Form** | The log-density form `phi_state` | `intermediate.log_density_form` differs per state |

Concrete schemes typically pick one axis:

- `NoTempering`: identity on both.
- `LikelihoodTemperingViaForm`: form axis only — emulator is
  state-invariant.
- `LikelihoodTemperingViaTarget`: target axis only — emulator refits
  per state on rescaled `Y`.

Combining both is mathematically definable but rarely useful; sabi
doesn't ship a combined scheme.

The scheme's `is_invariant_target_function(state_a, state_b)` and
`is_invariant_form(state_a, state_b)` flags advertise which axis is
state-invariant. The loop uses these to skip redundant work.

## 3. Cases

In all the examples below, `f` is the (un-tempered) target function,
`phi` is the log-density form, and `pi_0` is the prior. The
intermediate distribution at round `t` is

`log p_t(x) = phi_t(x, f_t(x), pi_0)`

with `phi_t` and `f_t` determined by the chosen `TemperingScheme`
and the round's state.

### Case 1: No tempering

The simplest case: nothing depends on `t`.

- **Intermediate**: `log p(x) = phi(x, f(x), pi_0)` (unchanged across
  rounds).
- **Scheme**: `NoTempering` (default).
- **Schedule**: `UntemperedSchedule` (always returns `(None, True)`).

```python
algorithm = Algorithm(
    emulator_factory=lambda: GPEmulator(input_shape=(2,)),
    acquisition=ExpectedImprovement(...),
    # tempering_scheme defaults to NoTempering()
    # schedule defaults to UntemperedSchedule()
    # acquisition_target defaults to AcquisitionTarget.CURRENT
)
```

Loop behavior: the emulator is fit on `(X, Y_raw)` (since
`output_transform` is identity); the SP is built with the base form;
acquisition optimizes against the round's state, which equals the
terminal state (= None). One emulator fit per round; one form per
round.

### Case 2: Likelihood tempering with tempered target map

**Setting**: emulator approximates the log-likelihood directly
(`f = log L`, `phi = LogLikPlusPrior`). Tempering scales the
emulator's training target by `beta_t`.

- **Intermediate**: `log p_t(x) = log pi_0(x) + beta_t * log L(x)`.
- **f_state**: `f_t = beta_t * log L`. Emulator refits per round on
  `Y_train_t = beta_t * Y_raw`.
- **phi_state**: unchanged (`LogLikPlusPrior`).
- **Scheme**: `LikelihoodTemperingViaTarget`.

```python
algorithm = Algorithm(
    emulator_factory=lambda: GPEmulator(input_shape=(2,)),
    acquisition=ExpectedImprovement(...),
    tempering_scheme=LikelihoodTemperingViaTarget(),
    schedule=FixedSchedule(states=(0.1, 0.5, 1.0)),
    acquisition_target=AcquisitionTarget.NEXT,  # SMC-style look-ahead
)
```

Loop behavior at round `t`:

1. `current_state = beta_t`, `target_state = beta_{t+1}` (under
   `NEXT`).
2. Loop builds the look-ahead intermediate at `beta_{t+1}` and
   computes `Y_train_acq = beta_{t+1} * Y_raw`.
3. Refits emulator on `(X, Y_train_acq)`. (Issue #4 will add a
   cheap-update dispatch that avoids the full refit when only `Y` is
   rescaled.)
4. Builds SP for the acquisition at `(emulator_acq, phi)`.
5. Acquisition picks `x_new`; loop appends `y_new_raw = f(x_new)` to
   `Y_raw`.
6. Round-end: refits emulator at the current state `beta_t` on the
   augmented design (`Y_train_t = beta_t * Y_raw`) for metrics.

`is_invariant_target_function(beta_a, beta_b)` returns True iff
`beta_a == beta_b`; the look-ahead refit happens iff the scheme says
the target is non-invariant.

### Case 3: Likelihood tempering with constant target map

**Setting**: same intermediate distribution as Case 2, but the
emulator's target is invariant. The `beta` factor lives in the form.

- **Intermediate**: `log p_t(x) = log pi_0(x) + beta_t * log L(x)`.
- **f_state**: `f_t = f` (unchanged, e.g. `f = log L` for
  log-likelihood emulation, or `f = g(x)` for forward-model
  emulation).
- **phi_state**: `phi_t(x, y) = beta_t * <likelihood term> + log pi_0(x)`.
  Per-form-type math:
  - `LogLikPlusPrior(y) = y + log pi_0(x)` becomes
    `phi_t = beta_t * y + log pi_0(x)`.
  - `ForwardModel(y) = log L(x, y) + log pi_0(x)` becomes
    `phi_t = beta_t * log L(x, y) + log pi_0(x)`.
  - `Identity(y) = y` becomes the geometric bridge
    `phi_t = (1 - beta_t) * log pi_0(x) + beta_t * y`.
- **Scheme**: `LikelihoodTemperingViaForm`.

```python
algorithm = Algorithm(
    emulator_factory=lambda: GPEmulator(input_shape=(2,)),
    acquisition=ExpectedImprovement(...),
    tempering_scheme=LikelihoodTemperingViaForm(),
    schedule=FixedSchedule(states=(0.1, 0.5, 1.0)),
    acquisition_target=AcquisitionTarget.NEXT,
)
```

Loop behavior at round `t`:

1. `current_state = beta_t`, `target_state = beta_{t+1}`.
2. `is_invariant_target_function(beta_t, beta_{t+1})` returns True
   for this scheme — the emulator is the *same* across all states.
3. The acquisition's SP uses the same emulator as the round-end SP,
   but with a different form (`phi_{t+1}` vs `phi_t`).
4. No emulator refit between current and look-ahead — only the form
   changes.

This is the cheap case: one emulator fit per round (just for the new
data after the acquisition), and the form is rebuilt per state but
that's `O(1)`.

### When to pick Case 2 vs Case 3

Both encode the same intermediate distribution family. The choice is
about emulator fit characteristics:

- **Case 2** (`LikelihoodTemperingViaTarget`): the emulator sees data
  rescaled by `beta`. Hyperparameter optimization happens on the
  rescaled scale. Useful when the un-tempered scale of `Y_raw` is
  poorly conditioned for the GP and per-state rescaling helps.
- **Case 3** (`LikelihoodTemperingViaForm`): the emulator sees the
  un-tempered `Y_raw`. The `beta` factor is applied at SP-construction
  time. Useful when you want to amortize the GP fit across all states
  (no per-round refit cost) — natural when the target function is
  expensive to evaluate but `beta` changes are cheap.

The verification test
[`test_via_form_and_via_target_are_numerically_equivalent`](../tests/test_likelihood_tempering.py)
confirms both schemes produce identical intermediate log-densities at
every `(x, beta)` for a `LogLikPlusPrior` base — the only difference
is *where* the `beta` factor enters the computation.

### Forward-model emulation under tempering

For forward-model emulation, `f = g(x)` (the forward model output)
and `phi = ForwardModel(log_lik_from_outputs)`. Likelihood tempering
naturally falls into Case 3: scaling `f` by `beta` would scale the
forward-model output, which doesn't correspond to likelihood
tempering. Use `LikelihoodTemperingViaForm`; it dispatches on
`ForwardModel` and produces `phi_t = beta_t * log L(x, y) + log pi_0(x)`.

`LikelihoodTemperingViaTarget` raises if applied to a `ForwardModel`
base form — there's no sensible interpretation.

## 4. Acquisition target

`AcquisitionTarget` selects the state the acquisition optimizes
against:

- `CURRENT`: state of the current round (`state_t`). Default; simplest.
- `NEXT`: next round's state (`state_{t+1}`, clamped to terminal at
  the last round). Standard SMC-flavor look-ahead — acquisition picks
  points to inform the *next* intermediate.
- `TERMINAL`: schedule's terminal state. Acquisition optimizes
  toward the final target throughout, regardless of where the
  schedule is.

For untempered loops, all three collapse to the same state and the
choice has no effect.

For `NEXT` and `TERMINAL`, the loop builds a separate "look-ahead"
intermediate target at the resolved state and feeds it to the
acquisition's SP. When the scheme says the target axis is non-
invariant under the state change, the emulator is refit at the
look-ahead state (Step 5 / issue #4 will add a cheap-update dispatch
that avoids the full refit).

Richer policies (ESS-adaptive look-ahead, custom callable that
depends on loop state) are tracked as
[issue #5](https://github.com/arob5/sabi/issues/5).

## 5. Schedule + scheme compatibility

The schedule produces states; the scheme consumes them. They must
agree on the state PyTree type:

- `LikelihoodTemperingViaForm` / `LikelihoodTemperingViaTarget` expect
  a scalar `beta in [0, 1]`. Pair with `FixedSchedule(states=(0.1,
  0.5, 1.0))` or similar.
- `NoTempering` accepts any state (it ignores the value). Pair with
  `UntemperedSchedule` (state = None) or any schedule for ablation.
- A future `DataTempering` would expect subset indices; pair with a
  schedule that produces those.

There's no static check; the user / Hydra config is responsible for
the pairing. If they disagree, you'll see the mismatch at runtime
when the scheme tries to consume the state.

## 6. Summary table

| Setting | `tempering_scheme` | `schedule` | Notes |
|--|--|--|--|
| Untempered | `NoTempering()` | `UntemperedSchedule()` | Default. |
| Likelihood tempering, form-based | `LikelihoodTemperingViaForm()` | `FixedSchedule((b1, ..., 1.0))` | Emulator state-invariant. Compatible with `LogLikPlusPrior`, `ForwardModel`, `Identity`. |
| Likelihood tempering, target-based | `LikelihoodTemperingViaTarget()` | `FixedSchedule((b1, ..., 1.0))` | Emulator refits per state. Restricted to `LogLikPlusPrior` base. |

For all three, `acquisition_target` selects how look-ahead works:
default `CURRENT` (no look-ahead), `NEXT` (one-step look-ahead),
`TERMINAL` (always optimize toward final target).

## 7. References

- Source: [`sabi/tempering/`](../src/sabi/tempering/) (`base.py`,
  `likelihood.py`, `schedule.py`).
- Per-state intermediate: [`IntermediateTarget`](../src/sabi/problems/target_distribution.py).
- Loop integration: [`sabi/algorithms/loop.py`](../src/sabi/algorithms/loop.py).
- Tests: [`tests/test_tempering.py`](../tests/test_tempering.py),
  [`tests/test_likelihood_tempering.py`](../tests/test_likelihood_tempering.py),
  [`tests/test_acquisition_target.py`](../tests/test_acquisition_target.py).
- Open issues:
  - [#4](https://github.com/arob5/sabi/issues/4) — `update_emulator`
    cheap-path dispatch (avoid full refits in Case 2).
  - [#5](https://github.com/arob5/sabi/issues/5) — Adaptive
    `AcquisitionTarget` policies (ESS-adaptive look-ahead).
