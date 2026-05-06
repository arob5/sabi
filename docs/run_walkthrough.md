# Anatomy of a run

This page traces one execution of `run(problem, algorithm, key)` from
[`src/sabi/algorithms/loop.py`](../src/sabi/algorithms/loop.py). The
goal is to build the mental model — read this once, then the file
itself reads like the algorithm.

The code blocks here are *pseudocode*: they mirror the actual helper
shape so you can map them line by line, but they're not copy-pasted
(the real loop has more bookkeeping). The `loop.py:NNN` links go to
the source.

Prerequisites: you've read [`overview`](overview.md) (vocabulary) and
skimmed [`notation`](notation.md) (shape conventions). The running
example is `banana_2d` from
[Getting Started](getting_started.ipynb).

## The shape of the loop

Skipping over the bookkeeping, [`loop.py:117`](../src/sabi/algorithms/loop.py)
is structured like this:

```python
def run(problem, algorithm, key):
    target = problem.target_distribution

    # Round 0 — initial design, no acquisition.
    X, Y_raw, Y_train, emulator, emulator_state, round_0 = (
        _run_initial_round(problem, algorithm, ..., key_init, key_metric_0)
    )
    tempering_states = [emulator_state]
    per_round_metrics = [round_0]

    # Acquisition rounds — 1..n_rounds-1.
    for round_idx in range(1, algorithm.n_rounds):
        round_state = _resolve_round_state(algorithm, target, round_idx)
        emulator_for_acq, Y_train_for_acq = _resolve_acquisition_view(...)
        x_new, y_new_raw = _run_acquisition(...)
        X, Y_raw, Y_train, y_new_at_current = _append_round_evaluations(...)
        emulator, emulator_state = _update_round_end_emulator(...)
        per_round_metrics.append(_build_round_metrics_row(...))
        tempering_states.append(round_state.current_intermediate.state)

    # Post-loop — final eval at the un-tempered base.
    final_estimate, final_metrics = _run_final_eval(...)

    return RunResult(X, Y_raw, Y_train, emulator,
                     tempering_states, per_round_metrics,
                     final_estimate, final_metrics)
```

Each helper owns one phase of the round. The body of `run` is the
algorithm; the helpers carry every conditional that doesn't belong in
the algorithm itself (the
[contributing guide](contributing.md#main-loops-read-like-pseudocode)
is explicit about this).

The rest of this page expands each helper.

## 1. Initial round

[`_run_initial_round` — loop.py:291](../src/sabi/algorithms/loop.py)

Round 0 has no acquisition. The initial sampler draws a design, the
target is evaluated, the emulator is fit. Conceptually:

```python
def _run_initial_round(problem, algorithm, ...):
    X = algorithm.initial_sampler.sample(problem, key_init, algorithm.n_initial)
    Y_raw = problem.target_map(X)

    state_0, _ = algorithm.schedule.at(0)
    target_0 = algorithm.tempering_scheme.intermediate_target(target, state_0)
    Y_train = target_0.output_transform(state_0, X, Y_raw)

    emulator = algorithm.emulator_factory()
    emulator = emulator.fit(X, Y_train)

    round_0_metrics = _eval_round(... firing scheduled metrics at round 0 ...)
    return X, Y_raw, Y_train, emulator, state_0, round_0_metrics
```

After this returns, the shapes are:

```
X.shape       == (n_initial,) + problem.input_shape       # (8, 2) for banana_2d
Y_raw.shape   == (n_initial,) + problem.output_shape      # (8,)
Y_train.shape == (n_initial,) + problem.output_shape      # (8,) — equal to Y_raw under no tempering
```

(`n_initial`, `input_shape`, etc. — see [`notation`](notation.md).)

`Y_raw` is the *un-transformed* base target evaluated at `X`. It's
cached for the whole run; every subsequent round derives `Y_train`
from `Y_raw` via the round's `output_transform`. Under
`NoTempering` (the Getting Started default), `output_transform` is
identity and `Y_train == Y_raw`.

## 2. The per-acquisition-round body

The rest of the loop is rounds `1..n_rounds-1`. Each round runs five
phases in order. Every phase is one named helper.

### 2a. Resolve the round's state

[`_resolve_round_state` — loop.py:414](../src/sabi/algorithms/loop.py)

Two states matter per round: the round's **current** tempering state
(the one the round-end emulator will be fit at), and the
acquisition's **target** state (the one the acquisition optimizes
against). They differ when `algorithm.acquisition_target` is `NEXT`
or `TERMINAL`. Most algorithms use `CURRENT`, in which case the two
states are equal.

```python
def _resolve_round_state(algorithm, target, round_idx):
    current_state, _ = algorithm.schedule.at(round_idx)
    target_state = resolve_state(
        algorithm.acquisition_target, algorithm.schedule,
        round_idx, current_state,
    )
    current_intermediate = algorithm.tempering_scheme.intermediate_target(
        target, current_state
    )
    invariance = algorithm.tempering_scheme.invariance(current_state, target_state)
    target_intermediate = (
        current_intermediate if invariance.both
        else algorithm.tempering_scheme.intermediate_target(target, target_state)
    )
    return RoundState(round_idx, current_intermediate,
                      target_intermediate, invariance)
```

`InvarianceFlags` records *which axis of tempering changes* between
the two states. The downstream helpers branch on these flags to skip
work when nothing has changed.

### 2b. Resolve the acquisition's view

[`_resolve_acquisition_view` — loop.py:454](../src/sabi/algorithms/loop.py)

The acquisition needs a `(emulator, Y_train)` pair *at the
acquisition's target state*. Two cases:

- **Same state, no work.** If the target axis is invariant, the
  round-end emulator and `Y_train` already live at the right state.
  Reuse them.
- **Different state, cheap update.** Otherwise, materialize
  `Y_train_for_acq` via the target intermediate's `output_transform`,
  then dispatch a state-only `EmulatorUpdate` plan through
  [`update_emulator`](../src/sabi/emulators/dispatch.py). The
  dispatcher tries any registered fast paths and falls back to a full
  refit.

```python
def _resolve_acquisition_view(*, emulator, emulator_state,
                              round_state, X, Y_raw, Y_train, algorithm):
    if round_state.invariance.target_map:
        return emulator, Y_train

    Y_train_for_acq = round_state.target_intermediate.output_transform(
        round_state.target_intermediate.state, X, Y_raw
    )
    plan = _plan_round_update(
        round_state.target_intermediate.output_transform,
        state_prev=emulator_state,
        state_new=round_state.target_intermediate.state,
        X_new=None, Y_new_at_new_state=None,   # state-only, no new rows
    )
    emulator_for_acq = update_emulator(
        emulator, plan, factory=algorithm.emulator_factory,
        X_full=X, Y_full=Y_train_for_acq,
    )
    return emulator_for_acq, Y_train_for_acq
```

### 2c. Run the acquisition

[`_run_acquisition` — loop.py:500](../src/sabi/algorithms/loop.py)

Build the pre-round `SurrogateDistribution` from the acquisition's
emulator + the target intermediate's log-density form, hand the
acquisition an `AcquisitionState`, let it pick the next batch,
evaluate the target on those points.

```python
def _run_acquisition(*, problem, algorithm, emulator_for_acq,
                    X, Y_raw, Y_train_for_acq, round_state, key):
    pre_round_posterior = _build_surrogate_distribution(
        algorithm.surrogate_distribution_factory,
        emulator=emulator_for_acq, X=X, Y=Y_train_for_acq,
        log_density_form=round_state.target_intermediate.log_density_form,
        problem=problem,
    )
    acq_state = AcquisitionState(
        problem=problem,
        surrogate_distribution=pre_round_posterior,
        X=X, Y_raw=Y_raw, Y_train=Y_train_for_acq,
    )
    x_new = algorithm.acquisition.select_batch(acq_state, algorithm.q, key)
    y_new_raw = problem.target_map(x_new)
    return x_new, y_new_raw
```

`y_new_raw` is at the **un-transformed base target** — the round-end
helper will transform it to `current_state` before the emulator
refit. Caching the raw form makes future state changes
re-derivable.

### 2d. Append the new evaluations

[`_append_round_evaluations` — loop.py:546](../src/sabi/algorithms/loop.py)

Concatenate, then re-derive `Y_train` at the round's `current_state`
from the now-extended `Y_raw`. The new-rows-only block
`y_new_at_current` is computed alongside so the round-end fast path
can append rather than recomputing the whole transformed dataset.

```python
def _append_round_evaluations(X, Y_raw, x_new, y_new_raw, round_state):
    X = jnp.concatenate([X, x_new], axis=0)
    Y_raw = jnp.concatenate([Y_raw, y_new_raw], axis=0)

    cur = round_state.current_intermediate
    Y_train = cur.output_transform(cur.state, X, Y_raw)
    y_new_at_current = cur.output_transform(cur.state, x_new, y_new_raw)

    return X, Y_raw, Y_train, y_new_at_current
```

### 2e. Update the round-end emulator

[`_update_round_end_emulator` — loop.py:578](../src/sabi/algorithms/loop.py)

One `EmulatorUpdate` plan combines the *transform diff*
(`emulator_state → current_state`) with the *new-rows append*. The
dispatcher tries fast paths (e.g., rescaling outputs, appending rows
under fixed hyperparameters) and falls back to a full refit when no
fast path is registered.

```python
def _update_round_end_emulator(*, emulator, emulator_state,
                               round_state, X, Y_train,
                               x_new, y_new_at_current, algorithm):
    cur = round_state.current_intermediate
    plan = _plan_round_update(
        cur.output_transform,
        state_prev=emulator_state, state_new=cur.state,
        X_new=x_new, Y_new_at_new_state=y_new_at_current,
    )
    emulator = update_emulator(
        emulator, plan,
        factory=algorithm.emulator_factory,
        X_full=X, Y_full=Y_train,
    )
    return emulator, cur.state
```

After this returns, the emulator's fit state equals the round's
`current_state`. The next round's `_resolve_acquisition_view` uses
this as the "from" state when planning the next state-only diff.

### Round-end metrics

[`_build_round_metrics_row` — loop.py:618](../src/sabi/algorithms/loop.py)

Any `ScheduledMetric` that fires this round runs against the
post-update emulator. Construction of `SurrogateDistribution` and the
posterior estimate is **lazy** — when no metric fires this round,
nothing is built. The row gets bookkeeping fields (`round`,
`tempering_state`, `target_tempering_state`, `n_evals`) attached
unconditionally.

## 3. Final evaluation

[`_run_final_eval` — loop.py:356](../src/sabi/algorithms/loop.py)

After the loop exits, sabi always builds a final `SurrogateDistribution`
+ posterior estimate at the **un-tempered base** form — independent
of the round's last tempering state. This is what
`RunResult.final_estimate` carries, and it's how every algorithm's
output gets normalized to the same target distribution for fair
comparison.

```python
def _run_final_eval(*, problem, emulator, X, Y_raw, Y_train,
                   target, algorithm, scheduled, last_state, key):
    final_sp = _build_surrogate_distribution(
        algorithm.surrogate_distribution_factory,
        emulator=emulator, X=X, Y=Y_train,
        log_density_form=target.log_density_form,
        problem=problem,
    )
    final_estimate = algorithm.estimator(final_sp)

    final_scheduled = tuple(s for s in scheduled if s.final)
    if not final_scheduled:
        return final_estimate, {}

    final_metrics = _run_metrics_against_pair(
        metrics=final_scheduled, ...,
        metric_target=MetricTarget.TERMINAL,
        ...,
    )
    return final_estimate, final_metrics
```

`final_metrics` only contains values for metrics whose `final=True`
flag is set; everything else is in `per_round_metrics`. Final metrics
always evaluate against the un-tempered base, regardless of their
per-round `target` field.

## 4. What `RunResult` carries home

[`RunResult` — algorithm.py:54](../src/sabi/algorithms/algorithm.py)

| field | shape / type | what it is |
|---|---|---|
| `X` | `(n_initial + q*n_rounds,) + input_shape` | full design across all rounds |
| `Y_raw` | `(n_initial + q*n_rounds,) + output_shape` | un-transformed base-target evaluations on `X`. The cache that lets every state's `Y_train` be re-derived. |
| `Y_train` | `(n_initial + q*n_rounds,) + output_shape` | `Y_raw` transformed to the *terminal* round's state. What the final emulator was fit against. Equal to `Y_raw` under `NoTempering`. |
| `emulator` | `Emulator` | the emulator at the terminal state — the one a downstream user would call `predict` on |
| `tempering_states` | `list[Any]`, length `n_rounds` | one state per round (round 0 first) |
| `per_round_metrics` | `list[dict]`, length `n_rounds` | bookkeeping (`round`, `tempering_state`, `target_tempering_state`, `n_evals`) plus values from any metric that fired |
| `final_estimate` | `Distribution \| None` | the un-tempered base posterior estimate; built unconditionally so downstream tooling always has a target to score |
| `final_metrics` | `dict[str, float]` | values from metrics with `final=True` |

Metrics consume `final_estimate` (deterministic estimators that score
against the reference) and `surrogate_distribution` (when the metric
needs the surrogate's stochasticity, e.g., MMD on samples). Per-round
metrics get a freshly-built SP each round; the final SP is reused for
every `final=True` metric.

## Where to next

- [Getting Started](getting_started.ipynb) — the four-panel run that
  produced these structures end-to-end.
- [`design`](design.md) — the long-form design record. §4 ("Core
  abstractions") and §4.14 (the loop sketch) are this page's source
  of truth.
- [`tempering`](tempering.md) — when `output_transform` and `state`
  do non-trivial work. The Getting Started run uses `NoTempering`,
  so most of the round-state machinery here is no-op.
- [`scheduled_metrics`](scheduled_metrics.md) — how
  `_build_round_metrics_row` decides which metrics fire when.
