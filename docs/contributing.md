# Contributing to sabi

Conventions for code and documentation. Scope is intentionally narrow:
the things that come up often enough that having a written rule beats
re-arguing them in review.

For shape and symbol conventions (`x`, `y`, `X`, `Y`, `n`, `d`, `p`, `q`,
`f`), see [`notation.md`](notation.md). All sabi math docstrings use
those symbols without redefining them.

For the tempering layer (`TemperingScheme`, `IntermediateTarget`,
`AcquisitionTarget`) — the conceptual layering, the two-axes
decomposition, and worked case examples — see
[`tempering.md`](tempering.md).

## Documentation

### Math goes in docstrings

When a function, class, or method implements a non-trivial mathematical
formula, the formula belongs in the docstring. Code-as-the-only-source-
of-truth makes review harder than it needs to be — and "trivial" usually
isn't, once a reader who didn't write the code is trying to understand it.

Use `r"""..."""` raw strings and reST math directives so Sphinx can render
them. Two flavors:

- Inline: `:math:`\\mu(x)``  (renders as $\mu(x)$).
- Block:

  ```
  .. math::

      \mathrm{EI}(x) = (\mu(x) - f^*)\, \Phi(z) + \sigma(x)\, \phi(z),
      \quad z = \frac{\mu(x) - f^*}{\sigma(x)}.
  ```

Reference [`notation.md`](notation.md) for the meaning of standard
symbols rather than redefining them per-file.

### Cross-reference, don't duplicate

If a class's behavior depends on a contract documented elsewhere
(e.g. `PointwiseScoredAcquisition`'s scoring contract), say so and
point to it rather than restating. Docstrings should add what's
specific; the shared contract lives in one place.

### Update the spine and concepts when behavior changes

PRs that modify or introduce user-facing functionality must update the
corresponding **spine** page(s) and concepts page(s). The spine is the
curated read path for new users:

- [`getting_started`](getting_started.ipynb) — the end-to-end run.
- [`overview`](overview.md) — the five abstractions and how they
  compose.
- [`notation`](notation.md) — shape, symbol, and naming conventions.
- [`run_walkthrough`](run_walkthrough.md) — what `run()` does, helper
  by helper.

"User-facing" means anything visible on the docs site: a public class
or function signature, a flag a user might set in a Hydra config, a
default that affects behavior, or a shape contract. The auto-generated
API reference picks up docstring changes for free — but spine prose,
example notebooks, and concepts pages do not. If the change makes
existing prose wrong or stale, fix it in the same PR.

If the change is genuinely spine-irrelevant (purely internal refactor,
perf-only change, test-only edit), say so in the PR description so
reviewers don't have to guess.

### Notebook outputs are cached

Tutorial notebooks under `docs/` ship with their cell outputs already
populated. CI builds the docs with `nb_execution_mode = "off"` (see
`docs/conf.py`), so the rendered site reflects whatever outputs the
notebook was last committed with — Sphinx does not re-execute. This
is a temporary trade-off: sabi tracks in-flight ProbPipe APIs that
are not always present on the public `TARPS-group/prob-pipe` `main`,
so a fresh CI clone cannot reliably import sabi yet. Once ProbPipe
stabilizes (post-overhaul), `nb_execution_mode` will flip back to
`"force"` and CI will catch staleness automatically.

Until then, **authors who edit a notebook's code cells, or who change
sabi behavior that any notebook exercises, must re-execute the
affected notebooks locally before committing**:

```bash
./scripts/python -m jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  docs/getting_started.ipynb
```

Then `git add` the notebook with its refreshed outputs as part of the
same PR. Reviewers should treat absent / stale outputs the same as
broken docs.

## Coding conventions

### Modern Python type hints

sabi targets Python 3.12+. Use the modern hint conventions:

- Import `Callable`, `Iterable`, `Iterator`, `Mapping`, `Sequence`, etc.
  from `collections.abc`, **not** from `typing`.
- Use `X | Y` for unions, **not** `Union[X, Y]` or `Optional[X]`.
- Use built-in generics: `list[int]`, `dict[str, Array]`, `tuple[int, ...]`,
  not `List`, `Dict`, `Tuple` from `typing`.
- Don't quote type hints unless you have to (forward references in a
  cycle, or to keep an annotation cheap to import). `from __future__
  import annotations` at the top of every module makes most quoting
  unnecessary.

```python
# good
from collections.abc import Callable
from jax import Array

def run(fn: Callable[[Array], Array], items: list[Array]) -> dict[str, float]:
    ...

# avoid
from typing import Callable, Dict, List, Optional, Union

def run(fn: "Callable[[Array], Array]", items: List[Array]) -> Dict[str, float]:
    ...
```

### JAX-traceability

Code that runs inside a `jax.vmap`, `jax.jit`, or `jax.grad` boundary
must be traceable. Concretely:

- No Python `if` on traced values; use `jnp.where`, `jax.lax.cond`,
  or `jax.lax.select` instead.
- No Python `for` over traced shapes; use `jax.vmap` /
  `jax.lax.scan` / `jax.lax.fori_loop`.
- Pure functions: no module-level mutable state read inside the trace.

When a function isn't expected to be traced (e.g., one-shot construction
of a `Problem`), Python control flow is fine — but call out the
expectation in the docstring if the boundary isn't obvious.

`PointwiseScoredAcquisition._score_single` is the canonical example
of a function that **must** be traceable: `ContinuousMultiStartOptimizer`
takes its gradient.

### ProbPipe imports

Prefer top-level imports from `probpipe`:

```python
# good
from probpipe import sample, log_prob, mean, variance
from probpipe.core.constraints import Constraint
from probpipe.core._distribution_base import Distribution
```

The top-level `probpipe` namespace re-exports the stable surface (the
ops). Reach into `probpipe.core.*` only for types that don't have a
top-level alias (`Distribution`, `Constraint`, `RandomFunction`,
`NumericRandomMeasure`, etc.). Don't reach into private modules
(`probpipe.core._distribution_base` is a fact of life today; future
refactors will move things and we update with them).

### Dataclasses for value objects

Component classes that are configuration bundles (`Algorithm`,
`AcquisitionState`, `ExpectedImprovement`, `CandidateSetOptimizer`,
…) are `@dataclass(frozen=True)`. Frozen because they're often passed
to JAX (which expects no mutation), and `frozen=True` makes hashability
work for caching keys.

For ABCs whose subclasses are dataclasses (e.g. `PointwiseOptimizer`,
`BatchSampler`), declare them as plain ABCs (`abc.ABC`) and let
subclasses opt into `@dataclass(frozen=True)`. Don't decorate the
ABC itself.

## Algorithmic invariants

### Main loops read like pseudocode

The main algorithm loops — currently `run()` in
[`src/sabi/algorithms/loop.py`](../src/sabi/algorithms/loop.py), and any
future top-level loop bodies that play the same role — must read like a
paper-style pseudocode description of the algorithm. An informed
reader should be able to scan the loop body once and recover the
algorithmic skeleton: state resolution → acquisition view →
acquisition → target evaluation → round-end update → metrics →
bookkeeping. Each step should be one named operation, ideally one line.

Concretely, the per-round body should *not* expose:

- Branching on tempering invariance, `output_transform` plumbing, or
  cheap-update vs. refit dispatch — push these into named helpers.
- Manual PRNG-key splitting interleaved with algorithmic steps —
  resolve keys at the top of the round or inside the helper that
  consumes them.
- Inline construction of `SurrogateDistribution` /
  `AcquisitionState` / `MetricContext` payloads — these are
  bookkeeping, not algorithm.
- Per-axis or per-target conditional reuse logic (e.g.
  "if `invariance.both` then reuse the current intermediate") — wrap
  in a helper whose name describes the *what*, not the *how*.

The helpers carrying that complexity should have docstrings that
explain the conditional logic and trade-offs. The loop body itself
explains the algorithm.

When reviewing a PR that touches `run()` or a similar loop, ask:
"could a reader who knows the math but not this codebase trace the
algorithm by reading only the loop body?" If the answer is no, the
change needs to push complexity into helpers before it lands.

## Tests

- One test file per source module where practical (`tests/test_loop.py`
  for `algorithms/loop.py`, `tests/test_optim.py` for
  `acquisitions/optim.py`, etc.).
- Prefer `scripts/python -m pytest` over the bare `pytest` command —
  the wrapper threads `PYTHONPATH` for the worktree + ProbPipe pin.
  See `.claude/worktree_probpipe.md` for why.
- Numerical assertions: pick tolerances that survive seed-dependent
  variance with the configured `n_initial` / `n_rounds` / sample
  budgets. A 5% slack on a top-level metric is usually right; tighter
  if you're testing exactness, looser only with a written reason.
