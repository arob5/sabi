# Contributing to sabi

Conventions for code and documentation. Scope is intentionally narrow:
the things that come up often enough that having a written rule beats
re-arguing them in review.

For shape and symbol conventions (`x`, `y`, `X`, `Y`, `n`, `d`, `p`, `q`,
`f`), see [`notation.md`](notation.md). All sabi math docstrings use
those symbols without redefining them.

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

## Tests

- One test file per source module where practical (`tests/test_loop.py`
  for `algorithms/loop.py`, `tests/test_optim.py` for
  `acquisitions/optim.py`, etc.).
- Prefer `scripts/python -m pytest` over the bare `pytest` command —
  the wrapper threads `PYTHONPATH` for the worktree + ProbPipe pin.
  See [`.claude/worktree_probpipe.md`](../.claude/worktree_probpipe.md)
  for why.
- Numerical assertions: pick tolerances that survive seed-dependent
  variance with the configured `n_initial` / `n_rounds` / sample
  budgets. A 5% slack on a top-level metric is usually right; tighter
  if you're testing exactness, looser only with a written reason.
