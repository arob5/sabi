---
name: audit-tests
description: Audit the sabi test suite for correctness, coverage gaps, duplicates, misleading or "cheating" tests, and missing mathematical-correctness checks. Invoke as `/audit-tests` for the whole suite, or `/audit-tests <path-or-glob>` to scope to a subset (e.g. `/audit-tests tests/test_loop.py`, `/audit-tests metrics`, `/audit-tests acquisitions`). Reports findings without modifying tests.
---

# Audit the sabi Test Suite

Use this skill when the user invokes `/audit-tests` (optionally with a path / module / glob argument). The deliverable is a written audit presented to the user — do **not** edit tests unless the user explicitly asks you to fix something specific afterward.

The audit is a quality-of-tests review, not a coverage-percentage check. The bar is: would this test catch a real regression, and does its name truthfully describe what it asserts?

## Source-of-truth docs

Test conventions (file-per-source-module, `scripts/python -m pytest`,
numerical-tolerance defaults) and notation conventions (shape strings,
public-batched / private-single-point, "target" vocabulary) live in:

- [`docs/contributing.md`](../../../docs/contributing.md) — Tests section
- [`docs/notation.md`](../../../docs/notation.md)

**Read both before producing an audit.** Cite the relevant section
when flagging a convention violation. Don't restate the rules here —
when the docs change, the skill should keep working without edits.

## 1. Scope the audit

Determine the audit scope from the argument:

- No argument → full suite under `tests/`.
- A path (e.g. `tests/test_loop.py`) → just that file.
- A bare module name (e.g. `metrics`, `acquisitions`, `tempering`) → tests covering that source module: typically `tests/test_<module>.py` plus any test files importing from `sabi.<module>`. Use `grep -l "sabi.<module>" tests/` to identify them.
- A glob (e.g. `tests/test_*tempering*.py`) → expand directly.

State the scope explicitly back to the user as the first line of the audit, so they can confirm before reading findings.

Then locate the corresponding source files and read both sides. A test audit without reading the code under test is shallow — you can't tell whether an assertion is meaningful unless you know what the function is supposed to do.

For full-suite audits, read all of `tests/` and `src/sabi/` upfront if context allows; for huge suites, prioritize the load-bearing modules (`algorithms/loop.py`, `acquisitions/`, `metrics/`, `tempering/`, `surrogate/`). The "one test file per source module" rule in `contributing.md` is your map.

Run the relevant tests once before auditing, so failures surface immediately. Use the test-runner command documented in `contributing.md`.

If tests are failing, surface that at the top of the audit before going further. A skill that reports "all looks good" while the suite is red is worse than useless.

## 2. Run the standard checks

For each test file in scope, evaluate the following. Cite specific test names (`tests/test_foo.py::test_bar`) for every finding.

### Correctness

- Does each assertion actually verify the documented behavior, or does it just verify *something* the code happens to do?
- Are tolerances reasonable per the rule in `contributing.md`? Flag tests that use a tolerance so loose it would pass even on a broken implementation.
- Are random seeds fixed where determinism matters? Are they varied where the test is *meant* to probe seed-dependent behavior?
- Does the test set up the right precondition? A test of `_score_single` (a private single-point hook per `notation.md`) that passes a batched array is testing the wrong contract.

### Misleading or "cheating" tests

These are the highest-value findings. Look for:

- **Names that don't match assertions.** `test_emulator_handles_nan` that never feeds NaN to the emulator. `test_acquisition_target_terminal` that asserts on `CURRENT`.
- **Vacuous assertions.** `assert result is not None` on a function that can't return None. `assert len(out) >= 0`. `assert isinstance(x, type(x))`.
- **Tautological assertions.** Asserting on a value the test itself just constructed and passed in unchanged.
- **Round-trip-on-itself.** Calling the function under test twice and comparing the outputs — confirms determinism, not correctness.
- **Mocked-into-success.** A mock or fixture that returns the exact value the assertion checks for, so the test would pass even if the code under test were `pass`.
- **Skip-on-failure.** `pytest.skip()` or `pytest.xfail()` used to paper over real failures rather than to mark genuinely platform-specific tests.
- **Try/except that swallows.** `try: ...; except Exception: pass` inside a test, hiding regressions.
- **Setup that does the work.** A test where the fixture computes the expected value using the same code path the test claims to verify — common when fixtures grow over time.

For each finding, state *what* the test claims to verify (from the name or docstring) vs. *what* it actually verifies. The gap is the bug.

### Duplicate / redundant tests

- Multiple tests asserting the same property on the same input with different names.
- Parametrized tests where most parameter rows exercise the same code path (the rows are symbolically distinct but functionally identical).
- A test that's a strict subset of another, more thorough, test.

Recommend consolidation, but don't be aggressive — duplicate tests are cheap, and sometimes redundancy is intentional (e.g. a focused unit test plus an integration test covering the same property).

### Coverage gaps

For each public function / class / method in scope, check whether at least one test exercises:

- The happy path with realistic inputs.
- Edge cases relevant to the function's contract:
  - Single-element / empty inputs (where supported).
  - `n_rounds == 1`, `q == 1`, `n_initial` minimal.
  - Scalar vs. vector outputs (`output_shape == ()` vs `(p,)` with `p > 1`) — see the shape conventions in `notation.md`.
  - Unbounded prior support (algorithms that need bounded support should error clearly).
  - Tempering-on (non-`NoTempering`) variants.
  - JAX trace boundaries: if a function is traced, is there a test that actually runs it under `jax.jit` / `jax.vmap`?
- Documented error paths: each `raise` in the source should have a test that triggers it.

If a public surface has no tests at all, that's a finding by itself.

### Mathematical correctness

This is sabi-specific and load-bearing. For every test of a function with a math formula in its docstring, check:

- Is there at least one test that verifies the math against a closed-form value or a reference implementation? Smoke tests (`returns the right shape`) don't count.
- For estimators / metrics: does the test cover a case where the *correct* value is known analytically (e.g. MMD between two equal distributions = 0, expected improvement at `f^* = mean` reduces to `sigma * phi(0)`)?
- For samplers: is there a test that the empirical mean / variance / quantile of a large sample matches the analytical moment?
- For tempering: do tests exist for both endpoints of the schedule (the un-tempered base and the fully-tempered terminal) where the math collapses to known limits?
- For acquisitions: does the test verify the *score*, not just that `select_batch` returns the right shape?

Flag math-bearing functions whose tests don't include at least one closed-form check. These are the tests most likely to silently rot.

### Test gaps surfaced by recent changes

- Skim `git log --oneline -- src/sabi/` for the scoped module since the last test addition. Bug fixes without a regression test are gaps.
- Look for `# TODO test ...` / `# pragma: no cover` markers in source. Each is a flag worth checking.

### Convention conformance

Walk the test files against the Tests section of `contributing.md` and the relevant parts of `notation.md`, and flag any violation. Cite the violated section by anchor.

## 3. Write the audit

Present findings in this structure. Be specific — every finding should cite `tests/<file>::<test_name>` and the source it covers (or fails to cover).

```
# Test audit: <scope>

**Suite status**: <N tests, M passing, K failing/skipped>.

## Critical
- `tests/test_foo.py::test_bar` — <misleading/cheating finding, one or two sentences>.
- ...

## Coverage gaps
- `src/sabi/<module>.py::<function>` — <what's untested and why it matters>.
- ...

## Mathematical-correctness gaps
- <function> — <what closed-form check is missing>.
- ...

## Duplicates / consolidation candidates
- ...

## Minor
- ...

## What's solid
- <areas with strong, meaningful test coverage — keep this honest>.

## Recommendation
<top 1–3 things worth fixing first, with a one-line rationale each>.
```

Notes:
- Use clickable markdown links for file paths: `[tests/test_foo.py:42](tests/test_foo.py:42)`.
- Don't pad. If a category has no findings, say "none" or omit the section.
- Prioritize: a single misleading test that masks a real bug matters more than ten naming nits.
- The recommendation is a *short* punch list. The user will decide what to fix.

After delivering the audit, offer to: open follow-up issues for each gap, write the missing tests for a specific finding, or rerun the suite with extra options. Wait for the user to choose.
