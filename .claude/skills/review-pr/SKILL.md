---
name: review-pr
description: Review a sabi GitHub PR by number (e.g. `/review-pr 20`). Performs a thorough code-quality pass: correctness, cleanliness, documentation, redundancy, edge cases, test coverage, and conformance to `docs/contributing.md` and `docs/notation.md`. Also enforces the "main loops read like pseudocode" invariant and flags excessive AI-generated commentary. Reports findings without modifying the PR.
---

# Review a sabi Pull Request

Use this skill when the user invokes `/review-pr <number>` (or asks to review a PR by number). The argument is the GitHub PR number to review. The deliverable is a written review presented to the user — do **not** post the review to GitHub unless the user explicitly asks for that.

The goal is a careful, calibrated review: catch real problems, don't pad the report with nits. If a section has nothing to flag, say so rather than inventing concerns.

## Source-of-truth docs

The substantive rules — coding conventions, math-doc style, type-hint
conventions, ProbPipe import policy, dataclass policy, shape /
notation / "target" vocabulary, public-batched / private-single-point
convention, the "main loops read like pseudocode" invariant, test
conventions, numerical-tolerance defaults — all live in:

- [`docs/contributing.md`](../../../docs/contributing.md)
- [`docs/notation.md`](../../../docs/notation.md)

**Read both before producing a review** and keep them open as you go.
Cite the relevant section back to the user when flagging a violation
(e.g. "violates `contributing.md#jax-traceability`"). Don't restate
the rules in this skill — when the docs change, the skill should keep
working without edits.

## 1. Fetch the PR and orient

Run these in parallel with the `Bash` tool:

- `gh pr view <number>` — title, body, author, state, branch.
- `gh pr view <number> --comments` — discussion so far (decisions, scope changes, pushback).
- `gh pr diff <number>` — the actual change.
- `gh pr checks <number>` — CI status.

Identify the PR's stated goal (from title + body), the linked issue if any (`gh issue view <linked>`), and the base/head branches. If the PR is huge, list the changed files with `gh pr view <number> --json files -q '.files[].path'` and prioritize the load-bearing files for deep reading.

Read the changed files at their PR-head version, not at `main`. If a changed file's logic depends on unchanged context (a base class, a helper), read that too — reviews based only on the diff miss bugs that live in the seam between changed and unchanged code.

## 2. Run the standard checks

Cover each of the following. For each, decide "no issues", "minor", or "blocking", and cite specific file:line locations.

### Correctness

- Does the code do what the PR / issue says it should?
- Are there logic bugs, off-by-one errors, swapped arguments, or wrong defaults?
- Are mathematical formulas correct? Cross-check against the docstring math.
- JAX trace boundaries: if the change runs inside a traced region, does it satisfy the JAX-traceability section of `contributing.md`?

### Edge cases and failure modes

- Empty inputs, single-element inputs, single-round runs (`n_rounds == 1`), zero-batch acquisitions (`q == 0`), unbounded prior support, scalar vs. vector outputs (`output_shape == ()` vs `(p,)`).
- NaN / inf propagation. Degenerate inputs (e.g. duplicate rows in `(X, Y)`).
- What happens when an optional dependency / protocol is missing? Are the error messages clear and at the right boundary?
- Tempering / `NoTempering` defaults: does the change still behave correctly when the schedule is degenerate?

### Cleanliness and redundancy

- Dead code, unused imports, unused parameters.
- Duplicated logic that should be factored, OR over-abstracted helpers introduced for hypothetical future use.
- Premature feature flags, backwards-compat shims, or `# removed: ...` comments instead of deletions.
- Variables that shadow notation symbols — see `notation.md` for the reserved set.

### Conformance to `contributing.md` and `notation.md`

Walk the PR diff against each subsection of those two docs and flag any violation. The docs are the spec; the review just enforces them. Cite the violated section by anchor in your finding.

The "main loops read like pseudocode" invariant (in `contributing.md`)
deserves an explicit mention because it's easy to violate
incrementally: any change touching `run()` in
`src/sabi/algorithms/loop.py` (or another top-level loop body) should
be checked against it. Don't flag pre-existing violations the PR
doesn't touch, but **do** flag any new code that makes the situation
worse.

### Excessive "AI thinking" commentary

Flag comments that are clearly AI-generated rationalization rather than load-bearing context. Patterns to watch for:
- Multi-paragraph block comments above a 3-line function explaining what the function does in prose.
- Comments that narrate the obvious (`# increment counter`, `# return result`).
- Comments that reference the chat that produced them (`# as discussed`, `# to address the user's concern about ...`).
- Comments that hedge (`# this might break if ...` followed by no actual handling).
- Comments restating what well-named identifiers already say.

Recommend deletion or compression for any AI-thinking-style block found.

### Test coverage

This is a quick check — a deep audit is the `audit-tests` skill's job. Just verify:
- New behavior has tests, ideally in the right `tests/test_<module>.py` file.
- Bug fixes include a regression test that fails on the old code.
- Public API changes update the relevant tests.
- Numerical tolerances are picked per the rule in `contributing.md`.

If test coverage looks thin, mention it and suggest the user run `/audit-tests` for a deep pass.

### Other useful checks

- **CI status**: if `gh pr checks` shows failures, mention them and look at the failure briefly.
- **Commit hygiene**: very large unrelated changes mixed in, untouched files re-formatted, accidentally committed secrets / large binaries.
- **Linked issue alignment**: does the PR actually solve the linked issue, or has scope drifted? Note drift if any.
- **Breaking changes**: public API removed or renamed without deprecation. Flag and ask whether intentional.

## 3. Write the review

Present findings to the user in this structure. Keep each item terse — file:line + one sentence + a citation to the violated doc section is usually enough. Group by severity, not by category.

```
# Review: PR #<number> — <title>

**Summary** (1–2 sentences): what the PR does and overall impression.

**CI status**: pass / fail / skipped checks.

## Blocking
- [`path/to/file.py:42`](path/to/file.py:42) — <issue, one sentence> (`contributing.md#<section>`).
- ...

## Minor
- ...

## Nits / suggestions
- ...

## What's good
- ...

## Recommendation
<approve | request changes | comment>, with a one-line rationale.
```

Notes on writing the report:
- Cite file paths as clickable markdown links: `[path/to/file.py:42](path/to/file.py:42)`.
- Don't mention checks that found nothing — silence on a category means no issues.
- Keep "What's good" honest; if the PR is mediocre, say "nothing notable to highlight" rather than fabricating praise.
- The recommendation is your honest read. If the PR is correct and well-tested but has minor cleanup items, "approve with minor comments" is fine.

After delivering the review, offer to: post it as a GitHub PR comment (`gh pr comment <number> --body-file ...`), open follow-up issues for non-blocking items, or dig deeper into any specific finding. Wait for the user to choose.
