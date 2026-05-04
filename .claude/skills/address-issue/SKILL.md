---
name: address-issue
description: Begin work on a GitHub issue by number (e.g. `/address-issue 12`). Creates a feature branch with a human-readable name, gathers current context for the issue (the codebase may have drifted since the issue was filed), and produces a comprehensive plan for review. Does not edit any code until the user approves the plan.
---

# Address GitHub Issue

Use this skill when the user invokes `/address-issue <number>` (or similar phrasing referencing a GitHub issue number). The argument is the GitHub issue number to work on.

Follow the three steps below in order. Do not skip ahead — in particular, do not edit any code until step 3's plan has been approved by the user.

## 1. Create a feature branch with a human-readable name

- Fetch the issue with `gh issue view <number>` to get the title and body.
- Verify the working tree is clean (`git status`). If it is not, stop and ask the user how to proceed before branching.
- Make sure you branch from an up-to-date `main` (or the repo's default branch). Run `git fetch origin` and base the new branch on `origin/<default>`.
- Choose a branch name that a human would recognize at a glance. Format: `<type>/issue-<number>-<short-slug>`, where:
    - `<type>` is one of `feat`, `fix`, `refactor`, `docs`, `test`, `chore` — pick whichever best matches the issue.
    - `<short-slug>` is 2–5 lowercase, hyphenated words derived from the issue title. Strip filler words; keep the meaningful nouns/verbs.
    - Example: issue #12 titled "MMD estimator returns NaN for degenerate samples" → `fix/issue-12-mmd-nan-degenerate-samples`.
- Do **not** use a randomly generated worktree-style name (e.g. `claude/elastic-pike-...`). The branch name should be self-explanatory months from now.
- Create and check out the branch: `git checkout -b <branch-name>`.

## 2. Gather current context

The issue may have been filed weeks or months ago and the codebase has likely moved since then. Before planning, build a fresh picture of:

- **What the issue is actually asking for.** Read the full issue body and every comment (`gh issue view <number> --comments`). Note any decisions, scope changes, or pushback recorded in the comments — the latest comment often supersedes the original ask.
- **Linked PRs, issues, and discussions.** Follow any references in the issue. Check whether part of the work has already been done or attempted.
- **Current state of the relevant code.** Locate the files, functions, and modules the issue touches. Read them as they exist *now*, not as the issue describes them — names, signatures, and structure may have changed. Note any drift between the issue's description and current reality.
- **Recent history.** Skim `git log` for the affected files since the issue was opened. Look for commits that may already address part of the issue, change the surrounding design, or invalidate the issue's premise.
- **Tests and conventions.** Identify existing tests covering the affected area and any project conventions (CLAUDE.md, contributing guides, neighboring code style) that the change should follow.

If, after this investigation, the issue appears partially or fully resolved, out of date, or ambiguous, surface that to the user before producing a plan.

## 3. Produce a plan and wait for approval

Write a concrete plan and present it to the user. The plan should include:

- **Restatement of the goal** in one or two sentences, reflecting the *current* understanding (which may differ from the original issue text).
- **Drift notes** — anything that has changed in the codebase since the issue was filed that affects the approach.
- **Proposed approach** — the design or strategy at a high level, including trade-offs you considered and why you chose this path.
- **Concrete change list** — a step-by-step breakdown of files to add/modify/delete and what each change does. Be specific (file paths, function names, key signatures).
- **Tests** — what tests you'll add or update, and how you'll verify the fix locally.
- **Out of scope** — anything tempting but deliberately excluded, with a one-line reason.
- **Open questions** — anything the user needs to decide before you proceed.

After presenting the plan, **stop and wait for explicit approval**. Do not edit any source files (other than the branch creation in step 1) until the user has approved the plan or asked for revisions. If the user asks for changes, revise the plan and present it again — still without editing code — until they approve.

Once approved, proceed with implementation following the plan.
