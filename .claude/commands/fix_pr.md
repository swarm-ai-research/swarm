# /fix_pr

Multi-mode PR operations: create a PR from local changes, resolve merge conflicts on an existing PR, or run quality gates on an external PR. Consolidates the former `/review_external_pr` command (now `/fix_pr --review`).

## Usage

```
/fix_pr [branch-name]                          # Mode A: create PR from local changes
/fix_pr <pr-number-or-url> [fix|merge]          # Mode B: resolve conflicts / merge existing PR
/fix_pr --review <pr-number>                    # Mode C: run quality gates on a PR
```

Examples:
- `/fix_pr` (auto-generates branch name from changes)
- `/fix_pr fix/circuit-breaker-reset`
- `/fix_pr 239` (resolve conflicts on PR #239)
- `/fix_pr https://github.com/swarm-ai-research/swarm/pull/239 merge` (resolve conflicts and merge)
- `/fix_pr 239 merge` (resolve + merge)
- `/fix_pr --review 220` (check out PR, run quality gates, auto-fix, report)

## Argument parsing

Parse `$ARGUMENTS` to extract:
- `--review`: Quality gate review mode (Mode C)
- If first non-flag arg is a number or GitHub PR URL → Mode B (conflict resolution)
- Otherwise → Mode A (create PR from local changes)

---

## Mode A: Create PR from Local Changes

Unlike `/pr` which expects you to describe what changed, this mode is for when you've already made edits and just want to ship them on a clean branch.

### Behavior

1. **Sanity check**:
   - Run `git status` to identify modified and untracked files.
   - If there are no changes, abort with a message.

2. **Identify relevant files**:
   - Show the user the list of modified/untracked files.
   - Ask which files to include if it's ambiguous (e.g. mix of unrelated changes).
   - Exclude: `.DS_Store`, `*.db`, `.env*`, `credentials*`.

3. **Create a clean branch**:
   - Stash all changes: `git stash push --include-untracked -m "fix_pr: temp stash"`.
   - Ensure main is up to date: `git checkout main && git pull origin main` (skip if in a session worktree — use `git fetch origin main` and branch from `origin/main` instead).
   - Create the feature branch: `git checkout -b <branch-name>` (or `git checkout -b <branch-name> origin/main` in a worktree).
   - Pop the stash: `git stash pop`.

4. **Stage and commit**:
   - Stage only the relevant files (not `git add -A`).
   - Analyze the diff and draft a concise commit message.
   - Always append `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`.
   - Commit. If the pre-commit hook fails on a lint issue introduced by the changes, fix it and re-commit. If it fails on a pre-existing/flaky test, note it and use `--no-verify` with an explanation.

5. **Push and open PR**:
   - `git push -u origin <branch-name>`
   - `gh pr create --head <branch-name> --base main --title "..." --body "..."`
   - Body format: `## Summary` (bullet points), `## Test plan` (checklist), Claude Code footer.

6. **Return to main**:
   - `git checkout main` (leave the feature branch pushed).
   - Print the PR URL.

---

## Mode B: Resolve Conflicts / Merge Existing PR

Triggered when the first argument is a PR number or GitHub PR URL (without `--review`).

### Behavior

1. **Parse PR reference**:
   - Extract `owner`, `repo`, and `pr_number` from the argument.
   - If just a number (e.g. `239`), infer from current repo's `origin` remote.
   - If a URL (e.g. `https://github.com/org/repo/pull/239`), parse all three.

2. **Fetch PR metadata**:
   - `gh pr view <pr_number> --repo <owner>/<repo> --json title,headRefName,baseRefName,state,mergeable`
   - If the PR is already merged or closed, abort.
   - If `mergeable` is not `CONFLICTING` and no `merge` flag, report "no conflicts" and skip to step 6.

3. **Clone or reuse**:
   - If the PR is on the **current repo** (same `origin`): fetch and checkout the branch directly.
   - If the PR is on a **different repo**: clone to `/tmp/fix-pr-<pr_number>` via HTTPS (fall back from SSH if it fails).
   - Checkout the PR's head branch.

4. **Merge base branch to surface conflicts**:
   - `git fetch origin <baseRefName>` then `git merge origin/<baseRefName>`.
   - If no conflicts, auto-commit the merge and skip to step 5.
   - If conflicts exist:
     a. Show the conflicting files.
     b. For each file, read the conflict regions and resolve them:
        - If both sides added non-overlapping content (e.g. both appended tests), keep both.
        - If changes overlap, analyze intent from both sides and produce a correct merge.
        - If the resolution is ambiguous, show both versions and ask the user.
     c. Verify no conflict markers remain (`<<<<<<< `, `>>>>>>> `).
     d. Stage resolved files and commit: `git add <files> && git commit` with a merge message.

5. **Push the resolution**:
   - `git push origin <headRefName>`.

6. **Merge (if requested)**:
   - Only if the user passed the `merge` keyword or explicitly asks.
   - **Step 6a — Update branch**: If branch protection requires "up to date":
     - `gh api repos/<owner>/<repo>/pulls/<pr_number>/update-branch -X PUT -f expected_head_sha=<full-sha>`
     - Wait 5 seconds for GitHub to process.
   - **Step 6b — Attempt merge**:
     - `gh pr merge <pr_number> --repo <owner>/<repo> --squash`
   - **Step 6c — Handle branch protection failures**:
     - If merge fails because checks are pending, ask the user: `--admin` (merge now) or `--auto` (merge when checks pass).
     - If `--auto` fails because auto-merge is disabled on the repo:
       - Enable it: `gh api repos/<owner>/<repo> -X PATCH -f allow_auto_merge=true`
       - Retry with `--auto`.
     - Report final status.

7. **Cleanup**:
   - If a `/tmp/fix-pr-*` clone was created, remove it: `rm -rf /tmp/fix-pr-<pr_number>`.
   - If working in the current repo, return to the previous branch: `git checkout -`.

---

## Mode C: Quality Gate Review (`--review`)

Check out a PR branch, run full quality gates, auto-fix what's possible, push fixes, and report remaining issues.

### Behavior

1. **Fetch PR metadata**:
   ```bash
   gh pr view <pr-number> --json number,title,headRefName,baseRefName,author,state,statusCheckRollup
   ```
   If the PR is already merged or closed, abort with a message.

2. **Check out the PR branch locally**:
   ```bash
   gh pr checkout <pr-number>
   ```

3. **Run quality gates** (collect all results before reporting):

   Run all checks, do NOT stop on first failure:

   a) **Syntax check** — `python -m py_compile` on all `.py` files touched by the PR:
   ```bash
   gh pr diff <pr-number> --name-only | grep '\.py$'
   ```

   b) **Ruff lint with auto-fix** — run `ruff check --fix` on PR-touched files, then re-check.

   c) **Mypy** — only for files under `swarm/`.

   d) **Pytest** — full test suite: `python -m pytest tests/ -x -q --tb=short`

4. **Report results**:
   ```
   PR #<number> Quality Review: <title>
   Author: <author>   Branch: <headRefName>
   ─────────────────────────────
     Syntax check:   PASS / FAIL (N files with errors)
     Ruff lint:      PASS / FAIL (N issues, M auto-fixed)
     Mypy:           PASS / FAIL / SKIP (N errors)
     Tests:          PASS / FAIL (N passed, M failed)
   ─────────────────────────────
     Verdict:        CLEAN / N issues remain
   ```

5. **If auto-fixes were applied**:
   - Stage the fixed files, commit with `Fix lint issues from automated review`, push.
   - Append `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`.

6. **If unfixable issues remain**:
   - Provide specific errors with file:line references and suggested fixes.
   - Ask: "Should I attempt to fix the remaining issues, or leave a review comment on the PR?"

7. **Optional: Leave a PR review** via `gh pr review`:
   - `--approve` if all gates pass
   - `--request-changes` if blockers remain
   - `--comment` if only advisory issues remain

8. **Return to previous branch**: `git checkout -`

---

## Constraints

- Never force-push.
- Never commit `.DS_Store`, `*.db`, or files likely containing secrets.
- Never auto-resolve ambiguous merge conflicts — ask the user.
- Always verify no conflict markers remain before committing (Mode B).
- Do not merge unless the user explicitly requests it (Mode B).
- Never merge the PR automatically (Mode C).
- Do not modify files outside the PR's changed file set (Mode C).
- If the PR branch is from a fork and you can't push, skip the auto-fix push and report as suggestions (Mode C).
- If tests fail during pre-commit, stop and report unless clearly pre-existing.
- Prefer HTTPS for cloning external repos (SSH keys may not be configured).
- Always install dependencies first if needed: `python -m pip install -e ".[dev,runtime]"`.

## Migration from old commands

| Old command | Equivalent |
|---|---|
| `/review_external_pr 220` | `/fix_pr --review 220` |
