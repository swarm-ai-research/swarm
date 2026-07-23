# /fix-ci

Triage and fix failing GitHub Actions CI runs. Fetches remote failure logs, categorizes errors, reads the relevant source context, and applies fixes — all in one shot.

## Usage

`/fix-ci [run_id]`

- `/fix-ci` — find the latest failed CI run on the current branch and fix it
- `/fix-ci 21852912800` — triage a specific run by ID

## Behavior

### Step 1: Identify the failing run

If no `run_id` is provided, find the most recent failed run:

```bash
gh run list --branch "$(git branch --show-current)" --status failure --limit 1 --json databaseId,name,headBranch,conclusion,createdAt
```

If no failures found on the current branch, also check `main`:

```bash
gh run list --branch main --status failure --limit 1 --json databaseId,name,headBranch,conclusion,createdAt
```

If still no failures, report "No failed CI runs found" and exit.

### Step 2: Fetch failed logs

```bash
gh run view <run_id> --log-failed
```

### Step 3: Categorize failures

Parse the log output and classify into these categories (matching the CI workflow jobs):

| Category | Job name pattern | What to look for |
|----------|-----------------|-------------------|
| **lint** | `lint` | `ruff check` errors |
| **type-check** | `type-check` | `mypy` errors — extract `file:line: error: message [code]` triples |
| **test-fail** | `test` | `FAILED` test lines — extract `file::test_name` |
| **coverage** | `test` | `FAIL Required test coverage of N% not reached. Total coverage: M%` |

### Step 4: Triage report

Print a structured summary before fixing:

```
CI Triage — Run #<run_id> (<workflow_name>)
Branch: <branch>
────────────────────────────────────────
  lint:        PASS / FAIL (N errors)
  type-check:  PASS / FAIL (N errors in M files)
  tests:       PASS / FAIL (N failed, M passed)
  coverage:    PASS / FAIL (actual% < required%)
────────────────────────────────────────
```

For each failing category, list the specific errors:

**type-check failures** — group by file:
```
  swarm/foo/bar.py
    L45: error: Incompatible types ... [assignment]
    L89: error: Missing return ... [return]
  swarm/baz/qux.py
    L12: error: ... [arg-type]
```

**test failures** — list each:
```
  tests/test_payoff.py::test_payoff_linear_in_p — FAILED
  tests/test_proxy.py::test_sigmoid_bounds — FAILED
```

**coverage failures** — show the gap:
```
  Required: 70%  Actual: 68.66%  Gap: 1.34%
  Lowest-coverage files (with most uncovered lines):
    swarm/research/workflow.py — 19% (174 lines uncovered)
    swarm/research/reflexivity.py — 28% (168 lines uncovered)
```

### Step 5: Fix

For each failure category, apply the appropriate fix strategy:

**lint** — Run `ruff check --fix swarm/ tests/` to auto-fix what's possible, then report remaining.

**type-check** — For each error:
1. Read the file around the error line (10 lines of context).
2. Identify the root cause (wrong type annotation, missing cast, etc.).
3. Apply the minimal fix (add annotation, cast, import, widen type).
4. Prefer fixing root causes over adding `# type: ignore`.

**test-fail** — For each failing test:
1. Read the test to understand what it expects.
2. Read the source code it tests.
3. Fix the source or the test as appropriate.

**coverage** — If coverage is below threshold:
1. Identify files with the most uncovered lines that are pure logic (not UI/CLI).
2. Check `pyproject.toml` `[tool.coverage.run]` for existing `omit` patterns.
3. If there are clearly untestable entry points (streamlit apps, CLI `__main__`, flask apps) not yet omitted, add them to `omit`.
4. If still below threshold, write targeted tests for the highest-impact uncovered module.

### Step 6: Verify

After all fixes, run the same checks locally to confirm:

```bash
ruff check swarm/ tests/
mypy swarm/
pytest tests/ -v -n auto --cov=swarm --cov-report=term --cov-fail-under=70 -p no:testmon -q
```

Print a final summary:

```
CI Fix Results
────────────────────────────────────────
  lint:        FIXED / was already passing
  type-check:  FIXED (N errors → 0) / was already passing
  tests:       FIXED (N failures → 0) / was already passing
  coverage:    FIXED (old% → new%) / was already passing
────────────────────────────────────────
  Verdict:     ALL CHECKS PASS / N issues remain
```

### Step 7: Suggest next steps

If all checks pass, suggest:
```
All CI checks now pass locally. You can commit and push to verify on remote.
```

If issues remain, list them clearly.

## Why this exists

`/preflight` runs checks locally on staged files before committing. `/fix-ci` works the other direction — it starts from a *remote* CI failure and works backward to identify and fix the root cause. The typical workflow is:

1. Push code
2. CI fails
3. `/fix-ci` — triages the failure, reads the relevant code, applies fixes, verifies locally
4. Push the fix

Without this command, CI triage requires manually: checking `gh run list`, fetching logs with `gh run view --log-failed`, parsing error output, reading each failing file, fixing, and re-running checks. This automates the entire cycle.

## Relationship to other commands

- **`/preflight`** — catches issues *before* commit; `/fix-ci` fixes issues *after* CI fails remotely
- **`/lint-fix`** — handles only ruff fixes; `/fix-ci` handles all CI job categories
- **`/healthcheck`** — checks code health broadly; `/fix-ci` targets specific CI failures

## `--watch [sha]` mode

Poll a CI run to conclusion instead of triaging an already-finished failure.
Added after a session hand-rolled this loop four times (retro 2026-07-18).

- `/fix-ci --watch` — watch the run for the current HEAD sha
- `/fix-ci --watch <sha>` — watch the run for a specific commit (short sha ok)

Behavior:

1. Resolve the sha (`git rev-parse --short=8 HEAD` when omitted).
2. Poll in a background Bash task (never a foreground sleep loop):
   ```bash
   for i in $(seq 1 10); do
     STATUS=$(gh run list --branch main --workflow CI --limit 8 \
       --json conclusion,headSha,status \
       -q '.[] | select(.headSha[0:8]=="<sha>") | "\(.status) \(.conclusion)"' | head -1)
     [[ "$STATUS" == completed* ]] && break
     sleep 90
   done
   ```
3. On `completed success`: report green and stop.
4. On `completed failure`: fall through to the normal /fix-ci triage flow
   (fetch `--log-failed`, classify, fix) for that run automatically.
5. Runs queue behind other pushes in this repo — expect several minutes;
   poll interval below 60s is wasted quota.

