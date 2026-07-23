# /sync_artifacts

Stage, commit, and push changes in the `swarm-artifacts` repo. Handles pre-commit validation and Co-Authored-By automatically.

## Usage

`/sync_artifacts [message]`
`/sync_artifacts --migrate <owner/repo> <path>`

Examples:
- `/sync_artifacts` (auto-generates commit message from diff)
- `/sync_artifacts "Add new claim cards from collusion study"`
- `/sync_artifacts --migrate swarm-ai-research/swarm runs/20260221_104316_langgraph_governed`

## Behavior

### 1. Locate the artifacts repo

Check these paths in order:
1. `$SWARM_ARTIFACTS_DIR` environment variable (if set)
2. `/Users/raelisavitt/swarm-artifacts` (default)
3. `../swarm-artifacts` relative to the main repo root

If none exist, report: "Cannot find swarm-artifacts repo. Set $SWARM_ARTIFACTS_DIR or clone it."

### 2. Check for changes

```bash
cd <artifacts-repo>
git status --short
```

If no changes (staged or unstaged), report "swarm-artifacts is clean, nothing to sync." and stop.

### 3. Show what changed

```bash
git diff --stat        # unstaged
git diff --cached --stat  # staged
```

Report the summary to the user.

### 4. Stage files

Stage all changed files, excluding:
- `.DS_Store`
- `*.db`
- `.env*`
- `credentials*`

```bash
git add -A -- ':!.DS_Store' ':!*.db' ':!.env*' ':!credentials*'
```

### 5. Commit

If the user provided a message via `$ARGUMENTS`, use it. Otherwise, auto-generate from the staged diff:
- Count files by category (vault/, scripts/, schemas/, runs/)
- Summarize: "Update N vault notes, M scripts" or similar

Always append `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`.

```bash
git commit -m "$(cat <<'EOF'
<message>

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

If pre-commit hooks fail:
- Read the error output
- If it's an index consistency error (notes not in `_index.md`), auto-fix by adding the missing references to `vault/_index.md`, re-stage, and retry the commit
- If it's a validation error (schema, YAML), report the errors and stop

### 6. Push

```bash
git push
```

If push fails with non-fast-forward:
- `git pull --rebase`
- Retry push once
- If still fails, report error and stop

### 7. Report

```
swarm-artifacts synced:
  Commit:  <short-sha> <message>
  Files:   <N> changed
  Push:    origin/main
```

---

## `--migrate` mode

Migrate misplaced artifacts from another repo into swarm-artifacts and remove them from the source.

### Usage

`/sync_artifacts --migrate <owner/repo> <path>`

- `<owner/repo>`: GitHub repo where the artifacts were mistakenly committed (e.g. `swarm-ai-research/swarm`)
- `<path>`: Path within that repo to migrate (e.g. `runs/20260221_104316_langgraph_governed`)

### Behavior

#### 1. Inspect source

Use the GitHub API (`mcp__github__get_file_contents` or `gh api`) to:
- List all files under `<path>` recursively
- Confirm the path exists and isn't empty
- Report what will be migrated (file count, total size estimate)

#### 2. Check source .gitignore

Fetch the source repo's `.gitignore` and check if `<path>` should have been ignored. Report whether this was a force-add situation.

#### 3. Locate artifacts repo

Same resolution as the default mode (steps 1-3 from `$SWARM_ARTIFACTS_DIR`, default path, or relative path). If not found locally, shallow-clone to `/tmp/sync_artifacts_migrate/swarm-artifacts`.

#### 4. Download files

For each file in the source path:
```bash
curl -sL "https://raw.githubusercontent.com/<owner/repo>/main/<file-path>" \
  -o <artifacts-repo>/<file-path>
```

Create intermediate directories as needed with `mkdir -p`.

#### 5. Push to swarm-artifacts

Stage all downloaded files. Use `git add -f` to handle files that match the artifacts repo's `.gitignore` (run artifacts like `*.png`, `*.parquet` are commonly ignored but valid in `runs/`).

```bash
git add -f <path>/
git commit -m "Migrate <path> from <owner/repo>

Moved from <owner/repo> (was force-added despite gitignore).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
git push
```

If push fails with non-fast-forward, `git pull --rebase` and retry once.

#### 6. Remove from source

Clone the source repo (shallow, to `/tmp/sync_artifacts_migrate/source`):
```bash
gh repo clone <owner/repo> -- --depth 1
```

Remove the migrated path and push:
```bash
git rm -r <path>
git commit -m "Remove force-added artifacts (migrated to swarm-artifacts)

<path> is gitignored — these were force-added by mistake.
Migrated to swarm-ai-research/swarm-artifacts.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
git push
```

#### 7. Cleanup

```bash
rm -rf /tmp/sync_artifacts_migrate
```

#### 8. Report

```
Migration complete:
  Source:  <owner/repo> <path>
  Target:  swarm-artifacts/<path>
  Files:   <N> migrated
  Source commit: <short-sha> (removed)
  Target commit: <short-sha> (added)
```

### Edge cases

- If the artifacts repo already contains files at `<path>`, warn the user and ask before overwriting.
- If the source repo push fails (e.g. branch protection), report the error and note that the files are already safe in swarm-artifacts — the user can remove from source manually.
- If `gh auth status` fails, stop early with an auth error.
- Binary files (`.png`, `.parquet`, `.pdf`) must be downloaded with `curl` (not read via API content endpoints which may truncate).

---

## Constraints

- Never force push.
- Never modify files in the main distributional-agi-safety repo — this command only operates on swarm-artifacts.
- Always return to the original working directory after finishing.
- If pre-commit index fixes are applied, mention what was auto-fixed.
