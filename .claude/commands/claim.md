# /claim

Atomic work-start. **Claim a task before doing any work on it** — this refuses
if another session already holds it, which is the one thing that physically
prevents two sessions from building the same feature (the 2026-07-22 duplicate-
7ge5 incident).

## Usage

`/claim <bead-id>` — claim a task and start work
`/claim status` — show what you and every other session currently hold
`/claim release <bead-id>` — release when done (or handing off)
`/claim check <bead-id>` — verify you still hold it (used by the pre-commit gate)

## Why this exists

`bd update --status=in_progress` is advisory — it doesn't stop a second session
from picking up the same bead. `/claim` routes work-start through an **atomic**
claim in the shared cross-worktree coordination DB
(`$MAIN_REPO_ROOT/runs/runs.db`): the first session to claim a task wins, and
every other session is **refused** with the holder's name. No two sessions can
hold the same task, so no two sessions can duplicate the work.

## Behavior

When you invoke `/claim <bead-id>`:

1. Run `python -m swarm.agentgit claim claim <bead-id>`.
2. **If it prints `REFUSED: … already claimed by <session>`** (exit 2): STOP.
   Another session is on this. Run `/claim status`, pick different `bd ready`
   work, or coordinate. Do **not** start the work.
3. **If it prints `CLAIMED …`** (exit 0): a marker is written to
   `.agentgit/current-claim.json` and the bead is set `in_progress`. Proceed.
   The pre-commit hook will verify you still hold it at commit time — a
   collision there means someone took your task mid-flight.
4. When the task is shipped (or you hand off), run `/claim release <bead-id>`.

## When to use it

**Every time you pick up a bead to do real work** — this is the first step of
work-start, before reading code or editing. It replaces the bare
`bd update <id> --status=in_progress`. See CLAUDE.md → "Session Work-Start
Protocol". For read-only investigation you don't need to claim; claim once you
intend to change files.

## Notes

- Identity is `$SESSION_ID` (set per worktree) or `main-checkout@<host>` outside
  a worktree. Concurrent code work in one checkout is separately flagged by the
  pre-commit worktree tripwire — the real fix there is an isolated worktree.
- Re-claiming a task you already hold is idempotent (safe to run again).
- `--force`-style overrides are intentionally absent: stealing a live claim is
  the failure mode, not a feature.
