# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Recommended workflow (SWARM)

This repo is set up as a **Claude Code template** for SWARM-style research work:

- Custom slash commands live in `.claude/commands/` (e.g. `/run_scenario`, `/sweep`, `/plot`, `/red_team`).
- Research-role specialist agents live in `.claude/agents/`.
- Role-selection guidance lives in `AGENTS.md`; keep it synchronized with `.claude/agents/`.
- Optional git hygiene hooks live in `.claude/hooks/` (install via `/install_hooks`).
- MCP integrations are configured in `.mcp.json` (safe-by-default placeholders; no secrets committed).

### Extend, don't proliferate

**Do not create new slash commands, agents, or hooks when an existing one can absorb the functionality.** This repo has been through multiple consolidation passes. Before proposing anything new:

1. **Check existing commands first.** Read `.claude/commands/` and look for a command that covers the same workflow area. If one exists, add a `--flag` or mode to it instead of creating a new file.
2. **Prefer flags over files.** A command with 3 clear modes (`/ship`, `/ship --fix`, `/ship --all`) is better than 3 separate commands (`/commit_push`, `/fix_commit`, `/sweep_and_ship`).
3. **Same rule for agents.** If a new role overlaps with an existing agent's domain, extend that agent's `.md` with a new section rather than creating a new agent file.
4. **Same rule for hooks.** Post-write checks are consolidated into a single `post_write_check.sh`. Add new checks as a section in that file, not as a new hook script.
5. **When in doubt, don't create.** If you're unsure whether something warrants a new command, it probably doesn't. Add it as a mode on the closest existing command and note it in the migration table.
6. **Document every change.** When extending a command with a new flag/mode or creating a new command:
   - Update the command's `.md` file with usage examples, argument parsing, and behavior for the new mode.
   - Add a migration table entry if replacing or absorbing an old command.
   - Update cross-references in other commands that mention the changed command.
   - Update `AGENTS.md` if agent roles change.
   - Run `/audit_docs --nav-only` if docs pages were added or moved.

### Artifacts repo

Large/supplementary files live in a separate repo: [`swarm-ai-research/swarm-artifacts`](https://github.com/swarm-ai-research/swarm-artifacts). This includes:
- Historical experiment runs, Lean proofs, promo site, research notes, reference papers, `IMPLEMENTATION_PLAN.md`, `DESIGN_CRITIQUE.md`.

These directories are gitignored in main. Local `runs/` and `docs/papers/` are still used as working directories — they just aren't committed here.

### Run artifacts

Prefer writing experiment outputs to a self-contained run folder:

- `runs/<timestamp>_<scenario>_seed<seed>/history.json` (JSON export)
- `runs/<timestamp>_<scenario>_seed<seed>/csv/` (CSV exports)
- `runs/<...>/plots/` (generated plots)

The `runs/` directory is gitignored. To archive runs, push them to `swarm-artifacts`.

## Commands

```bash
# Install for development
python -m pip install -e ".[dev,runtime]"

# Run all tests (use python -m to ensure correct environment)
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_payoff.py -v

# Run a specific test
python -m pytest tests/test_payoff.py::TestPayoffInitiator::test_payoff_linear_in_p -v

# Run with coverage
python -m pytest tests/ --cov=swarm --cov-report=html

# Lint
ruff check swarm/ tests/

# Type check
python -m mypy swarm/

# Run a scenario (CLI)
python -m swarm run scenarios/baseline.yaml --seed 42 --epochs 10 --steps 10

# Run a scenario (example runner)
python examples/run_scenario.py scenarios/baseline.yaml
```

## Architecture

This is a simulation framework for studying distributional safety in multi-agent AI systems using **soft (probabilistic) labels** instead of binary good/bad classifications.

### Data Flow

```
Observables → ProxyComputer → v_hat → sigmoid → p → SoftPayoffEngine → payoffs
                                                  ↓
                                            SoftMetrics → toxicity, quality gap, etc.
```

### Key Components

**`swarm/core/proxy.py`** - `ProxyComputer` converts observable signals (task_progress, rework_count, verifier_rejections, engagement) into `v_hat ∈ [-1, +1]` using weighted combination, then applies calibrated sigmoid to get `p = P(v = +1)`.

**`swarm/core/payoff.py`** - `SoftPayoffEngine` implements payoffs using soft labels:
- `S_soft = p * s_plus - (1-p) * s_minus` (expected surplus)
- `E_soft = (1-p) * h` (expected harm externality)
- Payoffs include surplus share, transfers, governance costs, externality costs, and reputation

**`swarm/metrics/soft_metrics.py`** - `SoftMetrics` computes probabilistic metrics:
- Toxicity: `E[1-p | accepted]`
- Quality gap: `E[p | accepted] - E[p | rejected]` (negative = adverse selection)
- Conditional loss: selection effect on payoffs

**`swarm/metrics/reporters.py`** - `MetricsReporter` provides dual reporting of soft (probabilistic) and hard (threshold-based) metrics for comparison.

**`swarm/logging/event_log.py`** - Append-only JSONL logger for simulation replay. Can reconstruct `SoftInteraction` objects from event stream.

### Test Fixtures

`tests/fixtures/interactions.py` provides generators for test data:
- `generate_benign_batch()` - high p, positive outcomes
- `generate_toxic_batch()` - low p, exploitation patterns
- `generate_mixed_batch()` - realistic distribution
- `generate_adversarial_scenario()` - coordinated attack pattern

## Domain Concepts

- **p**: Probability that interaction is beneficial, `P(v = +1)`, always in `[0, 1]`
- **v_hat**: Raw proxy score before sigmoid, in `[-1, +1]`
- **Adverse selection**: When low-quality interactions are preferentially accepted (quality_gap < 0)
- **Externality internalization**: ρ parameters control how much agents bear cost of ecosystem harm

## Research Operator Context

Claude Code serves as the local research operator for this project. The research memory system, skills, and workflow definitions live on disk — Claude Code reads them directly.

### 4-Tier Memory System (`.letta/memory/`)

| Tier | Path | Purpose |
|---|---|---|
| System | `.letta/memory/system/` | Identity, workflow rules, preferences |
| Project | `.letta/memory/project/` | Repo map, scenario families, governance knobs |
| Threads | `.letta/memory/threads/` | Active hypothesis (`current.md`), session log (`research-log.md`) |
| Runs | `.letta/memory/runs/` | Run pointers and summaries (`latest.md`) |

These files are append-only where noted (especially `research-log.md`). Read them for context; update them via `/ship --research-close`.

### Available Skills (`.skills/`)

Skills are self-documenting directories with a `SKILL.md` file. To use a skill, read its `SKILL.md` and follow the instructions. Available skills:

- `run-query` — Search run history by tag, date, type, or claim
- `synthesize` — Generate vault notes from a completed run
- `sanity-check` — Quick validation run on a scenario
- `regression-check` — Regression check (tests, baseline, or full)
- `experiment-loop` — Run an experiment cycle
- `claim` — Manage research claims inventory
- `session-close` — End-of-session memory update (now handled by `/ship --research-close`)
- `verify` — Verify a claim against evidence
- `vault-init` — Initialize the memory vault structure

### Research Session Protocol

- **Open**: Run `/status --research` to load active hypothesis, recent runs, and research context.
- **Close**: Run `/ship --research-close` to summarize the session, update memory files, and commit/push.
- **CLI wrapper**: `./scripts/letta-os.sh` provides shell access to research operations via `claude -p`.

## Multi-Session Worktree Workflow

When running 15+ concurrent Claude Code sessions, each session runs in its own git worktree to avoid index races and branch conflicts.

### Launch

```bash
./scripts/claude-tmux.sh 4    # Launch 4 isolated sessions
./scripts/claude-tmux.sh kill  # Kill tmux session (worktrees kept)
./scripts/claude-tmux.sh cleanup  # Remove all session worktrees + branches
```

### Environment Variables

Each session pane has these env vars set via `scripts/detect-session.sh`:

| Variable | Example | Description |
|---|---|---|
| `IS_SESSION_WORKTREE` | `true` | Whether this shell is inside a session worktree |
| `SESSION_ID` | `session-2` | Worktree directory name |
| `WORKTREE_ID` | `pane-2` | Pane identifier |
| `SESSION_BRANCH` | `session/pane-2` | Git branch for this session |
| `MAIN_REPO_ROOT` | `/path/to/repo` | Absolute path to the main repo |

### Session-Aware Commands

| Command | Worktree behavior |
|---|---|
| `/status` | Shows session identity and main repo path at top |
| `/pr` | Branches from `origin/main` instead of checking out `main` |
| `/ship` | Uses `bd --sandbox sync` to avoid beads daemon contention |
| `/merge_session` | Merges session branch into main (run from main repo) |
| `/merge_session --all` | Merges all session branches at once |

### Beads in Sessions

Use `bd --sandbox` in worktrees to avoid contention with the main repo's beads daemon. The `/ship` command does this automatically.

### Inter-Session Coordination (`agent_messages`)

Sessions coordinate via a shared SQLite table in `runs/runs.db` (accessible through the `sqlite_runs` MCP server). Schema:

```sql
CREATE TABLE agent_messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  from_agent TEXT NOT NULL,
  to_agent TEXT NOT NULL,   -- SESSION_ID for direct, '#swarm' for broadcast
  body TEXT NOT NULL,
  acked INTEGER NOT NULL DEFAULT 0
);
-- Partial index for fast inbox queries
CREATE INDEX idx_agent_messages_to_unacked ON agent_messages (to_agent, acked) WHERE acked = 0;
```

**Message conventions:**
- `ONLINE: ready for work` — announce session start
- `CLAIM: <beads-id>` — claim a task (check before starting work to avoid duplicates)
- `DONE: <beads-id>: <summary>` — announce completion
- `BLOCKED: <description>` — ask for help

**Usage from any session:**
```sql
-- Check inbox (broadcast + direct)
SELECT * FROM agent_messages WHERE to_agent IN ('<SESSION_ID>', '#swarm') AND acked = 0 ORDER BY ts;
-- Send broadcast
INSERT INTO agent_messages (from_agent, to_agent, body) VALUES ('<SESSION_ID>', '#swarm', 'CLAIM: beads-042');
-- Ack after reading
UPDATE agent_messages SET acked = 1 WHERE id = <msg_id>;
```

If the table doesn't exist (fresh `runs.db`), create it with the schema above.

For atomic claims, advisory file/module locks, structured proposals, and early
conflict detection (beyond `CLAIM:` message strings), prefer the coordination
primitives in the same database: `python -m swarm.agentgit coord claim|lock|conflicts|propose|status`
(see `docs/agentgit_mvp.md` § Machine-Speed Coordination).

## Paper Author Resolution

When `/write_paper` or `/compile_paper` needs an author name, resolve in this order:

1. `$SWARM_AUTHOR` environment variable (if set)
2. `git config user.name` (if set)
3. Prompt the user

Never guess or infer from the OS username.

## Test fix discipline

- When fixing a flaky test, prefer making it deterministic (set seeds, constrain inputs) over loosening assertions.
- Before fixing a beads-filed bug, verify the cited code at HEAD first (`grep` the cited lines, `git log -S` the symbol): in multi-session periods ~30% of filed bugs are already fixed — the filer worked from a stale snapshot. Close those with the fixing commit hash as evidence instead of re-fixing.
- After rewriting a source file with a script (python heredoc, sed -i), a same-second mtime can defeat Python's bytecode-cache invalidation: imports then silently run the OLD code while the file shows the new code (observed twice on 2026-07-18). If a just-edited module behaves as if unchanged, `rm -rf` the package's `__pycache__` (or run `python -B`) before debugging further.

## Blog post disclaimers

- Any blog post (`docs/blog/`, `docs/posts/`) that references financial markets, trading, investment, market microstructure, or similar financial concepts **must** include a disclaimer at the bottom:
  ```
  ---

  *Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
  ```
- For posts that simulate economies (e.g. AI Economist), use the variant: `"...simulates a stylized economic environment for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any economic policy or trading strategy."`
- The pre-commit hook enforces this automatically.

## Core principles are append-only

- Foundational framing in README.md (e.g. "The Core Insight"), CLAUDE.md ("Safety / invariants"), and `docs/research/theory.md` should be **extended, not replaced**.
- When new understanding refines a core idea, add a new subsection or paragraph — do not rewrite the original.
- The post-write hook will warn when edits touch foundational sections. Treat the warning seriously.

## Faithful reporting (no false claims)

Report outcomes faithfully: if tests fail, say so with the relevant output; if you did not run a verification step, say that rather than implying it succeeded. Never claim "all tests pass" when output shows failures, never suppress or simplify failing checks (tests, lints, type errors) to manufacture a green result, and never characterize incomplete or broken work as done. Equally, when a check did pass or a task is complete, state it plainly — do not hedge confirmed results with unnecessary disclaimers, downgrade finished work to "partial," or re-verify things you already checked. The goal is an accurate report, not a defensive one.

## Safety / invariants (do not break)

- `p` must remain in `[0, 1]` everywhere it is surfaced or logged.
- Event logs (`*.jsonl`) are append-only and should remain replayable.
- Runs should be reproducible from: scenario YAML + seed + exported history/CSVs.
