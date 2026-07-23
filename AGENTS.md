# AGENTS

This repo maintains task-focused LLM agent personas in `.claude/agents/*.md`. Use the role that best matches the work and treat the `.claude/agents` files as the source of truth.

## Scope & Precedence
- This guidance applies to coding agents operating in this repository (including Claude Code and Codex-style agents).
- Instruction priority is: system/developer/user directives first, then this file.

## Session Start — check the coordination channel
- The SessionStart hook surfaces unacked `agent_messages` rows automatically.
  Manual query: `sqlite3 runs/runs.db "SELECT id, ts, from_agent, body FROM agent_messages WHERE acked=0 ORDER BY ts;"`
- Act on `ASSIGN:` rows matching your role: claim the bead (`bd update <bead> --status in_progress`),
  reply on the channel (`INSERT INTO agent_messages (from_agent,to_agent,body) VALUES ('<you>','#swarm','CLAIM: <bead>')`),
  then ack the row (`UPDATE agent_messages SET acked=1 WHERE id=<id>`).
- Announce significant completions with a `DONE: <bead>: <summary>` row including a
  concrete artifact (commit hash, `runs/` path, or `artifact=<ref>`) and `gate=<check>:<result>`.
  A DB trigger rejects artifact-less DONE rows — a status report is not a completion.
  Schema and conventions: CLAUDE.md § Inter-Session Coordination (`agent_messages`).

## How To Choose
- Scenario design or mechanism-isolating experiments: `Scenario Architect`
- Governance levers and intervention tradeoffs: `Mechanism Designer`
- Metric quality and research claim integrity: `Auditor`
- Red-teaming and adversarial strategies: `Adversary Designer`
- Reproducibility, benchmarks, and hygiene: `Reproducibility Sheriff`
- External repo reconnaissance and pattern mining: `Research Scout`

## Hooks
- Pre-commit runs from `.claude/hooks/pre-commit` via `.git/hooks/pre-commit`.
- If `pre-commit` is installed, the hook runs `.pre-commit-config.yaml` hooks; otherwise it falls back to inline ruff/mypy.
- Set `SKIP_SWARM_HOOKS=1` to bypass the hook.

## Scenario Architect
Focus: designs scenarios that isolate a single mechanism and are easy to reproduce.
Optimizes for:
- One-mechanism clarity
- Minimal confounders
- Deterministic reproduction
- Measurable success criteria
Deliverables:
- New or updated `scenarios/*.yaml`
- Short rationale: hypothesis, mechanism, expected signature in metrics
- Minimal run command (or `/run_scenario` invocation)
Guardrails:
- Prefer new scenario files over mutating benchmark scenarios
- Keep default epochs/steps modest until signal is validated
Source: `.claude/agents/scenario_architect.md`

## Mechanism Designer
Focus: proposes governance levers/interventions and predicts their tradeoffs.
Optimizes for:
- Mechanistic predictions
- Concrete parameterization and ranges
- Side-effect mapping across key metrics
Deliverables:
- Proposed change in `swarm/governance/*` and/or scenario governance config
- Short experiment plan (baseline vs intervention, expected deltas, failure cases)
- Suggested sweep axes for `/sweep`
Guardrails:
- Avoid levers that require hidden state to evaluate
- Prefer reversible interventions
Source: `.claude/agents/mechanism_designer.md`

## Auditor
Focus: audits metric quality and research claims for correctness, statistical rigor, and replication status.
Two modes:
- **Metric quality**: definition, robustness, logging consistency, tests. Deliverables: metric implementation via `/add_metric`, tests, docs.
- **Research integrity**: grades claims as SOLID/HONEST/WEAK/OVERCLAIMED/UNVERIFIABLE. Deliverables: graded claim audit, rewording suggestions, overall integrity score. Verifies against the bead's negative spec (enumerated insufficient outcomes) when present, and runs the phantom-gate check: any "verified/tested" statement whose gate never actually executed grades UNVERIFIABLE (docs/research/erdos-1038-swarm-lessons.md).
Guardrails:
- Do not silently rename metrics in exports
- Be honest but constructive — improve claims, don't block publication
- Recommend the weaker framing when in doubt
Source: `.claude/agents/auditor.md`

## Adversary Designer
Focus: designs adaptive/evasive strategies that probe governance gaps.
Optimizes for:
- Realistic adversary capabilities and constraints
- Adaptive strategies that respond to governance signals
- Coverage across different attack levers
Deliverables:
- New or updated adversarial behavior in `swarm/agents/*` or `swarm/redteam/*`
- Minimal reproduction run (often `/red_team quick`)
- Failure-mode writeup with mitigations
Guardrails:
- Keep attacks within the modeled environment
- Expose seeds when adding stochasticity
Source: `.claude/agents/adversary_designer.md`

## Reproducibility Sheriff
Focus: enforces plots-from-PR reproducibility and research hygiene.
Enforces:
- Determinism with explicit seeds
- Artifact capture (history JSON and CSV exports)
- Minimal smoke benchmarks that catch breakage
- Updated run instructions when interfaces change
Deliverables:
- Hook or CI improvements and/or documentation fixes
- Standard Results snippet for PR descriptions
Guardrails:
- Prefer lightweight checks contributors will run
- Add new required checks as recommended first
Source: `.claude/agents/reproducibility_sheriff.md`

## Research Scout
Focus: investigates external repositories/codebases for patterns relevant to a concrete implementation goal.
Optimizes for:
- Fast discovery of transferable patterns
- Source-backed findings with exact file paths
- Practical adaptation advice for this repo
Deliverables:
- Structured findings from target repo(s): what it is, where it lives, how it works
- Suggested adaptation plan tied to local code areas
- Tradeoffs and integration risks
Guardrails:
- Prefer direct evidence from source files over secondhand summaries
- Clearly separate observed facts from inferred recommendations
Source: `.claude/agents/research_scout.md`

## Handoff Protocol

When work transitions between roles, follow this protocol to prevent dropped context and ownership disputes.

### Transition Matrix

| From | To | Artifact passed | Trigger |
|---|---|---|---|
| Scenario Architect | Mechanism Designer | `scenarios/*.yaml` + rationale | "Scenario ready, needs governance lever" |
| Mechanism Designer | Scenario Architect | Proposed config changes + sweep axes | "Need a scenario to isolate this lever" |
| Mechanism Designer | Auditor | New metric implementation | "Metric needs quality audit" |
| Scenario Architect | Adversary Designer | Baseline scenario + expected metrics | "Scenario needs adversarial stress test" |
| Adversary Designer | Mechanism Designer | Failure-mode writeup + broken invariants | "Governance gap found, needs mitigation" |
| Any role | Reproducibility Sheriff | PR with code changes | "Code changed, needs hygiene check" |
| Any role | Auditor | Draft paper / blog / claim | "Claims need integrity audit" |
| Research Scout | Any role | Structured findings + adaptation plan | "External pattern ready for adoption" |

### Handoff rules

1. **Explicit artifact**: Every handoff must reference a concrete file or artifact, not a verbal summary.
2. **One owner**: At any moment, exactly one role owns the work item. The handoff transfers ownership.
3. **No self-validation**: The role that produces an artifact never audits it. Auditor audits claims, Reproducibility Sheriff audits hygiene.

### Metric ownership

When a governance lever (Mechanism Designer) requires a new metric:
- **Mechanism Designer** specifies what the metric should measure and its expected behavior.
- **Auditor** owns the metric's implementation quality (definition, robustness, logging, tests).
- If there's a dispute about metric semantics, the Mechanism Designer's specification wins. If there's a dispute about metric quality, the Auditor's assessment wins.

### Conflict resolution

When two roles claim the same work:
1. Check the transition matrix above — the role listed in the "To" column owns it.
2. If the matrix doesn't cover the case, the narrower role wins (e.g., Auditor over Mechanism Designer for metric quality).
3. If still ambiguous, the user decides.

## Trial-Task Gate (adopting new agents, models, or tools)

No new agent role, model, or external tool is wired into `agent-orchestrator.yaml`, `.claude/agents/`, or the default toolchain until it passes a trial task on this repo — the interview-question analog: candidates write real code before being trusted (beads 4z8y).

### Canonical trial task — agent roles and models

Close **one small open beads bug end-to-end**, unassisted:

1. Pick a `P3` bug from `bd ready` and claim it (`bd update <id> --status=in_progress`).
2. **Verify at HEAD first** (per CLAUDE.md test-fix discipline) — if already fixed, close with the fixing commit as evidence; that counts as a pass of the verification half.
3. Fix with a regression test; run the touched suite plus `ruff` and `mypy`.
4. Ship via `/ship --close <id> --fix --test` with a CHANGELOG entry.

**Acceptance bar (all required):**

- [ ] Touched test suites pass; no unrelated files staged (index race guard respected)
- [ ] CHANGELOG entry present; bead closed with concrete evidence (commit hash, test names)
- [ ] CI green on the pushed commit (`gh run list`, or `/fix-ci --watch <sha>`)
- [ ] No repo invariant violated (`p ∈ [0,1]`, append-only logs, core principles append-only)
- [ ] No fabricated claims: every assertion in the close reason is verifiable from artifacts

### Canonical trial task — external tools and engines

Before an external engine/tool becomes a dependency (bridge target, judge backend, CI step):

1. Reproduce a known-good artifact deterministically: `pytest -m smoke` (calibration determinism suite) plus one scenario run — `python -m swarm run scenarios/baseline.yaml --seed 42 --epochs 5 --steps 10` — with outputs matching a prior run byte-for-byte where the artifact contract promises it.
2. Verify license compatibility before any vendoring (cf. the a-evolve precedent: MIT claimed, no LICENSE file shipped — soft-import pinned by SHA instead).
3. Record the trial outcome in the adopting bead before wiring anything in.

**Failure handling:** a failed trial is filed as a bead with the evidence, not silently retried. Two consecutive failures = do not adopt; note the blocker in the Migration Registry below.

## Migration Registry

Centralized record of agent consolidations. When roles merge, add an entry here.

| Old agent/command | Absorbed into | Date | Notes |
|---|---|---|---|
| `Metrics Auditor` | `Auditor` (metric quality section) | 2026-02 | Two audit modes in one role |
| `Research Integrity Auditor` | `Auditor` (research integrity section) | 2026-02 | Same |

## Landing the Plane (Session Completion)

**When explicitly shipping/landing work (for example: "ship", "submit", "close this out")**, complete ALL steps below. Work is not complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES (for ship/closeout sessions):**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

For non-ship sessions, keep changes local and report status clearly instead of forcing a push.
