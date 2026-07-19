# AgentGit MVP

AgentGit is a thin provenance layer over normal git. The first MVP lets an
agent evaluate its current worktree diff against a task-scoped policy and emit
a signed JSON bundle for CI, reviewers, or another agent to inspect.

```bash
python -m swarm.agentgit attest \
  --task issue-123 \
  --agent codex \
  --policy examples/agentgit_policy.yaml \
  --check pytest=pass \
  --output .agentgit/provenance.json
```

The bundle records:

- agent and task identity
- base commit and changed files
- additions/deletions by file
- path and size policy decisions
- required check results
- a `provenance` block of *what happened* producing the diff (see below)
- sealed admissibility receipt with the payload hash

Policy failures return a non-zero exit code by default. Use `--warn-only` when
you want to capture the bundle without blocking the current command.

## Provenance Contents

Beyond the diff and policy verdict, schema `agentgit.provenance.v1` records a
`provenance` block describing how the change was produced:

- `commands` — commands executed (binary + args, `return_code`, OS `isolation`
  backend, `duration_seconds`, and a `timed_out` flag). Build these from a
  worktree `CommandResult` via `CommandRecord.from_command_result(result)`.
- `environment` — model / runtime / version of the producing agent.
- `dependency_changes` — manifest/lockfile edits (`requirements.txt`,
  `pyproject.toml`, `package-lock.json`, `Cargo.lock`, `go.sum`, …) detected
  **automatically** from the diff.
- `sources` — external sources consulted.
- `reviews` — reviewer decisions.
- `overrides` — human overrides.

```python
from swarm.agentgit import CommandRecord, build_bundle

bundle = build_bundle(
    ...,
    commands=[CommandRecord(command=["pytest", "-q"], return_code=0, isolation="bwrap")],
    environment={"model": "claude-opus-4-7", "runtime": "python3.13"},
    sources=["https://example.com/issue/123"],
    reviews=[{"reviewer": "security", "decision": "approve"}],
)
```

The `provenance` block is folded into the signed receipt payload, so it is
**tamper-evident**: editing a recorded command or hiding a dependency change
makes `verify_bundle` fail the `payload_hash` check. Older `v0` bundles (hashed
without provenance) still verify — `verify_bundle` reconstructs the payload per
schema version.

## Durable Storage

A loose JSON file under `.agentgit/` is easy to lose and easy to rewrite
silently. `--store` writes each attested bundle (including failing ones —
a policy failure is provenance too) into one or both durable backends:

```bash
python -m swarm.agentgit attest \
  --task issue-123 --agent codex \
  --policy examples/agentgit_policy.yaml \
  --store log --store git-notes
```

- **`log`** — appends a hash-chained entry to `.agentgit/log.jsonl`. Each
  entry commits to the previous entry's hash, so edits, drops, and reorders
  break the chain. The log is append-only and replayable, matching the repo's
  event-log invariants.
- **`git-notes`** — attaches the bundle to the attested commit under
  `refs/notes/agentgit` (override with `--notes-ref`, target a different
  commit with `--store-commit`). Notes are appended as JSONL, so re-attesting
  a commit accumulates bundles instead of overwriting earlier provenance.
  Provenance then travels *with history*:

```bash
git push origin refs/notes/agentgit        # publish provenance
git fetch origin refs/notes/agentgit:refs/notes/agentgit   # retrieve it
git log --notes=agentgit                   # view inline with commits
```

Inspect and verify stored provenance with `history`:

```bash
python -m swarm.agentgit history --verify
```

`--verify` checks the log's hash chain, then re-verifies every stored
bundle's receipt signature (dev-key fallback, same as `verify`) — so a forged
entry that was re-chained consistently still fails on its signature. Non-zero
exit on any failure makes it CI-friendly.

## Reputation: Track Records

A track record attaches not to a bare agent name but to a **reputation
key** — identity + model + runtime + toolchain + prompt/config version —
because the same agent under a different config is a different actor.
`swarm.agentgit.ReputationLedger` (append-only JSONL over the existing
`PerformanceTracker` substrate) records:

- **attestations** — policy verdict, panel review burden, blocking
  security findings, and domains touched, all derived from the bundle;
- **ground-truth outcomes** — merged / rolled back / tests broken /
  security incident, supplied later by CI or the operator (the agent
  never self-attests its merge success).

```bash
python -m swarm.agentgit reputation record --bundle .agentgit/provenance.json
python -m swarm.agentgit reputation outcome codex@a1b2c3d4e5f6 --merged --task issue-9
python -m swarm.agentgit reputation show
```

`track_record()` folds the stream into rates (merge success, rollback,
test breakage, review burden, domains of competence) plus a **trust
score** in `[0, 1]` — a Beta-smoothed estimate that starts at 0.5 for
unknown actors and weights failures by severity (security incident >
rollback > broken tests).

**Low reputation → restricted scope**: policies get a `trust_below`
condition, and the CI gate reads trust from its own ledger — never from
the (agent-supplied) bundle:

```yaml
rules:
  - id: low-trust-needs-review
    when: { trust_below: 0.5 }
    action: require_review
```

```bash
python -m swarm.agentgit gate --bundle ... --policy ... \
  --reputation-ledger .agentgit/reputation.jsonl
```

Unknown trust (no track record) counts as below-threshold — fail-closed,
because "no history" must not outrank "bad history".

## Operational Memory

Code, issues, and PRs don't capture operational knowledge: "module X is
fragile because...", "never use lib Y here", "deploys fail if migration
order changes", "the previous agent failed this because...".
`swarm.agentgit.MemoryStore` gives agents a durable place for exactly that:

- **Structured** — every memory has a kind (`fragility`, `constraint`,
  `failure`, `practice`, `context`) and a subject (path/module/topic).
- **Versioned** — storage is append-only JSONL; updates append new versions,
  retiring appends a tombstone, history stays replayable.
- **Scoped** — `repo` (`.agentgit/memory.jsonl`, travels with clones and is
  reviewable in PRs), `org` (`~/.agentgit/org-memory.jsonl`, shared across
  repos), `agent` (`~/.agentgit/agent-memory/<agent>.jsonl`, private).

```bash
python -m swarm.agentgit memory remember swarm/core/proxy.py \
  --kind fragility --body "sigmoid calibration breaks if weights are renormalized"
python -m swarm.agentgit memory recall swarm/core   # hierarchical match
python -m swarm.agentgit memory history --id fragility-swarm-core-proxy-py --scope repo
python -m swarm.agentgit memory retire --id fragility-swarm-core-proxy-py --reason "refactored"
```

Recall is subject-aware and hierarchical: a memory about `swarm/core`
surfaces when asking about `swarm/core/proxy.py` and vice versa. Before
touching a diff, `store.recall_for_paths([f.path for f in
snapshot.changed_files])` answers "what should I know before changing these
files" — fragility notes, constraints, and prior agents' failures, most
specific scope (agent → repo → org) first.

## Machine-Speed Coordination

GitHub issues/PRs are human-shaped. When many agent sessions work one repo
concurrently, `swarm.agentgit.CoordinationBoard` (SQLite, defaults to the
same `runs/runs.db` as `agent_messages`) provides machine-semantics
primitives, all mirrored into an append-only `coordination_events` audit
stream:

- **claim / yield / done** — atomic task ownership; a contested claim tells
  you who holds it instead of racing message strings.
- **lock / release** — advisory locks on files/directories/module prefixes;
  a lock on `swarm/core` conflicts with one on `swarm/core/proxy.py` (and
  vice versa), siblings don't.
- **propose / respond** — structured plan proposals, review requests, and
  subtask delegations (`--to` an agent or `#swarm` broadcast); first
  responder wins.
- **conflicts** — check the paths you're about to touch against other
  agents' active locks *before* the work collides in a merge.

```bash
python -m swarm.agentgit coord claim beads-042        # agent = $SESSION_ID
python -m swarm.agentgit coord lock swarm/core --reason "payoff refactor"
python -m swarm.agentgit coord conflicts swarm/core/proxy.py   # exit 1 on conflict
python -m swarm.agentgit coord propose "plan: split epic" --kind plan --to session-2
python -m swarm.agentgit coord status
```

Claim/lock acquisition uses `BEGIN IMMEDIATE` transactions, so it is atomic
across processes. Merging compatible diffs is deliberately out of scope for
v1 — locks plus early conflict detection make merges boring.

## Multi-Agent Review

Instead of one end-of-task review, a panel of specialized reviewers examines
the same diff from different angles and the output is *synthesized*: humans
see where reviewers disagree, which findings are blocking, and which spots
are high-risk or low-confidence — not every line every reviewer looked at.

```bash
python -m swarm.agentgit review              # text synthesis, exit 1 on block
python -m swarm.agentgit review --json       # machine-readable synthesis
python -m swarm.agentgit attest ... --review # record panel outcome in bundle
```

The v1 panel ships the boring-valuable reviewers:

- **test-coverage** — warns on source changes with no test changes
  (confidence scales with diff size); blocks test deletions.
- **dependency** — warns on any manifest/lockfile change; blocks
  lockfile-only changes (dependency graph moved without declared intent —
  the classic supply-chain smell).
- **security** — blocks sensitive paths (`.env`, keys, CI workflows) and
  secret-looking added lines (private-key material, AWS keys, hardcoded
  credentials); warns on risky calls (`shell=True`, `eval`/`exec`,
  `pickle.load`, `os.system`). Content rules only see lines *added* by the
  diff, so pre-existing code is never attributed to the change.

Reviewers are deterministic — pure functions of the diff — so panel verdicts
are reproducible facts. With `attest --review`, per-reviewer reports plus the
panel synthesis land in the bundle's `provenance.reviews`, inside the signed
payload: forging a reviewer's decision after the fact fails `verify`.

Custom reviewers subclass `swarm.agentgit.Reviewer` and return
`ReviewFinding`s; pass them to `run_review_panel(repo, reviewers=[...])`.

## Conditional Policy & CI Gate

Beyond the fixed limits (allowed/denied paths, file/line caps, required checks),
a policy may carry **conditional rules** — `when <condition> then <action>`:

```yaml
rules:
  - id: deps-need-supply-chain-scan
    when: {dependency_changed: true}
    action: require_check
    check: supply-chain-scan
  - id: auth-needs-security-review
    when: {paths_match: ["*auth*", "*security*"]}
    action: require_review
  - id: tests-must-pass
    when: {check_failed: pytest}
    action: deny
```

Conditions (ANDed): `paths_match` (fnmatch globs), `dependency_changed`,
`added_lines_gt`, `changed_files_gt`, `check_failed`, `check_passed`. Actions:
`deny`, `require_check` (needs `check:`), `require_review`. A firing blocking
rule **passes if a human override names its id** in `provenance.overrides`
(`{rule: <id>, by: ..., reason: ...}`) — this is the "block unless override"
escape hatch. `severity: warning` rules surface but never block.

Rules evaluate at attest time (folded into the signed bundle `decisions`), so
`verify` already enforces them. They also run as a **CI gate** against an
*already-attested* bundle, using a policy CI/the org controls — judging what the
agent actually did against what's allowed, independent of the policy the agent
self-attested with:

```bash
python -m swarm.agentgit gate \
  --bundle .agentgit/provenance.json \
  --policy .github/agentgit.policy.yaml   # exits non-zero on any blocking rule
```

`gate` first **verifies the bundle's signature** (failing closed on tampering or
malformed input), then evaluates the policy against the bundle's recorded facts
(changed files, totals, dependency changes) — so a stricter CI policy catches
violations even if the agent attested against a lax one. Dependency facts are
derived from the **signed diff** (`git.changed_files`), not the `provenance`
block, so `dependency_changed` rules fire even on older `v0` bundles where
provenance is unsigned. The gate also **requires an explicit signing key**
(`--signing-key` or `AGENTGIT_SIGNING_KEY`) and fails closed if neither is set —
it never falls back to the public dev key, so a misconfigured CI job can't accept
a dev-key-signed bundle as authentic. Crucially, two agent-supplied fields are
**not trusted** at the gate:

- `provenance.overrides` — an agent could otherwise pre-approve the rule meant
  to catch it. A CI override must be supplied explicitly with
  `--override <rule-id>` from a CI-controlled source.
- `checks` — the agent authored these, so a check-based rule (e.g.
  `tests-must-pass`: `when: {check_failed: pytest} then deny`) would be defeated
  by an agent self-attesting `checks={"pytest": true}`. At gate time, supply the
  CI-authoritative result with `--check <name=pass|fail>`; unsupplied checks
  **fail closed** (the rule blocks until CI vouches for the result).

```bash
python -m swarm.agentgit gate \
  --bundle .agentgit/provenance.json \
  --policy .github/agentgit.policy.yaml \
  --signing-key "$AGENTGIT_SIGNING_KEY" \
  --check pytest=pass \
  --override deps-need-supply-chain-scan
```

This repo now runs that gate in its own CI: the `agentgit-gate` job in
`.github/workflows/ci.yml` gates every provenance bundle committed under
`.agentgit/` against `.github/agentgit.policy.yaml` and feeds the required
`quality-gate` check. The pytest result is wired from the CI `test` job
(`--check pytest=pass|fail`), never read from the bundle; with no committed
bundles the job is a no-op pass. Verdict parity between attest time and gate
time is pinned by the `test_attest_and_gate_verdicts_match_*` tests in
`tests/test_agentgit_policy_engine.py` — an agent cannot see a different
verdict locally than CI enforces, given the same policy and trusted inputs.

## Worktree Loop

AgentGit also plugs into the worktree sandbox bridge:

```bash
python -m swarm.bridges.worktree create codex
python -m swarm.bridges.worktree exec codex -- python -m pytest tests/test_agentgit.py
python -m swarm.bridges.worktree attest codex \
  --task issue-123 \
  --policy examples/agentgit_policy.yaml \
  --check pytest=pass \
  --output .agentgit/provenance.json
python -m swarm.agentgit verify .agentgit/provenance.json
```

That gives the first complete local loop:

```text
delegate task -> isolate worktree -> execute checks -> attest diff -> verify bundle
```

## Cryptographic Identity & Delegation

The MVP signs bundles with a shared HMAC key, which proves a bundle was sealed
by *someone holding the key* but not *which agent* produced the change. The
`swarm.agentgit.identity` module adds verifiable identity with Ed25519
(asymmetric) signatures.

- **`AgentKeypair`** — an Ed25519 keypair. Its `did` is `did:key:ed25519:<hex>`,
  so the public key is embedded in the identifier and verifiers need no key
  registry.
- **`AgentIdentity`** — the agent's DID plus owner/org and model/runtime/version
  provenance and its `allowed_tools`.
- **`DelegationChain`** — an ordered, individually-signed `human -> org -> agent`
  chain. `verify()` checks every link's signature, that the chain is connected
  (each link's subject issues the next), that permissions only *narrow* down the
  chain, and that no link has expired.

When `identity` + `agent_keypair` (and optionally `delegation`) are passed to
`build_bundle`, the agent's key signs the receipt `payload_hash`, binding a
verifiable identity to that exact diff:

```python
from swarm.agentgit import AgentIdentity, AgentKeypair, build_bundle, sign_link, DelegationChain

org = AgentKeypair.generate()
agent = AgentKeypair.generate()
identity = AgentIdentity.for_keypair(agent, owner="alice", org="acme", allowed_tools=["read", "test"])
chain = DelegationChain(links=[sign_link(org, subject_did=agent.did, permissions=["read", "test", "open_pr"])])

bundle = build_bundle(..., identity=identity, agent_keypair=agent, delegation=chain)
```

`verify_bundle` then additionally checks the identity signature, the delegation
chain, that the chain's final subject is the signing identity, and that the
identity's `allowed_tools` stay within the delegated grant. Bundles built
without identity blocks still verify (backward compatible).

> The CLI (`attest`/`verify`) does not yet manage keypairs — that key-storage
> surface is tracked as a follow-up. Today identity is wired through the library
> API.

## Capability Enforcement

Verifying a delegation chain is still *advisory* — it proves what an agent was
allowed to do without stopping it from doing more. `swarm.agentgit.capabilities`
turns a verified chain into the command allowlist the worktree sandbox
*physically* enforces, closing the loop identity → delegation → enforcement.

`CAPABILITY_COMMANDS` maps permission tokens to the command binaries they
authorize (`read` → `ls/cat/grep/…`, `test` → `pytest/python`, `vcs` → `git`,
etc.). `enforced_allowlist_for_chain` verifies the chain and returns the granted
commands — or, on any verification failure, an empty allowlist (**deny by
default**).

```python
from swarm.bridges.worktree.config import WorktreeConfig
from swarm.bridges.worktree.policy import WorktreePolicy

policy = WorktreePolicy(WorktreeConfig())
ok, errors = policy.apply_delegation("codex", chain, expected_subject_did=agent.did)
# Now only the delegated capabilities execute:
policy.evaluate_command("codex", ["pytest", "tests/"]).allowed   # True  (test granted)
policy.evaluate_command("codex", ["git", "status"]).allowed       # False (vcs not granted)
```

An invalid, expired, or over-scoped chain installs an empty allowlist, so the
agent can run *nothing* until a valid delegation is supplied. Unconditional
hard-blocks (ssh/scp, `git push|fetch|pull|clone`) still apply regardless of
what was delegated.

This slice enforces **command** capabilities (which binary may start).

## OS-Level Isolation

Gating *which* binary starts is not enough: `subprocess.run(cmd, cwd=sandbox)`
runs an ordinary child process, so an allowlisted `python` can still write
anywhere and open sockets. `swarm.bridges.worktree.sandbox_launch` wraps the
executed command in a real OS confinement that limits **filesystem writes to
the sandbox** and **blocks network egress**:

- macOS → `sandbox-exec` with an SBPL profile (deny `file-write*` outside the
  sandbox + temp, deny `network*`).
- Linux → `bwrap` (read-only root, read-write bind on only the sandbox subtree,
  private empty network namespace via `--unshare-net`).

Opt-in via `WorktreeConfig`:

```python
WorktreeConfig(os_isolation_enabled=True)          # wrap when a backend exists
WorktreeConfig(os_isolation_enabled=True, require_os_isolation=True)  # fail-closed
```

When enabled but no backend is available (e.g. CI/Linux without `bwrap`), the
command still runs and `CommandResult.isolation` is recorded as `"none"` — the
isolation status is **never silent**. Set `require_os_isolation=True` to instead
**deny** execution when no backend exists. Reads are not restricted in this
slice (interpreters need their stdlib); a stronger read-confining jail and
short-lived scoped git push tokens remain follow-ups.
