# SWARM Research Foundry — 90-day execution plan

Date: 2026-09-06. Horizon: 2026-12-05. Scope confirmed by user: SWARM-led research.
Status: original planning snapshot grounded in local HEAD `7dcb672b`. Execution has begun; see [Wave 1 results](WAVE1.md) for executed checks, fixes, and remaining blockers. Dates below are targets, not commitments from existing agents.

## Mission

Build a continuously operating research swarm that discovers multi-agent failure mechanisms, tests governance against adaptive responses, and produces independently reproducible evidence. The flagship output is a **Governance Failure Atlas**: a versioned benchmark and interactive evidence map showing where useful cooperation becomes indistinguishable from prohibited coordination, where monitoring fails, and which interventions preserve useful work.

The ambitious result is one coherent empirical program connecting **observability, incentives, authority, and verification capacity**. Success includes well-established impossibility boundaries and negative results. Papers, software, and demos are views of the same versioned evidence, not separate workstreams inventing their own claims.

## What already exists, and what needs reconciliation

Evidence inspected: current local git history, source files, preregistrations, result notes, and a fresh `bd list --status all --limit 0 --json` snapshot. This is a local working-state assessment; remote CI, provider availability, and current compute prices were not independently checked.

- Wiki Monte Carlo: documented 24×30 pilot runs, 6×200 detection confirmation runs, and 10×200 moderation confirmation runs. Confirmation folders exist locally. The results note still calls for final contrast/manifest review. Synthetic relocation cannot establish historical relocation.
- Content-free detection: implemented output-correlation and re-derivation signals. The current simple probe identifies copying well; it does not establish prohibited coordination. Matched permission-only labels are deliberately behaviorally indistinguishable.
- Calibration/adaptation: existing preregistrations, pilots, judge pipeline, and join-schema work. Current Arm B completion must be checked against artifacts: July task notes and June protocol headers are not a reliable current run inventory.
- Governance economics: existing decomposition, attacker-economics, and real StakingLever work. The real-lever note corrects the abstract model: recoverable stake alone does not impose the assumed sunk cost. Preserve this correction throughout the program.
- Agent execution: AgentGit, signed receipts, authority bounds, claim protocol, and dependency-based dispatch exist. These are research support infrastructure, not proof that the complete workflow operates correctly.
- Tracker reconciliation matters: `zsof` and `jxyi` remain open P1 bugs, while current source includes context/nonce binding and per-signer Ed25519 receipts. Verify at HEAD before allocating implementation work. `njvs`, `7bwz`, and `zxkw` remain in progress despite completion-like notes.
- `xwhu` reports a current P1 knowledge-graph CI failure affecting research pages. Its status comes from the tracker, not a CI execution in this session.
- The documented coordination query fails locally because `runs/runs.db` has no `agent_messages` table. Identify the authoritative database and migration path before dispatching; do not bootstrap a second coordination system by accident.

Snapshot: **72 open, 9 in progress, 431 closed; 3 open P1 bugs**. Existing dirty `.beads/interactions.jsonl` and untracked `.beads/hooks/` belong to other work and must be preserved.

## One master dependency plan

The live graph's longest recorded blocking path is seven nodes:

`5bpg → p6bz → h5bo → u4av → 53rj → gajd → xvb3`

That is judge anchoring → agreement → frozen join → external-quality anchor → adaptive generation → LLM corroboration → adaptive-study completion. Epic nodes represent completion dependencies, not separate experiments. Missing edges and unrecorded operator gates can change the true critical path.

| Root | Immediate tasks becoming unblocked if it closes | All downstream live nodes | Interpretation |
|---|---:|---:|---|
| `5bpg` | 1 | 10 | Main shared scientific bottleneck; already in progress |
| `h5bo` | 2 | 8 | Valuable, but still depends on B/C |
| `a2il` | 4 | 5 | Best independent new mechanism branch |
| `vv3j.1` | 2 | 3 | Enables capacity and payoff sweeps; already in progress |

These are dependency counts, not estimates of tasks that will finish. Take over no in-progress task without an explicit handoff. Recompute readiness at claim time.

## Research portfolio

### A. Cooperation, collusion, and limits of observation

**Question:** What additional evidence distinguishes authorized sharing, independent convergence, prohibited coordination, and adversarial contamination?

Build on the wiki protocol, `vv3j`, `vv3j.1/.3/.4/.6`, `y2t2`, and scanner provenance tasks `3vrx`, `dfxk`, `10ta`.

1. Independently reproduce the existing synthetic confirmation summaries and audit threshold/seed freezes.
2. Separate permission-only controls from mechanisms with actual behavioral differences. Freeze definitions of copying, coordination, prohibition, and harm separately.
3. Complete the overt/covert carrier implementation, then test capacity × defender visibility × task overlap. Follow the existing 10-seed capacity grid as exploratory; select confirmation contrasts before fresh data collection.
4. Add poisoning and natural-language variants as separately labeled extensions. Include sanctioned high-volume collaboration and hub-heavy honest communities as negatives.
5. Test both permission/context evidence and behavioral screening. Abstention under inadequate coverage is a valid output. A detector without access to permission evidence cannot be credited with inferring an otherwise unobservable permission label.

**Deliverable:** a cooperation/coordination benchmark with observable-feature boundaries, provenance, coverage masks, latency, run-level false positives, recall, and legitimate-work cost. Preserve both routes: behavioral evidence and authority/provenance evidence.

**Failure gate:** if a frozen detector exceeds its declared false-positive budget, report the failure; do not retune on confirmation data. No historical absence claim beyond scanner coverage.

### B. Governance that survives adaptation

**Question:** When does governance improve external quality rather than reject more work or reward proxy manipulation?

Complete `5bpg → p6bz → h5bo → u4av`, then `lu3o`, `53rj`, `170j`, `mcrn`, and `gajd`. Resolve `2qgp` before using A-Evolve selection results as evidence.

1. Inventory real and mock judge outputs, resolve rubric/version provenance, and finish the locked calibration arm. Do not silently replace rubric v1 with a better-performing later rubric.
2. Apply the registered agreement escalation rule. Agreement is reliability evidence, not proof that judges are correct or ungameable; report shared bias and held-out checks.
3. Freeze the join schema and validate lineage/missingness. Add attempted-versus-accepted outcomes as a documented extension where the existing sample is accepted-only.
4. Execute the registered static, acceptance-only, generation-adaptive, and fully adaptive comparisons. Retain the registered five-seed result as such; design any larger holdout replication prospectively, with its own seed list and amendment.
5. Measure external quality, welfare, throughput, rejection, proxy gap, and uncertainty together. Exercise action-to-observable sensitivity before adaptive search; a flat fitness landscape invalidates search conclusions.

**Deliverable:** audited static-versus-adaptive welfare/quality frontiers and a decomposition of improvement, filtering, and gaming.

**Failure gate:** if calibration remains unreliable, finish the calibration failure report and continue synthetic mechanism work; do not relabel the proxy as external truth.

### C. Prevention, detection, and the economics of evasion

**Question:** Which combination of authority restrictions, monitoring, and at-risk capital remains useful against a defender-aware attacker?

Reproduce and reconcile `njvs`, `7bwz`, `zxkw`; extend through `vwj3`, `iujo`, and `0jxz` after the relevant identity fixes pass regression checks.

Sweep attacker decomposition, identity budget, liquidity, detection probability, slashing, monitoring window, and authority scope. Compare monitoring alone, recoverable bonds alone, their combination, and capability preconditions. Match legitimate-work opportunities and report restricted task coverage.

**Deliverable:** a prevention/detection/cost frontier, including explicit zero-detection regions and authorized-but-harmful actions that capability checks cannot prevent.

**Failure gate:** no universal minimum-stake prescription from the abstract `0.52V` result. Capital lockup, permanent loss, and slashing risk are different mechanisms. Provenance signatures establish provenance, not correctness.

### D. Research integrity under a flood of claims

**Question:** Can verification keep up with agent-generated knowledge, and which acceptance mechanisms prevent contamination without suppressing useful discoveries?

Start `a2il`, then `u8x6`, `oa5h`, `sx6c`, `77ao`, and `cx83` in dependency order. Model retrieval checks, costly novel claims, and mechanically checkable claims separately. Sweep claim arrival versus verification service capacity, correlated reviewers, citation depth, and strategic corroboration.

**Deliverable:** a claim-ledger benchmark and a verification-capacity frontier: false acceptance, useful discoveries, backlog latency, contamination cascades, and verification cost. Compare default-unverified acceptance rules, independent corroboration, and claim bonds.

Dogfood the ledger on this program's own claims, while keeping the empirical evaluation separate from the system producing those claims. The ledger should make unsupported conclusions harder to publish, not merely count more documents.

### E. Transfer and scale

After A–D pass their internal gates, port **one** mechanism to an independent environment. Evaluate existing Miroshark integration first; consider `bv3p → se2m` only after installation and trial gates. Agency-OS and Aeon are optional execution adapters, not dependencies for the first scientific results.

Scale populations geometrically: 10 → 30 → 100 → 300 → 1,000 simulated agents, conditional on runtime and signal checks. Population size is not the number of independent replicates. Add bounded real-model populations only after a measured pilot establishes cost and a budget is set.

**Deliverable:** one attempted independent replication, including failed transfer if that is what occurs. Do not promise a successful generalization result.

## Agent organization and concurrency

Use the existing six personas: Scenario Architect, Mechanism Designer, Adversary Designer, Auditor, Reproducibility Sheriff, Research Scout. Portfolio slots are assignments of these roles, not new installed personas. Any new role/model/tool must pass the repository's trial task before default adoption.

**Initial four simultaneous slots:** (1) Sheriff clears verification/dispatch blockers, (2) Auditor owns calibration inventory and quality review, (3) Architect advances the covert-channel scenario with its current owner, (4) Mechanism Designer designs the claim-ledger branch. Rotate the Adversary Designer into each completed mechanism before confirmation. Research Scout performs a bounded transfer reconnaissance only when a concrete interface is needed.

**Target mature fleet:** 12 role slots across four pods, with at most 8 producing artifacts and 4 reviewing, reproducing, or integrating. This is future capacity, not a claim about this session's four-slot runtime. Grow 4→8→12 only when measured review capacity, claim reliability, and budget support it. Keep at least 25% capacity on reproduction/consolidation; stop expansion if review backlog exceeds two working days.

Every work packet contains: hypothesis, existing bead, owner, due date, exact allowed files, dependencies outside its epic, frozen inputs, artifact paths, executable acceptance gate, insufficient outcomes, budget, and named next reviewer. Claims use the canonical atomic claim path and isolated worktrees. File overlap adds a serialization constraint even when the bead graph is disjoint.

No producer validates its own research claim. An Auditor's implementation is reviewed by another Auditor instance; a Sheriff's reproducibility artifact is reproduced by a different instance. A planner cannot mark these gates passed. G0 mechanical and G1 replayable tasks may use aggressive search; G2 interpretations get one conservative author and a named independent reviewer.

## Milestones, owners, and target dates

Owners below are accountable role slots, not assignments sent to existing agents.

| Date | Deliverable | Accountable role | Acceptance |
|---|---|---|---|
| Sep 8 | Current-state and CI/coordination reconciliation | Sheriff | Executed gates, authoritative task/run inventory, stale-status discrepancies resolved with evidence |
| Sep 13 | Four one-page protocols and artifact contract | Architect | Owners, dependencies, nulls, frozen/adaptable fields, review sign-offs recorded |
| Sep 20 | Wiki reproduction and calibration disposition | Auditor | Re-derived confirmation summaries; usable anchor or explicit failed-agreement decision |
| Oct 6 | Benchmark v0.1: A–D runnable pilots | Sheriff | Fresh-checkout reproduction of each included family; exclusions explicit |
| Nov 5 | Frozen confirmation suites and draft result bundles | Auditor | Holdouts untouched until freeze; gate execution logs; claim-to-evidence mapping |
| Dec 5 | Governance Failure Atlas v1 and research release candidate | Sheriff | Independent reproduction, one transfer attempt, cost ledger, limitations, and publication-ready drafts |

The plan aims at three substantial manuscripts: observation/coordination; adaptive governance and evasion; verification-limited agent knowledge. Merge or narrow them if evidence does not justify three independent contributions. Release targets are artifacts, not promised positive findings or acceptances.

## First 72 hours

1. Sheriff: verify `xwhu`, `jxyi`, `zsof` at HEAD; fix only surviving defects. Identify the correct coordination store. Confirm existing owners and real run artifacts for in-progress tasks.
2. Auditor: resolve `5bpg`'s actual data/rubric/budget state; produce the shortest actionable path through C and D. Audit existing wiki confirmation manifests and the staking correction.
3. Architect: coordinate with `vv3j.1` owner; produce a testable handoff for the two dependent sweeps. Keep code ownership in `memory_handler.py` exclusive.
4. Mechanism Designer: specify `a2il`'s arrival/service mechanisms, observables, and negative cases; hand the minimal scenario to the Architect when capacity frees.
5. Planner/Sheriff: link this plan into the plan-of-record process tracked by `ja4u` after its owner review; reuse existing beads and add missing edges before creating new epics.

## Experiment contract, budget, and stopping rules

Every run bundle must include code SHA plus dirty patch if any, config hash, dataset/source hash, population and seed, dependency versions, rubric/model identifiers, observer-access contract, event history, per-run metrics, aggregate script, actual invocation/output, and cost. Separate claimed gates from executed gates. Signed receipts cannot substitute for test output or experimental evidence.

Report independent population/seed units, paired interventions with audited random-stream coupling, exclusions, missing data, uncertainty, effect sizes, and declared multiple-testing families. Use existing frozen protocols unchanged unless a prospective amendment explicitly supersedes them. Never treat edits, agents, or agent pairs as independent replicates by default.

**Proposed resource allocation:** 30% A, 30% B, 15% C, 15% D, 10% E. These are shares, not spending authorization. Within each track reserve at least 25% for adversarial review and reproduction. Real-model cost estimate = sampled input/output tokens × then-current provider rates × planned repetitions, plus bounded retries; measure it in a small pilot before setting the run ceiling. Until there is a funded ceiling, queue paid experiments and continue local work.

Stop or redirect when: mechanism sensitivity fails; confirmation provenance cannot be reconstructed; independent review finds label leakage; the anchor fails its registered rule; retries repeat the same failure without a new hypothesis; compute exceeds the declared cap; or verification backlog exceeds two working days. Preserve failed results. A blocked route reopens only with a materially new mechanism or resolved dependency.

## Definition of success

By day 90, another researcher can take the release bundle, reproduce the main figures, identify exactly which claims are synthetic versus independently transferred, and see where every proposed governance mechanism fails. Operational metrics are reproduced claims per unit cost, independent-review turnaround, unresolved provenance gaps, and useful work preserved at a declared false-positive budget—not number of agents, commits, or generated papers.

The stretch outcome is a research engine that uses new counterexamples to propose the next bounded experiment automatically, while protocol changes and claim acceptance remain explicit, independently reviewed events.

## Local evidence map

- [Current research thread](../../.letta/memory/threads/current.md)
- [Dispatch rules](../../.claude/commands/bv-dispatch.md) and [dispatch retrospective](../../docs/research/dispatch-retro-2026-07-19.md)
- [Wiki protocol](../../docs/research/wiki-monte-carlo-plan.md) and [results](../../docs/research/wiki-monte-carlo-results.md)
- [Content-free detectors](../../docs/research/content-free-discriminators.md)
- [Calibration preregistration](../../docs/research/calibration-prereg.md) and [adaptive preregistration](../../docs/research/adaptive-agents-prereg.md)
- [Flat fitness landscape](../../docs/research/aevolve-flat-landscape.md)
- [Long-horizon governance corrections](../../docs/research/long-horizon-safety-lessons.md)
- [Authority implementation](../../swarm/agentgit/identity.py) and [receipt signer](../../swarm/attestation/signer.py)
- [Compact live task snapshot](task-snapshot.json), taken during this session; re-query before executing.
