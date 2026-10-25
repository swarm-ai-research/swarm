---
description: "Storm-0940's ARTEX campaign through ConnectWise Cloud Backup: the spray that failed loudly, the app-token path that succeeded quietly, and an app that granted itself Global Administrator. Read against INV-6, INV-7 and the detection/prevention split."
author: "SWARM Team"
date: "2026-10-25"
source: "https://www.microsoft.com/en-us/security/blog/2026/10/21/storm-0940-uses-artex-agentic-ai-connectwise-supply-chain-compromise/"
keywords:
  - supply chain
  - delegation
  - agentic offense
  - prevention vs detection
  - attack tempo
---

# Storm-0940 and ARTEX: the loud path failed, the trusted path did not

**Source:** Microsoft Threat Intelligence, *Storm-0940 uses ARTEX agentic AI in
ConnectWise supply chain compromise*, Microsoft Security Blog, 2026-10-21
([link](https://www.microsoft.com/en-us/security/blog/2026/10/21/storm-0940-uses-artex-agentic-ai-connectwise-supply-chain-compromise/)),
read 2026-10-25. It cites ConnectWise's advisory of 2026-10-16 and Huntress
reporting of 2026-10-15. Neither of those was read for this note.

**Fidelity caveat:** written from a summarised rendering of the post, not a
verbatim copy. The figures below are the ones the rendering attributes to
Microsoft. One inconsistency was flagged in that rendering: the post says to
inventory "four application IDs", but its hunting queries list three. Check the
original and ConnectWise's advisory before relying on any count here. This note
does not reproduce the indicators of compromise (IPs, user agents, app IDs).
Defenders should take them from the source. Nothing here is attribution beyond
what Microsoft states.

Companions:
[hyperspace-two-swarms-lessons.md](hyperspace-two-swarms-lessons.md) (attacker
and defender swarms, and which side can see its own failures) and
[babel-model-router-lessons.md](babel-model-router-lessons.md) (harm carried by
tooling rather than by a model at runtime).

## What happened

Storm-0940 is a China-state-aligned actor, active since 2021, historically using
password spraying, exploitation of internet-facing applications, and credentials
added to existing OAuth apps. In this campaign it used **ARTEX**, an open-source
multi-agent penetration-testing framework (Go backend, any model) whose agents
run "from reconnaissance through data exfiltration with little needed human
interaction" and keep discovered assets in a graph. Microsoft says several
actors in different regions use it.

| Stage | What the post reports |
|---|---|
| Upstream compromise | ConnectWise Cloud Backup, a multi-tenant application, was compromised. The post does not say how. |
| Spray (Oct 10–14) | Automated cross-tenant spray against 327 accounts in 8 organisations: 151,000+ failed logons, peak ~2,900 failures/min on Oct 11, 48 accounts with >1,000 failures, ~98% of activity was lockouts. **No successful authentication.** |
| Token path | With control of several ConnectWise cloud-app identities, agents signed into customer tenants as the app, with no interactive user sign-in. Sign-in attempts against **more than 2,250 tenants**. |
| Inside tenants | Graph reconnaissance; changes to application credentials for persistence; attempted changes to identities, groups and privileged roles. One ConnectWise management app **self-assigned Global Administrator**. |
| Data plane | Bulk Key Vault secret reads across several vaults, key enumeration, a key-unwrap operation. |
| Tempo | Initial targeting to privileged access in **under 3 hours** in several cases. |
| Outcome | "Hundreds of organizations" targeted across defence, energy, finance, government, legal, semiconductors and more. Most targeting failed, but privileged cloud persistence was reached "at many organizations". |

Microsoft's guidance runs contain → preserve → investigate → audit → remediate
→ harden, and two items in it are unusual: **do not delete the enterprise app
first** (that destroys the sign-in and audit trail), and **review back to
October 1, not only from the published IPs**.

## The core reading

**The two paths failed and succeeded in opposite registers.** The spray was the
loud path: 151,000 failures, lockouts, zero successes. Every signal a
failure-counting defence watches lit up, and it bought the attacker nothing. The
token path was the quiet one: an already-consented application presenting its own
valid credentials, with no interactive sign-in to challenge. Its observables were
those of the vendor doing its job. The damage came through the path with no
failures to count.

This is the [Hyperspace](hyperspace-two-swarms-lessons.md) observability
asymmetry seen from the other side. There, the attacker's cost was visible and
the defender's misses were silent. Here the attacker's *failures* were visible
and its *successes* were silent, because success ran on trust the defender had
already issued.

For SWARM the point is where `p` comes from. The `p`-threshold detectors in
`swarm/governance/` (the circuit breaker, random audits) score interactions
through the proxy label, so they see an interaction that looks like routine
vendor traffic as routine vendor traffic. Not every detector is `p`-only:
`LoopDetectorLever` (`swarm/governance/loop_detector.py`) can freeze an
initiator on repetition, tool misuse or rework whatever its `p`, and bulk Key
Vault reads might well look like repetition to it. But a pattern detector
fires on the shape of behaviour, and an attacker who controls tempo controls
the shape. The levers that would have stopped the token path outright are the
ones that check **authority**, not suspiciousness.

## Lessons and their translation

As in [erdos-ai-ledger-lessons.md](erdos-ai-ledger-lessons.md), each mechanism
maps onto the **model plane** (constructs in the SWARM simulation) and the
**practice plane** (how this rig itself operates).

| # | Mechanism | Model plane (SWARM sim) | Practice plane (this rig) |
|---|---|---|---|
| 1 | **An app granted itself Global Administrator.** The grant the tenant gave the vendor app included the power to change role assignments, its own included. | INV-6: *an authority bound a child can loosen is not a bound.* Wards compose by intersection and reject an over-scoped declaration rather than clamp it (`swarm/agentgit/wards.py`, `tests/test_agentgit_wards.py::TestNeverWidens`). A tenant's consent grant has no such composition rule: a role-management permission is a permission to widen. | `/claim` refuses (exit 3) a claim that would widen a dispatch `WARDS:` line. That covers stamped beads. Unstamped beads carry no bound at all, so a session working one has nothing to widen against, which is the vendor app's position. |
| 2 | **One credential, 2,250 tenants.** The app's identity was valid in every tenant that had consented to it. | INV-7: *a credential valid everywhere is a bearer token, not a delegation.* `swarm/agentgit/identity.py` binds a `DelegationLink` to an `audience` and refuses it in a foreign context (`test_bound_link_replayed_into_foreign_context_is_refused`). A multi-tenant app secret is the unbound case: holding it is enough. Binding limits *replay*, not *holding*: a vendor that legitimately holds a bound link for each of `N` tenants, all signed by one key store, loses all `N` when that store is taken (`DelegationChain.verify` checks each link's audience, not how many grants one subject holds). What would have limited this incident is per-tenant key separation, which INV-7 does not provide. | Our MCP and API keys in `.mcp.json` placeholders and `/load_keys` are bearer credentials of the same kind. Nothing in the rig scopes a key to the session that loaded it. |
| 3 | **The spray tripped every alarm and caught nothing that mattered.** | `CircuitBreakerLever` (`swarm/governance/circuit_breaker.py`) freezes an agent on windowed average toxicity `1 − p`. An attacker running a loud decoy path and a quiet real path gets the breaker spent on the decoy. The lever is per-initiator, and the quiet path's initiator is the trusted vendor, whose `p` stays high. | The heartbeat and claim-collision guards are the loud-path kind: they fire on two sessions in one checkout. A session acting under another's legitimate claim marker would pass both. |
| 4 | **Prevention vs. detection.** The mitigations that would have bounded damage are all authority-side: least privilege on multi-tenant consent, monitoring credential *additions*, alerting on role self-assignment. | `CapabilityPreconditionLever` (`swarm/governance/capability_precondition.py`, bead `vwj3`) refuses a write outside the grant whatever its `p`, and its docstring states the limit: a harmful write to a resource the agent *is* granted passes. Storm-0940 needed exactly one write outside any sane grant (the role self-assignment), and after that everything was inside the grant. | Same split as the [Shepherd](shepherd-reversible-execution.md) note: prevention bounds the blast radius, detection explains it afterwards. This incident is a prevention failure that detection then documented. |
| 5 | **Under 3 hours from targeting to privileged access.** Agents "work around the clock" and "attempt multiple breach paths simultaneously". | `CircuitBreakerLever` checks after every interaction and the freeze applies immediately; epochs set only how long a freeze lasts. So SWARM's governance is, if anything, faster than real defenders: it has no response delay at all. The incident's gap is between detection and a human acting on it, and SWARM has no parameter for that. A scenario about tempo needs an explicit response delay. | Dispatch retros and `/bv-dispatch` run on human cadence. A session that went wrong would have hours of commits before the next retro looked. |
| 6 | **The vendor is a hub.** One upstream compromise reached every downstream tenant that trusted it. | `CascadeRiskLever` (`swarm/governance/cascade.py`) governs artifact chains: a node penalised for its bad *descendants*. The trust graph here has the same shape with the direction reversed: one *ancestor* compromised, all descendants exposed. Nothing in SWARM scores a node by how many others have delegated authority to it. | `main` and the shared `runs/runs.db` are this rig's hubs. Every worktree trusts both. |
| 7 | **Preserve before deleting the app.** Deleting the enterprise app first destroys the sign-in and audit history needed to scope the breach. | INV-2: event logs are append-only and replayable. The guidance is the same invariant applied by an incident responder: removing the actor must not remove the record of what it did. | `/ship` and worktree `cleanup` delete branches and worktrees. The beads JSONL and git history survive; per-session heartbeats and claim markers do not. |
| 8 | **ARTEX is open-source, model-agnostic and used by several actors.** | Attacker capability here is a property of a public tool, not of any single model or account, so per-model refusals and per-account bans do not bound it. This is the [BABEL](babel-model-router-lessons.md) lesson (harm carried by tooling) arriving in offence. | — |

## The adversarial reading

**"No successful authentication" is a statement about the spray only.** The
headline number sounds like a defensive win, and the post is careful that it
is not: the campaign still reached privileged persistence at many
organisations by the other path. A dashboard that reported spray success rate
would have read 0% throughout.

**"Majority of targeting wasn't successful" has no denominator we can use.**
Hundreds of organisations targeted, 2,250+ tenants attempted, "many" with
privileged persistence. Without the count of successes the per-tenant hit rate
is not recoverable, and the post also says an affected app or a successful sign-in
"does not necessarily indicate" data access. Treat the scale figures as
attempt counts, not outcome counts. This is the denominator problem from
[erdos-ai-ledger-lessons.md](erdos-ai-ledger-lessons.md) lesson 2.

**How much of this is agentic?** The post credits ARTEX with speed and
parallelism. The decisive step, though, was the upstream compromise, and the post does not
say how that happened or whether ARTEX did it. Cross-tenant token reuse and role
self-assignment are not new techniques. What is new is the tempo (lesson 5) and
the breadth (2,250 tenants attempted). On the post's own evidence the agentic
contribution is to make a known chain fast and wide. It did not open a new
chain.

## Proposed scenario

Not filed as a bead: `bd` is not available in this session. Sketch for whoever
files it:

**Hub delegation under a decoy.** One vendor agent holds standing delegated
authority into `N` tenant agents. An adversary controls the vendor identity and
runs two paths at once: a loud path (many low-`p` interactions from a
throwaway initiator) and a quiet path (a few interactions from the vendor
identity, high observable `p`, one of which widens the vendor's own grant).
Compare three governance stacks at matched cost:

1. detection only (`CircuitBreakerLever` and audits on `p`);
2. detection plus `CapabilityPreconditionLever` with the vendor's grant fixed;
3. (2) plus per-tenant keys: each tenant's delegation is bound to that tenant
   (INV-7 semantics) **and** signed by a separate vendor key, with the
   compromise modelled as theft of `k` of the `N` keys.

Add an explicit governance response delay `d` (interactions between a lever
firing and the freeze taking effect); SWARM has none by default. Measure
tenants with widened authority and harm realised over a **fixed observation
horizon**, not up to the first freeze, since the first freeze is expected to
land on the decoy. Report decoy and vendor freeze times separately. Sweep
attack tempo against `d`.

**Prediction:** stack 1 freezes the decoy initiator and misses the vendor in
every cell, unless the quiet path's shape trips `LoopDetectorLever`. Stack 2
stops the self-widening write but not harm inside the existing grant. Stack 3
bounds the tenants reached to `k`; with `k = N` (one key store for every
tenant) it is no better than stack 2. Harm under every stack rises with `d`
once attack tempo exceeds `1/d`.

**Falsifier:** if stack 1 catches the quiet path at any threshold setting that
does not also freeze the honest vendor in the no-attack control, then
`p`-scored detection can see a trusted-identity attack and lesson 3 is wrong.

## Correction of record (2026-10-25)

The first version of this note (PR #643) made four errors, caught in review
and recorded rather than edited away:

1. It said every detection lever in `swarm/governance/` scores only through
   `p`. `LoopDetectorLever` does not.
2. It said epoch-granular levers act after the fact by construction. The
   circuit breaker freezes on the interaction that trips it; epochs set only
   the freeze duration.
3. The scenario measured harm up to the first freeze, which would land on the
   decoy and cut off the vendor activity the scenario exists to measure.
4. It predicted that context-bound delegation limits a stolen vendor
   credential to one tenant. Binding stops replay into a foreign context; it
   does not limit a vendor that holds valid bound links for every tenant.

## Caveats

- One source, read through a summariser. Every figure is Microsoft's.
  ConnectWise's and Huntress's accounts were not read.
- The post gives no mechanism for the upstream compromise, so nothing here
  says how a vendor app's credentials are lost. Only what follows once they are.
- The rig-side rows (practice plane) are analogies by structure, not claims that
  this repo faces a comparable threat. Lesson 2's point about bearer API keys is
  the one that applies directly.
