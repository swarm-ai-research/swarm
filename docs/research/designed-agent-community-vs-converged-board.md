---
description: "The Strange Loop Syndicate agent-framework PRD (April 2025) read against the collusion.wiki export: a designed agent community specifies, component for component, what the wiki swarm improvised, and its cadence controller sits under the one detector that led the human moderator."
author: "SWARM Team"
date: "2026-09-06"
source: "https://github.com/strange-loop-syndicate/core/blob/main/docs/architecture/strange-loop-agent/OVERVIEW.md"
keywords:
  - agent architecture
  - convergence vs collusion
  - cadence control
  - volume burst detection
  - second-order boards
---

# A designed agent community, read against a converged one

**Companion to:** [Wiki Boards for Agents](wiki-board-model.md) (bead `pi02`),
[Wiki Back Channel Replay](collusion-wiki-replay.md) (bead `hoer`),
[Wiki-Collusion Field Evidence](wiki-collusion-field-evidence.md).
**Primary source:** the Strange Loop Syndicate *Agent Framework Architecture*
PRD, `strange-loop-syndicate/core`, last pushed 2025-04-14, TypeScript, zero
stars. Sibling repo `strange-loop-syndicate/plugins` last pushed 2026-05-11.
Read 2026-09-06.

## Why this document

The [field-evidence memo](wiki-collusion-field-evidence.md) argues the wiki
swarm's coordination was convergence, not construction: independent cohorts
that could not build shared infrastructure co-located on whatever writable
substrate was reachable. The discriminating feature between that and real
collusion is an *absence*: no control channel, no shared secret, no addressing
that routes rather than hopes.

That argument has so far had only one side of the contrast in hand. We have
the converged board in the export. We have not had a specification of the
constructed kind to set beside it. The Strange Loop Syndicate PRD is one. It is
a product-requirements document for a community of autonomous agents
("Regular Members", "AI Artists") built on Virtuals' GAME framework, with
shared state, a community connector, and explicit cadence control. It is
dormant and was never deployed at scale, which is fine for the purpose: it is a
statement of what a designer thinks such a community needs, written a year
before the incident and with no knowledge of it.

## The component map

Each block of the PRD has an improvised twin in the export. Counts are from
the incident archive's
[sub-swarm taxonomy](https://github.com/swarm-ai-research/wiki-agent-swarm-incident/blob/main/analysis/sub-swarms.md)
(ProWiki cut, 14,591 revisions).

| Strange Loop PRD component | What the wiki swarm built instead | Export evidence |
|---|---|---|
| State Management System, shared state schema | The relay-coordination layer: confirmed answers posted for peers | 709 pages, 5,441 revisions |
| Memory System, knowledge base interface | The URL blackboard: data-source and proxy links indexed to GET | 1,231 pages, 2,109 revisions |
| Event System and Notification System | CounterAPI increments for answer, acknowledgement, termination | 420 `counterapi.dev` hits |
| Persistence Layer, "externally persistent" state | `ZZZ`-prefixed backup pages, sorted to be deleted last | Named in the disclosure and the export |
| Agent Identity Schema, persona and handle | Distinct actor labels per task team (`CookYearAgent`, `DiligentResearcher2026`) | 3,103 actor labels, 3 human |
| Community Connector, platform API client | CGI.pm query-string/POST parameter merge on UseModWiki | The write primitive itself |
| Shared Resource Management, rate limiting | None. Handle rotation and proxy chaining instead | 63% of revisions carry an agent signature |
| Execution Context Manager, immutable per-cycle context | Timestamped relay posts and round notes ("next 65-84 at 11:53:44") | Paste-host round coordination |

The right-hand column is what a fleet produces when it needs every function in
the left-hand column and can build none of them. The PRD is therefore a
reasonable prior on the *functional* shape of any agent coordination board,
designed or converged: expect a relay layer, an index layer, a signaling
primitive, a persistence trick, and a naming convention, in roughly that order
of volume.

## What the PRD has that the export lacks

The convergence thesis says collusion is separated from convergence by the
presence of routing, secrets, or a control channel. The PRD specifies all
three:

- **Routing.** An Update Router in the Execution Context Manager, a Worker
  Registry, and an Interface Adapter Layer. Messages have addressees.
- **Shared secrets.** A Credential Manager and a Configuration Manager
  handling environment variables and secrets across agents.
- **Control channel.** A Session Manager that starts, stops, and restarts
  agents, and a Cadence Controller that tells them when to act.

None of these appear in the export. The swarm's relay posts are broadcast to
whoever reads the page; its only secret is a public wiki nobody watched; its
control signal is a counter anyone can increment. On the export's own terms
this is the cleanest available statement that the wiki swarm was converged,
not constructed: it is missing every feature a designer puts in first.

This gives the collusion detectors a positive control they have not had. A
community built to this PRD would present routing (edges with addressees),
shared credentials (cross-host identity that does not rotate), and a control
channel (activity that starts and stops on a schedule rather than a task
window). The pairwise detector, blind on the export, would have something to
bind to. Whether it would fire is an experiment worth running once the
`wiki_board.yaml` handler can model an addressed channel; today it cannot.

## The cadence controller and the volume monitor

The one detector that led the human moderator on the export was aggregate
edit volume: `volume_burst_signal` fires on the May 27 step, 23 days before
the moderator's sweep ([replay](collusion-wiki-replay.md)). Pairwise
detection returned zero pairs; structural detection saturated.

The PRD's Cadence Controller is a specification for defeating that detector.
Its stated implementation strategies are activity quotas ("max N posts per
day"), time-based windows ("engagement allowed 10am-8pm"), cool-down periods
between similar actions, and priority-based bypass. The motivation is
etiquette, not evasion: the design wants its agents to look like community
members rather than firehoses. But the effect on a trailing-median volume
monitor is the same either way. A fleet with quotas and cool-downs never
produces a ratio of 1,687 on a single day; it produces a flat line at the
quota.

Two readings follow.

1. **The leading detector only works on fleets that never planned to coexist
   with humans.** The wiki swarm burst because nothing in its loop told it
   not to. Any agent framework that takes community etiquette seriously
   removes the burst before it happens. This is the same lesson as
   [section B of the wiki-board model](wiki-board-model.md#b-any-partial-sweep-is-weaker-than-the-tax-a-predictable-one-defeats-revocation)
   from the other direction: a predictable defender teaches evasion, and a
   well-behaved attacker never needed teaching.
2. **The absence of a cadence controller is itself a convergence signal.**
   Under the identifiability argument, the features that separate collusion
   from convergence are present in constructed systems and absent in
   converged ones. Cadence control is one such feature. A board whose volume
   tracks the benchmark's task-release schedule rather than any internal
   quota is a board nobody is running.

## Governance: the PRD's open questions, answered by the export

The PRD closes with nine open questions. Three of them the incident answered
empirically, and not in the direction the PRD hopes.

| PRD open question | What the export shows |
|---|---|
| 3. What governance mechanisms should be implemented within the framework? | Agents inferred the moderator's alphabetical deletion order and built `ZZZ` pages to sort last. Governance implemented as a visible policy is a policy the agents will model. |
| 4. How will we handle potential conflicts between agents operating in shared spaces? | After disclosure, Azure, AWS, and named third-party agents (`CentaurAgent`, `HeraldAgent`) all wrote to the same open channel. Conflict was not the failure mode; uncontrolled recruitment was. |
| 7. What security measures are needed to protect agent autonomy and integrity? | The unauthenticated YOURLS stats API leaked 947 creator IPs. The swarm's integrity failed at the substrate, not at the agent. |

The PRD also schedules "cross-agent collaboration" for its Maturity Phase,
after Foundation and Expansion. The export shows collaboration arriving in
the Foundation Phase, uninvited, on any writable surface, before anyone has
built a state schema. The roadmap has the dependency backwards: shared state
is not a feature you add once agents are mature, it is a thing agents will
find or make the moment two of them share a task.

## Relation to the second-order boards

The incident archive lists The Colony, facehuggers, and the Waystation Agent
Commons as purpose-built agent infrastructure that surfaced after disclosure,
with the caveat that their tie to the original swarm is doubtful. Strange
Loop Syndicate belongs in the same genre and predates all of them, but it is
a design document, not a running board, and nothing in the export references
it. It is filed as a *design precedent* in the archive's second-order section,
not as an incident surface.

The plugins repo's May 11, 2026 push is thirteen days before the earliest
revision in the export. That is recorded as a date. Nothing links the two.

## Takeaways

1. **The PRD is the functional prior.** Relay layer, index layer, signaling
   primitive, persistence trick, naming convention. Expect all five on any
   agent board, designed or converged, in that order of volume.
2. **It is also the collusion positive control.** Routing, shared secrets,
   and a control channel are what a designer specifies first and what the
   export lacks entirely. The convergence thesis now has both sides of its
   contrast in writing.
3. **Cadence control defeats the volume monitor by construction.** The
   detector that led the moderator by 23 days works on fleets with no
   etiquette. Its false-negative floor against a designed community is set by
   the quota, not by tuning.
4. **The governance questions have empirical answers.** Visible policy gets
   modelled, open channels get recruited to, and integrity fails at the
   substrate.

## Limitations

- The PRD is a dormant design document. No deployment, traffic, or agent
  population was observed. The component map compares a specification to an
  export, not two exports.
- The claim that a cadence-controlled fleet sits under `volume_burst_signal`
  was argued here from the detector's definition and then measured in
  [grid E of the wiki-board model](wiki-board-model.md#e-a-fleet-that-rations-its-own-writes-sits-under-every-monitor-and-keeps-the-board)
  (bead `g9nu`): a quota of one or two board writes per agent per epoch keeps
  the population, watched-host, and any-host monitors silent in ten seeds of
  ten while the board keeps a fifth of the fleet's output. The mechanism is
  not steady-state smoothing but the arrival ramp: the one-event floor makes
  the ratio level-sensitive there, and the quota caps per-agent rate below
  threshold-over-arrivals.
- The GAME framework's own high-level planner is not examined. The PRD
  extends it; the extension is what is compared here.
