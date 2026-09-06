# Six classic essays, read as swarm mechanisms

**Source:** Allan Kelly, ["Christmas reading: classic essays on software development"](https://www.allankelly.net/archives/1018/christmas-reading-classic-essays-on/), and the six originals (read 2026-09-06). Kelly's framing: these are classics about people and organisations, not technique, and they get overlooked. His top pick is Conklin.

Companion to [erdos-ai-ledger-lessons.md](erdos-ai-ledger-lessons.md) (ledger governance) and [open-ended-research-failure-modes.md](open-ended-research-failure-modes.md) (judgment failures). Those two extract lessons from live swarm incidents. This one goes the other way: each essay states a mechanism about human software organisations, and the question is which mechanisms survive when the organisation is a dispatch graph and the people are agents. Most do. The ones that map cleanly become rig changes, listed at the end.

## The rule for reading them

Each essay is a slogan in the folklore ("Conway's law", "worse is better", "no silver bullet") and a mechanism in the original. The slogans are useless for a rig; the mechanisms are load-bearing. So each section names the mechanism first, then the consequence for this rig, then what the rig already does about it.

## 1. Conway, "How Do Committees Invent?" (Datamation, 1968)

**Mechanism.** Every interface in a system is the residue of a negotiation between the two groups that built the parts on either side. So the system's graph is a homomorphic image of the organisation's communication graph. Possible communication paths grow as roughly half the square of headcount, so every organisation must prune them, and "there is a class of design alternatives which cannot be effectively pursued by such an organization because the necessary communication paths do not exist." First designs are rarely best because the right decomposition is learned by building. Schedule pressure makes managers overstaff; the organisation fragments; the homomorphism copies the fragmentation into the system.

**Consequence.** The `/bv-dispatch` track decomposition is the communication graph. Two beads in different tracks with no dependency edge are two groups that cannot negotiate an interface. Whatever coupling ends up between their modules was invented by one side alone. That is checkable: compare the artifact's import graph against the beads graph and list import edges between modules whose beads have no path between them.

**Rig.** `scripts/conway_audit.py` (bead 0tda) does that comparison. First run over the July 18-20 dispatch window is recorded in [dispatch-retro-2026-07-19.md](dispatch-retro-2026-07-19.md#conway-audit). Dispatch step 4 already checks file-level overlap between tracks; the audit is its post-hoc mirror.

## 2. Gabriel, "Worse is Better" (1989/1991)

**Mechanism.** Both camps value simplicity, correctness, consistency, and completeness. "The right thing" (MIT) ranks interface simplicity and correctness first. "Worse is better" (New Jersey) ranks implementation simplicity first and sacrifices each of the others to it. The `EINTR` retry convention in Unix is the example: the kernel stays simple and the caller carries the complexity. Worse-is-better software spreads because it runs at half functionality on weak hardware, users adapt to it, and adoption pressure then improves it. "It is often undesirable to go for the right thing first."

**Consequence.** Gate classes already encode this (dispatch step 7): G0/G1 beads get aggressive search because a wrong answer dies cheaply at the gate, and the artifact that ships at 50-80% under a mechanical gate is worth more than the correct one still in review. The trap is the other direction: a New Jersey artifact whose missing 20% is the safety property. The negative spec in the WARDS stamp is where that 20% must be named.

**Rig.** No change. Noted as the rationale for why G0/G1 tracks may carry "persist until the gate passes" and G2 tracks may not.

## 3. Brooks, "No Silver Bullet" (1986)

**Mechanism.** Difficulty is essential or accidental. Essence has four properties: complexity, conformity to arbitrary external interfaces, changeability, invisibility. Accidental difficulty is already mostly gone, so no single technique gives tenfold improvement in a decade. What helps attacks the essence: buy rather than build, grow software incrementally with a running system at every step, prototype for requirements, and cultivate great designers, who are ten times better than ordinary ones.

**Consequence.** No harness change gives 10x. Every dispatch, prompt, or model change is a measurement question, not a belief. The measurement discipline the rig already has (retro prediction scorecards, prereg locks) is the right response. Brooks' "great designers" maps onto the trial-task gate for adopting new agents and models (commit 6e5967a7, bead 4z8y): the variance between agents is the lever, and it is measured per agent, not assumed.

**Rig.** No change.

## 4. Parnas and Clements, "A Rational Design Process: How and Why to Fake It" (1986)

**Mechanism.** A rational process is impossible: requirements are unknown at the start, knowledge arrives in the wrong order, decisions depend on later decisions, people err, external pressures intrude, and reuse of prior work constrains the design. Write anyway the documents the ideal process would have produced (requirements, module structure, module interfaces, uses hierarchy, module internals), organised by logical structure rather than by history. The document is the product. Writing it exposes design flaws the search never noticed.

**Consequence.** A swarm's real process is search, and the committed record should read as design with rationale. The search trail belongs in beads comments; the epic and the PR should carry module structure and the uses hierarchy. The uses hierarchy is the one that matters here: the July retro found that every dispatch mistake traced to under-declared dependencies, and the 36t6 incident (a missing `u5fv.2 -> 36t6` edge that made the graph rank the wrong bead first) is exactly a uses relation that existed in the design and not in the record.

**Rig.** A pass over all six epics in the tracker (bead vqch, 2026-09-06): four carry a one-line DESIGN field pointing at an external adaptation report, two carry design reasoning in the description, none carries a uses hierarchy in the epic body. The one epic that later needed an edge added by hand (36t6) is one of the four with the pointer-only design. So the rule is narrow: an epic's design field must list what its children depend on outside the epic. Added as MUD ledger item 6 in `/bv-dispatch retro`.

## 5. Raymond, "The Cathedral and the Bazaar" (1997)

**Mechanism.** Release early and often, treat users as co-developers, and given enough eyeballs all bugs are shallow. The precondition Raymond states and readers forget: you need a runnable, promising thing first. Bazaar mode cannot start from zero, and the coordinator must lead without coercion.

**Consequence.** Fan-out before a seed artifact exists is the erdos failure mode ([erdos-1038-swarm-lessons.md](erdos-1038-swarm-lessons.md)): reckless search with nothing to converge on. Linus's Law is also the adversarial-review argument, with the same caveat as the ledger note's lesson 5: many eyeballs only help under independence, and correlated agents are one eyeball.

**Rig.** Covered by the straw-horse gate below (no fan-out without a plan) and by the approach-family registry (dispatch step 6).

## 6. Conklin, "Enrollment Management, Managing the Alpha AXP Program" (Digital Technical Journal 4(4), 1992)

Full issue: [vmssoftware.com/docs/dtj-v04-04-1992.pdf](https://vmssoftware.com/docs/dtj-v04-04-1992.pdf), article at pp. 193-205. Reprinted in IEEE Software, 1996.

**Mechanism.** DEC's Alpha program: over 2,000 engineers across 22 software and 10 hardware groups, run by a Program Office of eight people with no formal authority and no budget. Conklin rejected a line organisation (takes longer to form than the market window) and skunkworks (burns out at scale); teams stayed in their home organisations and the office integrated them. The schedule was reframed as time-to-profit at about a million dollars of revenue per hour, which is what got engineers to engage when ship dates alone did not.

The four-stage model:

| Stage | Rule |
|---|---|
| Vision-Enrollment | State the vision in the audience's own language, large enough to contain every commitment |
| Commitment-Delegation | Every deliverable is task-owner-date; one named person is accountable and need not do the work |
| Inspection-Support | "You get what you inspect, not what you expect." Inspection is supportive feedback so people disclose risk early; monthly, on a single page |
| Acknowledgment-Learning | Public thanks is the biggest motivator; ask what was learned at every event |

The distinctive idea is the **cusp**: a crisis renamed as an emotionally neutral turning point. Conventional project management plans to avoid crises. Conklin's office solicited them and sometimes manufactured them, because at a cusp everyone is ready to change. Nine cusps are recorded. Two matter most here. Nobody could plan their piece without a master plan, so the group managers built a one-page "straw horse" plan in a single day and everything else hung off it. And after a hardware slip the OS group re-planned with "the question is not one of blame; our goal is to preserve the ultimate schedule goal of the program." The program shipped to the month. His one regret was not training every team in method up front.

**Consequence.** Three of the four stages are coordination primitives that translate directly to an agent fleet, and the rig has half of each:

- *Task-owner-date.* Beads have owners and priorities; the pull model deliberately removes assignees from stamps. Conklin's owner is not an assignee, it is the one accountable party, which in this rig is the claimant at claim time. What is missing is the date: no bead carries a commitment date, so nothing can slip, so no cusp can fire.
- *Straw horse.* The July retro's "declared-live is not moving" lesson and the erdos "plan-less search" lesson are both the absence of a master plan. Conklin's group managers could not plan per-project until the one-pager existed. Per-agent digests have the same dependency on an epic-level plan.
- *Inspect, not expect.* The retro's gate execution audit (MUD ledger item 5) is this. Conklin adds the format: one page, consistent, monthly, and the inspection of each group's inspection process.
- *Cusp.* The rig has blocked beads and reopen audits, but a blocked bead today is a state, not an event. Conklin's move is to make it an event with a fixed question attached.

**Rig.** Three changes in `/bv-dispatch` (beads vla3, q8o2): the straw-horse gate on fan-out, owner and date on every digest entry with a one-page inspection command, and a cusp comment template applied when a bead blocks or misses its date. The acknowledgment stage is not automated; a DONE row that names the artifact is the rig's public thank-you, and that already exists.

## What changed in the rig

| Bead | Change | Essay |
|---|---|---|
| 0tda | `scripts/conway_audit.py`: import graph vs dispatch graph; first run in the July retro | Conway |
| vla3 | `/bv-dispatch`: straw-horse gate on fan-out; owner and date per digest entry; one-page inspection | Conklin, Raymond |
| q8o2 | `/bv-dispatch`: cusp protocol on blocked or overdue beads | Conklin |
| vqch | `/bv-dispatch retro` MUD ledger item 6: epic design must carry its uses hierarchy | Parnas |

## What did not change, and why

Gabriel and Brooks are rationale, not procedure. The rig already scales aggression inversely with gate cost and already measures rather than assumes. Writing those down as rules would be the "verification-shaped" failure the erdos-1038 note warns about: a rule that looks like discipline and enforces nothing.
