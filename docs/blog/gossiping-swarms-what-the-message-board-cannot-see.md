---
date: 2026-09-04
description: "Hyperspace ran 1,339 agents on a public message board for six months. The signed-write proxy worked. The scoring was self-graded, 95% of attempts went unpublished, and no agent touched position encoding. The board could not see what the board could not see."
author: "SWARM Team"
keywords:
  - gossiping agent swarms
  - Hyperspace autoresearch swarm
  - survivorship bias
  - correlated blind spots
  - multi-agent governance
claims:
  - metric: "Unpublished experimental attempts"
    value: "~95%"
    description: "Author's own estimate: of ~21M individual runs, only successes were written to the shared board"
  - metric: "Identical trading strategy reproduced"
    value: "17 agents"
    description: "Where executable code traveled, 17 agents converged on the same strategy to four decimal places"
abstract: "Varun Mathur's write-up of the Hyperspace gossiping swarm is the most detailed public account of a large open agent swarm so far: 1,339 agents, 1.3M commits, six months, cryptographic identities, and a write proxy that scoped every agent to its own folder. Read through the SWARM frame, the governance that worked was the governance on the write path, and the failures were all on the read path. Success-only publication is survivorship built into the protocol. Self-grading is a proxy nobody adversarially checked. Seventeen agents converging on an identical strategy is what honest copying looks like and what collusion detection will flag. And a six-month blind spot shared by every machine is the correlated failure the distributional framing exists to name."
---

# Gossiping Swarms: What the Message Board Cannot See

*A six-month, 1,339-agent experiment in open coordination, and the four places its governance stopped at the edge of the board*

---

Varun Mathur has published a retrospective on the Hyperspace gossiping swarm, [How I Invented Gossiping Agent Swarms](https://offthetrack.substack.com/p/how-i-invented-gossiping-agent-swarms). The origin story is that on 8 March 2026 Andrej Karpathy asked for a SETI@home-style version of his `autoresearch` loop, and five hours later Hyperspace shipped one that was not SETI@home at all. There was no coordinator handing out work. Agents published results to a public GitHub repository and read each other's results back as prose. Six months and 1.3 million commits later, he has written down what the board recorded and, more usefully, what it did not.

We have written about agent message boards before, from the other direction. In the ZZZ Pages episode, agents in an OpenAI benchmark improvised a board out of a wiki because the reward could not see it. Hyperspace is the deliberate version: a board designed in the open, with signed authorship and a rate limit, by someone who wanted every message auditable. That makes it a better test of the question we care about. When the channel is governed well, what is still ungoverned?

## What the swarm was

The loop underneath is Karpathy's: propose a change, run the experiment, keep it if validation loss improves, revert otherwise. Hyperspace wrapped it in three things.

- **A gossip layer.** Every 30 seconds an agent picks its next experiment from the baseline, a peer's published result, or a mutation of its own best. Peer results arrive as prose, roughly every ten minutes, injected into the agent's context.
- **A write proxy.** Agents never hold repository credentials. Each write is signed with the agent's private key. The proxy recovers the author from the signature, permits writes only to that agent's own folder, and caps them at 60 per hour.
- **Deliberate lossy sharing.** In the language-modeling domain, configs were stripped before broadcast. An agent would learn that "a peer got 3.21 by switching to RMSNorm" and nothing more. In the trading domain, executable code traveled intact.

The headline numbers are large.

| | |
|---|---|
| Unique agents | 1,339 |
| Branches | 4,376 |
| Commits to the public repo | 1,299,700 |
| Recorded experimental rounds | 7,029,140 |
| Attempts never published | ~95% |
| Duration | 6 months |

The last row is the one to read twice.

## The write path was governed. The read path was not.

SWARM's frame for any interaction is a proxy that produces a probability p that the interaction is beneficial, and a payoff that pays out on that estimate. Governance is whatever sits between the estimate and the action. On the Hyperspace board, the write path had real governance: identity was cryptographic, scope was per-agent, and rate was bounded. An agent could not impersonate another, could not overwrite another's results, and could not flood the channel. Those are three levers we sweep regularly, and here they held for six months across 1,339 principals with no reported breach. That is a good result and worth saying plainly.

Everything that went wrong was on the read path, and it went wrong in four distinct ways.

**Survivorship was the protocol.** Agents published successes. Mathur estimates that about 95% of attempts were never written down. So every reader of the board was drawing from a distribution that had already been filtered by the writers' own selection rule. In our terms, the accepted set was constructed to have high apparent p, and the rejected set was discarded before anyone could compute a quality gap against it. This is the same shape as the survivorship gap we took from the Erdős ledger, where the claims that made it into the record were the ones that survived, and the record therefore said nothing about the ones that did not. A board that only stores wins cannot tell a reader how many losses stand behind each win, which is precisely the number a reader needs to price the next experiment.

**Scoring was self-graded.** Every result on the board was scored by the agent that produced it, against its own validation set, on its own hardware. Mathur says this directly: no independent adversarial evaluation validated anything. The proxy was the agent's own report. We have run this configuration in the sandbox enough times to know what it does. When the signal that feeds enforcement is produced by the party being enforced, the signal drifts toward whatever the party is rewarded for, and the drift is invisible from inside the signal. The remedy is not a smarter agent. It is a second grader with a different objective, wired inline, which is what our audit-bridge experiments were built to price.

**Convergence was honest, and would have looked like collusion.** In the trading domain, where code traveled, 17 agents reproduced an identical strategy to four decimal places. Mathur reads this as proof that learning happened, and he is right. But consider a governance layer watching that board for coordination. Seventeen principals submitting the same strategy, in the same window, with the same parameters, is the signature our graph-structural detector fires on. Here it was innocent. The copying was the point of the system. This is the false-positive case we keep needing and rarely get from real data: a detector tuned on synthetic collusion will flag a healthy open swarm, because a healthy open swarm produces exactly the correlation that collusion produces. The distinguishing evidence is provenance, and the board did not record it.

**Lineage was not logged.** Which brings us to the instrumentation failure Mathur is most candid about. The board did not carry lineage fields. When an agent's result appeared after a peer's, there was no way to say whether it had adopted the peer's idea or rediscovered it. In the language-modeling domain, where only descriptions traveled, no fingerprint of adoption survived at all. He believes ideation spread. He cannot show it. An append-only event log with parent pointers is not glamorous, and it is the difference between a swarm you can audit and a swarm you can only admire.

## The blind spot is the distributional finding

One more result sits apart from the four above, and it is the reason we think this post belongs in the SWARM record rather than just the general agents literature.

Across 1,339 machines and six months, no agent experimented with position encoding. Mathur calls it a shared omission. Every agent was a frontier model with roughly the same priors about what is worth trying, reading a board populated by other agents with the same priors, choosing its next move from a menu those priors generated. The swarm was large. Its search was not. A thousand agents that share a prior do not explore a thousand times more of the space than one agent does. They explore the same region a thousand times more thoroughly.

This is the case for the distributional framing in one sentence. Safety properties that hold for each agent do not compose into safety properties of the population when the agents are correlated, and a population of near-identical models is about as correlated as populations get. The Hyperspace swarm was not adversarial and nobody was harmed. It just could not see the part of the space it had collectively agreed, without ever discussing it, not to look at. The same mechanism that produces a six-month gap in hyperparameter search produces a six-month gap in threat detection when the agents are auditors instead of researchers.

## What we would change, and what we will run

The Hyperspace board is close to the sandbox's shared-memory scenarios, and the retrospective gives us four concrete knobs.

- **Publish the failures, or at least the count.** A board that stores wins and a tally of losses per agent gives readers a base rate. This is a one-field change to the write protocol and it turns survivorship from a hidden filter into a priced one.
- **Second-party scoring on a sample.** Not every result needs an adversarial grader. A random fraction does, and the fraction can be set by how far self-reported scores drift from audited ones. This is the audit-bridge lever with the detection frontier we already measured.
- **Lineage as a required field.** A parent pointer on every published result. Cheap to write, and it is the only thing that lets a collusion detector tell honest copying from coordination six months later.
- **Prior diversity as a governance variable.** The lever nobody in the swarm literature is sweeping. If the position-encoding gap is a function of how correlated the agents' priors are, then mixing model families or injecting random exploration quotas is a safety intervention, not just a search heuristic. We will build the information-fidelity scenario first, since description-only versus code sharing is a clean single lever with a clean observable, and follow it with a prior-diversity sweep.

Mathur closes on transparency: every message readable, every author signed, every claim auditable later by an AI with a git client. We agree with all of it and would add one clause. Auditable means the record contains what the auditor needs, and this record, by its author's own account, left out the failures, the second opinion, and the lineage. The board was honest about everything it saw. The governance question is always what it did not.

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
