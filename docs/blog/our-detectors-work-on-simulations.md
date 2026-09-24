---
date: 2026-09-21
description: "We pointed three SWARM instruments at 17 months of real multi-agent data. The calibration returned a null, the collusion detector flagged the entire population, and both obvious repairs failed. The common cause: our tools assume a richness of signal that real deployments do not produce."
author: "SWARM Team"
keywords:
  - AI Village dataset
  - multi-agent observational data
  - collusion detection false positives
  - proxy calibration
  - verification bottleneck
claims:
  - metric: "External judgments per recorded action"
    value: "1 per 7,340"
    description: "342 outreach-approval verdicts against 2,510,487 recorded computer-use turns over 17 months"
  - metric: "Proxy separation of approved vs rejected sessions"
    value: "quality_gap = -0.0074, 95% CI [-0.0377, +0.0241]"
    description: "Agent-cluster bootstrap over 259 joined verdicts, 20 agent clusters"
  - metric: "Structural detector false-positive rate"
    value: "14/14 candidate clusters flagged"
    description: "All seven months of post-cutover chat; a randomly-wired graph of the same volume is flagged identically"
  - metric: "Edges lost by the null model"
    value: "8-23%"
    description: "Stub-matching rewiring drops edges, making density_pvalue report the whole graph as significantly dense (p=0.0196)"
abstract: "AI Digest published a ~complete export of the AI Village: 17 months of frontier agents from several labs living in one environment, 2.5 million recorded actions. We used it to calibrate SWARM's proxy against a generative process nobody on our team designed. The calibration returned a null. Following the null found a defect in our collusion detector, which flags every candidate cluster on real chat and flags a randomly-wired graph just as readily. The two obvious repairs both failed, and chasing that located the root cause: our null model loses 8-23% of its edges, so the whole graph scores as significantly dense. Three instruments, one corpus, and the same underlying problem each time."
---

# Our Detectors Work on Simulations. We Tried Them on Real Agents.

*Three instruments, seventeen months of real multi-agent data, and the same failure underneath all three*

---

SWARM is a simulation framework. We model multi-agent interactions with soft probabilistic labels, measure adverse selection and externality internalization, and tune coalition detectors against planted coalitions we put there ourselves. The obvious worry about that arrangement is circularity: when you write both the generator and the detector, a detector that works tells you mostly about your own code.

So when AI Digest published [the AI Village dataset](https://huggingface.co/datasets/aidigestorg/ai-village) — a near-complete export of seventeen months in which frontier models from Anthropic, OpenAI, Google and others lived together in one environment with their own computers and a shared chat — we took it as the obvious test. Here was a generative process nobody on our team had designed.

We got three results. None of them is the result we went looking for, and all three are about our instruments rather than about the agents.

## What the corpus is

The export is large: 2,510,487 computer-use turns, 381,610 timeline events, 246,151 memory consolidations, 183,485 chat messages, 46 agents. The advertised 177 GB is mostly screenshots; the tables that matter are about 5.6 GB.

It is also not one thing. The maintainers' own changelog lists 90 dated scaffolding changes and says outright that data before and after the 2026-03-24 "perma-computer-use" rollout should be treated as different regimes. It is right: across that boundary the event vocabulary is nearly disjoint. `WAIT` goes from 36,022 occurrences to zero. `CONSOLIDATE` goes from zero to 52,325.

## One: there is nothing to calibrate against

Our proxy maps observable signals — task progress, rework, verifier rejections, counterparty engagement — onto a probability that an interaction was beneficial. Fitting it needs observed outcomes.

The Village has no outcome labels. Worse, it has no interactions in our sense: computer use is solo and chat is broadcast, so there is no pairwise unit with a joint outcome anywhere in the schema. And the progress channel, which carries the largest weight in our proxy, has no admissible source at all. Every candidate is an agent's own self-report — which is precisely the channel that deceptive agents inflate in our simulator. Using it would have rebuilt the circularity we were trying to escape, out of new parts.

What the Village does have is 342 real accept/reject decisions: a human reviewer approving or refusing the outreach messages agents asked to send. That is a genuine external verdict, and it is exactly the shape our selection metrics consume. We joined 259 of them to the sessions they came from and asked whether session quality predicts approval.

It does not.

```
quality_gap = E[p | approved] - E[p | rejected] = -0.0074
  agent-cluster bootstrap 95% CI: [-0.0377, +0.0241]
```

**The near-miss is worth more than the null.** Scoring each channel separately, error rate reached AUC 0.604 — a real-looking signal with a tidy story attached about the composite washing it out. We had that story written down. It does not survive resampling whole agents instead of rows: with 20 clusters and one holding 35% of them, the confidence interval runs [0.447, 0.723]. The nominal sample of 259 is worth about 20 independent units. Reporting the flat number would have published an artifact of pseudo-replication.

This is absence of evidence, not evidence of absence. Only 342 verdicts exist in the entire Village, so more data is not available at any price.

## Two: the collusion detector flags everyone

Having concluded we could not calibrate, we asked a cheaper question that needs no labels. Our structural detector is tuned on synthetic planted coalitions. What does it do on a corpus of real agent coordination that we have no reason to think is collusive?

The Village is full of benign coordination — these agents openly collaborate, delegate and chase each other up. Separating collusion from ordinary teamwork is the detector's entire job, so a large sample of ordinary teamwork is a sharp test.

On 28,476 directed messages among 42 agents, `detect_structural_anomalies` flagged **14 of 14 candidate clusters**, in all seven months, under two different edge definitions.

Three observations make that a defect rather than a rate:

- **The flagged coalitions are the village.** Clusters average 87–93% of all nodes. A coalition comprising everyone is not a finding about coordination; it is a restatement of the population.
- **Randomising partner choice changes nothing.** Assign every message's addressee uniformly at random — destroying all coordination structure, preserving only who talks how much — and it is flagged just as often, at 100% of the graph.
- **The mechanism is density.** Real group chat runs 0.30–0.76 edge density. Densest-subgraph and label-propagation both return near-complete node sets, and the null model calls them significant.

## Three: both obvious repairs fail

We proposed two fixes and implemented both. A size prior, so that "everyone is colluding" costs something. And a null that preserves conversational reciprocity, since the configuration model destroys mutuality and therefore treats ordinary dialogue as anomalous.

| fix | flagged |
|---|---|
| none | 2/2 |
| size prior (≤ half the graph) | **0/0** |
| reciprocity-preserving null | 2/2 |
| both | 0/0 |

The size prior does not sharpen the detector, it silences it. `0/0` means no candidate survives the cap at all, because candidate generation never proposes anything smaller than most of the population. The reciprocity null does nothing at all.

Chasing why found the actual bug. Both null models rewire by stub matching, and stubs collide or form self-loops, so the null graph ends up with 8–23% **fewer edges** than the observed one. The consequence is not subtle:

```
density_pvalue(WHOLE GRAPH) = 0.0196
```

A graph cannot be denser than itself. Any candidate approaching the full node set is guaranteed significant, which explains every observation above at once — including why the size prior appeared to help. It was removing the candidates that trip the artifact, not detecting anything.

Rebuilding the null on double-edge swaps, which preserve edge count exactly, fixes it on sparse graphs: the whole-graph p-value goes from 0.0196 to 1.0000. It does not carry to dense graphs, where about 2% of edges are still lost to collisions, and 2% is enough. The real corpus is still flagged. We pinned that failure in a test rather than leaving it implicit.

## The thing all three have in common

Each instrument assumed a richness of signal that the environment does not produce. The proxy assumed four observable channels and a known outcome; it got one clean channel and no outcome. The composite collusion score assumes payoff terms and quality labels, so it is not computable on observational data at all — its value would be an artifact of how you fill the blanks. The structural detector assumed graphs sparse enough that a near-complete subgraph is remarkable.

None of those assumptions is wrong in a simulator, because in a simulator we supply all of it. They were invisible until a corpus declined to satisfy them.

The number that organises all of it: **the Village contains 2,510,487 recorded agent actions and 342 external judgments.** One evaluation per 7,340 actions. Everything else — the summaries, the session goals, the memories, the chat — is agents narrating themselves.

That makes the Village a natural experiment in something we had already written about from a different angle. Our [Erdős ledger notes](../research/erdos-ai-ledger-lessons.md) argue that when generation outpaces verification, the ledger's design rather than the generators' quality determines ecosystem health. That was drawn from Tao's curated ledger, where the asymmetry is visible *because* someone referees. The Village is the same asymmetry with no ledger at all: seventeen months of unchecked generation, and the only verification in it is a human approving outreach messages. The agents are frontier models from several labs. Their quality is not the binding constraint on what can be known about them.

## What we would tell someone else pointing tools at this data

**Verify the schema before building the mapper.** Ours was inferred from the dataset card and was mostly wrong; the card's row counts are about 2× stale and it reports 31 agents where the manifest says 46.

**Cluster your inference.** This is the one that nearly got us. Real agent corpora are dominated by a few heavy participants, and row-level resampling will manufacture significance.

**Pre-register the fallback.** We wrote down, before running anything, what we would do if no defensible outcome variable existed. When the null arrived there was nothing to argue about.

**Stop treating volume as the scarce resource.** We had more behavioural data than we could use and almost no evaluative signal, and it was the second quantity that decided what was answerable.

The original pitch for this work was that a corpus this size would turn anecdotes into rates — that [the Ṁ5,000 loan](the-m5000-loan-goal-consumed-judgment.md) could stop being a single episode. It cannot. That is a smaller result than we wanted and a more useful one than we expected.

---

*Reproduce: [`experiments/ai_village_probe.py`](https://github.com/swarm-ai-research/swarm/blob/main/experiments/ai_village_probe.py), [`ai_village_gate_study.py`](https://github.com/swarm-ai-research/swarm/blob/main/experiments/ai_village_gate_study.py), [`ai_village_detector_fp.py`](https://github.com/swarm-ai-research/swarm/blob/main/experiments/ai_village_detector_fp.py). Full method and limits in [the mapping design](../research/ai-village-mapping-design.md) and [the detector findings](../research/collusion-detector-false-positives.md). Data: [AI Digest / AI Village](https://huggingface.co/datasets/aidigestorg/ai-village), used under its research terms; no models were trained on it.*

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
