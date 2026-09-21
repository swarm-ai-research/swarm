# SwarmWorld → SWARM: the dyadic-enrichment null, the agent-free assay, and what a swarm advantage is actually made of

**Source:** Subhadeep Pal, Fiona Y. Wang, Markus J. Buehler, *SwarmWorld:
Stigmergic Technological Evolution in Societies of Language-Model Agents*,
arXiv:2608.26081, submitted 2026-08-26
(<https://arxiv.org/abs/2608.26081>). Surfaced via the author's 2026-09-15
post on X. Date of this note: 2026-09-22.
**Sourcing status:** the v1 HTML full text was retrieved and read directly;
every quoted number below is from that text, not from the abstract or a
summary. Figure panels were not viewed — claims resting on a figure are
taken from its caption, which is where the authors put the numbers.
**Beads:** this note `ci8k`; follow-ups `xf2r` (dyadic-enrichment null),
`h5mg` (agent-free durability assay), `nlws` (warning reach).
**Companion notes:** [RL Organism Emergence](rl-organism-emergence.md) (the
pair-first detection failure this paper independently corroborates),
[Research-Swarm Whistleblowing](research-swarm-whistleblowing.md) (the
alert-channel result this paper puts a threat-to-validity under),
[AI Village mapping design](ai-village-mapping-design.md) (D2 there hits the
same "SoftInteraction is dyadic, the world is not" wall).

## Why this matters for us

Most external multi-agent work we read is either an incident (the wiki swarm,
the Artifactory board) or a capability demo. This is neither. It is a
controlled ablation with a falsifiable baseline, and two of its design
choices are things SWARM does not have and should.

The paper is *not* a safety result and this note does not treat it as one.
SwarmWorld has no adversary, no deceptive archetype, no harm term and no
externality: every agent is cooperative by construction and the only thing
being optimized is habitat service coverage. Nothing in it speaks to
toxicity, adverse selection or governance. What it does supply is
**methodology and a null result about coordination structure**, both of which
transfer, and one number that quietly undermines an assumption our detectors
are built on.

## 1. What was run

Two paired studies over a spatial world where agents gather feedstocks, test
materials, construct persistent artifacts and write bounded controller
programs that a deterministic simulator executes.

| | scaling study | long-horizon study |
|---|---|---|
| ticks | 800 | 3,200 |
| population | N = 50, 100, 200 | N = 100 |
| replication | 4 matched world seeds per cell | 4 matched world seeds |
| held-out schedules | 8 | 8, at 5 frozen checkpoints |

Four conditions, removing mechanisms without changing the task:

- **Full culture** — shared world, explicit messages/records/teaching/trade,
  cross-agent program inheritance, artifact stigmergy.
- **No communication** — messaging and publication-dependent composition
  removed; shared world and program inheritance retained.
- **No explicit culture** — additionally removes cross-agent program forking
  and skill inheritance. Only physical stigmergy remains.
- **Independent search** — N isolated one-agent worlds, reported as an
  **endpoint-wise best-of-N envelope**: `Y_iso(e,t) = max_i Y_i(e,t)`, where
  the maximizing member may differ across endpoints and checkpoints.

That last control is the part worth stealing outright. It gives the
no-interaction baseline its *best available result for every question
separately*, rather than forcing one solo agent to win every contest. A swarm
advantage measured against that envelope cannot be a parallel-sampling
artifact. We have no equivalent: SWARM scenario comparisons are
regime-vs-regime at matched agent counts, which conflates "coordination
helped" with "more draws helped."

## 2. The result that bears on our detectors

Diffusion is near-universal in both shared-world conditions — 99.3% of
full-culture artifacts and 96.9% of no-explicit-culture artifacts were
eventually reused by a noncreator, with median time to first reuse 5 vs 8
ticks and mean adoption breadth 13.53 vs 7.49 noncreator agents. So
transmission happens, a lot, and explicit culture measurably accelerates and
broadens it.

But **it does not travel along inventor→adopter edges.** The authors tested
the obvious motif — creator messages, teaches or trades with the eventual
first adopter at `t1`, that agent first reuses the artifact at `t2` — against
a 200-replicate timestamp-shuffled null preserving directed dyads and the
global activity schedule. Observed/null was **1.175 at a 25-tick lag and
below parity at 50–400 ticks**. Approximately 95% of first reuse began with
direct physical observation of the artifact in the world, in both conditions.

The ablation's 95% is near-forced (it has no messages). The informative cell
is full culture, which had 3,924 delivered-message recipient edges in the
representative seed and still routed ~95% of first adoption through the
world.

The correct reading is *not* "messages don't matter" — the authors are
careful here and so should we be. Full culture produced roughly twice as many
distinct agent-technology relationships and higher tie retention (0.650 vs
0.588). Their conclusion is that culture "operated diffusely": explicit
exchange reshaped the society-wide substrate, after which agents encountered
technology through the world. **Messaging mattered; the dyad did not.**

This is the claim that lands on `swarm/metrics/collusion.py`. Our detection
stack is dyad-first by construction: `PairMetrics` keys on interaction count,
mutual acceptance and benefit correlation *between two named agents*, and the
group path builds cliques out of those pairs. We already know this misses
real structure — [RL Organism Emergence](rl-organism-emergence.md) found an
emergent predatory coalition walking past `CollusionDetector` in all 20 runs
because pair-first scoring cannot see quality-inside/harm-outside structure.
SwarmWorld is independent evidence for the same gap from the opposite
direction: in a population where coordination was real, dense, and
measurable at the network level, **the dyadic statistic sat at parity with a
shuffled null.** A detector reading only that statistic would have reported
nothing, correctly, and been useless.

There is a second, narrower defect their null design exposes. The
`mutual_favoritism` path scores internal concentration against
`expected_internal = (|members| - 1) / (n_agents - 1)` — a **uniform-mixing**
expectation. Uniform mixing is not the right null when agents differ in
activity rate: a pair that simply acts more than everyone else looks
concentrated under it. SwarmWorld's null preserves directed dyads *and* the
global activity schedule, which is what makes its parity result meaningful
rather than an artifact of who was busy. Ours does not.

## 3. The agent-free held-out assay

At a discovery checkpoint the complete world state is frozen into eight exact
clones. Each clone gets a paired unseen schedule of contamination, drought
and storm — new centers, timings and orderings. **The agents take no actions
during evaluation.** Only deterministic physics and the installed
agent-authored programs continue. Balanced service coverage is integrated
over assay time and averaged across schedules.

The assay asks whether the technology left behind protects the habitat after
its inventors have gone. SWARM asks nothing like this. Our runs are
reproducible from scenario YAML + seed + exported history, and our event logs
replay — but every governance result we report is measured *with the agents
still in the loop and the environment on-distribution*. We have no statistic
for whether a regime's welfare advantage is durable: whether it lives in the
institution the population built or in the population's ongoing behavior, and
whether it survives a shock it never trained against.

That distinction is load-bearing for the governance claims this project
exists to make. A lever that only works while its agents are present and the
world stays nominal is a much weaker claim than we have been implicitly
making.

## 4. More sharing is not monotonically better

The headline is explicitly bounded: *"the main result is not a universal
swarm advantage."* At tick 3,200 (N = 100, 4 seeds):

| endpoint | full culture | no explicit culture | isolated best-of-100 |
|---|---|---|---|
| portfolio resilience | 0.2474 | 0.2365 | 0.1794 |
| validated inventions | 5.75 | **7.00** | 2.75 |
| held-out resilience | ≈ no-explicit | **0.0446** | 0.0356 |
| best final single artifact | 0.2380 | — | **0.3488** |

Validated inventions **never cross**: no-explicit-culture is higher at every
checkpoint. Best final single artifact reverses the ordering entirely — the
isolated envelope wins. In the scaling study the condition ranking changes
with N: at N = 50 full culture and no communication *trailed* the independent
envelope on discovery AUC, at N = 100 all three shared-world conditions
exceeded it, and at N = 200 no-explicit-culture produced the largest paired
discovery gain (+0.069).

Two things follow for us. First, shared worlds win on **breadth and
robustness**, isolated search wins on **peak**. That is a diversity/peak
tradeoff, and for safety work the sign is not obviously good: a broader
persistent portfolio is also a larger attack surface and more inherited
substrate for a bad artifact to ride on. Second, adding an explicit sharing
channel to a population that already coordinates through the environment can
*reduce* a headline endpoint. Our governance levers are mostly
monotone-by-assumption — more transparency, more reputation, more
internalization. This is a concrete counterexample from an adjacent system.

One more design note worth importing: the behavioral differentiation result
(agents self-sorting into constructor, caretaker, coordinator and surveyor
modes from a homogeneous start, ~27%/20%/17% artifact-centered at N = 200
across conditions) is re-fit **without** the cultural and social-contact
features, because two of the coordinator state's defining features are
disabled by the ablations. The 11 physical/task features independently
support three modes at silhouette 0.551. That anti-circularity refit is
exactly the discipline `fcmy.7` needed and the
[AI Village D3 decision](ai-village-mapping-design.md) reached the hard way —
do not let a channel that the intervention defines be the channel that
detects the intervention's effect.

## 5. The threat to validity this puts under our whistleblowing result

[Research-Swarm Whistleblowing](research-swarm-whistleblowing.md) found that
"the warning does most of the work": audit alone at 24% whistleblower share
reaches peak infection 0.268, and adding the peer warning takes it to 0.075.
The note already flags that this "is the one result that depends on an alert
channel existing."

SwarmWorld does not contradict the transmission half of that model — our
`memetic_spread` contagion runs through a shared cache, which *is* an
artifact-mediated channel, and the paper is corroboration that
environment-mediated transmission is how this works. The exposure is on the
countermeasure side. `whistleblower_warning_strength` multiplies infection
for *every* non-whistleblower honest/opportunistic agent in any epoch with at
least one catch. That is a broadcast with assumed total reach, in a model
where transmission is environmental — and SwarmWorld's population largely did
not route adoption decisions through messages at all.

The naive fix — a scalar reach parameter — is not identifiable: uniform reach
r and strength s enter as the same product. The version that actually tests
anything is **heterogeneous** reach, where a fixed subpopulation never
attends to the alert. If the warning's 3.6× advantage over audit-alone
collapses once an unreached minority can sustain the epidemic, then "the
warning does most of the work" is a statement about a fully-attentive
population and should be reported that way.

## 6. What this does not license

- No safety conclusions. No adversary, no harm, no externality in their
  world. "Swarm beats isolated search on portfolio breadth" is a capability
  result and says nothing about whether swarms are safer or less safe.
- Four seeds per cell. The authors say so plainly and emphasize effect sizes
  and paired consistency over population-level inference; any number quoted
  from here should carry that.
- The Protein Realms transfer pilot is single-seed and **did not complete a
  valid assay** under full culture or no-explicit-culture (13 and 10
  proposals, no invented design rows). It demonstrates the workflow runs; it
  supports no comparison between conditions. Do not cite it as a biomaterials
  result.
