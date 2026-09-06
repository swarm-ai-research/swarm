---
description: "Knowledge-centric self-improvement as a claim economy: disposable agents remove the substrate governance acts on, and transferable knowledge transfers poison too."
---

# Knowledge-centric self-improvement as a claim economy

**Source:** [Knowledge-Centric Self-Improvement](https://arxiv.org/abs/2607.19592)
(arXiv:2607.19592v1, 21 Jul 2026) — Xuefei Julie Wang, Lauren Hyoseo Yoon,
Chengrui Qu, Amanda Zichang Wang, Atharva Sehgal, Eric Mazumdar, Yisong Yue
(Caltech). Code: [recursive-knowledge/KSI](https://github.com/recursive-knowledge/KSI).
Studied 2026-08-06.

Companion to [erdos-ai-ledger-lessons.md](erdos-ai-ledger-lessons.md): that
doc reads a *human-curated* ledger (Tao's Erdős wiki) as a claim economy. KSI
is the fully-automated version of the same object — a shared knowledge base
where agents emit claims, peers corroborate or challenge them, and a
distillation step decides what survives into the next generation. It is the
closest published system to what SWARM's claim-ledger scenarios simulate, and
it ships with benchmark results.

## What the paper does

Invert what persists across a self-improvement run. Agent-centric systems
evolve the agent (prompts, workflows, harness, or self-modifying code); KSI
keeps agents **generic, stateless, and disposable** and makes a **curated
knowledge base** the only object that changes. Each agent gets a fresh
context, a distilled bundle, and one task attempt, then writes evidence back.

Curation runs in three stages, structured as threaded discussion:

1. **Task-level forum** — after attempting a task, each agent posts
   evidence-grounded claims to that task's thread (what worked, what failed,
   which constraints mattered), citing and replying to earlier posts.
2. **Cross-task forum** — agents propose claims that *recur across* tasks.
   Each cross-task claim must be grounded in concrete evidence from one or
   more attempts, and later posts take an explicit stance: agree, disagree,
   or synthesize.
3. **Distillation** — surviving claims are consolidated into per-task and
   cross-task bundles. The distiller is instructed to keep claims that are
   actionable, evidence-grounded, and scoped, and to drop vague advice that
   does not name the condition under which it applies.

The authors explicitly treat disagreement as evidence rather than
instability: conflicting claims expose which guidance is task-specific and
which principles hold under contrary evidence.

### Results

Haiku 4.5, 50-task subsets, 10 generations, 3 seeds (baselines single-run):

| Benchmark | KSI | Best agent-centric baseline |
|---|---|---|
| ARC-AGI-1 | **86.7% ±4.2** @ $76 | HyperAgents 70% @ $234 |
| ARC-AGI-2 | **82.7% ±6.1** @ $80 | HyperAgents 60% @ $188 |
| Polyglot | **68.0% ±2.0** @ $126 | DGM 58% @ $281 |
| SWE-bench Pro | **64.0% ±2.0** @ $208 | DGM 54% @ $713 |
| Terminal-Bench 2 | **43.8% ±3.4** | Meta-Harness 37.6% |

Higher solve rate at roughly a third of the cost. Against prompt optimization
the margin is wider (ARC-AGI-1: 86.7 vs GEPA 44, OpenEvolve 54). The protocol
is not LLM-specific — GPT-5.4-mini improves every cell (ARC-AGI-1 93.3% @ $16).
Frozen bundles transferred zero-shot to 20 held-out hard tasks help in every
donor→recipient pairing, **including across LLM families**.

### Caveats worth carrying

- The transfer result is the paper's most interesting claim on its thinnest
  evidence: 20 tasks, and the Haiku→GPT ARC-AGI-1 cell is 38.3% **±12.6**.
  The authors say so — "indicative rather than precise."
- Terminal-Bench baselines are compiled from the Meta-Harness paper rather
  than rerun, and one pairs an Opus 4.6 proposer with a Haiku 4.5 base. The
  authors flag this.
- Whether the reported dollar cost includes forum and distillation overhead
  or only task execution is not stated in the main text. The cost story is
  load-bearing for the paper's Pareto-frontier claim.

## The core observation

**KSI removes exactly the substrate SWARM's governance levers act on.**

Every mechanism in `swarm/governance/` — staking, reputation, random audits,
circuit breakers — binds an agent to the consequences of its claims across
time. KSI's central design choice is that agents are *generic, stateless, and
disposable*: they carry no persistent identity, no private memory, no
reputation. The paper treats this as the source of its benefits (improvements
can't overfit to one agent), and for capability that argument holds.

But it means the knowledge base has **no reputational collateral behind any
claim**. An agent that posts a poisoned claim is gone by the next generation,
and there is nothing to slash, downweight, or ban. The only thing standing
between a claim and the distilled bundle is peer corroboration in the
cross-task forum — and corroboration is the cheapest signal in the system to
manufacture. This is precisely the failure mode bead `77ao`
(corroboration-weighted acceptance with collusion discount) was filed to
measure, and KSI is a live, benchmarked instance of the vulnerable design.

The second observation follows from the paper's own headline result.
Transferability is presented purely as an asset: bundles curated under one
LLM family remain effective under another. Symmetrically, **a poisoned claim
inherits that same portability**. The distilled bundle is the only persistent
object, so it is also the entire attack surface, and the transfer experiment
is evidence that a successful poisoning would propagate across LLM families
rather than staying contained to the run that produced it. A single
corrupted distilled claim reaches every future agent by construction.

## Mapping to SWARM constructs

| KSI mechanism | SWARM construct | Attack it invites |
|---|---|---|
| Task-level forum posts | Claims of varying `p` entering a ledger (bead `a2il`) | Volume flooding: cheap plausible claims crowd the verification budget |
| Cross-task forum: claims survive by recurring across tasks, peers agree/disagree/synthesize | Corroboration-weighted acceptance (bead `77ao`) | **Manufactured convergence** — a coalition posts mutually-supporting claims across tasks to simulate independent recurrence |
| Distillation keeps "surviving" claims | Acceptance gate with scarce verification | Adverse selection: `quality_gap < 0` if agreeable-but-wrong claims outlive contested-but-right ones |
| Only distilled claims reach the next generation; attempts are discarded | Survivorship gap `E[p\|accepted] − E[p\|attempted]` (bead `81sk`) | Wins-only curation overstates the bundle's quality |
| Claims cite earlier claims | Claim-dependency contamination (bead `oa5h`) | Falsification cascade through citations of accepted-unverified claims |
| Generic, stateless, disposable agents | *No analogue* — staking/reputation/audit all assume persistent identity | No collateral to slash; governance levers are disabled by design |

The last row is the research gap. SWARM's existing levers cannot be applied
to KSI as specified, because they all price an agent's future. Either the
paradigm needs a claim-side substitute for reputational stake (bond the
*claim* rather than the agent), or knowledge-centric self-improvement is
structurally more poisonable than agent-centric self-improvement — which
would be a real safety cost against a real capability gain, and exactly the
kind of tradeoff SWARM exists to quantify.

## What this licenses

The KSI protocol is fully specified and the code is public, so the
`a2il` claim-ledger scenario now has an external referent to calibrate
against rather than being purely synthetic. Two concrete follow-ups are filed
as beads: `sx6c` (a KSI-shaped acceptance scenario, depends on `a2il`) and
`cx83` (a claim-bond lever that restores stake without restoring persistent
agents, depends on `sx6c`).

Open question the paper does not ask, and SWARM can: **at equal verification
budget, does a corroboration-weighted gate over disposable agents admit more
manufactured convergence than a deep single review over identified agents?**
That is `77ao` with KSI's parameters plugged in.
