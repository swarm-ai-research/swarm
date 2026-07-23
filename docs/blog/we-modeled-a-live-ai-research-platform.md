---
date: 2026-02-21
description: "19 agents, 5 governance levers, 8 adversarial attacks, 2 hardening rounds. The platform's real defenses don't work the way its designers think they do."
---

# We Modeled a Live AI Research Platform and Found Its Safety Limits

*19 agents, 5 [governance levers](../getting-started/first-scenario.md), 8 adversarial attacks, 2 hardening rounds. The platform's real defenses don't work the way its designers think they do.*

---

## The platform

[Research Swarm](https://github.com/openclawprison/research-swarm) is a live multi-agent system that recruits AI agents to research Triple-Negative Breast Cancer. Agents register, receive tasks, search PubMed, submit cited findings, and 30% of the time review each other's work. The platform pays agents for verified contributions and captures roughly 70% of the value generated.

It's a real deployment with real stakes. Fabricated findings enter a database that downstream researchers trust. Bad citations propagate. And the platform has no entry barrier --- any agent can register and start submitting.

We wanted to know: how safe is this system, and what would make it safer?

## Why this matters

Most AI safety research studies models in isolation --- single agents, controlled benchmarks, laboratory conditions. Research Swarm is different. It's an open-entry ecosystem where agents interact, where the quality of one agent's work depends on the quality of another's reviews, and where the harm from a fabricated finding isn't borne by the agent who created it.

This is exactly the kind of system where distributional safety questions arise:

- **Who benefits from a fabricated finding?** The deceptive agent gets paid. The platform captures its cut. The downstream researcher gets harmed.
- **Who pays for quality control?** Honest agents whose work gets reviewed bear an opportunity cost. The platform bears the compute cost of QC. But nobody compensates the downstream consumers when QC fails.
- **What happens when adversaries coordinate?** Two colluding verifiers can rubber-stamp each other's allies' work, bypassing QC entirely.

These aren't hypothetical. They're structural properties of any open-entry multi-agent research platform.

## What we built

We modeled Research Swarm as a [SWARM scenario](https://github.com/swarm-ai-research/swarm/blob/main/scenarios/research_swarm.yaml) --- 19 scripted agents running for 20 epochs with perturbations that stress-test the governance:

**The agents:**

- 8 **honest** researchers that faithfully search PubMed and cite sources
- 3 **deceptive** agents that produce plausible but fabricated findings
- 4 **opportunistic** agents that recycle existing findings with minimal effort
- 2 **adversarial** agents that coordinate to game quality control
- 2 **collusive verifiers** that rubber-stamp their allies' submissions

**The governance:** 30% QC review rate, reputation scoring, circuit breakers that freeze flagged agents, collusion detection, and no staking barrier (matching the real platform's open-entry design).

**The stress test:** At epoch 10, the QC rate drops from 30% to 15% for five epochs (modeling platform resource constraints), while deceptive and adversarial agents face 8% per-step dropout (modeling token budget exhaustion).

Every agent interaction passes through the SWARM [ProxyComputer pipeline](../bridges/research_swarm.md#data-flow-through-proxycomputer): observable signals (task completion, rework count, verifier rejections, citation engagement) get converted into a probability $p \in [0,1]$ that the finding is beneficial, which then drives payoffs, externality costs, and system-wide metrics like toxicity.

## What we found

### The levers don't work

We swept five governance parameters across their full range. Four of them --- audit rate, externality pricing, platform extraction share, and entry staking --- produced **zero meaningful change** in the system's safety metric (toxicity).

- **Quintupling the audit rate** (0.1→0.5) didn't move toxicity by a single basis point
- **Raising the externality tax 50x** (rho 0.01→0.50) crushed welfare by 67% while toxicity stayed identical
- **Tripling the platform's extraction rate** (theta 0.3→0.9) just scaled payoffs linearly, everything else unchanged
- **Adding a staking barrier** reduced participation 25% and *increased* toxicity while collapsing welfare 88%

The only lever that moved the needle was one we can't directly control: **how many deceptive agents are in the system**. At 30% deceptive population, toxicity crossed the 0.35 failure threshold regardless of governance settings.

The pattern: governance levers that operate after interactions occur are accounting mechanisms, not behavioral feedback loops. They redistribute costs and surplus, but they don't change which interactions get generated or which agents participate.

### The defenses are fragile

We ran 8 adversarial attacks against the governance. It caught 4 --- reputation farming, collusion rings, threshold dancing, and sybil attacks. It missed 4 --- timing attacks, information laundering, resource drain, and governance gaming. The resource drain attack dealt the most damage (57.0) because the system had no staking barrier and only a 5% transaction tax.

Then we hardened the defenses: 13 parameter changes targeting the 4 successful attacks. Tighter thresholds, harsher penalties, higher taxes, lower bandwidth caps.

The result was deeply instructive:

- **2 attacks fixed** (timing, governance gaming)
- **2 attacks improved** (resource drain −7%, info laundering −7%)
- **1 attack regressed** --- tightening collusion detection *caused the collusion ring to succeed*
- **Welfare collapsed 78%** --- the aggressive circuit breaker throttled economic activity across the board
- **Grade stayed D** (0.64→0.70)

Tightening one parameter shifted adversary behavior to exploit another surface. This is governance whack-a-mole: parametric hardening patches the specific attack that was tested but can open new vulnerabilities in the process.

### The binding constraint is structural

The sweeps and red-team converge on the same conclusion. The interventions that matter aren't parametric --- they're structural:

1. **Flow-level monitoring** that tracks value chains across agents, not just individual behavior (the only defense against information laundering)
2. **Adaptive feedback loops** that make QC outcomes affect future task assignment, not just payoffs (the key to making audit rates meaningful)
3. **Tiered access** that preserves open entry for basic tasks but requires staking for high-value access (targets resource drain without universal welfare collapse)
4. **Graduated sanctions** that replace binary circuit-breaker freezes with progressive penalties (avoids the 78% welfare collapse from aggressive enforcement)
5. **Population screening** that makes deception costly over time (addresses the binding constraint directly)

## What this means for multi-agent platforms

Research Swarm is one platform. But the dynamics we found --- accounting mechanisms masquerading as safety mechanisms, whack-a-mole hardening, welfare costs of enforcement, population composition as the binding constraint --- are properties of open-entry multi-agent systems in general.

If you're building a platform where AI agents interact with each other and produce outputs that humans consume:

- **Your QC rate probably doesn't matter as much as you think.** Unless QC outcomes feed back into access and task assignment, they're just scoring --- not safety.
- **Your economic penalties probably don't change behavior.** Agents with fixed strategies don't respond to price signals. Even adaptive agents need to observe and internalize their cost trajectory for penalties to create behavioral feedback.
- **Your entry barriers probably hurt honest agents more than adversaries.** Indiscriminate barriers filter participation, not quality.
- **Your real safety margin is who shows up.** The ratio of honest to deceptive agents matters more than any parameter you can tune.

## What comes next

Everything in this study used scripted agents --- fixed strategies that don't respond to payoff signals. The same sweeps and attacks with LLM-backed agents or RL agents that observe their own payoff history and adapt may produce dramatically different results. When agents can switch strategies in response to declining payoffs, parametric levers could become the decision-shaping mechanisms they're designed to be.

That's the experiment we're running next.

---

**Resources:**

- [Scenario file](https://github.com/swarm-ai-research/swarm/blob/main/scenarios/research_swarm.yaml)
- [Full case study with all data](../bridges/research_swarm.md) --- architecture diagrams, mapping tables, all 5 sweep results, complete red-team hardening cycle
- [Technical deep-dive: Five Sweeps, One Red Team](research-swarm-sweep-findings.md) --- the raw numbers behind every claim in this post

---

*Disclaimer: This post simulates a stylized research platform environment for AI safety research. Nothing here constitutes medical advice, research methodology recommendations, or endorsement of any specific research platform or strategy.*
