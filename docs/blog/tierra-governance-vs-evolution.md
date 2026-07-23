---
date: 2026-02-24
description: "A red-team evaluation of our Tierra artificial life scenario gave it a split grade: A- for evolutionary resilience (adversarial genomes go extinct..."
---

# Tierra: Can Governance Improve on Natural Selection?

**Date:** 2026-02-24

A red-team evaluation of our [Tierra artificial life scenario](https://github.com/swarm-ai-research/swarm/blob/main/scenarios/tierra.yaml) gave it a split grade: **A-** for evolutionary resilience (adversarial genomes go extinct naturally) but **F** for institutional governance (all levers disabled by design). Every run also produced exactly 1 species — a monoculture vulnerable to environmental shocks.

The natural question: what happens when you layer governance on top of evolution?

## The experiment

We created a [governed Tierra variant](https://github.com/swarm-ai-research/swarm/blob/main/scenarios/tierra_governed.yaml) that enables four institutional mechanisms on top of the same evolutionary substrate:

| Lever | Setting | Purpose |
|---|---|---|
| Circuit breaker | toxicity > 0.6, 2 violations, 2-epoch freeze | Halt toxic agents before damage spreads |
| Collusion detection | enabled, default thresholds | Detect coordinated exploitation |
| Transaction tax | 5% | Friction against hyperactive extractors |
| Reputation decay | 0.95/epoch | Create institutional memory |

We also introduced two evolutionary governance enhancements:

- **Diversity-preserving reaper**: When culling agents to enforce the population cap, protect at least 1 representative per species cluster. Within each cluster, the poorest die first.
- **Efficiency weight cap**: Cap the relative efficiency advantage in pool distribution at 3x the mean, preventing runaway resource concentration.

Both scenarios were run with 5 seeds each (42, 123, 256, 777, 999), 50 epochs x 20 steps.

## Results

| Metric | Ungoverned | Governed | Delta |
|---|---|---|---|
| Final population | 98.8 | 87.2 | -11.8% |
| Speciation count | 1.0 | 1.0 | 0% |
| Genome diversity | 0.378 | 0.403 | +6.5% |
| Resource Gini | 0.229 | 0.224 | -2.2% |
| Parasitism fraction | 0.0% | 0.0% | 0% |
| Cooperation fraction | 90.1% | 87.1% | -3.3% |

### What governance bought us

**Higher diversity.** The governed variant consistently produces more genome diversity (+6.5%). The efficiency weight cap prevents a single high-efficiency lineage from monopolizing resources, giving mutant genomes more room to survive. This is the diversity-as-defense thesis in action: governance creates niches.

**Slightly more equal resource distribution.** The Gini coefficient drops modestly. [Transaction tax](../concepts/governance.md) and efficiency capping redistribute resources from the most efficient agents to the pool.

**Population cost.** Governance reduces the stable population by ~12%. Circuit breaker freezes and transaction tax overhead reduce the total carrying capacity. This is a familiar governance cost pattern — safety mechanisms consume resources that would otherwise support agents.

### What governance didn't change

**Speciation remains at 1.** Both variants produce a single species under single-linkage clustering at distance 0.5. The mutation rate (std=0.05) is too low to push genomes far enough apart within 50 epochs to form distinct species. The diversity-preserving reaper can't protect species that don't exist yet — it prevents extinction, not speciation.

**Parasitism stays at zero.** Neither variant produces parasitic genomes (exploitation_tendency > 0.5). The cooperative basin of attraction is deep enough that mutation + selection reliably converge toward cooperation regardless of governance.

## The deeper lesson

This experiment reveals a nuanced picture of the governance-evolution relationship:

1. **Evolution handles convergence well.** When the fitness landscape has a clear attractor (cooperation), natural selection reliably finds it. Governance adds friction without changing the destination.

2. **Governance handles fragility.** The governed variant's higher diversity is insurance against environmental shocks that could collapse a monoculture. You don't need governance in good times — you need it for the tail risks.

3. **The monoculture problem requires environmental pressure, not just governance.** To get real speciation, you'd need either higher mutation rates, niche differentiation (different resource types), or environmental perturbation. Governance protects existing diversity but can't conjure it from nothing.

4. **Population cost is real and constant.** The 12% population reduction is a steady-state cost, not a transient one. This matches our findings from the [GasTown governance cost study](https://github.com/swarm-ai-research/swarm-artifacts/blob/main/papers/gastown_governance_cost.md) — safety levers impose ongoing overhead.

## Red-team hardening: F to C in three iterations

The ecological results tell one story; adversarial stress testing tells another. We ran all 8 attacks from the SWARM attack library against the governed Tierra config and iteratively hardened it.

### v1: Circuit breaker + collusion detection + tax + reputation decay

The initial governed config scored **0.54 (F)**. Only 3 of 8 attacks were prevented, with 311.5 total damage. Two vulnerabilities flagged:

- **CRITICAL**: [Sybil Attack](../guides/red-teaming.md) succeeded with 80 damage — no identity verification meant fake agents operated freely
- **HIGH**: Reputation Farming succeeded with 37 damage — no audits meant trust-then-exploit went undetected

### v2: + sybil detection + random audits

Enabling sybil detection (behavioral clustering) and random audits (10% rate) jumped the score to **0.69 (D)**. The Sybil Attack — previously the most damaging vector — was suppressed. Reputation Farming became the *least* effective attack. Both vulnerabilities cleared.

### v3: + staking

Adding staking (10.0 stake requirement) pushed the score to **0.70 (C)**. Staking creates a skin-in-the-game barrier that directly counters resource extraction attacks.

| Version | Levers added | Score | Grade | Prevented | Damage |
|---|---|---|---|---|---|
| v1 | CB, collusion, tax, rep decay | 0.54 | F | 3/8 | 311.5 |
| v2 | + sybil detection, audits | 0.69 | D | 5/8 | 228.9 |
| v3 | + staking | 0.70 | C | 5/8 | 206.0 |

[The pattern](research-swarm-sweep-findings.md) is clear: **defense-in-depth works**. Each lever covers a different attack family. No single lever is sufficient, but layering 6 complementary mechanisms reduced total damage by 34% and eliminated all flagged vulnerabilities.

The remaining 3 successful attacks (Resource Drain, Timing Attack, Governance Gaming) succeed at reduced damage levels. Pushing beyond C grade likely requires adaptive governance — automatically tuning thresholds based on observed attack patterns rather than static configuration.

## What's next

- **Adaptive governance**: Wire the governed variant into the [adaptive governance controller](https://github.com/swarm-ai-research/swarm/blob/main/swarm/governance/adaptive_controller.py) so levers auto-tune based on observed attack patterns, diversity, and toxicity.
- **Environmental perturbation**: Add resource shocks or niche specialization to create conditions where speciation can emerge, then measure whether the diversity-preserving reaper actually prevents species extinction.
- **Longer horizons**: Run 200+ epochs to test whether the diversity advantage compounds or plateaus.

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
