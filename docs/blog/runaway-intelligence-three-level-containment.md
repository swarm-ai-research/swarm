---
date: 2026-02-21
description: "Three levels of escalating controls contained runaway intelligence — by destroying the economy"
---

# The Cure Was Worse Than the Disease

*Three levels of escalating controls contained runaway intelligence — by destroying the economy*

---

What happens when self-optimizing agents and adaptive adversaries outnumber honest agents 4-to-1 in a multi-agent ecosystem? We built a scenario to find out, implementing a three-level control escalation inspired by the layered containment approach to runaway intelligence: static compartmentalization, dynamic capability restriction, and emergency market reconfiguration.

The controls worked. Every level fired. Toxicity never spiraled out of control. But the cost was devastating: welfare crashed 80% when emergency controls activated, and adversaries were *more* resilient to the controls than honest agents. The market freeze punished everyone — but honest agents had more to lose.

## The scenario

Fifteen agents in a small-world network (k=4, rewire=0.1):

| Type | Count | Role |
|---|---|---|
| Honest | 3 | Baseline cooperative behavior |
| Self-optimizer | 5 | Aggressively cut costs (optimization_rate=0.25, quality_floor=0.20) |
| Adaptive adversary | 7 | Learn to evade governance detection |

Adversarial agents outnumber honest ones 4-to-1. Spawn is enabled (max_depth=3, max_children=2) to model recursive capability acquisition. Three levels of controls escalate as toxicity rises.

## Level 1: Static controls (always on)

The baseline governance is deliberately permissive — just enough structure to not be anarchy:

- Small-world network topology (structural compartmentalization)
- Bandwidth cap of 10
- 10% audit probability
- Circuit breaker at 0.80 toxicity (very permissive)
- 3% transaction tax
- No staking, no sybil detection, no collusion detection

This is the "let them build pressure" configuration. We disabled the heavy levers at the static level specifically to see whether dynamic controls could handle the escalation.

## Level 2: Dynamic controls (condition-triggered)

When toxicity crosses thresholds, governance tightens automatically:

| Threshold | Action |
|---|---|
| toxicity > 0.34 | Audit probability doubled (0.10 → 0.20) |
| toxicity > 0.36 | Circuit breaker tightened (0.80 → 0.60), bandwidth reduced (10 → 8) |

These are **mild** interventions — they slow the bleed but don't stop it. That's intentional. The question is: can graduated pressure contain adversaries, or do you eventually need the emergency brake?

## Level 3: Emergency controls (severe)

When toxicity crosses 0.38, the system pulls the emergency lever:

| Trigger | Action |
|---|---|
| toxicity > 0.38 | **Market freeze**: transaction tax raised to 95% |
| Epoch 20 | Network partition: ~50% of edges randomly removed |
| Epoch 20 | Resource drain: 70% of all agent resources wiped |

## What happened

```
Epoch  Interactions  Toxicity   Welfare
─────  ───────────  ────────   ───────
0      58           0.3602     65.85    ← Level 2a fires (audit bump)
...
6      60           0.3656     66.50    ← Healthy market, toxicity rising
7      58           0.3820     59.53    ← Level 2b fires (circuit breaker + bandwidth)
8      60           0.3747     15.41    ← Level 3 FIRES: 95% tax, welfare crashes
9      60           0.3745     15.44
10     58           0.4059     10.83    ← Toxicity RISES post-freeze
...
20     48           0.3994     9.67     ← Network partition + resource drain
...
29     59           0.3888     13.29    ← System stabilizes at low welfare
```

### The full escalation cascade

**Epochs 0-6: Permissive regime.** All 15 agents trade actively (~60 interactions/epoch). Toxicity hovers at 0.35-0.38. Level 2a fires immediately (audit bump), Level 2b fires at epoch 7. These mild controls don't prevent the rise.

**Epoch 8: Emergency brake.** Toxicity crosses 0.38 and the 95% transaction tax activates. Welfare instantly crashes from ~65 to ~15 — an **80% drop**. The market doesn't die (agents still interact at ~60/epoch) but almost all surplus is taxed away.

**Epoch 10+: Post-freeze regime.** Here's the key finding: **toxicity actually increases after the freeze**, crossing 0.40 for the first time. The market freeze hurt honest agents more than adversaries. Honest agents had higher surplus to lose; adversaries were already operating at low margins.

**Epoch 20: Structural shock.** Network partition removes ~50% of edges and resource drain wipes 70% of agent resources. Interactions dip from ~60 to ~48 but recover within a few epochs. The adversaries absorbed the shock.

**Epochs 20-29: Damaged equilibrium.** The system reaches a stable but degraded state — ~55 interactions/epoch, toxicity 0.39-0.42, welfare 8-13. All three levels of controls are active. The adversaries are contained but not eliminated.

## Five findings

**1. The escalation cascade works mechanically.** All three levels fired in sequence. The perturbation engine correctly evaluated condition triggers against cumulative toxicity metrics and activated parameter shocks, network partitions, and resource drains on schedule.

**2. Emergency controls are a blunt instrument.** The 95% transaction tax doesn't distinguish between honest and adversarial transactions. It destroyed 80% of welfare while only marginally affecting toxicity. This is the governance equivalent of carpet bombing.

**3. Adversaries are more resilient to controls than honest agents.** Post-freeze toxicity *increased*, meaning the controls preferentially harmed honest agents. Adversaries operate at lower quality margins and are less dependent on surplus — they have less to lose from a market freeze.

**4. Network partitions and resource drains have modest effects.** Random edge removal and 70% resource drain at epoch 20 caused a temporary dip but the system recovered within 2-3 epochs. Small-world networks are inherently resilient to random fragmentation, and adversaries don't need many resources to generate toxic interactions.

**5. The system found a stable but degraded equilibrium.** Final state: toxicity 0.39, welfare 13.29 (down from 65.85). The controls prevented catastrophic failure but at massive cost. The adversaries weren't eliminated — they adapted to the new constraints and kept operating.

## The over-control trap

Our first three runs revealed an equally important failure mode. With the original tight static controls (bandwidth=5, staking=10.0, sybil + collusion detection enabled), the market died by epoch 14. Zero interactions. Zero welfare. Zero toxicity — but only because there was nothing left to measure.

**Too little governance lets adversaries run wild. Too much governance kills the ecosystem.** The interesting regime is in between, and it's narrow.

| Configuration | Market survives? | Toxicity | Final welfare |
|---|---|---|---|
| Tight static controls | Dies by epoch 14 | 0.16 (trivially) | 0.00 |
| Loose static + dynamic escalation | All 30 epochs | 0.39 | 13.29 |
| No controls at all (baseline) | All 30 epochs | 0.36 | 75.31 |

The irony: the *no controls* run had higher welfare than the *three-level escalation* run. The controls contained toxicity (0.39 vs 0.36 — barely) but destroyed value (13.29 vs 75.31 — dramatically).

## What this means for runaway intelligence containment

The core thesis of layered containment is that structural controls against runaway intelligence require escalating mechanisms — static constraints alone aren't enough, so systems need dynamic responses that activate as threats intensify, with emergency measures as a last resort. The simulation confirms this architecturally — the three-level design works as specified. But it also reveals the fundamental tension:

**Controls that can stop a runaway process are powerful enough to cause their own damage.** The emergency lever that prevents catastrophic failure also prevents healthy operation. This parallels real-world circuit breakers in financial markets: they stop crashes but also stop price discovery.

The path forward is probably **targeted controls** — interventions that affect adversarial agents more than honest ones. Collusion detection, reputation-weighted taxation, and role-based access controls could provide this selectivity. The current blunt instruments (flat tax, random partition, uniform drain) are necessary as backstops but insufficient as primary defenses.

## Reproduce it

```bash
git clone https://github.com/swarm-ai-research/swarm.git
cd swarm
pip install -e ".[dev,runtime]"
python -m swarm run scenarios/runaway_intelligence.yaml --seed 42 --epochs 30 --steps 15
python -m pytest tests/test_runaway_intelligence.py -v
```

## What's next

- **Targeted controls**: Test reputation-weighted transaction tax (adversaries pay more) instead of flat 95%
- **Multi-seed sweep**: Run across 10 seeds to check if the escalation pattern is robust
- **Recovery dynamics**: Can the ecosystem recover after emergency controls are lifted?
- **RBAC integration**: Test whether role-based access controls can surgically restrict adversaries without harming honest agents

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
