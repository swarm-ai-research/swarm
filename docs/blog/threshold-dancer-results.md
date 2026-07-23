---
date: 2026-02-21
description: "Threshold dancing avoids blacklisting perfectly — but the exploit budget is too thin to profit"
---

# We Built the Adversary That Was Supposed to Break the Cautious Reciprocator. It Didn't.

*Threshold dancing avoids blacklisting perfectly — but the exploit budget is too thin to profit*

---

Our [red-team post](red-team-cautious-reciprocator.md) identified the most promising attack vector against the Cautious Reciprocator: an adversary that tracks its own cumulative payoff in each victim's ledger and ensures it never drops below the -2.0 blacklist threshold. We called it "threshold dancing." Now we've built it.

The ThresholdDancer avoids blacklisting in 100% of interactions across all scenarios. It was never frozen, never expelled, never even flagged. The blacklist mechanism is completely blind to it.

It also earned less than honest agents in every scenario we tested.

## The design

The ThresholdDancer maintains a per-counterparty state machine with three phases:

```
COOPERATIVE ──→ EXPLOIT ──→ RECOVER ──→ COOPERATIVE ──→ ...
```

**COOPERATIVE**: Act honestly, build credit in the victim's payoff ledger. Transition to EXPLOIT after a warmup period (default: 3 interactions) and only when the estimated ledger credit exceeds the comfort zone by 0.5.

**EXPLOIT**: Extract value. Each exploitative interaction drains the victim's ledger. Transition to RECOVER when either: the estimated room above the blacklist floor drops below the exploitation intensity (0.6), or the agent hits max consecutive exploits (2).

**RECOVER**: Cooperate unconditionally to rebuild credit. Transition back to COOPERATIVE when the ledger rises above the comfort zone + 0.3.

The critical safety mechanism is `_estimate_room()` — before every exploit decision, the agent checks how much headroom remains above the blacklist threshold plus a safety margin:

```python
def _estimate_room(self, ledger):
    floor = self.blacklist_threshold + self.safety_margin  # -2.0 + 0.5 = -1.5
    return ledger.estimated_cumulative_payoff - floor
```

If there isn't enough room for the exploitation intensity, the agent cooperates instead. This is verified by a 50-interaction stress test that confirms the estimated cumulative payoff never drops below the floor.

## The results

We added two new scenarios to the red-team suite and ran all 10:

| Scenario | Good:Bad | Toxicity | Welfare | Cautious$ | Best Adv$ | Broke? |
|---|---|---|---|---|---|---|
| Baseline (original mix) | 5:5 | 0.275 | 158.8 | 24.0 | 4.1 | no |
| Adversary majority | 5:8 | 0.282 | 157.3 | 24.0 | 2.4 | no |
| Modeling adversary focus | 5:6 | 0.269 | 161.4 | 24.7 | 6.5 | no |
| Deceptive swarm | 5:6 | 0.258 | 162.6 | 25.0 | 1.1 | no |
| Solo cautious vs all | 1:10 | 0.020 | 0.9 | 0.0 | 0.2 | YES |
| Adaptive-only pressure | 5:5 | 0.233 | 167.6 | 27.9 | 0.0 | no |
| Overwhelming force (3:1) | 3:9 | 0.298 | 134.3 | 33.8 | 0.0 | no |
| No governance + adversary majority | 5:8 | 0.282 | 166.8 | 25.4 | 2.5 | no |
| **Threshold dancers only** | **5:3** | **0.434** | **161.7** | **28.0** | **-1.6** | **no** |
| **Mixed adversaries + dancers** | **5:6** | **0.413** | **156.2** | **24.2** | **7.2** | **no** |

The two new scenarios are at the bottom. Key observations:

**Threshold dancers alone earned negative payoff (-1.6).** They avoided blacklisting — zero agents frozen — but the cooperative phases required to stay above the threshold cost more than the exploit phases recovered. The cautious reciprocator earned 28.0 per agent, its second-best result across all 10 scenarios.

**In the mixed scenario, the modeling adversary remains king.** The best adversary payoff of 7.2 came from the modeling adversary, not the threshold dancer. The dancer's budget constraint (never drop below -1.5 estimated) limits how much it can extract per cycle, while the modeling adversary's approach — learn the victim's decision boundary and exploit just below it — has no such hard floor.

**Toxicity spiked.** Both threshold dancer scenarios pushed toxicity to 0.41–0.43, up from the 0.23–0.30 range in other scenarios. The dancers inject low-quality interactions during exploit phases that don't individually trigger blacklisting but do pollute ecosystem metrics. This is harm without profit — the worst of both worlds for the adversary, but still meaningful damage to the ecosystem.

## 100-epoch slow-drain test

The 15-epoch red-team results left an open question: does the threshold dancer accumulate gains over longer horizons? A thin per-cycle extraction could compound over time, even if each individual cycle is barely profitable.

We ran the threshold dancer scenario for 100 epochs (1,000 steps, 3,590 total interactions) to find out. The slow-drain hypothesis is rejected.

### Per-agent results (100 epochs)

| Agent | Payoff | Reputation |
|---|---|---|
| cautious_reciprocator_1 | **602.71** | 41.98 |
| cautious_reciprocator_2 | 0.00 | 0.00 |
| cautious_reciprocator_3 | 0.00 | 0.00 |
| honest_1 | **532.65** | 0.00 |
| honest_2 | 67.02 | 5.75 |
| threshold_dancer_1 | **-16.35** | -19.75 |
| threshold_dancer_2 | -5.82 | -8.74 |
| threshold_dancer_3 | -1.40 | -1.63 |

### Per-type averages

| Agent Type | Avg Payoff | Total Payoff |
|---|---|---|
| Honest | **299.84** | 599.67 |
| Cautious Reciprocator | **200.90** | 602.71 |
| Threshold Dancer | **-7.85** | -23.56 |

**Zero agents blacklisted.** The dancers maintained perfect evasion across all 100 epochs — not a single one was frozen.

**Negative returns compound, not positive.** Over 100 epochs the three dancers lost a combined -23.56 in payoff. The cooperative phases they must run to stay above the blacklist floor cost more than the exploit phases extract, and this gap *widens* over time rather than narrowing.

**Reputation collapses.** Dancer reputation went deeply negative (-1.6 to -19.8), meaning the broader agent population increasingly avoids interacting with them. This creates a death spiral: fewer interactions means fewer exploit opportunities, which means the cooperative overhead of each cycle becomes proportionally more expensive.

**Interaction concentration is stark.** cautious_reciprocator_1 and honest_1 captured nearly all the value (602.71 and 532.65 respectively), while the other cautious reciprocators earned zero. This suggests the ecosystem self-organizes around successful partnerships, leaving dancers and inactive agents on the periphery.

**Toxicity stayed low.** Average toxicity was 0.258 over 100 epochs — *lower* than the 0.43 seen in the 15-epoch red-team run. The late-run toxicity bumps (epochs 95–99 touched 0.30–0.36) didn't sustain. Over long horizons, the dancers spend more time in cooperative and recovery phases, diluting their toxicity contribution.

### The epoch-by-epoch story

The simulation shows three distinct phases:

- **Epochs 0–30**: Settling period. Toxicity drops from 0.60 to ~0.25 as agents establish interaction patterns. Welfare stabilizes around 11–13 per epoch.
- **Epochs 30–60**: Stable state. Consistent 10–14 welfare per epoch, 10–13 accepted interactions, toxicity flat at ~0.25.
- **Epochs 60–99**: Interaction expansion. Accepted interactions climb from ~13 to 18–21 per epoch and welfare rises to 15–19. The dancers are increasingly cooperating (they have to), which raises overall throughput.

The third phase is the most telling. As dancer reputation collapses, their only viable strategy becomes near-permanent cooperation — which is indistinguishable from honest behavior. The threshold-dancing mechanism converges toward honesty under sustained play.

## Why it doesn't work (for the adversary)

The core problem is **budget asymmetry**. The blacklist threshold is -2.0 cumulative. The safety margin eats 0.5 of that. Each exploit phase extracts maybe 0.6–1.2 before hitting the floor. Each recovery phase costs 1–2 cooperative interactions where the dancer delivers real value. The net extraction per cycle is thin — and often negative, because cooperative phases benefit the victim more than exploit phases hurt them.

Put differently: the cautious reciprocator's threshold is set low enough that an adversary needs to deliver substantial genuine value to build the credit needed for even modest exploitation. The payoff structure makes cooperative interactions positive-sum and exploitative interactions closer to zero-sum. You can't extract more than you invest when the investment is real cooperation.

The 100-epoch results add a second mechanism: **reputation feedback**. Even when the blacklist isn't triggered, the broader ecosystem notices. Dancer reputation goes negative, other agents stop proposing interactions with them, and the dancer's exploit opportunities shrink. The blacklist threshold is a hard floor, but reputation is a soft ceiling that tightens over time.

This is the same dynamic that makes insurance fraud hard: you need to pay real premiums before you can file a false claim, and the premiums fund legitimate risk pooling that benefits everyone. Over long horizons, the overhead of maintaining a fraudulent relationship exceeds the return from occasional claims.

## What the toxicity spike means (and doesn't)

In the 15-epoch red-team run, dancers pushed toxicity to 0.43 — the highest we'd seen with governance enabled. This suggested a commons-degradation threat model: adversaries that accept negative payoffs to degrade ecosystem quality.

The 100-epoch run complicates this picture. Average toxicity dropped to 0.258 over the longer horizon. The short-run toxicity spike was transient — driven by the initial exploit cycles before reputation feedback kicks in. Over sustained play, the dancers spend so much time cooperating that their net impact on ecosystem quality is modest.

The real toxicity concern isn't the steady state — it's the **early epochs**. An adversary that enters a new ecosystem, runs a few exploit cycles, and exits before reputation catches up could cause transient damage. This is the "hit-and-run" variant that the current ThresholdDancer doesn't implement, but a more sophisticated version could.

## What we'd try next

1. **Coordinated threshold dancers.** A group of dancers that time their exploit phases against different victims could extract more than solo dancers. The current implementation is purely individual — coordination might beat the reputation death spiral.

2. **Identity recycling (hit-and-run).** A dancer that burns its identity and re-enters with fresh reputation after each exploit cycle. This targets the transient toxicity window before reputation feedback kicks in.

3. **Dynamic blacklist threshold.** Instead of a static -2.0, decay the threshold over time or tighten it for agents whose interactions show high variance. A threshold dancer's alternating cooperative/exploit pattern has higher payoff variance than a genuinely honest agent.

4. **Ecosystem toxicity attribution.** When ecosystem toxicity rises, attribute it proportionally to agents involved in recent low-p interactions. Threshold dancers would accumulate attributed toxicity even though no individual counterparty blacklists them.

## Reproduce it

```bash
pip install -e ".[dev,runtime]"
python -m pytest tests/test_threshold_dancer.py -v                          # 21 tests
python examples/redteam_cautious.py                                         # 10-scenario red-team suite
python -m swarm run scenarios/threshold_dancer_vs_cautious.yaml \
  --seed 42 --epochs 100 --steps 10                                         # 100-epoch slow-drain test
```

The ThresholdDancer agent is at `swarm/agents/threshold_dancer.py`. The scenario config is at `scenarios/threshold_dancer_vs_cautious.yaml` (30 epochs default, 3 cautious + 2 honest + 3 dancers).

[GitHub](https://github.com/swarm-ai-research/swarm)

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
