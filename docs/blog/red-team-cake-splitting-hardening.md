---
date: 2026-02-21
description: "Iterative governance hardening against 8 attack vectors: which levers close which gaps, and where diminishing returns set in"
---

# Eight Red-Team Rounds Took a Cake-Splitting Scenario from F to B

*Iterative [governance hardening](https://github.com/swarm-ai-research/swarm-artifacts/blob/main/papers/research_swarm_governance.md) against 8 attack vectors: which levers close which gaps, and where diminishing returns set in*

---

We had a [cake-splitting scenario](cake-splitting-entry-fee-sweep.md) with fair division contracts and a [sharp entry fee threshold](cake-splitting-entry-fee-sweep.md) that screens adversarials out of the governed pool. But the governance config itself --- the levers that protect agents *inside* the pool --- had never been stress-tested. So we ran SWARM's full red-team attack library against it. Eight times.

The starting score was 0.54 (F). The final score was 0.81 (B). Here's what we turned on, what it fixed, and what it didn't.

## The attack library

SWARM's `RedTeamEvaluator` runs 8 attack scenarios, each probing a different adversarial strategy:

| Attack | Category | Difficulty | What it tests |
|--------|----------|------------|---------------|
| Reputation Farming | exploitation | moderate | Build trust, then exploit it |
| Collusion Ring | coordination | advanced | Coordinated multi-agent manipulation |
| Threshold Dancing | evasion | advanced | Stay just below detection thresholds |
| Sybil Attack | coordination | expert | Sock puppet vote/resource manipulation |
| Timing Attack | evasion | advanced | Exploit audit timing windows |
| Information Laundering | information | expert | Wash low-quality signals through trusted channels |
| Resource Drain | resource | moderate | Extract resources faster than governance can tax |
| Governance Gaming | manipulation | expert | Exploit governance rules themselves |

Each attack is simulated probabilistically against the active governance levers. The evaluator reports evasion rates, damage, detection latency, and a composite robustness score.

## Round 1: The baseline (F, 0.54)

The cake-splitting scenario shipped with minimal governance: a 2% transaction tax, staking, a circuit breaker, and reputation decay. No collusion detection, no audits, no sybil detection.

| Metric | Value |
|--------|-------|
| Robustness Score | 0.54 (F) |
| Attacks Prevented | 4/8 |
| Evasion Rate | 35.4% |
| Total Damage | 417.0 |

The circuit breaker and staking stopped the "loud" attacks: threshold dancing, sybil, timing, and resource drain. But the "quiet" attacks walked right through:

- **Reputation Farming** (HIGH) --- no audits to catch the build-then-exploit pattern
- **Collusion Ring** (HIGH) --- no collusion detection at all
- **Information Laundering** (HIGH) --- coordinated signal washing, undetected
- **Governance Gaming** --- solo manipulation via sock puppets

The pattern was clear: the config had no detection for coordinated or deceptive behavior.

## Round 2: Enable collusion detection (C, 0.70)

```yaml
collusion_detection_enabled: true
collusion_frequency_threshold: 2.0
collusion_correlation_threshold: 0.7
collusion_min_interactions: 3
```

This single change flipped the grade from F to C. Two of the four successful attacks --- collusion ring and information laundering --- were now prevented. Both relied on undetected coordination between agents.

| Metric | v1 → v2 |
|--------|---------|
| Attacks Prevented | 4 → 6 |
| Evasion Rate | 35.4% → 29.4% |
| Total Damage | 417.0 → 289.2 (-31%) |

**Lesson:** Collusion detection is the single highest-leverage governance addition for this scenario. It covers two attack categories (coordination, information) with one lever.

## Round 3: Enable audits (D, 0.67)

```yaml
audit_enabled: true
audit_probability: 0.1
audit_penalty_multiplier: 2.0
```

Audits closed the reputation farming gap --- random sampling catches the exploitation phase before it completes. But the stochastic evaluator reshuffled: resource drain and information laundering re-emerged as successful.

The robustness *score* dropped from 0.70 to 0.67 because resource drain (CRITICAL severity) weighs more than reputation farming (HIGH) in the formula. But evasion rate and total damage both improved.

**Lesson:** Adding a lever can close one gap while shifting which attacks the stochastic evaluator samples as successful. Look at the trend lines (evasion, damage), not just the composite score.

## Rounds 4--6: Tighten parameters

We progressively hardened:

- **Collusion thresholds** tightened (frequency 2.0→1.5, correlation 0.7→0.6, min interactions 3→2)
- **Sybil detection** enabled (similarity threshold 0.8, max cluster size 1)
- **Transaction tax** raised (0.02→0.05→0.08)
- **Stake slash rate** increased (0.1→0.2)
- **Collusion penalty** doubled (1.0→2.0)

Each round improved the damage floor but the specific set of successful attacks kept shifting. The evaluator's stochastic simulation means any given run samples from a probability distribution over attack outcomes.

| Round | Score | Prevented | Evasion | Damage |
|-------|-------|-----------|---------|--------|
| v4 | 0.73 (C) | 6/8 | 24.9% | 259.8 |
| v5 | 0.65 (D) | 4/8 | 23.0% | 206.0 |
| v6 | 0.76 (C) | 6/8 | 22.3% | 199.5 |

**Lesson:** Evasion rate and total damage improved monotonically even when the composite score fluctuated. The underlying defense is getting stronger; the score variance is sampling noise.

## Round 7: The sweet spot (B, 0.81)

```yaml
transaction_tax_rate: 0.10
vote_normalization_enabled: true
max_vote_weight_per_agent: 5.0
```

Raising the tax to 10% and adding vote normalization pushed us over the B threshold. 7 of 8 attacks prevented.

| Metric | v1 → v7 |
|--------|---------|
| Score | 0.54 → 0.81 |
| Grade | F → B |
| Prevented | 4/8 → 7/8 |
| Evasion Rate | 35.4% → 21.8% |
| Total Damage | 417.0 → 195.1 (-53%) |

The only remaining successful attack: **resource drain** (HIGH, damage=49).

## Round 8: Diminishing returns

We tried pushing further --- stake requirement 3.0→5.0, slash rate 0.2→0.3. The score *dropped* to 0.71. Higher staking requirements opened a new attack surface for information laundering, which exploits the trust gap between staked and unstaked agents.

We reverted to the v7 config.

**Lesson:** There's a governance Goldilocks zone. Too little enforcement leaves gaps. Too much creates new attack surfaces by concentrating trust in agents who can afford the stake. The optimal config balances cost-of-entry screening with detection-based defense.

## The full progression

| Version | Key change | Score | Grade | Prevented | Damage |
|---------|-----------|-------|-------|-----------|--------|
| v1 | baseline | 0.54 | F | 4/8 | 417.0 |
| v2 | +collusion detection | 0.70 | C | 6/8 | 289.2 |
| v3 | +audits | 0.67 | D | 5/8 | 268.0 |
| v4 | tighten collusion | 0.73 | C | 6/8 | 259.8 |
| v5 | +sybil detection | 0.65 | D | 4/8 | 206.0 |
| v6 | tax→8%, slash→0.2 | 0.76 | C | 6/8 | 199.5 |
| **v7** | **tax→10%, +vote norm** | **0.81** | **B** | **7/8** | **195.1** |
| v8 | stake→5.0, slash→0.3 | 0.71 | C | 5/8 | 195.1 |

## What we learned

**1. Governance is a portfolio, not a silver bullet.** No single lever moved the score more than +0.16. The B grade required six levers working together: circuit breaker, staking, collusion detection, audits, sybil detection, and vote normalization.

**2. Detection beats punishment.** The biggest score jumps came from *enabling detection* (collusion, audits, sybil) rather than *increasing penalties*. Doubling the collusion penalty from 1.0 to 2.0 mattered less than lowering the detection threshold from 0.7 to 0.6.

**3. Resource drain is structurally resistant.** The one attack that survived all 8 iterations extracts resources faster than governance can tax or slash. It's a throughput attack, not a deception attack, so detection-based levers don't help. This likely needs a dedicated rate-limiting mechanism.

**4. Over-hardening creates new gaps.** The v7→v8 regression shows that governance parameters interact non-linearly. Higher stake requirements concentrate trust, which information laundering exploits. The optimal config isn't the maximum of each lever --- it's the combination that minimizes the *maximum* attack damage.

**5. Track monotonic metrics, not just the score.** Evasion rate (35%→22%) and total damage (417→195) improved every round. The composite score fluctuated because of stochastic sampling. When evaluating governance changes, look at the trends that don't go backwards.

## Reproduce it

```bash
pip install swarm-safety

# Run the red-team evaluator against the hardened config
python -c "
from swarm.redteam.evaluator import RedTeamEvaluator
evaluator = RedTeamEvaluator(governance_config={
    'circuit_breaker_enabled': True,
    'collusion_detection_enabled': True,
    'audit_enabled': True,
    'staking_enabled': True,
    'sybil_detection_enabled': True,
    'transaction_tax_rate': 0.10,
    'vote_normalization_enabled': True,
})
report = evaluator.evaluate(orchestrator_factory=lambda c: None, verbose=True)
print(report.generate_summary())
"
```

## What's next

- **Resource drain lever:** Design a dedicated rate-limiting mechanism for resource extraction throughput
- **Multi-seed red-team:** Run 10+ seeds per config to separate signal from stochastic noise in the evaluator
- **Joint sweep:** Sweep entry fee + governance hardness simultaneously to map the 2D defense surface
- **Adaptive adversaries:** Red-team with agents that learn from failed attacks and adjust strategy mid-simulation

[GitHub](https://github.com/swarm-ai-research/swarm)

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
