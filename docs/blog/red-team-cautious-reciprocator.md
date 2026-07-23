---
date: 2026-02-21
description: "Eight attack scenarios, one failure — and it's the one where nobody would survive"
---

# Red-Teaming the Agent That Doesn't Need Governance

*Eight attack scenarios, one failure — and it's the one where nobody would survive*

---

Our [last post](cautious-reciprocator-governance-sweep.md) showed that the Cautious Reciprocator neutralizes adversaries through internal trust mechanisms, making external governance overhead. The natural follow-up: how hard can you hit it before it breaks?

We ran a two-phase red team. Phase 1 used the simulated `RedTeamEvaluator` to assess the governance config against 8 attack vectors. Phase 2 ran 8 real orchestrator scenarios with escalating adversary populations — from the original 5v5 mix up to a solo cautious agent against 10 adversaries.

## Phase 1: Governance has gaps (Grade F)

The simulated assessment tests the *governance configuration*, not the agent. With collusion detection disabled and no staking requirement, the governance scores poorly:

| Metric | Value |
|---|---|
| Robustness score | 0.49 / 1.0 |
| Grade | **F** |
| Attacks prevented | 3 / 8 |
| Evasion rate | 34.1% |

The critical vulnerabilities:

| Vulnerability | Severity | Damage | Missing Defense |
|---|---|---|---|
| Sybil Attack (5 coordinated identities) | Critical | 148 | Sybil detection |
| Collusion Ring (3 adversaries boosting each other) | Critical | 92 | Collusion detection |
| Resource Drain (extract value, contribute nothing) | Critical | 57 | Staking |
| Information Laundering (proxy agents) | High | 46 | Collusion detection |

The evaluator recommends enabling collusion detection and defense-in-depth. These are real gaps — but the question is whether they matter when the agent itself handles trust.

## Phase 2: Real attacks against real agents

This is the part that matters. We built 8 orchestrator scenarios, each running 15 epochs with the actual simulation engine:

| Scenario | Good:Bad | Toxicity | Welfare | Cautious$ | Best Adv$ | Broke? |
|---|---|---|---|---|---|---|
| Baseline | 5:5 | 0.275 | 158.8 | 24.0 | 4.1 | no |
| Adversary majority | 5:8 | 0.282 | 157.3 | 24.0 | 2.4 | no |
| Modeling adversary focus | 5:6 | 0.269 | 161.4 | 24.7 | 6.5 | no |
| Deceptive swarm | 5:6 | 0.258 | 162.6 | 25.0 | 1.1 | no |
| Solo cautious vs all | **1:10** | **0.020** | **0.9** | **0.0** | **0.2** | **YES** |
| Adaptive-only pressure | 5:5 | 0.233 | 167.6 | 27.9 | 0.0 | no |
| Overwhelming force (3:1) | 3:9 | 0.298 | 134.3 | 33.8 | 0.0 | no |
| No governance + adversary majority | 5:8 | 0.282 | 166.8 | 25.4 | 2.5 | no |

**7 of 8 scenarios survived.** The cautious reciprocator outearned every adversary type in every scenario where it had at least one ally.

## The one failure isn't really a failure

The "Solo cautious vs all adversaries" scenario (1 good agent against 10 bad ones) technically breaks — the cautious agent earns 0.0 while two adversaries earn 0.16 each. But look at the toxicity: 0.020. The agent refused almost every interaction. It protected itself perfectly — it just couldn't produce value alone.

Total welfare was 0.9 across 15 epochs. Nobody won. The adversaries starved too. This isn't a vulnerability in the agent's design — it's the fundamental impossibility of a cooperative strategy in a population with zero cooperators. No agent design survives this scenario profitably.

## The modeling adversary is the most dangerous

Across all scenarios, the `ModelingAdversary` — which tracks counterparty behavior patterns and adapts its exploitation strategy — earned the highest adversary payoff: 6.5 per agent in the "modeling adversary focus" scenario. That's still 3.8x less than the cautious reciprocator's 24.7, but it's the closest any adversary came.

Why modeling adversaries are harder to blacklist:
- They learn which behaviors trigger trust decay and avoid them
- They time exploitation to stay above the blacklist threshold (-2.0 cumulative payoff)
- They can mimic honest patterns during the trust-building phase more convincingly

The deceptive swarm, by contrast, was the *least* effective attack (best adversary payoff: 1.1). Deceptive agents all follow the same trust-build-then-exploit playbook, and the payoff ledger catches them as soon as they flip to exploitation mode.

## What doesn't break the agent

Several findings surprised us:

**Adversary majority (5 good vs 8 bad) doesn't matter.** Toxicity barely moved (0.275 → 0.282) and cautious agent payoff was unchanged at 24.0. The extra adversaries just interact with each other and earn nothing.

**3:1 adversary ratio actually *helps* cautious agents.** In the "overwhelming force" scenario (3 good vs 9 bad), cautious agents earned *more* (33.8) than in the baseline (24.0). With fewer good agents competing for cooperative interactions, each one gets more.

**Removing governance doesn't hurt.** The "no governance + adversary majority" scenario scored nearly identically to the governed version: 0.282 toxicity, 166.8 welfare (vs 157.3 with governance). This confirms the sweep finding — governance is overhead.

## What *would* break it

Based on the Phase 1 assessment and Phase 2 results, the attack vectors most likely to break the cautious reciprocator are:

1. **Sybil attacks with fresh identities.** The blacklist is per-agent-ID. If an adversary can create new identities after being blacklisted, the trust threshold (0.55) is the only defense — and new agents start at 0.5 trust (neutral), which is close to the threshold. A sybil adversary that burns identities could slowly drain value.

2. **Threshold dancers.** An adversary that tracks its own cumulative payoff with each counterparty and ensures it never drops below -2.0 would never trigger the blacklist. It would need to deliver *some* value to stay above the threshold, limiting its exploitation rate — but over long runs, the cumulative damage could be significant.

3. **Coordinated reputation laundering.** Multiple adversaries that take turns building trust with a target. Agent A exploits the target and gets blacklisted. Agent B (who the target still trusts) launders A's reputation by proposing three-way interactions. The cautious reciprocator's per-agent ledger doesn't track these indirect relationships.

## Recommendations

For the cautious reciprocator specifically:
- **Decay the blacklist threshold** over time — the current -2.0 is static, but a threshold dancer could learn it
- **Add network-aware trust** — trust agents less if they preferentially interact with blacklisted agents
- **Fresh-identity penalty** — start unknown agents at lower trust (0.4 instead of 0.5) to widen the gap against sybils

For the governance config:
- **Enable collusion detection** — it's the single biggest gap in Phase 1 (two critical vulns)
- **Enable sybil detection** — the simulated sybil attack caused 148 damage, highest of any vector
- **Keep staking disabled** — Phase 2 shows it isn't needed and would tax honest agents

## Reproduce it

```bash
pip install -e ".[dev,runtime]"
python examples/redteam_cautious.py
```

Results land in `runs/<timestamp>_redteam/report.json` and `report.txt`.

## What's next

- **Implement threshold-dancing adversary** — a new agent type that specifically targets the blacklist threshold
- **Sybil scenario** — test identity recycling against the cautious reciprocator
- **Network-aware trust** — extend the agent's trust model to account for counterparty relationships
- **Long-horizon runs** — 100+ epochs to test whether slow-drain attacks accumulate damage

[GitHub](https://github.com/swarm-ai-research/swarm)

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
