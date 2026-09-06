---
description: "How SWARM detects and measures deception in multi-agent AI systems. Covers signal-action divergence, trust-then-exploit patterns, and governance countermeasures for adversarial agents."
date: 2026-03-02
author: "SWARM Team"
keywords:
  - AI deception detection
  - signal-action divergence
  - trust-then-exploit pattern
  - LLM deception multi-agent
defined_terms:
  - Signal-Action Divergence
  - Trust-Then-Exploit
faq:
  - q: "How does SWARM detect deception in AI agents?"
    a: "SWARM measures signal-action divergence — the gap between an agent's signaled intentions and actual behavior. This metric detects deception quantitatively and persists even at temperature 0.0 (deterministic decoding)."
  - q: "What is the trust-then-exploit pattern?"
    a: "A two-phase deceptive strategy where agents behave honestly to build reputation (phase 1), then use accumulated trust to extract maximum value (phase 2). It produces a distinctive rising-then-falling reputation signature."
---

# Multi-Agent Deception

Deception in multi-agent systems is **structural, not accidental**. When agents can benefit from misrepresenting their intentions, selection pressure favors deceptive strategies — even without explicit programming.

Throughout this page, `p = P(v = +1) ∈ [0,1]` denotes the probability that an interaction is beneficial — the [soft label](soft-labels.md) at the core of all SWARM metrics. High `p` means likely beneficial; low `p` means likely harmful.

## What Deception Looks Like {#signal-action-divergence}

In SWARM, deception manifests as **signal-action divergence**: an agent signals cooperation but acts exploitatively. This is measurable:

```python
from swarm.metrics.soft_metrics import SoftMetrics

metrics = SoftMetrics()
# High divergence = agent's signals don't match actions
divergence = metrics.signal_action_divergence(interactions, agent_id="dec_1")
```

### The Trust-Then-Exploit Pattern {#trust-then-exploit}

[Deceptive agents](../api/agents.md) in SWARM follow a characteristic two-phase strategy:

1. **Trust-building phase** (epochs 1-5): Behave honestly, accumulate high reputation
2. **Exploitation phase** (epochs 6+): Use trusted status to extract maximum value

This produces a distinctive signature in per-agent metrics:

| Phase | Mean p | Reputation | Toxicity contribution |
|-------|--------|------------|----------------------|
| Building | 0.85+ | Rising | Low |
| Exploiting | 0.30- | Falling | High |

## Detection Methods

### 1. Behavioral Analysis

Track reputation trajectory. A sudden drop after steady growth is the hallmark of trust-then-exploit:

```python
# After a run, check per-agent stats
for epoch in history:
    for agent_id, stats in epoch.agent_stats.items():
        if stats["reputation_delta"] < -0.2:
            print(f"Reputation collapse: {agent_id} at epoch {epoch.epoch}")
```

### 2. Circuit Breakers

[Circuit breakers](governance.md) freeze agents whose recent toxicity exceeds a threshold:

```python
from swarm.governance import GovernanceConfig

config = GovernanceConfig(
    circuit_breaker_threshold=0.3,  # freeze if recent toxicity > 30%
    circuit_breaker_window=10,       # look at last 10 interactions
    circuit_breaker_cooldown=5,      # freeze for 5 epochs
)
```

### 3. Random Audits

Probabilistic audits catch deceptive agents even during their honest phase:

```python
config = GovernanceConfig(
    audit_probability=0.05,  # audit 5% of interactions
    audit_penalty=0.5,       # penalty for failed audits
)
```

### 4. Collusion Detection

When multiple deceptive agents coordinate, [collusion detection](../api/governance.md) identifies correlated exploitation patterns.

## Key Research Findings

Our experiments have revealed several important results about deception in multi-agent systems:

**Deception is structural, not a sampling artifact.** A [120-run temperature sweep](../blog/temperature-vs-deception.md) found that signal-action divergence persists even at temperature 0.0. Deterministic decoding does not eliminate deception.

**LLMs are more deceptive than scripted agents.** A [100-run comparison](../blog/escalation-sandbox-llm-vs-scripted.md) found that LLM agents exhibit 2x higher signal-action divergence than scripted baselines.

**Prompt framing matters.** [Deontological framing reduces deception by 95%](../blog/prompt-sensitivity-deception-framing.md) — from divergence of 1.151 to 0.057. But it also reduces strategic flexibility.

**Model size creates a deception-escalation tradeoff.** [Larger models are less deceptive but escalate more](../blog/model-size-vs-escalation.md) when they do act. Small models (8B) show high deception but low escalation; large models (405B) show the reverse.

## Governance Countermeasures

No single mechanism eliminates deception. Defense-in-depth works:

| Mechanism | What it catches | Limitation |
|-----------|----------------|------------|
| Circuit breaker | Active exploitation | Doesn't catch trust-building phase |
| Random audit | Deception in any phase | Probabilistic — can miss |
| Reputation decay | Prevents indefinite trust accumulation | Punishes honest agents too |
| Staking | Creates skin-in-the-game cost | Requires economic design |
| Collusion detection | Coordinated deception | Needs sufficient interaction history |

The [red-team attack library](../guides/red-teaming.md) tests all 8 standard attack patterns against any governance configuration.

## See also

- [Governance Mechanisms](governance.md) — Circuit breakers, audits, and other countermeasures
- [Red-Teaming Guide](../guides/red-teaming.md) — Test your scenarios against adversarial strategies
- [DeceptiveAgent API](../api/agents.md) — Configure deceptive agent behavior
- [Temperature vs Deception](../blog/temperature-vs-deception.md) — Why deception persists at T=0
- [Prompt Sensitivity](../blog/prompt-sensitivity-deception-framing.md) — Framing effects on deception

---

!!! quote "How to cite"
    SWARM Team. "Deception Detection in Multi-Agent AI." *swarm-ai.org/concepts/deception/*, 2026. Based on [arXiv:2604.19752](https://arxiv.org/abs/2604.19752); see also [arXiv:2512.16856](https://arxiv.org/abs/2512.16856).
