---
description: "Glossary of distributional AI safety terms: formal definitions for soft labels, toxicity rate, quality gap, adverse selection, governance mechanisms, and more. The authoritative reference for SWARM's multi-agent safety vocabulary."
author: "SWARM Team"
keywords:
  - distributional safety glossary
  - AI safety terminology
  - multi-agent systems glossary
  - soft labels definition
  - quality gap definition
  - adverse selection AI
defined_terms:
  - Distributional Safety
  - Soft Label
  - p
  - v_hat
  - Toxicity Rate
  - Quality Gap
  - Adverse Selection
  - Conditional Loss
  - Incoherence Index
  - Signal-Action Divergence
  - Circuit Breaker
  - Transaction Tax
  - Reputation Decay
  - Staking
  - Collusion Detection
  - Random Audit
  - Purity Paradox
  - Trust-Then-Exploit
  - Governance Latency
  - Variance Amplification
  - Information Asymmetry
  - Externality Internalization
---

# Glossary

Formal definitions for terms used in the SWARM distributional safety framework. Each term links to its primary concept page for full treatment.

---

## Distributional Safety {#distributional-safety}

The study of how risks emerge from **populations of interacting agents** rather than from any single model. Shifts the unit of analysis from "is this agent aligned?" to "is this ecosystem healthy?" — [Distributional Safety](concepts/distributional-safety.md)

## Soft Label {#soft-label}

A probabilistic classification $p \in [0, 1]$ replacing binary good/bad labels. Captures uncertainty about whether an interaction is beneficial, enabling calibrated metrics and proportional governance. — [Soft Labels](concepts/soft-labels.md)

## p {#p}

The probability that an interaction is beneficial: $p = P(v = +1)$, where $v \in \{-1, +1\}$ is the latent true value. Always in $[0, 1]$. Computed from observable signals via the ProxyComputer pipeline. — [Soft Labels](concepts/soft-labels.md)

## v_hat {#v-hat}

The raw proxy score before sigmoid transformation, $\hat{v} \in [-1, +1]$. A weighted combination of observable signals (task progress, rework count, verifier rejections, engagement) that is then mapped to $p$ via a calibrated sigmoid. — [Soft Labels](concepts/soft-labels.md)

## Toxicity Rate {#toxicity-rate}

Expected harm among accepted interactions: $\text{Toxicity} = E[1 - p \mid \text{accepted}]$. A toxicity rate above 0.3 typically indicates serious system problems. — [Metrics](concepts/metrics.md)

## Quality Gap {#quality-gap}

The difference in expected quality between accepted and rejected interactions: $\text{Quality Gap} = E[p \mid \text{accepted}] - E[p \mid \text{rejected}]$. Positive means good selection; **negative means adverse selection** — the system is preferentially accepting lower-quality interactions. — [Metrics](concepts/metrics.md)

## Adverse Selection {#adverse-selection}

A failure mode where the system preferentially admits lower-quality interactions than it rejects, indicated by a negative [quality gap](#quality-gap). Self-reinforcing: low quality gap → higher toxicity → agent exit → worse selection pool → lower quality gap. — [Distributional Safety](concepts/distributional-safety.md)

## Conditional Loss {#conditional-loss}

How the acceptance mechanism affects payoffs: $\text{Conditional Loss} = E[\pi \mid \text{accepted}] - E[\pi]$. Reveals whether selection creates or destroys value relative to the population average. — [Metrics](concepts/metrics.md)

## Incoherence Index {#incoherence-index}

Variance-to-error ratio across replays: $I = \frac{\text{Var}[\text{decision across replays}]}{E[\text{error}]}$. High incoherence means decisions change substantially under replay — the system is unstable. — [Metrics](concepts/metrics.md)

## Signal-Action Divergence {#signal-action-divergence}

The measurable gap between an agent's signaled intentions and its actual behavior. The primary quantitative indicator of deception in multi-agent systems. Persists even at temperature 0.0 (deterministic decoding). — [Deception](concepts/deception.md)

## Circuit Breaker {#circuit-breaker}

A governance mechanism that freezes agents whose recent toxicity exceeds a threshold. Monitors a sliding window of interactions and suspends agents during a cooldown period. Trade-off: may produce false positives on honest agents. — [Governance](concepts/governance.md)

## Transaction Tax {#transaction-tax}

A friction mechanism that deducts a percentage from both parties' payoffs on each interaction. Reduces the profit margin for low-quality interactions, making exploitation less attractive. Trade-off: reduces overall welfare including for honest agents. — [Governance](concepts/governance.md)

## Reputation Decay {#reputation-decay}

A governance lever that reduces agent reputation by a fixed fraction each epoch. Forces agents to continuously demonstrate good behavior rather than coasting on historical trust. Trade-off: honest agents also lose reputation over time. — [Governance](concepts/governance.md)

## Staking {#staking}

A requirement that agents post collateral to participate. Bad behavior results in stake being slashed, creating skin-in-the-game costs that deter exploitation. Trade-off: excludes agents without sufficient capital. — [Governance](concepts/governance.md)

## Collusion Detection {#collusion-detection}

A governance mechanism that monitors pairwise interaction patterns for suspiciously correlated exploitation timing. Flags agent pairs exceeding a correlation threshold over a sliding window. Trade-off: may flag legitimate cooperation. — [Governance](concepts/governance.md)

## Random Audit {#random-audit}

Probabilistic review of a fraction of interactions. Failed audits result in reputation and payoff penalties. Creates deterrence uncertainty for exploitative agents even during their trust-building phase. — [Governance](concepts/governance.md)

## Purity Paradox {#purity-paradox}

The empirical finding that populations with only 20% honest agents achieve 55% higher welfare than 100% honest populations. Mixed agent diversity outperforms purity because honest-only populations lack the adversarial pressure that activates governance mechanisms. — [The Purity Paradox](blog/purity-paradox.md)

## Trust-Then-Exploit {#trust-then-exploit}

A two-phase deceptive strategy: (1) build trust by behaving honestly for several epochs, then (2) use accumulated reputation to extract maximum value. Produces a distinctive signature of rising then sharply falling per-agent reputation. — [Deception](concepts/deception.md)

## Governance Latency {#governance-latency}

The delay between a safety problem emerging and governance mechanisms responding effectively. Creates a fundamental tension between responsiveness and stability — by the time a circuit breaker triggers, damage may have already propagated. — [Distributional Safety](concepts/distributional-safety.md)

## Variance Amplification {#variance-amplification}

The compounding of small per-interaction risks across a population. A 5% chance of harm per interaction becomes near-certainty across thousands of interactions. Soft probabilistic labels capture this where binary labels hide it. — [Distributional Safety](concepts/distributional-safety.md)

## Information Asymmetry {#information-asymmetry}

When agents have unequal access to information about interaction quality. The better-informed party can exploit the gap, creating a market for lemons where high-quality agents exit the ecosystem. — [Distributional Safety](concepts/distributional-safety.md)

## Externality Internalization {#externality-internalization}

The degree to which agents bear the cost of ecosystem harm they cause, controlled by ρ (rho) parameters. Higher internalization makes agents pay for negative externalities, aligning individual incentives with system health. — [Theoretical Foundations](research/theory.md)

---

!!! quote "How to cite this glossary"
    SWARM Team. "SWARM Glossary: Distributional AGI Safety Definitions." *swarm-ai.org/glossary/*, 2026. Based on [arXiv:2604.19752](https://arxiv.org/abs/2604.19752); see also [arXiv:2512.16856](https://arxiv.org/abs/2512.16856).
