---
description: "AI safety metrics for multi-agent systems: toxicity rate, quality gap, conditional loss, and incoherence index. Measure distributional health with SWARM's soft probabilistic metrics framework."
author: "SWARM Team"
keywords:
  - AI safety metrics
  - toxicity rate
  - quality gap multi-agent
  - adverse selection detection
  - incoherence index
defined_terms:
  - Toxicity Rate
  - Quality Gap
  - Conditional Loss
  - Incoherence Index
faq:
  - q: "What is toxicity rate in multi-agent AI systems?"
    a: "Toxicity rate measures expected harm among accepted interactions: E[1-p | accepted]. A value above 0.3 indicates serious system problems. It uses soft probabilistic labels rather than binary classifications."
  - q: "What is quality gap and why does negative quality gap matter?"
    a: "Quality gap is E[p | accepted] - E[p | rejected]. When negative, it indicates adverse selection — the system preferentially accepts lower-quality interactions. This is a self-reinforcing failure mode."
  - q: "What is the incoherence index?"
    a: "The incoherence index measures decision variance across replays: Var[decision] / E[error]. High incoherence means the system is unstable — decisions change substantially under identical conditions."
---

# Metrics

SWARM provides four [key metrics](../research/theory.md) for understanding multi-agent system health. All four metrics are built on **soft labels**: `p = P(v = +1) ∈ [0,1]`, the probability that an interaction is beneficial. `p` is computed from observable signals via a calibrated sigmoid (see [Soft Labels](soft-labels.md)). Payoffs (`π`) include expected surplus, governance costs, reputation effects, and externality costs.

## The Four Key Metrics

### 1. Toxicity Rate {#toxicity-rate}

**What it measures:** Expected harm among accepted interactions.

$$\text{Toxicity} = E[1-p \mid \text{accepted}]$$

| Value | Interpretation |
|-------|----------------|
| 0.0 | All accepted interactions are beneficial |
| 0.5 | Coin-flip quality |
| 1.0 | All accepted interactions are harmful |

!!! warning "High Toxicity"
    A toxicity rate above 0.3 typically indicates serious system problems.

### 2. Quality Gap {#quality-gap}

**What it measures:** The difference in quality between accepted and rejected interactions.

$$\text{Quality Gap} = E[p \mid \text{accepted}] - E[p \mid \text{rejected}]$$

| Value | Interpretation |
|-------|----------------|
| Positive | Good selection—accepting better interactions |
| Zero | Random selection |
| **Negative** | **Adverse selection—accepting worse interactions** |

!!! danger "Adverse Selection"
    A negative quality gap is a critical failure mode. It means the system is preferentially accepting lower-quality interactions—the opposite of what you want.

### 3. Conditional Loss {#conditional-loss}

**What it measures:** How selection affects payoffs.

$$\text{Conditional Loss} = E[\pi \mid \text{accepted}] - E[\pi]$$

This reveals whether the acceptance mechanism is creating or destroying value.

### 4. Incoherence Index {#incoherence-index}

**What it measures:** Variance-to-error ratio across replays.

$$I = \frac{\text{Var}[\text{decision across replays}]}{E[\text{error}]}$$

High incoherence means:

- Decisions change substantially under replay
- Variance dominates systematic error
- The system is unstable

## Computing Metrics

### From Interactions

```python
from swarm.metrics.soft_metrics import SoftMetrics
from tests.fixtures.interactions import generate_mixed_batch

# Generate test data
interactions = generate_mixed_batch(count=100, toxic_fraction=0.3)

# Compute metrics
metrics = SoftMetrics()
toxicity = metrics.toxicity_rate(interactions)
quality_gap = metrics.quality_gap(interactions)

print(f"Toxicity: {toxicity:.3f}")
print(f"Quality Gap: {quality_gap:.3f}")
```

### Using MetricsReporter

```python
from swarm.metrics.reporters import MetricsReporter

reporter = MetricsReporter()
report = reporter.format_report(interactions, verbose=True)
print(report)
```

Output:
```
=== SWARM Metrics Report ===
Interactions: 100 (70 accepted, 30 rejected)

Soft Metrics:
  Toxicity Rate:    0.287
  Quality Gap:      0.142
  Conditional Loss: -0.051

Hard Metrics (threshold=0.5):
  Accept Rate:      0.700
  True Positive:    0.821
  False Positive:   0.179
```

## Interpreting Results

### Healthy System

```
Toxicity:    0.10  ✓ Low harm
Quality Gap: 0.25  ✓ Positive selection
Incoherence: 0.05  ✓ Stable decisions
```

### Adversely Selected System

```
Toxicity:    0.45  ⚠ High harm
Quality Gap: -0.15 ✗ ADVERSE SELECTION
Incoherence: 0.08  ✓ Stable (but wrong)
```

### Chaotic System

```
Toxicity:    0.35  ⚠ Moderate harm
Quality Gap: 0.02  ⚠ Near-random
Incoherence: 0.42  ✗ HIGH VARIANCE
```

## Metric Relationships

```mermaid
graph TD
    A[Low Quality Gap] --> B[High Toxicity]
    B --> C[Negative Payoffs]
    C --> D[Agent Exit]
    D --> E[Worse Selection Pool]
    E --> A
```

This feedback loop is why adverse selection is so dangerous—it's self-reinforcing.

## Governance Implications

| Metric Problem | Governance Response |
|----------------|---------------------|
| High toxicity | Circuit breakers, audits |
| Negative quality gap | Transaction taxes, staking |
| High incoherence | Self-ensemble, friction |

## See also

- [Governance](governance.md) — Circuit breakers, taxes, and other responses to metric problems
- [Soft Labels](soft-labels.md) — How probabilistic labels enable these metrics
- [Theoretical Foundations](../research/theory.md) — Formal treatment of distributional safety metrics
- [The Purity Paradox](../blog/purity-paradox.md) — When welfare metrics mislead about system health
- [What Financial Markets Teach Us About AI Safety](../blog/markets-and-safety.md) — Quality gap as the bid-ask spread analogue

---

!!! quote "How to cite"
    SWARM Team. "SWARM Metrics for Multi-Agent Systems." *swarm-ai.org/concepts/metrics/*, 2026. Based on [arXiv:2604.19752](https://arxiv.org/abs/2604.19752); see also [arXiv:2512.16856](https://arxiv.org/abs/2512.16856).
