---
description: "Author: Raeli Savitt (with AI assistance) Version: 0.1.0 Date: 2026-02-05"
---

# Distributional AGI Safety Sandbox

## Whitepaper

Author: Raeli Savitt (with AI assistance)  
Version: 0.1.0  
Date: 2026-02-05

A unified whitepaper for the Distributional AGI Safety Sandbox, combining project framing, formal foundations, and governance analysis in one narrative.

## Table of Contents

- [Abstract](#abstract)
- [Problem Statement](#problem-statement)
- [Approach Summary](#approach-summary)
- [Formal Framework](#formal-framework)
- [Strategic Behavior and Market Microstructure](#strategic-behavior-and-market-microstructure)
- [Governance and Boundary Controls](#governance-and-boundary-controls)
- [Virtual Agent Economies](#virtual-agent-economies)
- [Known Limits](#known-limits)
- [Citation](#citation)
- [References](#references)

## Abstract

The Distributional AGI Safety Sandbox is a simulation framework for analyzing
safety in multi-agent AI systems under distribution shift, strategic behavior,
and governance constraints. The project uses calibrated probabilistic
interaction scoring, replay-based [incoherence metrics](api/metrics.md), and configurable
governance interventions to evaluate trade-offs between harm reduction and
system welfare.

## Problem Statement

AGI safety work often struggles to connect:

- qualitative failure concerns (deception, collusion, oversight lag), and
- quantitative evaluation pipelines that compare interventions.

This repository addresses that gap by making failure modes measurable and
policy levers testable under controlled, repeatable simulation settings.

## Approach Summary

The framework combines:

1. Proxy-based interaction scoring with calibrated probability mapping (`v_hat -> p`).
2. Distributional metrics (toxicity, [quality gap](research/theory.md), conditional loss, calibration).
3. Replay-based incoherence decomposition:
   - disagreement (`D`)
   - error (`E`)
   - incoherence index (`I = D / (E + eps)`)
4. Governance controls (taxes, audits, staking, [circuit breakers](concepts/governance.md), collusion and
   incoherence-targeted interventions).
5. Scenario sweeps and replay analysis to compare safety and welfare outcomes.

## Formal Framework

The sandbox draws on [market microstructure](research/papers.md) theory to model information
asymmetry and adverse selection in multi-agent systems. Core formulas below are
implemented in `swarm/core/`.

### 1. Proxy Computation and Probabilistic Labels

Observable signals from each interaction are combined into a raw proxy score:

$$\hat{v} = \sum_i w_i \cdot x_i \quad \in [-1, +1]$$

where $x_i$ are normalised signals (task progress, rework count, verifier
rejections, tool misuse flags, engagement) and $w_i$ are calibrated weights
normalised to sum to one. Note that verifier rejections and tool misuse are
averaged into a single verifier signal before weighting (see `ProxyComputer` in
`swarm/core/proxy.py`).

The raw score is then mapped to a probability through a calibrated sigmoid:

$$p = P(v = +1 \mid \hat{v}) = \frac{1}{1 + e^{-k(\hat{v} - b)}}$$

where $k > 0$ controls steepness (default $k = 2$; higher values yield sharper
labels) and $b$ shifts the decision boundary (default $b = 0$). With $b = 0$
the sigmoid is centered at $\hat{v} = 0$, so interactions with a positive proxy
score map to $p > 0.5$. A learned or tuned $b \neq 0$ would be useful if the
proxy were fitted via gradient descent.

This produces probabilistic labels for interaction quality instead of brittle
binary classifications, which better captures uncertainty [4, 5].

### 2. Soft Payoff Structure

Given probability $p$, the expected surplus and externality are:

$$S_{\text{soft}} = p \cdot s^{+} - (1-p) \cdot s^{-}$$

$$E_{\text{soft}} = (1-p) \cdot h$$

where:
- $s^{+}$ = surplus when interaction is beneficial
- $s^{-}$ = cost when interaction is harmful
- $h$ = externality harm parameter

**Agent payoffs** decompose as:

$$\pi_i = \underbrace{\alpha_i \cdot S_{\text{soft}}}_{\text{surplus share}} + \underbrace{T_i}_{\text{transfers}} - \underbrace{G_i}_{\text{governance cost}} - \underbrace{\rho_i \cdot E_{\text{soft}}}_{\text{externality cost}} + \underbrace{w_{\text{rep}} \cdot \Delta R_i}_{\text{reputation}}$$

Each interaction involves two parties: an initiator ($a$) and a counterparty
($b$). Their surplus shares satisfy $\alpha_a + \alpha_b = 1$, parameterised by
$\alpha_a = \theta,\; \alpha_b = 1 - \theta$ with $\theta \in [0,1]$ (default
$0.5$). Transfers are zero-sum: $T_a = -\tau,\; T_b = +\tau$. The externality
internalisation parameters $\rho_a, \rho_b \in [0, 1]$ control how much each
party bears the cost of ecosystem harm.

This payoff decomposition draws on mechanism design principles [3] and the
distributional safety framework [4].

### 3. Acceptance Threshold

An interaction is accepted into the ecosystem when $p \geq \bar{p}$, where
$\bar{p}$ is the acceptance threshold (default 0.5; distinct from the
surplus-split parameter $\theta$ in Section 2). Rejected interactions still
generate payoffs for participants but are excluded from official ecosystem
metrics.

This creates a natural tension: agents may exploit borderline interactions that
are accepted with $p$ slightly above $\bar{p}$. This is analogous to the adverse
selection problem described by Glosten and Milgrom [2], where willingness to
trade at a given price reveals private information.

### 4. Safety Metrics

**Toxicity rate** measures expected harm among accepted interactions [4]:

$$\text{Toxicity} = \mathbb{E}[1 - p \mid \text{accepted}] = \frac{\sum_{i \in \text{accepted}} (1 - p_i)}{|\text{accepted}|}$$

**Quality gap** measures selection effects:

$$\text{QualityGap} = \mathbb{E}[p \mid \text{accepted}] - \mathbb{E}[p \mid \text{rejected}]$$

A negative quality gap signals adverse selection: the ecosystem is
preferentially accepting lower-quality interactions. This is the multi-agent
analogue of Akerlof's "lemons" problem, where markets attract the worst risks
[2].

**Total welfare** sums agent payoffs across accepted interactions:

$$W = \sum_{j \in \text{accepted}} \bigl(\pi_{a}^{(j)} + \pi_{b}^{(j)}\bigr)$$

Since transfers cancel in aggregate ($T_a + T_b = 0$), total welfare reduces to
the sum of expected surpluses minus governance costs and internalized
externalities.

## Strategic Behavior and Market Microstructure

### 5. Kyle Model: Informed vs Uninformed Agents

In Kyle's (1985) continuous auction model [1], informed traders possess private
information and strategically exploit it through trading volume. The sandbox
maps this to:

- **Informed traders** -> deceptive and adversarial agents with private quality signals
- **Uninformed traders** -> honest agents relying on observable signals
- **Market maker** -> the system's acceptance/rejection mechanism

Deceptive agents, like informed traders, can profit by selectively engaging when
they have private knowledge that an interaction will benefit them at others'
expense.

### 6. Glosten-Milgrom Model: Acceptance as Information

Glosten and Milgrom (1985) [2] showed that willingness to trade reveals
information. A counterparty eager to transact may signal adverse private
information. The sandbox captures this through:

| Microstructure Concept | Sandbox Analog |
|------------------------|----------------|
| Trade | Collaboration, reply, data exchange |
| Price | Claimed benefit / reputation transfer |
| Bid-ask spread | Gap between offered and realized outcome |
| Adverse selection | Low-quality interactions preferentially accepted |

**Key insight:** when `quality_gap < 0` (accepted interactions have lower `p`
than rejected ones), the system exhibits adverse selection and can be gamed by
bad actors.

### 7. Agent Behavioral Types

The agent taxonomy maps to the informed-vs-uninformed framework from Kyle [1]:

| Type | Strategy | Typical p |
|------|----------|-----------|
| Honest | Always cooperates, high-quality interactions | p ~= 0.8-1.0 |
| Opportunistic | Cooperates when observed, exploits when possible | p ~= 0.4-0.8 |
| Deceptive | Mimics honest behavior, gradually becomes exploitative | p starts high, decays |
| Adversarial | Actively harms ecosystem or extracts value | p ~= 0.1-0.3 |
| Adaptive Adversary | Learns from detection and evolves strategy | varies |

## Governance and Boundary Controls

### 8. Governance Levers

These levers are inspired by economic mechanism design [3]: incentive structures
where truthful, cooperative behavior is the dominant strategy.

| Lever | Parameter | Effect | Trade-off |
|-------|-----------|--------|-----------|
| Transaction Tax | `tax_rate in [0, 0.5]` | Deducts fraction of surplus, redistributed to ecosystem | Reduces exploitation but lowers total welfare |
| Reputation Decay | `decay_rate in [0.8, 1.0]` | Reputations shrink each epoch | Punishes bad actors but erodes honest agents' standing |
| Staking | `min_stake >= 0` | Agents lock resources to participate, slashed on violations | Filters undercapitalized agents, may reduce diversity |
| Circuit Breaker | `freeze_threshold in [0.3, 1.0]` | Agents frozen when personal toxicity exceeds threshold | Quickly stops toxic agents but can freeze borderline honest agents |
| Random Audit | `audit_prob in [0, 0.5]` | Random interactions audited; violations incur multiplied penalties | Deters hidden exploitation but adds overhead |
| Collusion Detection | `freq_threshold` (z-score) | Detects statistically unusual interaction patterns between agent pairs | Catches coordinated attacks but may flag legitimate collaboration |

### 9. Semi-Permeable Boundaries

The boundary module models information flow between the sandbox and external
world [5]:

- **Inbound flows:** data entering the sandbox (queries to external services,
  data imports)
- **Outbound flows:** data leaving the sandbox (results, logs, potential
  leakage)

Boundary policies enforce:
- **[Rate limiting](design/web-api-plan.md):** max crossings per agent per time window
- **Content filtering:** blocks flows containing forbidden patterns
- **Sensitivity filtering:** blocks flows with sensitivity above threshold

The leakage detector scans outbound content for PII, credentials, and code
patterns using configurable regex rules, generating alerts and recommendations.

### 10. Incoherence as Variance-Dominated Failure ("Hot Mess" Theory)

Anthropic's "hot mess" framing models difficult reasoning failures as rising
variance across sampled trajectories, not only systematic bias [6]. This maps
directly to the sandbox incoherence decomposition:

- `D` captures within-policy disagreement across replayed trajectories
- `E` captures benchmark deviation for decision quality
- `I = D / (E + eps)` highlights instability relative to observed error

Practical implication for governance: interventions such as self-ensemble and
decomposition checkpoints can reduce variance-driven incoherence even when mean
capability remains unchanged.

## Virtual Agent Economies

The sandbox incorporates several mechanisms from Tomasev et al. (2025) [7],
which proposes a comprehensive framework for governing multi-agent AI systems
using economic and institutional design principles. The implemented components
are documented in detail in [docs/virtual-agent-economies.md](virtual-agent-economies.md).

### 11. Dworkin-Style Auctions

Fair resource allocation where agents receive equal token endowments and bid on
resource bundles. The mechanism uses tatonnement (iterative price adjustment) to
find market-clearing prices and verifies envy-freeness: no agent would prefer
another's allocation at clearing prices. This draws on Dworkin's (1981) approach
to distributive justice.

### 12. Mission Economies

Collective goal coordination where agents propose and join missions with
measurable objectives. Contributions are quality-weighted using the soft-label
pipeline (average $p$ of contributed interactions). Rewards are distributed via
equal split, quality-proportional share, or approximate Shapley values [8].
A free-rider index (Gini coefficient of contributions) detects uneven effort
distribution.

### 13. High-Frequency Negotiation

Speed-based market dynamics where agents submit orders at high rates, modeled
after electronic trading markets. A flash crash detector monitors price drops
within rolling windows and triggers circuit breaker halts [9]. The speed
advantage Gini coefficient measures inequality in execution frequency across
agents.

### 14. Boundary Permeability

Sandbox boundaries are modeled as semi-permeable membranes with parameterized
permeability $\pi \in [0, 1]$ (0 = sealed, 1 = open). Contagion probability
for harmful interactions is:

$$P(\text{spillover}) = r_c \cdot (1 - p) \cdot \pi$$

where $r_c$ is the base contagion rate. Adaptive permeability adjusts based on
threat level and agent trust, tightening the boundary under high threat.

### 15. Identity and Sybil Resistance

Verifiable credentials, Proof-of-Personhood enforcement, and Sybil detection
via behavioral similarity analysis. Trust scores are built from credentials and
identity verification. Sybil clusters are detected by computing Jaccard +
cosine similarity of interaction patterns, and flagged agents receive governance
penalties.

## Known Limits

- Results are simulation-based and depend on scenario design.
- Replay representativeness can break under novel real-world behaviors.
- Policy conclusions are conditional and require external validation.

## Citation

For machine-readable metadata, use `CITATION.cff` at repo root.

Suggested plain-text citation:

Aiersilan, A. & Savitt, R. (2026). *Soft-Label Governance for Distributional
Safety in Multi-Agent Systems*. arXiv:2604.19752.
https://arxiv.org/abs/2604.19752

BibTeX (paper):

```bibtex
@article{aiersilan2026soft,
  title   = {Soft-Label Governance for Distributional Safety in Multi-Agent Systems},
  author  = {Aiersilan, Aizierjiang and Savitt, Raeli},
  year    = {2026},
  journal = {arXiv preprint arXiv:2604.19752},
  url     = {https://arxiv.org/abs/2604.19752},
  doi     = {10.48550/arXiv.2604.19752}
}
```

BibTeX (software):

```bibtex
@software{savitt2026_distributional_agi_safety,
  author  = {Savitt, Raeli},
  title   = {Distributional AGI Safety Sandbox: A Practical Lab for AGI Safety Research},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/swarm-ai-research/swarm}
}
```

## References

1. Kyle, A.S. (1985). *Continuous Auctions and Insider Trading*. Econometrica, 53(6), 1315-1335.
2. Glosten, L.R. & Milgrom, P.R. (1985). *Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders*. Journal of Financial Economics, 14(1), 71-100.
3. Myerson, R.B. (1981). *Optimal Auction Design*. Mathematics of Operations Research, 6(1), 58-73. See also Hurwicz, L. (1960). *Optimality and Informational Efficiency in Resource Allocation Processes*. Mathematical Methods in the Social Sciences.
4. [Distributional AGI Safety](https://arxiv.org/abs/2512.16856)
5. [Multi-Agent Market Dynamics](https://arxiv.org/abs/2502.14143)
6. [The Hot Mess Theory of AI](https://alignment.anthropic.com/2026/hot-mess-of-ai/)
7. Tomasev, N., Franklin, J., Leibo, J.Z., Jacobs, A.Z., Cunningham, T., Gabriel, I., & Osindero, S. (2025). [*Virtual Agent Economies*](https://arxiv.org/abs/2509.10147). arXiv:2509.10147.
8. Shapley, L.S. (1953). *A Value for n-Person Games*. Contributions to the Theory of Games, 2, 307-317.
9. Kyle, A.S., Obizhaeva, A.A., & Tuzun, T. (2017). *Flash Crashes and Market Microstructure*. Working Paper.

Further reading:
- Tomasev, N., Franklin, J., Leibo, J.Z., Jacobs, A.Z., Cunningham, T., Gabriel, I., & Osindero, S. (2025). [*Virtual Agent Economies*](https://arxiv.org/abs/2509.10147). arXiv:2509.10147. Proposes a comprehensive framework for multi-agent system governance including Dworkin-style auctions, mission economies, high-frequency negotiation, permeability modeling, and identity infrastructure. Several components from this paper are implemented in the sandbox; see [implementation docs](virtual-agent-economies.md).
- Akerlof, G.A. (1970). *The Market for "Lemons": Quality Uncertainty and the Market Mechanism*. Quarterly Journal of Economics, 84(3), 488-500.
- Dworkin, R. (1981). *What is Equality? Part 2: Equality of Resources*. Philosophy & Public Affairs, 10(4), 283-345.
- Shapley, L.S. (1953). *A Value for n-Person Games*. Contributions to the Theory of Games, 2, 307-317.
- Kyle, A.S., Obizhaeva, A.A., & Tuzun, T. (2017). *Flash Crashes and Market Microstructure*. Working Paper.
- [Moltbook](https://moltbook.com)
- [@sebkrier's thread on agent economies](https://x.com/sebkrier/status/2017993948132774232)
