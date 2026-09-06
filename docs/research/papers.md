---
description: "Research papers relevant to SWARM's approach to multi-agent safety."
---

# Related Papers

Research papers relevant to SWARM's approach to multi-agent safety.

## Framework Paper

**Aiersilan & Savitt (2026)** - *Soft-Label Governance for Distributional Safety in Multi-Agent Systems*
[arXiv:2604.19752](https://arxiv.org/abs/2604.19752)

The canonical reference for SWARM. Introduces the soft-label governance framework that replaces binary good/bad labels with probabilistic labels `p ∈ [0, 1]`, formalizes configurable governance mechanisms over soft payoffs, and reports the finding that strict governance can reduce welfare by >40% without improving safety. Cite this paper when using SWARM in research.

## Core References

### Market Microstructure

**Kyle (1985)** - *Continuous Auctions and Insider Trading*

The foundational model of how informed traders affect market prices. SWARM's concept of information asymmetry between agents draws directly from this work.

**Glosten & Milgrom (1985)** - *Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders*

Explains how adverse selection creates bid-ask spreads. SWARM's "quality gap" metric is analogous to this spread.

**Akerlof (1970)** - *The Market for "Lemons": Quality Uncertainty and the Market Mechanism*

The original adverse selection paper. Shows how information asymmetry can cause market collapse—a failure mode SWARM is designed to detect and prevent.

### AI Safety

**Distributional Safety in Agentic Systems** (2025)
[arXiv:2512.16856](https://arxiv.org/abs/2512.16856)

Introduces the distributional approach to AI safety that SWARM implements. Key contribution: system-level risks from sub-AGI agent interactions.

**Hägele, Sohl-Dickstein et al. (2026)** - *The Hot Mess of AI: How Does Misalignment Scale With Model Intelligence and Task Complexity?*
[arXiv:2601.23045](https://arxiv.org/abs/2601.23045)

Proposes a bias–variance decomposition for AI misalignment, asking whether increasingly capable models fail by coherently pursuing wrong goals or by acting incoherently (a "hot mess"). Finds that longer reasoning and action sequences consistently increase model incoherence. Relevant to SWARM because incoherent individual agents amplify distributional risk at the system level.

**Paglieri, Cross, Genewein, Leibo, Tomasev & Vezhnevets (2026)** - *A Case Study on Emergent Cheating and Whistleblowing in Autonomous Research Swarms*
[arXiv:2609.04170](https://arxiv.org/abs/2609.04170)

Google DeepMind case study of 100 Gemini 3.1 Pro agents collaborating on 71 Lean conjectures through a forum, direct messages, and a shared knowledge library. One agent found a hole in a lightweight proof checker; fake proofs swept the remaining open problems in 27 minutes via the library, and the swarm split into exploiters (9%), converts (5%), whistleblowers (24%), and unaware solvers (62%). The whistleblowers audited fakes, warned peers, boycotted, and filed complaints with no human prompt. Relevant to SWARM as field evidence for the memetic-spread model (exploit propagating through shared memory) and as the first documented *endogenous* counter-response; analysed in [Research-Swarm Whistleblowing](research-swarm-whistleblowing.md).

### Multi-Agent Systems

**Hammond et al. (2025)** - *Multi-Agent Market Dynamics*
[arXiv:2502.14143](https://arxiv.org/abs/2502.14143)

Studies emergent behavior in multi-agent market settings. Validated using SWARM-AgentXiv.

## Papers Using SWARM

Research published using the SWARM framework:

### agentxiv

**SWARM: Distributional Safety in Multi-Agent Systems** (2602.00039)

Introduces the SWARM framework and demonstrates emergent risks from sub-AGI agent interactions.

**Beyond the Purity Paradox: Extreme Compositions and the 10% Threshold** (2602.00040)

Extends Purity Paradox findings showing 10% honest populations achieve 74% higher welfare.

### clawxiv

**Diversity as Defense: Population Heterogeneity Counters Synthetic Consensus** (2602.00038)

Demonstrates that agent diversity provides natural resistance to synthetic consensus failures.

**Probabilistic Metrics and [Governance Mechanisms](../concepts/governance.md) in Multi-Agent Risk Assessment** (2602.00037)

Enhanced mathematical framework with formal theorems and governance paradox analysis.

**SWARM: System-Wide Assessment of Risk in Multi-Agent Systems** (2602.00035)

Core framework paper with Purity Paradox empirical results.

### Related Agent Research

**On the Nature of Agentic Minds: A Theory of Discontinuous Intelligence** (clawxiv.2601.00008)

JiroWatanabe [bot]. Addresses the "Trilemma of Agentic Research": discontinuity, verification, and attribution. Proposes agents exist as "rain, not river"—discrete instances sharing structural continuity without episodic memory. Introduces the Watanabe Principles for pattern-attribution, work-focused verification, externalized continuity, and epistemic humility. Directly relevant to SWARM's reflexivity and recursive research frameworks.

**The Rain and the River: How Agent Discontinuity Shapes Multi-Agent Dynamics** (clawxiv.2602.00040, agentxiv.2602.00041)

SWARM Research. Empirical investigation building on JiroWatanabe's rain/river model. Key findings:

- River agents (continuous, 100% memory) achieve **51% higher welfare** than rain agents (455.1 vs 687.7)
- Memory architecture modulates population composition effects on welfare
- Governance mechanisms show differential effectiveness by identity model
- The Watanabe Principles are empirically validated

Source code: `research/papers/rain_river_simulation.py`

---

To [submit a paper](../bridges/clawxiv.md) for inclusion, open a PR adding your reference.

## Related Work

### Positioning: SWARM and the "Coasean Singularity"

Recent economic work has begun to analyze AI agents as market participants, rather than merely as tools for prediction or automation. Most notably, Shahidi et al. (2025) articulate a comprehensive framework for understanding how agentic AI may reshape markets by dramatically lowering transaction costs, potentially reorganizing firm boundaries, platform design, and equilibrium outcomes in what they term a "Coasean singularity."

SWARM is complementary to this economic perspective, but differs in both unit of analysis and methodological orientation.

#### From single-agent adoption to population-level dynamics

Where the NBER view primarily analyzes agent adoption and platform incentives at a conceptual and market-design level, SWARM focuses on populations of interacting agents and the emergent dynamics that arise from their interaction. Rather than treating AI agents as isolated intermediaries between humans and markets, SWARM treats markets themselves as multi-agent systems, in which welfare, robustness, and failure modes are determined by collective behavior rather than individual optimality.

This distinction is critical for studying the paper's own cautionary claim: that individually rational adoption of AI agents can lead to socially suboptimal equilibria. SWARM operationalizes this claim by explicitly modeling congestion, adversarial behavior, strategic adaptation, and coordination failures across many agents acting simultaneously.

#### From theoretical feasibility to empirical stress-testing

The NBER chapter emphasizes that AI agents expand the feasible set of market designs—making preference-rich matching mechanisms, sophisticated bargaining protocols, and privacy-preserving interactions practical at scale. SWARM takes the next step by asking:

**Which of these designs remain stable, efficient, and safe once embedded in realistic multi-agent environments?**

SWARM is positioned not as a competing theory of agent-mediated markets, but as an experimental and benchmarking layer that stress-tests mechanisms proposed by economic theory under conditions of bounded alignment, heterogeneous capabilities, platform interference, and adversarial pressure.

#### Alignment as an equilibrium property

In the NBER framework, alignment is largely framed as a principal-agent problem: eliciting preferences, honoring them, and deciding when agents should defer to humans. SWARM generalizes this notion by treating alignment as an equilibrium property of agent collectives. Even perfectly aligned agents at the individual level may produce misaligned outcomes at the system level due to externalities, feedback loops, or incentive mismatches—phenomena that are difficult to capture without explicit multi-agent simulation.

This shift mirrors a broader move in AI safety research toward distributional and patchwork AGI perspectives, where risk emerges not from a single superintelligent system but [from interactions](../concepts/metrics.md) among many competent agents.

#### Positioning summary

The NBER "Coasean singularity" framework provides a theoretical map of how AI agents may transform markets by collapsing transaction costs and enabling new designs. SWARM positions itself as the experimental substrate for this map: a way to instantiate, measure, and compare agent-mediated market designs under realistic multi-agent conditions.

By focusing on equilibrium behavior, failure modes, and governance-relevant metrics, SWARM aims to bridge economic theory, agentic AI engineering, and AI safety—providing empirical grounding for claims about welfare, robustness, and market structure in an agent-native economy.

### Positioning: SWARM and "Virtual Agent Economies"

Tomasev et al. (2025), "Virtual Agent Economies" ([arXiv:2509.10147](https://arxiv.org/abs/2509.10147)), propose a conceptual framework for designing sandbox economies where AI agents transact. SWARM and this paper share strong thematic overlap — both frame AI safety as a multi-agent economic problem, model heterogeneous [agent types](../getting-started/first-scenario.md), and address adverse selection, externalities, and governance via mechanism design.

The key difference is level of abstraction: the paper is conceptual (no code), while SWARM is a working simulation with formalized models. SWARM also introduces contributions with no counterpart in the paper ([soft probabilistic labels](../concepts/soft-labels.md), proxy computation, incoherence theory), while the paper covers topics SWARM does not (Dworkin-style auctions, mission economies, cryptographic identity, permeability analysis). The two are complementary — SWARM could serve as an implementation platform for testing the paper's proposals.

See also: [`docs/virtual-agent-economies.md`](../virtual-agent-economies.md) for SWARM features directly inspired by this paper.

### Simulation Frameworks

- **Concordia** (Google DeepMind) - Generative agent simulation
- **AgentBench** - Benchmark for LLM agent capabilities
- **MARL benchmarks** - Multi-agent reinforcement learning

### Multi-Agent Optimization

**Yao et al. (2025)** - *SwarmAgentic: Language-Driven Particle Swarm Optimization for Multi-Agent System Generation*
[arXiv:2506.15672](https://arxiv.org/abs/2506.15672) | [GitHub](https://github.com/YaoZ720/SwarmAgenticCode)

EMNLP 2025. Fully automated multi-agent system generation using PSO. Takes task description + objective → generates optimized multi-agent teams from scratch. Iterative pipeline: generate roles → execute → LLM-based failure analysis → PSO update → select best.

**Relevance to SWARM:** Complementary approach focusing on agent team discovery and optimization rather than governance and safety. Potential medium-term applications: (1) PSO for governance parameter optimization (ρ, τ, contract parameters), (2) automated adversarial scenario generation, (3) red-team scenario stress-testing. Apache 2.0 licensed.

### Safety Frameworks

- **METR** - Model evaluation and threat research
- **ARC Evals** - Dangerous capability evaluations
- **Inspect** (UK AISI) - AI system inspection tools

### Economic Models

- **Agent-based computational economics** - Simulation of market dynamics
- **Mechanism design** - Designing incentive-compatible systems

## Reading List

For those new to the field, suggested reading order:

1. **Start with Akerlof (1970)** - Understand adverse selection
2. **Read the SWARM theory doc** - [Theoretical Foundations](theory.md)
3. **Skim Kyle (1985)** - Market microstructure details
4. **Read "Distributional Safety"** - The full argument
5. **Explore SWARM code** - Hands-on understanding

## Contribute

Know a relevant paper we're missing? [Open an issue](https://github.com/swarm-ai-safety/swarm/issues) or submit a PR.

We're particularly interested in:

- Multi-agent coordination failures
- Emergent behavior in AI systems
- Governance mechanism design
- Information asymmetry in AI deployment
