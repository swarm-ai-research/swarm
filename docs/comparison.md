---
description: "SWARM occupies a specific niche: governance simulation for multi-agent AI systems using soft probabilistic labels. Most other frameworks focus on agent..."
---

# SWARM vs Other Frameworks

SWARM occupies a specific niche: **governance simulation for multi-agent AI systems using [soft probabilistic labels](concepts/soft-labels.md)**. Most other frameworks focus on agent capabilities, benchmarking, or single-agent evaluation. SWARM focuses on what happens when agents interact — and when those interactions go wrong.

## Feature Comparison

| Feature | SWARM | Concordia | AgentBench | METR | Inspect (AISI) |
|---|---|---|---|---|---|
| **Primary focus** | Multi-agent governance & safety | LLM agent simulation | LLM capability benchmarks | Dangerous capability evals | AI system inspection |
| **Multi-agent interaction** | Core design | Yes | Limited | Limited | Limited |
| **Soft probabilistic labels** | p = P(beneficial) for every interaction | No | No | No | No |
| **Adverse selection metrics** | Toxicity, quality gap, conditional loss | No | No | No | No |
| **Governance levers** | 6 configurable (tax, reputation, staking, circuit breakers, audits, collusion detection) | None built-in | None | None | Compliance rules |
| **Collusion detection** | Pair-wise frequency & correlation monitoring | No | No | No | No |
| **Replay-based incoherence** | Yes — variance-to-error ratio across replays | No | No | No | No |
| **Agent behavioral types** | Honest, opportunistic, deceptive, adversarial, adaptive, LLM | LLM-driven | LLM-driven | LLM-driven | LLM-driven |
| **LLM agent support** | Yes (Anthropic, OpenAI, Ollama) | Yes (primary) | Yes (primary) | Yes | Yes |
| **Scenario configs** | 23 YAML scenarios | Custom | Benchmark suites | Task suites | Eval suites |
| **Network topology** | Small-world, complete, dynamic edge evolution | No | No | No | No |
| **Economic mechanisms** | Payoff engine, auctions, escrow, staking | No | No | No | No |
| **Red-teaming framework** | 8 attack vectors, automatic scoring | No | No | Core focus | No |
| **Framework bridges** | Concordia, OpenClaw, GasTown, Ralph, AgentXiv, ClawXiv | — | — | — | — |
| **License** | MIT | Apache 2.0 | MIT | Varies | MIT |

## Complementary, Not Competitive

SWARM is designed to work alongside other frameworks:

- **Concordia** provides rich LLM agent simulation with narrative-driven environments. SWARM's [Concordia bridge](bridges/concordia.md) lets you run Concordia agents through SWARM's governance and metrics layer — adding adverse selection measurement and governance stress-testing to Concordia's simulation capabilities.

- **AgentBench** evaluates what individual [LLM agents](guides/scenarios.md) can do. SWARM evaluates what happens when multiple agents (of any kind) interact in a shared environment with governance constraints.

- **METR** focuses on whether AI systems have dangerous capabilities. SWARM focuses on whether multi-agent *ecosystems* exhibit dangerous dynamics — a complementary concern that can emerge even from individually safe agents.

- **Inspect** provides compliance-oriented inspection of AI systems. SWARM provides simulation-based stress-testing of [governance mechanisms](concepts/governance.md) before deployment.

## Why This Matters for Deployment

Most agent evaluation frameworks measure whether agents behave well *within a single trajectory*. SWARM measures whether that good behavior survives replay, redistribution, and recursive composition — and whether humans can tell the difference.

Projects like [Infinite Backrooms](https://dreams-of-an-electric-mind.webflow.io/) document what happens when locally fluent AI systems are observed over time: interactions that *feel* coherent, reflective, and intentional turn out to be unstable across context switches, role permutations, and extended horizons. This is the default regime for current systems.

SWARM formalizes this via the **illusion delta** (`Δ = C_perceived − C_distributed`), which quantifies the gap between how coherent a system *appears* from single-run signals and how consistent it *actually is* under replay. A high Δ is the precise condition under which standard evaluations give false confidence and governance mechanisms ship untested.

| Approach | Question it answers |
|---|---|
| Single-agent evals (AgentBench, METR) | *"Does the agent perform well?"* |
| Compliance inspection (Inspect) | *"Does the system follow rules?"* |
| Multi-agent simulation (Concordia) | *"How do agents interact in context?"* |
| **SWARM** | *"Does the system still behave when humans stop noticing the cracks?"* |

## When to Use SWARM

Use SWARM when you need to answer questions like:

- At what adversarial fraction does my governance mechanism fail?
- Is my system experiencing adverse selection (preferentially accepting bad interactions)?
- Does [collusion detection](governance.md) extend the viable operating range?
- How does welfare scale with agent count under different governance configurations?
- What is the trade-off between governance aggressiveness and ecosystem throughput?

## Looking for Industry Comparisons?

For comparisons with industry platforms like Knostic, Lumenova AI, Gravitee, and the Cooperative AI Foundation, see [SWARM vs Alternatives](swarm-vs-alternatives.md).

## When to Use Something Else

- **Single-agent evaluation**: Use AgentBench or Inspect
- **Dangerous capability assessment**: Use METR
- **Narrative-driven LLM simulation** (without governance analysis): Use Concordia directly
- **Production monitoring**: SWARM is a research/simulation tool, not a production monitoring system (though its metrics could inform monitoring design)
