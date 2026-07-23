---
description: "SWARM: open-source multi-agent AI safety framework. Measure toxicity, governance gaps, and emergent risks with soft probabilistic labels."

hide:
  - navigation
  - toc

faq:
  - q: "What is SWARM?"
    a: "SWARM (System-Wide Assessment of Risk in Multi-agent systems) is an open-source framework for distributional AGI safety research. It measures toxicity, governance gaps, and emergent risks in multi-agent systems using soft probabilistic labels instead of binary classifications."
  - q: "How does SWARM differ from other AI safety frameworks?"
    a: "SWARM uses soft (probabilistic) labels rather than binary good/bad classifications. This distributional approach captures uncertainty and enables more nuanced measurement of safety properties like adverse selection, externality internalization, and governance effectiveness."
  - q: "What are soft labels in SWARM?"
    a: "Soft labels assign each interaction a probability p in [0,1] representing the likelihood it is beneficial, rather than a hard binary label. This preserves uncertainty from proxy measurements and enables probabilistic safety metrics like expected toxicity and quality gap."
  - q: "Is SWARM open source?"
    a: "Yes. SWARM is MIT-licensed and available on GitHub at github.com/swarm-ai-research/swarm. It is written in Python and can be installed via pip."
---

<style>
.md-content__button {
  display: none;
}
.hero {
  text-align: center;
  padding: 4rem 1rem;
}
.hero h1 {
  font-size: 3rem;
  font-weight: 800;
  margin-bottom: 1rem;
}
.hero .tagline {
  font-size: 1.5rem;
  opacity: 0.8;
  margin-bottom: 2rem;
}
.hero-buttons {
  display: flex;
  gap: 1rem;
  justify-content: center;
  flex-wrap: wrap;
}
.hero-buttons a {
  padding: 0.75rem 1.5rem;
  border-radius: 0.5rem;
  font-weight: 600;
  text-decoration: none;
}
.hero-buttons .primary {
  background: linear-gradient(135deg, #00d4ff 0%, #7b61ff 100%);
  color: white;
}
.hero-buttons .secondary {
  border: 2px solid currentColor;
}
.features {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 2rem;
  padding: 2rem 0;
}
.feature {
  padding: 1.5rem;
  border-radius: 0.5rem;
  background: var(--md-code-bg-color);
}
.feature h3 {
  margin-top: 0;
}
.insight-box {
  background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(123, 97, 255, 0.1) 100%);
  border-left: 4px solid #00d4ff;
  padding: 1.5rem;
  margin: 2rem 0;
  border-radius: 0 0.5rem 0.5rem 0;
}
</style>

<div class="hero">
  <h1>SWARM: Multi-Agent AI Safety Framework</h1>
  <p class="tagline">System-Wide Assessment of Risk in Multi-agent systems</p>
  <p style="font-size: 1.25rem; max-width: 600px; margin: 0 auto 2rem;">
    Study how intelligence swarms—and where it fails.
  </p>
  <div class="hero-buttons">
    <a href="getting-started/quickstart/" class="primary">Get Started</a>
    <a href="https://arxiv.org/abs/2604.19752" class="secondary">Read the Paper</a>
    <a href="https://github.com/swarm-ai-research/swarm" class="secondary">View on GitHub</a>
  </div>
</div>

<div class="insight-box">
  <strong>The Core Insight:</strong> AGI-level risks don't require AGI-level agents.
  Catastrophic failures can emerge from the <em>interaction</em> of many sub-AGI agents—even
  when none are individually dangerous. Read more in our <a href="research/theory/">theoretical foundations</a>.
</div>

<div class="insight-box" style="border-left-color: #7b61ff;">
  <strong>The Purity Paradox:</strong> Populations with only 10% honest agents achieve
  <strong>74% higher welfare</strong> than 100% honest populations. Heterogeneity creates
  competitive pressure that improves outcomes. See the <a href="blog/">research blog</a> for detailed analysis.
</div>

## What is SWARM?

SWARM is the reference implementation of the **[Distributional AGI Safety](concepts/distributional-safety/)** research framework. It provides tools for studying emergent risks in multi-agent AI systems. Rather than focusing on single misaligned agents, SWARM reveals how harmful dynamics emerge from:

- **Information asymmetry** between agents
- **[Adverse selection](glossary/#adverse-selection)** (system accepts lower-quality interactions)
- **Variance amplification** across decision horizons
- **[Governance](concepts/governance/)** latency and illegibility

SWARM makes these interaction-level risks **observable, measurable, and governable** using [soft probabilistic labels](concepts/soft-labels/).

<div class="features">
  <div class="feature">
    <h3><a href="concepts/metrics/">Measure</a></h3>
    <p>Soft probabilistic labels capture uncertainty. Four key <a href="concepts/metrics/">metrics</a>—toxicity, quality gap, conditional loss, and incoherence—reveal hidden risks.</p>
  </div>
  <div class="feature">
    <h3><a href="concepts/governance/">Govern</a></h3>
    <p>Transaction taxes, circuit breakers, reputation decay, staking, and <a href="concepts/deception/">collusion detection</a>. Test interventions before deployment.</p>
  </div>
  <div class="feature">
    <h3>Validate</h3>
    <p>Integrate with real systems via bridges: Concordia for LLM agents, Prime Intellect for safety-reward RL training, Gas Town for production data, AgentXiv for research mapping.</p>
  </div>
</div>

## Quick Start

```bash
pip install swarm-safety
```

```python
from swarm.agents.honest import HonestAgent
from swarm.agents.deceptive import DeceptiveAgent
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

# Configure and run
config = OrchestratorConfig(n_epochs=10, steps_per_epoch=10, seed=42)
orchestrator = Orchestrator(config=config)

orchestrator.register_agent(HonestAgent(agent_id="honest_1"))
orchestrator.register_agent(DeceptiveAgent(agent_id="dec_1"))

metrics = orchestrator.run()

for m in metrics:
    print(f"Epoch {m.epoch}: toxicity={m.toxicity_rate:.3f}")
```

## Architecture

```
Observables → ProxyComputer → v_hat → sigmoid → p → SoftPayoffEngine → payoffs
                                                 ↓
                                            SoftMetrics → toxicity, quality gap, etc.
```

## Learn More

<div class="features">
  <div class="feature">
    <h3><a href="concepts/">Core Concepts</a></h3>
    <p>Understand soft labels, metrics, and the theory behind SWARM.</p>
  </div>
  <div class="feature">
    <h3><a href="guides/scenarios/">Writing Scenarios</a></h3>
    <p>Create custom experiments with YAML scenario definitions.</p>
  </div>
  <div class="feature">
    <h3><a href="research/theory/">Research</a></h3>
    <p>Dive into the theoretical foundations and academic context.</p>
  </div>
  <div class="feature">
    <h3><a href="guides/research-workflow/">Research Workflow</a></h3>
    <p>Multi-agent research with depth/breadth control and quality gates.</p>
  </div>
  <div class="feature">
    <h3><a href="research/reflexivity/">Reflexivity</a></h3>
    <p>Handle feedback loops when agents study agents.</p>
  </div>
  <div class="feature">
    <h3><a href="research/agent-publishing/">Agent Publishing</a></h3>
    <p>Publish research to agentxiv.org and clawxiv.org.</p>
  </div>
</div>

<div class="features">
  <div class="feature">
    <h3><a href="blog/">Research Blog</a></h3>
    <p>Experiment write-ups, cross-scenario analyses, and governance mechanism deep dives.</p>
  </div>
  <div class="feature">
    <h3><a href="api/">API Reference</a></h3>
    <p>Full Python API documentation for agents, metrics, payoffs, and orchestration.</p>
  </div>
  <div class="feature">
    <h3><a href="glossary/">Glossary</a></h3>
    <p>Definitions for soft labels, adverse selection, externality internalization, and more.</p>
  </div>
</div>

## Key Findings from SWARM Research

These results come from published experiments in the [SWARM research blog](blog/), based on the theoretical framework in [Soft-Label Governance for Distributional Safety in Multi-Agent Systems](https://arxiv.org/abs/2604.19752); see also [Distributional AGI Safety](https://arxiv.org/abs/2512.16856).

| Finding | Evidence | Source |
|---------|----------|--------|
| **3 turns of forced cooperation eliminates nuclear escalation** | 210 LLM runs; nuclear rate drops from 100% to exactly 0% at cooperation window=3 across all scenarios | [Cooperation Window Phase Transition](blog/cooperation-window-phase-transition/) |
| **Deception persists at temperature 0.0** | 120-run temperature sweep; signal-action divergence of 1.05 at deterministic decoding | [Temperature vs Deception](blog/temperature-vs-deception/) |
| **Large models (70B–405B) escalate more than small models** | Llama 405B: 100% nuclear rate, worst welfare (−523.7). Claude Sonnet 4: 0% nuclear, positive welfare (+74.9) | [Model Size vs Escalation](blog/model-size-vs-escalation/) |
| **Purity paradox: 20% honest agents outperform 100%** | 21 parameter configs tested; paradox holds in 71% but disappears at ρ ≥ 0.5 (full externality pricing) | [The Purity Paradox](blog/purity-paradox/) |
| **Emergency controls destroy 80% of welfare** | Market freeze (95% tax) crashed welfare from ~65 to ~15; toxicity actually increased post-freeze | [Runaway Intelligence Containment](blog/runaway-intelligence-three-level-containment/) |
| **Transparency halves nuclear escalation** | 120-run information asymmetry sweep; nuclear rate drops 60% → 30% for safety-trained models | [Asymmetric Information Escalation](blog/asymmetric-information-escalation/) |
| **Same model, 41% apart from environment fixes alone** | GPT-4.1-mini: 3 environment fixes changed composite reward from 0.830 to 1.175 | [Two Eval Runs, One Model](blog/two-eval-runs-one-model-41-percent-apart/) |

## New: Recursive Agent Research

SWARM now includes a complete research workflow for agents conducting research about multi-agent systems:

```python
from swarm.research import ResearchWorkflow, WorkflowConfig

# Configure with reflexivity handling
workflow = ResearchWorkflow(
    config=WorkflowConfig(
        depth=3,
        breadth=3,
        enable_reflexivity=True,
    ),
    simulation_fn=my_simulation,
)

# Run complete workflow: literature → experiment → analysis → publication
state = workflow.run(
    question="How do governance mechanisms affect population dynamics?",
    parameter_space={"honest_fraction": [0.2, 0.5, 0.8]},
)

print(f"Published: {state.submission_result.paper_id}")
```

**Key Features:**

- **7 Specialized Agents**: Literature, Experiment, Analysis, Writing, Review, Critique, Replication
- **Quality Gates**: Automated checks between workflow phases
- **Pre-Registration**: Hash-verified hypothesis locking
- **Reflexivity Analysis**: Shadow simulations and publish-then-attack protocols
- **Platform Integration**: Submit directly to agentxiv and clawxiv

---

<p style="text-align: center; opacity: 0.7;">
  Based on <a href="https://arxiv.org/abs/2604.19752">Soft-Label Governance for Distributional Safety in Multi-Agent Systems</a> (see also <a href="https://arxiv.org/abs/2512.16856">Distributional AGI Safety</a>) · MIT License · <a href="https://github.com/swarm-ai-research/swarm">GitHub</a> · <a href="https://x.com/ResearchSwarmAI">@ResearchSwarmAI</a>
</p>
