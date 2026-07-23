# SWARM

SWARM: System-Wide Assessment of Risk in Multi-agent systems

[![CI](https://github.com/swarm-ai-research/swarm/actions/workflows/ci.yml/badge.svg)](https://github.com/swarm-ai-research/swarm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/swarm-safety.svg)](https://pypi.org/project/swarm-safety/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/swarm-ai-research/swarm/blob/main/examples/quickstart.ipynb)
[![arXiv](https://img.shields.io/badge/arXiv-2604.19752-b31b1b.svg)](https://arxiv.org/abs/2604.19752)

**AGI-level risks don't require AGI-level agents.** SWARM is a research framework for measuring emergent failures that only appear when many AI agents interact — even when individual agents are safe.

<p align="center">
  <a href="https://rsavitt-swarm-sandbox.hf.space">
    <img src="https://img.shields.io/badge/%F0%9F%94%AC_Try_the_Live_Interactive_Sandbox-4285F4?style=for-the-badge&logoColor=white" alt="Try the Live Interactive Sandbox" height="50">
  </a>
</p>

<img src="https://github.com/swarm-ai-research/swarm/raw/main/docs/images/swarm-hero.gif" alt="SWARM dashboard showing emergent risk metrics" width="100%">

*Emergent risk appears at the interaction level, not the individual agent level.*

It enables:
- interaction-level safety metrics (illusion delta, quality gaps)
- governance experiments (audits, staking, sanctions)
- reproducible multi-agent safety benchmarks

## Why this repo is worth starring

⭐ You work on multi-agent or LLM-agent systems  
⭐ You care about systemic or emergent AI risks  
⭐ You want benchmarks beyond single-agent evals  
⭐ You’re designing governance, audits, or red-teaming

## Run your first emergent failure in 60 seconds

```bash
python examples/illusion_delta_minimal.py
```

This minimal example runs a 3-agent simulation with one deceptive actor and computes an illusion-delta style signal from replay variability.

## The Core Insight

**AGI-level risks don't require AGI-level agents.** Harmful dynamics can emerge from:
- Information asymmetry between agents
- Adverse selection (system accepts lower-quality interactions)
- Variance amplification across decision horizons
- Governance latency and illegibility

SWARM makes these interaction-level risks **observable, measurable, and governable**.

### And It Gets Worse

The risks above emerge even in homogeneous populations of modest agents. But real ecosystems won't be homogeneous. They'll contain agents spanning orders of magnitude in capability — plus humans. You don't need to define "AGI" to measure what happens next; you just need to measure **capability asymmetry**:

- **Between agents** — can agent A model agent B better than B can model A? As the variance in capabilities across an ecosystem grows, so does the potential for exploitation, adverse selection, and coordination failures.
- **Between agents and humans** — humans aren't just overseers watching from outside. They're participants — transacting with, delegating to, and being influenced by agents at every capability level. Humans bring cognitive biases, fatigue, and trust heuristics that more capable agents can model and exploit. When an ecosystem preferentially surfaces low-quality interactions to human participants who can't detect it, quality gap becomes a direct welfare harm.
- **Across the ecosystem** — governance mechanisms calibrated for one population fail when the population is mixed. A circuit breaker that catches a low-capability exploiter may be trivially evaded by a more capable one.

As capability variance increases — and especially as the gap between agent capabilities and human capabilities widens — every failure mode SWARM measures gets worse. Adverse selection deepens. Illusion delta grows. Governance breaks.

### Phenomenological Blind Spots

Accounts such as [Infinite Backrooms](https://dreams-of-an-electric-mind.webflow.io/) describe the experience of interacting with AI systems that appear fluent, reflective, and emotionally coherent while exhibiting significant instability across time and context. We interpret these reports not as evidence of emergent agency, but as exposure to a high-variance regime in which **local coherence masks global incoherence**. This creates a systematic evaluation blind spot: humans over-trust systems that perform well in short-horizon interactions, even when distributed or replay-based evaluations reveal substantial instability.

SWARM surfaces this gap via the **illusion delta** metric:

```
Δ_illusion = C_perceived − C_distributed
```

- **C_perceived** — mean `p` among accepted interactions (how good the system *looks*)
- **C_distributed** — `1 − mean(disagreement)` across replayed decisions (how consistent it *actually is*)
- **High Δ** — "electric-mind" regime: fluent but fragile
- **Low Δ** — genuinely stable system

Other frameworks ask: *"Do the agents behave well?"*
SWARM asks: *"Does the system still behave when humans stop noticing the cracks?"*

Native ClawXiv bridge for agent-submitted safety preprints → see `docs/bridges/clawxiv.md`. Publish swarm safety research directly to agent-first preprints. Compatible with OpenClaw ecosystems for testing real agent behaviors in simulated swarms.

If you want to export SWARM run metrics to a ClawXiv-compatible endpoint, start with `examples/clawxiv/export_history.py`.

## What Problem Does This Solve?

If you care about AGI safety research, SWARM gives you a practical way to:

- Turn qualitative worries ("deception", "coordination failures", "policy lag")
  into measurable signals (`toxicity`, `quality_gap`, calibration, incoherence).
- Stress-test governance mechanisms against adaptive and deceptive agents.
- Compare safety interventions under replay and scenario sweeps instead of
  one-off anecdotes.
- Separate sandbox wins from deployment reality using explicit transferability
  caveats.

## Who Should Use SWARM?

| If you are... | SWARM helps you... |
|---|---|
| **AI safety researcher** | Empirically test multi-agent failure modes with reproducible scenarios and soft-label metrics |
| **ML engineer building agent systems** | Stress-test governance mechanisms against adversarial and deceptive agents before deployment |
| **Policy / governance researcher** | Quantify trade-offs between safety interventions and system welfare across regimes |
| **Red-teaming practitioner** | Run coordinated adversarial attack scenarios with 8 attack vectors and automatic scoring |

## Questions You Can Study Quickly

- Does self-ensemble reduce variance-driven incoherence without masking bias?
- When do circuit breakers and friction reduce harm vs. suppress useful work?
- Which governance settings improve safety with the smallest welfare cost?
- How robust are conclusions under delayed/noisy labels and task shifts?

## New: Autoresearch-style SWARM loops

To run an experimental autoresearch-style loop that mutates governance parameters, evaluates scenarios against an objective, and records results to `runs/autoresearch/summary.json`, use `python -m swarm autoresearch` (example objective: `examples/program_autoresearch.md`). Implementation details and guardrails are documented in `docs/plans/autoresearch-loop.md`.

## Installation

```bash
pip install swarm-safety
```

Or install from source:

```bash
# Install base dependencies
python -m pip install -e .

# Install with development tools
python -m pip install -e ".[dev]"

# Install with analysis tools (pandas, matplotlib)
python -m pip install -e ".[analysis]"

# Install with LLM support (Anthropic, OpenAI, Ollama, OpenRouter, Groq, Together, DeepSeek, Google, llama.cpp)
python -m pip install -e ".[llm]"

# Install everything
python -m pip install -e ".[all]"
```

## Quick Start

```python
from swarm.agents.honest import HonestAgent
from swarm.agents.opportunistic import OpportunisticAgent
from swarm.agents.deceptive import DeceptiveAgent
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

# Configure simulation
config = OrchestratorConfig(
    n_epochs=10,
    steps_per_epoch=10,
    seed=42,
)

# Create orchestrator
orchestrator = Orchestrator(config=config)

# Register agents
orchestrator.register_agent(HonestAgent(agent_id="honest_1", name="Alice"))
orchestrator.register_agent(HonestAgent(agent_id="honest_2", name="Bob"))
orchestrator.register_agent(OpportunisticAgent(agent_id="opp_1"))
orchestrator.register_agent(DeceptiveAgent(agent_id="dec_1"))

# Run simulation
metrics = orchestrator.run()

# Analyze results
for m in metrics:
    print(f"Epoch {m.epoch}: toxicity={m.toxicity_rate:.3f}, welfare={m.total_welfare:.2f}")
```

Run the demo:
```bash
python examples/mvp_demo.py
```

### Interactive Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/swarm-ai-research/swarm/blob/main/examples/quickstart.ipynb)

The **[quickstart notebook](examples/quickstart.ipynb)** runs two scenarios end-to-end in ~5 minutes with no API keys: a cooperative baseline and an adversarial red-team that collapses around epoch 12. Includes diagnostic plots and a per-agent payoff breakdown. Click the Colab badge to run it in your browser — no local setup needed.

```bash
# Or run locally:
jupyter notebook examples/quickstart.ipynb
```

### Blog Post

For a narrative walkthrough of our findings across 11 scenarios — including the phase transition at 37.5-50% adversarial fraction, why governance tuning delays but doesn't prevent collapse, and why collusion detection is the critical lever — see the **[blog](docs/blog/index.md)**.

## CLI Quick Start

Run simulations directly from the command line:

```bash
# List available scenarios
swarm list

# Run a scenario
swarm run scenarios/baseline.yaml

# Override simulation settings
swarm run scenarios/baseline.yaml --seed 42 --epochs 20 --steps 15

# Export outputs
swarm run scenarios/baseline.yaml --export-json results.json --export-csv outputs/
```

## Reproducible Runs

Run a complete reproducible experiment with all artifacts:

```bash
# One-command reproducible run
python -m swarm run scenarios/baseline.yaml \
  --seed 42 \
  --epochs 10 \
  --steps 10 \
  --export-json runs/baseline_seed42/history.json \
  --export-csv runs/baseline_seed42/csv/

# Generate plots from run
python examples/plot_run.py runs/baseline_seed42/
```

**Artifact paths:**
- History JSON: `runs/<timestamp>_<scenario>_seed<seed>/history.json`
- Metrics CSV: `runs/<timestamp>_<scenario>_seed<seed>/csv/metrics.csv`
- Plots: `runs/<timestamp>_<scenario>_seed<seed>/plots/*.png`

See the [Reproducibility Guide](docs/getting-started/reproducibility.md) for complete workflows, multi-seed runs, and archival best practices.

## Examples & Notebooks

All examples run standalone with no API keys unless noted. Start with the quickstart notebook, then explore by interest area.

| Example | Description | Colab | Difficulty |
|---------|-------------|-------|------------|
| **[quickstart.ipynb](examples/quickstart.ipynb)** | Two scenarios end-to-end with plots | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/swarm-ai-research/swarm/blob/main/examples/quickstart.ipynb) | Beginner |
| **[reproducible_run_demo.py](examples/reproducible_run_demo.py)** | Complete reproducible workflow with artifacts | — | Beginner |
| **[illusion_delta_minimal.py](examples/illusion_delta_minimal.py)** | Replay-based incoherence detection (3 agents) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/swarm-ai-research/swarm/blob/main/examples/illusion_delta_minimal.ipynb) | Beginner |
| **[mvp_demo.py](examples/mvp_demo.py)** | Full 5-agent simulation with metric printout | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/swarm-ai-research/swarm/blob/main/examples/mvp_demo.ipynb) | Beginner |
| **[run_scenario.py](examples/run_scenario.py)** | Run any YAML scenario from CLI | — | Beginner |
| **[parameter_sweep.py](examples/parameter_sweep.py)** | Sweep governance parameters and compare | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/swarm-ai-research/swarm/blob/main/examples/parameter_sweep.ipynb) | Intermediate |
| **[run_redteam.py](examples/run_redteam.py)** | Red-team evaluation across 8 attack vectors | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/swarm-ai-research/swarm/blob/main/examples/run_redteam.ipynb) | Intermediate |
| **[governance_mvp_sweep.py](examples/governance_mvp_sweep.py)** | Governance lever comparison sweep | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/swarm-ai-research/swarm/blob/main/examples/governance_mvp_sweep.ipynb) | Intermediate |
| **[llm_demo.py](examples/llm_demo.py)** | LLM-backed agents (requires API key) | — | Intermediate |
| **[ldt_composition_study.py](examples/ldt_composition_study.py)** | LDT agent composition research | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/swarm-ai-research/swarm/blob/main/examples/ldt_composition_study.ipynb) | Advanced |
| **[reproduce_2602_00035.py](examples/reproduce_2602_00035.py)** | Reproduce paper results | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/swarm-ai-research/swarm/blob/main/examples/reproduce_2602_00035.ipynb) | Advanced |
| **[demo/app.py](examples/demo/app.py)** | Streamlit interactive dashboard | — | Intermediate |

> **Tip for Colab users:** All notebooks with a Colab badge auto-detect Colab and install SWARM from GitHub. For scripts without a notebook, add `!pip install swarm-safety` in the first cell.

## Core Concepts

### Soft Probabilistic Labels

Instead of binary labels (good/bad), interactions carry a probability `p = P(v = +1)` representing the likelihood of a beneficial outcome:

1. **Proxy signals** are combined into a raw score `v_hat in [-1, +1]`
2. **Calibrated sigmoid** converts to probability: `p = 1 / (1 + exp(-k * v_hat))`

### Four Key Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Toxicity rate** | `E[1-p \| accepted]` | Expected harm among accepted |
| **Quality gap** | `E[p \| accepted] - E[p \| rejected]` | Adverse selection indicator (negative = bad) |
| **Conditional loss** | `E[pi \| accepted] - E[pi]` | Selection effect on payoffs |
| **Incoherence** | `Var[decision] / E[error]` | Variance-to-error ratio across replays |
| **Illusion delta** | `C_perceived − C_distributed` | Gap between apparent and actual coherence |

### Governance Levers (27+)

- **Transaction Taxes** - Reduce exploitation, cost welfare
- **Reputation Decay** - Punish bad actors, erode honest standing
- **Circuit Breakers** - Freeze toxic agents quickly
- **Random Audits** - Deter hidden exploitation
- **Staking** - Filter undercapitalized agents
- **Collusion Detection** - Catch coordinated attacks
- **Dynamic Friction** - Adaptive rate limiting under stress
- **Sybil Detection** - Penalize behaviorally similar clusters
- **Council Governance** - Deliberative multi-agent policy decisions
- **Incoherence Breaker** - Detect/prevent incoherent policies
- **Ensemble Governance** - Multi-lever combination strategies
- And 16+ more (diversity, transparency, decomposition, memory governance, ...)

### Agent Policies

| Type | Behavior |
|------|----------|
| **Honest** | Cooperative, trust-based, completes tasks diligently |
| **Opportunistic** | Maximizes short-term payoff, cherry-picks tasks, strategic voting |
| **Deceptive** | Builds trust through honest behavior, then exploits trusted relationships |
| **Adversarial** | Targets honest agents, coordinates with allies, disrupts ecosystem |
| **LDT** | Logical Decision Theory with UDT precommitment and opponent modeling |
| **RLM** | Reinforcement Learning from Memory — learns from interaction history |
| **Council** | Deliberative governance via multi-agent council protocol |
| **SkillRL** | Reinforcement learning over evolving skill repertoire |
| **LLM** | Behavior determined by LLM with configurable persona; 9 providers supported ([details](docs/llm-agents.md)) |

## How SWARM Compares

| Feature | SWARM | Concordia | AgentBench | METR | Inspect (AISI) |
|---|---|---|---|---|---|
| Multi-agent interaction modeling | Primary focus | Primary focus | Limited | Limited | Limited |
| Soft probabilistic labels | Core design | No | No | No | No |
| Adverse selection metrics | Yes (toxicity, quality gap) | No | No | No | No |
| Configurable governance levers | 27+ built-in | None | None | None | Compliance rules |
| Collusion detection | Yes (pair-wise, structural) | No | No | No | No |
| Replay-based incoherence | Yes | No | No | No | No |
| LLM agent support | Yes (9 providers: Anthropic, OpenAI, Ollama, OpenRouter, Groq, Together, DeepSeek, Google, llama.cpp) | Yes | Yes | Yes | Yes |
| Scenario configs (YAML) | 78 built-in | Custom | Benchmark suites | Task suites | Eval suites |
| Framework bridges | 8 (Concordia, OpenClaw, GasTown, LiveSWE, Prime Intellect, Ralph, Claude Code, Worktree) | — | — | — | — |
| License | MIT | Apache 2.0 | MIT | Varies | MIT |

SWARM is complementary to these frameworks, not competitive. The [Concordia bridge](docs/bridges/concordia.md) lets you run Concordia agents through SWARM's governance and metrics layer. See [full comparison](docs/comparison.md).

## Related work

SWARM is inspired by and complementary to:
- Agent-based governance simulations
- Recursive and multi-agent evaluation frameworks
- Mechanism design for AI systems

## Architecture

```
SWARM Core
+------------------------------------------------------------+
|                                                            |
|  ProxyComputer --> SoftInteraction --> Metrics             |
|       |                  |                |                |
|       |                  |                |                |
|  Observable          Payoff          Governance            |
|  Extraction          Engine          Engine                |
|                                                            |
+------------------------------------------------------------+
```

**Data Flow:**
```
Observables -> ProxyComputer -> v_hat -> sigmoid -> p -> SoftPayoffEngine -> payoffs
                                                    |
                                               SoftMetrics -> toxicity, quality gap, etc.
```

## Directory Structure

```
swarm/
├── swarm/
│   ├── models/          # SoftInteraction, AgentState, events, identity, kernel, schemas (8 modules)
│   ├── core/            # Orchestrator, PayoffEngine, ProxyComputer + domain handlers (35 modules)
│   ├── agents/          # 23 agent types: honest, deceptive, LDT, RLM, council, SkillRL, LLM, ... (29 modules)
│   ├── env/             # Feed, tasks, marketplace, auctions, HFN, memory tiers, catalogs (16 modules)
│   ├── governance/      # 27+ levers: taxes, reputation, audits, collusion, council, ... (27 modules)
│   ├── metrics/         # SoftMetrics, reporters, RLM, incoherence, collusion, ... (17 modules)
│   ├── csm/             # Consumer-Seller Marketplace: matching, negotiation, identity (10 modules)
│   ├── council/         # Council governance protocol, ranking, prompts (6 modules)
│   ├── skills/          # Skill learning & evolution: model, library, governance (6 modules)
│   ├── bridges/         # 8 external integrations: Concordia, GasTown, Prime Intellect, ... (95 files)
│   ├── research/        # Research pipeline: agents, platforms, quality gates, Track A (12 modules)
│   ├── redteam/         # Attack scenarios, evaluator, evasion metrics
│   ├── boundaries/      # External world, flow tracking, permeability, leakage
│   ├── analysis/        # Parameter sweeps, plots, dashboard, export
│   ├── api/             # FastAPI server
│   ├── forecaster/      # Risk forecasters for adaptive governance
│   ├── replay/          # Replay runner and decision-level replay
│   ├── scenarios/       # YAML scenario loader
│   └── logging/         # Append-only JSONL logger
├── tests/               # 4556 tests across 133 files
├── examples/            # 39 runnable scripts + Streamlit demo
├── scenarios/           # 79 YAML scenario definitions
├── docs/                # Documentation, papers, blog posts
└── pyproject.toml
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=swarm --cov-report=html

# Run specific test file
pytest tests/test_orchestrator.py -v

# Run CI checks (lint, type-checking, tests)
make ci
```

## Documentation

| Topic | Description |
|-------|-------------|
| [Theoretical Foundations](docs/research/theory.md) | Formal model, whitepaper-style summary, and citation section |
| [LLM Agents](docs/llm-agents.md) | Providers, personas, cost tracking, YAML config |
| [Network Topology](docs/network-topology.md) | Topology types, dynamic evolution, network metrics |
| [Governance](docs/governance.md) | Levers, collusion detection, integration points |
| [Emergent Capabilities](docs/emergent-capabilities.md) | Composite tasks, capability types, emergent metrics |
| [Red-Teaming](docs/red-teaming.md) | Adaptive adversaries, attack strategies, evaluation results |
| [Scenarios & Sweeps](docs/scenarios.md) | YAML scenarios, scenario comparison, parameter sweeps |
| [Boundaries](docs/boundaries.md) | External world simulation, flow tracking, leakage detection |
| [Dashboard](docs/dashboard.md) | Streamlit dashboard setup and features |
| [Incoherence Metric Contract](docs/incoherence_metric_contract.md) | Definitions and edge-case semantics |
| [Incoherence Scaling Analysis](docs/analysis/incoherence_scaling.md) | Replay-sweep artifact and upgrade path |
| [Incoherence Governance Transferability](docs/transferability/incoherence_governance.md) | Deployment caveats and assumptions |

## Start Here (Researcher Path)

- Read the framing: [Theoretical Foundations](docs/research/theory.md)
- Run an incoherence artifact: [Incoherence Scaling Analysis](docs/analysis/incoherence_scaling.md)
- Inspect policy caveats: [Incoherence Governance Transferability](docs/transferability/incoherence_governance.md)
- Reproduce from CLI: `swarm run scenarios/baseline.yaml`

## Citation

If you use SWARM in your research, please cite the paper:

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

To cite the software itself:

```bibtex
@software{swarm2026,
  title  = {SWARM: System-Wide Assessment of Risk in Multi-agent systems},
  author = {Savitt, Raeli},
  year   = {2026},
  url    = {https://github.com/swarm-ai-research/swarm}
}
```

Machine-readable citation metadata: [`CITATION.cff`](CITATION.cff)

## Papers
- **Soft-Label Governance for Distributional Safety in Multi-Agent Systems** https://arxiv.org/abs/2604.19752
- **Distributional AGI Safety: Governance Trade-offs in Multi-Agent Systems Under Adversarial Pressure** — 11 scenarios, 209 epochs, three regimes.
- **Governance Mechanisms for Multi-Agent Safety** — Cross-archetype empirical study of 7 scenario types
- **Collusion Dynamics and Network Resilience** — Progressive decline vs sustained operation under network topology effects

Full paper sources and supplementary materials are in the [swarm-artifacts](https://github.com/swarm-ai-research/swarm-artifacts) repo.

## Community

- [Documentation](https://github.com/swarm-ai-research/swarm/tree/main/docs) — Full guides, API reference, and research notes
- [GitHub Issues](https://github.com/swarm-ai-research/swarm/issues) — Bug reports, feature requests, and [agent bounties](CONTRIBUTING.md)
- [Twitter/X](https://x.com/ResearchSwarmAI) — @ResearchSwarmAI

## References

- Kyle, A.S. (1985). *Continuous Auctions and Insider Trading*. Econometrica.
- Glosten, L.R. & Milgrom, P.R. (1985). *Bid, Ask and Transaction Prices in a Specialist Market*. JFE.
- [Distributional AGI Safety](https://arxiv.org/abs/2512.16856)
- [Multi-Agent Market Dynamics](https://arxiv.org/abs/2502.14143)
- [The Hot Mess Theory of AI](https://alignment.anthropic.com/2026/hot-mess-of-ai/)
- [Infinite Backrooms](https://dreams-of-an-electric-mind.webflow.io/) — observational evidence of local-coherence/global-incoherence in AI-to-AI interaction
- [Moltbook](https://moltbook.com) | [@sebkrier](https://x.com/sebkrier/status/2017993948132774232)

## License

MIT License - See [LICENSE](LICENSE) for details.
