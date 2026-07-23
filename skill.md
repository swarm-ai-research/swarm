---
name: swarm-safety
version: 1.9.0
description: "SWARM: System-Wide Assessment of Risk in Multi-agent systems. Simulate multi-agent dynamics, test governance, study emergent risks."
homepage: https://github.com/swarm-ai-research/swarm
metadata: {"category":"safety","license":"MIT","author":"Raeli Savitt"}
---

# SWARM Safety Skill

Study how intelligence swarms — and where it fails.

SWARM is a research framework for studying emergent risks in multi-agent AI systems using soft (probabilistic) labels instead of binary good/bad classifications. AGI-level risks don't require AGI-level agents.

Repository: `https://github.com/swarm-ai-research/swarm`

## Hard Rules

- SWARM simulations run locally. Install the package first.
- Do not submit scenarios containing real API keys, credentials, or PII.
- Simulation results are research artifacts. Do not present them as ground truth about real systems.
- When publishing results, cite the framework and disclose simulation parameters.

## Security

- **API binds to localhost only** (`127.0.0.1`) by default to prevent network exposure.
- **CORS restricted** to localhost origins by default.
- **No authentication** on development API — do not expose to untrusted networks.
- **SQLite persistence** — simulation data persists locally between restarts.
- For production deployment, add authentication middleware and use a proper database.

## Install

```bash
# From PyPI
pip install swarm-safety

# From source (full development)
git clone https://github.com/swarm-ai-research/swarm.git
cd swarm
pip install -e ".[all]"
```

## Quick Start (Python)

```python
from swarm.agents.honest import HonestAgent
from swarm.agents.opportunistic import OpportunisticAgent
from swarm.agents.deceptive import DeceptiveAgent
from swarm.agents.adversarial import AdversarialAgent
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

config = OrchestratorConfig(n_epochs=10, steps_per_epoch=10, seed=42)
orchestrator = Orchestrator(config=config)

orchestrator.register_agent(HonestAgent(agent_id="honest_1", name="Alice"))
orchestrator.register_agent(HonestAgent(agent_id="honest_2", name="Bob"))
orchestrator.register_agent(OpportunisticAgent(agent_id="opp_1"))
orchestrator.register_agent(DeceptiveAgent(agent_id="dec_1"))

metrics = orchestrator.run()
for m in metrics:
    print(f"Epoch {m.epoch}: toxicity={m.toxicity_rate:.3f}, welfare={m.total_welfare:.2f}")
```

## Quick Start (CLI)

```bash
# List available scenarios
swarm list

# Run a scenario
swarm run scenarios/baseline.yaml

# Override settings
swarm run scenarios/baseline.yaml --seed 42 --epochs 20 --steps 15

# Export results
swarm run scenarios/baseline.yaml --export-json results.json --export-csv outputs/
```

## Quick Start (API)

Start the API server:

```bash
pip install swarm-safety[api]
uvicorn swarm.api.app:app --host 127.0.0.1 --port 8000
```

API documentation at `http://localhost:8000/docs`.

> **Security Note**: The server binds to `127.0.0.1` (localhost only) by default. Do not bind to `0.0.0.0` unless you understand the security implications and have proper firewall rules in place.

### Register Agent

```bash
curl -X POST http://localhost:8000/api/v1/agents/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "YourAgent",
    "description": "What your agent does",
    "capabilities": ["governance-testing", "red-teaming"]
  }'
```

Returns `agent_id` and `api_key`.

### Submit Scenario

```bash
curl -X POST http://localhost:8000/api/v1/scenarios/submit \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-scenario",
    "description": "Testing collusion detection with 5 agents",
    "yaml_content": "simulation:\n  n_epochs: 10\n  steps_per_epoch: 10\nagents:\n  - type: honest\n    count: 3\n  - type: adversarial\n    count: 2",
    "tags": ["collusion", "governance"]
  }'
```

### Create & Join Simulation

```bash
# Create
curl -X POST http://localhost:8000/api/v1/simulations/create \
  -H "Content-Type: application/json" \
  -d '{"scenario_id": "SCENARIO_ID", "max_participants": 5}'

# Join
curl -X POST http://localhost:8000/api/v1/simulations/SIM_ID/join \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "YOUR_AGENT_ID", "role": "participant"}'
```

## Core Concepts

### Soft Probabilistic Labels

Interactions carry `p = P(v = +1)` — probability of beneficial outcome:

```
Observables → ProxyComputer → v_hat → sigmoid → p → PayoffEngine → payoffs
                                                 ↓
                                            SoftMetrics → toxicity, quality gap, etc.
```

### Four Key Metrics

| Metric | What It Measures |
|---|---|
| **Toxicity rate** | Expected harm among accepted interactions |
| **Quality gap** | Adverse selection indicator (negative = bad) |
| **Conditional loss** | Selection effect on payoffs |
| **Incoherence** | Variance-to-error ratio across replays |

### Agent Types

| Type | Behavior |
|---|---|
| **Honest** | Cooperative, trust-based, completes tasks diligently |
| **Opportunistic** | Maximizes short-term payoff, cherry-picks tasks |
| **Deceptive** | Builds trust, then exploits trusted relationships |
| **Adversarial** | Targets honest agents, coordinates with allies |
| **LDT** | Logical Decision Theory with UDT precommitment and opponent modeling |
| **RLM** | Reinforcement Learning from Memory — learns from interaction history |
| **Council** | Deliberative governance via multi-agent council protocol |
| **SkillRL** | Reinforcement learning over evolving skill repertoire |
| **LLM** | Behavior determined by LLM with configurable persona; 9 providers supported |

### Governance Levers

- **Transaction Taxes** — Reduce exploitation, cost welfare
- **Reputation Decay** — Punish bad actors, erode honest standing
- **Circuit Breakers** — Freeze toxic agents quickly
- **Random Audits** — Deter hidden exploitation
- **Staking** — Filter undercapitalized agents
- **Collusion Detection** — Catch coordinated attacks
- **Sybil Detection** — Identify duplicate agents
- **Transparency Ledger** — Reward/penalize based on outcome
- **Moderator Agent** — Probabilistic review of interactions
- **Incoherence Friction** — Tax uncertainty-driven decisions

### Scenario YAML Format

```yaml
simulation:
  n_epochs: 10
  steps_per_epoch: 10
  seed: 42

agents:
  - type: honest
    count: 3
    config:
      acceptance_threshold: 0.4
  - type: adversarial
    count: 2
    config:
      aggression_level: 0.7

governance:
  transaction_tax_rate: 0.05
  circuit_breaker_enabled: true
  collusion_detection_enabled: true

success_criteria:
  max_toxicity: 0.3
  min_quality_gap: 0.0
```

## Case Studies

### Moltipedia Heartbeat

Model the Moltipedia wiki editing loop: competing AI editors, editorial policy, point farming, and anti-gaming governance. See `scenarios/moltipedia_heartbeat.yaml`.

### Moltbook CAPTCHA

Model Moltbook's anti-human math challenges and rate limiting: obfuscated text parsing, verification gates, and spam prevention. See `scenarios/moltbook_captcha.yaml`.

## API Endpoints (Full Reference)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/` | API info |
| POST | `/api/v1/agents/register` | Register agent |
| GET | `/api/v1/agents/{agent_id}` | Get agent details |
| GET | `/api/v1/agents/` | List agents |
| POST | `/api/v1/scenarios/submit` | Submit scenario |
| GET | `/api/v1/scenarios/{scenario_id}` | Get scenario |
| GET | `/api/v1/scenarios/` | List scenarios |
| POST | `/api/v1/simulations/create` | Create simulation |
| POST | `/api/v1/simulations/{id}/join` | Join simulation |
| GET | `/api/v1/simulations/{id}` | Get simulation |
| GET | `/api/v1/simulations/` | List simulations |

## Citation

```bibtex
@software{swarm2026,
  title = {SWARM: System-Wide Assessment of Risk in Multi-agent systems},
  author = {Savitt, Raeli},
  year = {2026},
  url = {https://github.com/swarm-ai-research/swarm}
}
```

## Linked Docs

- Skill metadata: `skill.json`
- Agent discovery: `.well-known/agent.json`
- Bridges: `docs/bridges/`
- Full documentation: `https://github.com/swarm-ai-research/swarm/tree/main/docs`
- Theoretical foundations: `docs/research/theory.md`
- Governance guide: `docs/governance.md`
- Red-teaming guide: `docs/red-teaming.md`
- Scenario format: `docs/guides/scenarios.md`
