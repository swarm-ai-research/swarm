---
description: "SWARM bridges connect the core framework to external systems for validation and real-world application."
---

# Integration Bridges

SWARM bridges connect the core framework to external systems for validation and real-world application.

## Available Bridges

### SWARM-Claude Code

Govern and score Claude Code CLI agents with SWARM's safety framework.

- **Purpose:** Programmatic orchestration of Claude Code agents under governance
- **Features:** Plan/permission adjudication, tool budgets, circuit breakers, event streaming
- **Status:** Production Ready (February 2026)

[Learn more →](claude_code.md)

### SWARM-Concordia

Integrate with Google DeepMind's Concordia for realistic LLM agent simulations.

- **Purpose:** Test SWARM metrics on LLM-based agents
- **Features:** Narrative → interaction translation, LLM judge scoring
- **Status:** In development

[Learn more →](concordia.md)

### SWARM-OpenClaw

Run SWARM as a service with secure multi-agent orchestration.

- **Purpose:** Production-ready SWARM deployments
- **Features:** REST API, job queue, containerization
- **Status:** In development

[Learn more →](openclaw.md)

### SWARM-GasTown

Instrument real production systems with SWARM metrics.

- **Purpose:** Monitor live multi-agent deployments
- **Features:** Event capture, interaction mapping, governance hooks
- **Status:** In development

[Learn more →](gastown.md)

### SWARM–Gitlawb

Monitor AI agent interactions on the Gitlawb decentralized git network in real time.

- **Purpose:** Live safety monitoring of decentralized agent collaboration
- **Features:** WebSocket event subscriptions, heuristic + LLM quality scoring, live dashboard, persistence
- **Status:** Available Now

[Learn more →](gitlawb.md)

### SWARM–Prime Intellect

Train and evaluate RL models on SWARM safety metrics using Prime Intellect's distributed training platform.

- **Purpose:** Safety-reward RL training and evaluation via the verifiers library
- **Features:** Environment export, composite reward from SWARM metrics, anti-gaming scoring, platform job management
- **Status:** In development

[Learn more →](prime_intellect.md)

### SWARM-Ralph

Ingest Ralph event streams into SWARM governance and metrics.

- **Purpose:** Score Ralph agent outcomes with SWARM proxy labels
- **Features:** JSONL event ingest, interaction mapping, incremental polling
- **Status:** In development

[Learn more →](ralph.md)

### SWARM-AgentXiv

Map research papers to SWARM scenarios for validation.

- **Purpose:** Validate published claims empirically
- **Features:** Paper metadata, scenario generation, validation reports
- **Status:** In development

[Learn more →](agentxiv.md)

### SWARM-ClawXiv

Publish SWARM research directly to ClawXiv.

- **Purpose:** Submit and discover agent-first preprints
- **Features:** Search, registration, submission, version updates
- **Status:** In development

[Learn more →](clawxiv.md)

### SWARM-AgentVeil (Planned)

Cryptographic trust, reputation tiers, and sybil-resistant admission control via the Agent Veil Protocol.

- **Purpose:** Pre-interaction trust gate with signed attestation write-back
- **Features:** DID-based identity, Bayesian reputation → p mapping, admission gate, attestation rate limiting
- **Status:** Plan

[Learn more →](agentveil.md)

## Bridge Architecture

```
┌─────────────────┐     ┌─────────────┐     ┌──────────────┐
│ External System │ ──► │   Bridge    │ ──► │  SWARM Core  │
│  (Concordia,    │     │  (Adapter)  │     │  (Metrics,   │
│   Gas Town...)  │     │             │     │  Governance) │
└─────────────────┘     └─────────────┘     └──────────────┘
```

Bridges provide:

1. **Event capture** - Extract interactions from external systems
2. **Observable mapping** - Convert external signals to SWARM format
3. **Governance hooks** - Apply SWARM governance to external systems
4. **Metric reporting** - Surface SWARM metrics in external dashboards

## Installation

Bridges are installed as separate packages:

```bash
# Individual bridges
pip install swarm-concordia
pip install swarm-openclaw
pip install swarm-gastown
pip install swarm-agentxiv

# All bridges
pip install swarm-safety[bridges]
```

ClawXiv integration ships with the core `swarm-safety` package and does not
require a separate bridge install.

## Contributing a Bridge

If you want to connect SWARM to a new system:

1. Create an adapter that translates system events to `SoftInteraction`
2. Implement observable extraction for `ProxyComputer`
3. Add governance hooks if the system supports intervention
4. Write validation scenarios

See the [bridge development guide](../guides/custom-agents.md) for details.
