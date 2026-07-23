---
description: "Get up and running with SWARM in minutes."
---

# Getting Started

Get up and running with SWARM in minutes.

<div class="grid cards" markdown>

-   :material-download: **[Installation](installation.md)**

    ---

    Install SWARM and its dependencies

-   :material-rocket-launch: **[Quick Start](quickstart.md)**

    ---

    Run your first simulation in 5 minutes

-   :material-file-document: **[Your First Scenario](first-scenario.md)**

    ---

    Build a custom scenario from scratch

-   :material-refresh: **[Reproducibility Guide](reproducibility.md)**

    ---

    Run reproducible experiments with proper artifact management

-   :material-notebook: **[Quickstart Notebook](https://colab.research.google.com/github/swarm-ai-research/swarm/blob/main/examples/quickstart.ipynb)**

    ---

    Run two scenarios in Colab with plots — no local setup needed [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/swarm-ai-research/swarm/blob/main/examples/quickstart.ipynb)

</div>

## Prerequisites

- Python 3.10+
- pip or uv package manager

## Quick Install

```bash
pip install swarm-safety
```

Or for development:

```bash
git clone https://github.com/swarm-ai-research/swarm.git
cd swarm
pip install -e ".[dev]"
```

## Next Steps

After installation, follow the [Quick Start](quickstart.md) guide to run your first multi-agent simulation and observe emergent risk dynamics.
