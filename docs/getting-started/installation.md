---
description: "Clone the repository and install in development mode:"
author: "SWARM Team"
keywords:
  - SWARM installation
  - swarm-safety pip install
  - multi-agent safety framework setup
---

# Installation

## Quick Install

Install SWARM from PyPI:

```bash
pip install swarm-safety
```

## Install from Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/swarm-ai-research/swarm.git
cd swarm
pip install -e ".[dev]"
```

## Optional Dependencies

SWARM has several optional dependency groups:

=== "Development"

    ```bash
    pip install swarm-safety[dev]
    ```

    Includes: pytest, pytest-cov, hypothesis, mypy, ruff

=== "Analysis"

    ```bash
    pip install swarm-safety[analysis]
    ```

    Includes: pandas, matplotlib, seaborn

=== "LLM Support"

    ```bash
    pip install swarm-safety[llm]
    ```

    Includes: anthropic, openai, httpx

=== "Dashboard"

    ```bash
    pip install swarm-safety[dashboard]
    ```

    Includes: streamlit, plotly

=== "Everything"

    ```bash
    pip install swarm-safety[all]
    ```

    All optional dependencies

## Verify Installation

```python
import swarm
print(swarm.__version__)
```

```bash
swarm --help
```

## System Requirements

- Python 3.10 or higher
- numpy, pydantic, pandas (installed automatically)

## See also

- [Quick Start](quickstart.md) — Run your first multi-agent simulation in 5 minutes
- [Your First Scenario](first-scenario.md) — Create a custom YAML experiment from scratch
- [Soft Labels](../concepts/soft-labels.md) — How SWARM uses probabilistic labels instead of binary classifications
- [Your First Governance Experiment](../tutorials/first-governance-experiment.md) — Step-by-step tutorial for running governance experiments
