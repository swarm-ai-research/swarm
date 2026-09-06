"""Counterfactual resampling for local-model wiki-board behaviour.

The bridge deliberately uses a local, in-memory board.  It does not connect to
MediaWiki or any other public service.  ``run_experiment`` accepts an injected
model client so the causal bookkeeping can be tested without an LLM runtime.
"""

from swarm.bridges.wiki_resampling.config import ExperimentConfig
from swarm.bridges.wiki_resampling.model import OllamaClient
from swarm.bridges.wiki_resampling.runner import run_experiment

__all__ = ["ExperimentConfig", "OllamaClient", "run_experiment"]
