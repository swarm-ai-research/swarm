"""SWARM -> Semantica export bridge.

Replays a run's event log into Semantica decision records (soft label p as
Decision.confidence) without importing semantica. See exporter.export_run.
"""

from swarm.bridges.semantica.client import SemanticaMCPClient, SemanticaMCPError
from swarm.bridges.semantica.config import SemanticaConfig
from swarm.bridges.semantica.exporter import ExportSummary, export_run
from swarm.bridges.semantica.mapper import (
    interaction_to_decision,
    proxy_weights_to_bridge_axioms,
    run_manifest,
)

__all__ = [
    "ExportSummary",
    "SemanticaConfig",
    "SemanticaMCPClient",
    "SemanticaMCPError",
    "export_run",
    "interaction_to_decision",
    "proxy_weights_to_bridge_axioms",
    "run_manifest",
]
