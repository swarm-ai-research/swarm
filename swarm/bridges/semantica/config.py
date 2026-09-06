"""Configuration for the Semantica export bridge."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SemanticaConfig:
    # Command that starts the Semantica MCP server (stdio transport).
    # Requires semantica installed in that environment; the bridge itself
    # never imports semantica.
    mcp_command: Tuple[str, ...] = ("semantica-mcp",)
    request_timeout_s: float = 60.0
    category_prefix: str = "swarm"
    # Recorded as prov agent on pushed decisions; the per-interaction
    # decision maker is always the accepting counterparty.
    client_name: str = "swarm-semantica-bridge"
    # Keys record_decision accepts over MCP (their schema has no metadata
    # field, so pushes strip everything else; the JSONL artifact keeps all).
    mcp_arg_keys: Tuple[str, ...] = field(
        default=(
            "category",
            "scenario",
            "reasoning",
            "outcome",
            "confidence",
            "entities",
            "decision_maker",
            "valid_from",
            "valid_until",
        )
    )
