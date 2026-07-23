"""Configuration for the SWARM–AgentVeil bridge.

Field defaults track the plan in ``docs/bridges/agentveil.md``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentVeilConfig:
    """Full bridge configuration.

    v1 is mock-only — ``mock_mode`` defaults to True and the registry URL
    is only consulted when ``mock_mode=False`` (v2).
    """

    # Registry
    registry_url: str = "https://agentveil.dev"
    mock_mode: bool = True

    # Identity
    orchestrator_did: str = ""

    # Admission
    min_tier: str = "basic"  # newcomer | basic | trusted | elite
    fail_mode: str = "closed"  # "open" or "closed" on registry error

    # Proxy calibration
    proxy_sigmoid_k: float = 2.0

    # Write-back
    writeback_enabled: bool = True
    writeback_positive_threshold: float = 0.7
    writeback_negative_threshold: float = 0.3

    # Rate limiting
    max_attestations_per_epoch: int = 100
    max_trust_checks_per_step: int = 50

    # Memory caps
    max_interactions: int = 50_000
    max_events: int = 50_000

    # Tier-amplification cap (failure-mode C4): the maximum contribution
    # an AVP tier alone is allowed to push v_hat by, before other signals.
    # Keeps reputation feedback from running away.
    max_tier_v_hat_contribution: float = 0.6
