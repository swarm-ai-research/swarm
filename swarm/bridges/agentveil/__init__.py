"""SWARM–AgentVeil Bridge.

Connects SWARM's governance and metrics framework to the Agent Veil
Protocol (AVP), enabling cryptographic identity, peer-attested reputation,
and sybil-resistant admission control on top of SWARM's probabilistic
labels.

See ``docs/bridges/agentveil.md`` for the full plan and failure-mode
catalog. v1 is mock-only: live registry HTTP, write-back attestations,
and the dispute/arbitration flow land in v2+.

Architecture:
    AVP Registry (agentveil.dev)        [live mode only; v2]
        |
    AVPClient (client.py)               [v1: mock_mode only]
        |
    AgentVeilBridge._process_event()
        |   AVPPolicy (admission gate, rate limiter, write-back gate)
        |
    AVPMapper -> ProxyObservables -> ProxyComputer -> (v_hat, p)
        |
    SoftInteraction -> EventLog (incl. ATTESTATION_SUBMITTED) -> metrics
"""

from swarm.bridges.agentveil.bridge import AgentVeilBridge
from swarm.bridges.agentveil.client import (
    AttestationReceipt,
    AVPClient,
    DIDDocument,
    ReputationSnapshot,
    TrustDecision,
)
from swarm.bridges.agentveil.config import AgentVeilConfig
from swarm.bridges.agentveil.events import (
    AgentVeilEvent,
    AgentVeilEventType,
    AttestationEvent,
    ReputationSnapshotEvent,
    TrustDecisionEvent,
)
from swarm.bridges.agentveil.mapper import AVPMapper
from swarm.bridges.agentveil.policy import AVPPolicy, PolicyDecision, PolicyResult

__all__ = [
    "AgentVeilBridge",
    "AttestationReceipt",
    "AttestationEvent",
    "AgentVeilConfig",
    "AgentVeilEvent",
    "AgentVeilEventType",
    "AVPClient",
    "DIDDocument",
    "ReputationSnapshot",
    "ReputationSnapshotEvent",
    "TrustDecision",
    "TrustDecisionEvent",
    "AVPMapper",
    "AVPPolicy",
    "PolicyDecision",
    "PolicyResult",
]
