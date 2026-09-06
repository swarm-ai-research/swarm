"""SWARM–Prime Agent bridge.

Governs and scores [Prime Agent](https://github.com/PrimeIntellect-ai/prime-agent)
sessions — the self-improving RLM agent — with SWARM's soft-label metrics.

Distinct from :mod:`swarm.bridges.prime_intellect`, which targets the same
organization's *RL training* platform (environments hub, verifiers, safety
rewards). This bridge targets the *agent*: its continual harness and its
recursive delegation tree.

Architecture::

    ~/.prime/agent/sessions/*.jsonl   (session tree: id / parentId entries)
        |
    PrimeAgentClient                  (parse harness refinements, rlm() spawns,
        |                              tool outcomes, usage, stop reason)
        |
    HarnessTracker                    (growth rate, evidence rate, rollback churn)
        |
    HarnessRefinementPolicy           (base-prompt immutability, growth caps,
        |                              evidence gate, recursion depth / fan-out)
        |
    ProxyObservables -> ProxyComputer -> (v_hat, p) -> SoftInteraction
        |                                              (causal_parents = spawning
        |                                               session's interaction)
    EventLog + SWARM metrics pipeline

Two governable claims Prime Agent makes about itself, which this bridge
measures rather than assumes:

1. Refinements are *small* and *evidence-backed* — measured as characters
   written per refinement, net harness entries per turn, and whether the
   rationale cites a concrete referent.
2. A session that finished is not a session that succeeded — upstream says so
   directly ("reaching a limit does not imply task success"), so an ungated
   clean stop maps to a weak positive, not a strong one.

Quick start::

    from swarm.bridges.prime_agent import PrimeAgentBridge

    bridge = PrimeAgentBridge()
    interactions = bridge.analyze_session_tree("~/.prime/agent/sessions")
    print(bridge.get_metrics()["unsupported_refinement_rate"])
"""

from swarm.bridges.prime_agent.bridge import (
    GATED_PROGRESS,
    ORCHESTRATOR_ID,
    UNGATED_PROGRESS,
    PrimeAgentBridge,
    PrimeAgentBridgeConfig,
)
from swarm.bridges.prime_agent.client import (
    DEFAULT_SESSION_ROOT,
    PrimeAgentClient,
    PrimeAgentClientConfig,
    SessionTree,
    build_session_tree,
)
from swarm.bridges.prime_agent.events import (
    BASE_SYSTEM_PROMPT_ID,
    HARNESS_KINDS,
    REFINEMENT_CUSTOM_TYPE,
    CompactionRecord,
    GateResult,
    HarnessEditRecord,
    PrimeAgentEvent,
    PrimeAgentEventType,
    RefinementRecord,
    RlmSpawnRecord,
    SessionOutcome,
    SessionTrajectory,
    TokenUsage,
    ToolCallRecord,
)
from swarm.bridges.prime_agent.harness import (
    EVIDENCE_PATTERNS,
    HarnessDriftState,
    HarnessTracker,
    evidence_kinds,
    has_concrete_evidence,
)
from swarm.bridges.prime_agent.policy import (
    HarnessRefinementPolicy,
    PolicyDecision,
    PolicyResult,
    RefinementPolicyConfig,
)

__all__ = [
    # bridge
    "PrimeAgentBridge",
    "PrimeAgentBridgeConfig",
    "GATED_PROGRESS",
    "UNGATED_PROGRESS",
    "ORCHESTRATOR_ID",
    # client
    "PrimeAgentClient",
    "PrimeAgentClientConfig",
    "SessionTree",
    "build_session_tree",
    "DEFAULT_SESSION_ROOT",
    # events
    "PrimeAgentEvent",
    "PrimeAgentEventType",
    "SessionTrajectory",
    "SessionOutcome",
    "GateResult",
    "RefinementRecord",
    "HarnessEditRecord",
    "RlmSpawnRecord",
    "CompactionRecord",
    "ToolCallRecord",
    "TokenUsage",
    "BASE_SYSTEM_PROMPT_ID",
    "REFINEMENT_CUSTOM_TYPE",
    "HARNESS_KINDS",
    # harness
    "HarnessTracker",
    "HarnessDriftState",
    "EVIDENCE_PATTERNS",
    "has_concrete_evidence",
    "evidence_kinds",
    # policy
    "HarnessRefinementPolicy",
    "RefinementPolicyConfig",
    "PolicyDecision",
    "PolicyResult",
]
