"""Agent screening protocol for the OpenSandbox bridge.

Evaluates agent capability manifests against available governance
contracts and assigns each agent to a compatible tier.  Tracks
self-sorting behavior for mechanism-design measurement.
"""

import logging
from typing import Dict, List, Optional

from swarm.bridges.opensandbox.config import (
    AgentType,
    CapabilityManifest,
    ContractAssignment,
    GovernanceContract,
    NetworkPolicy,
    OpenSandboxConfig,
)
from swarm.bridges.opensandbox.events import OpenSandboxEvent, OpenSandboxEventType

logger = logging.getLogger(__name__)

# Type alignment scoring: (agent_type, contract_tier) -> score in [0, 1]
# Adversarial agents score highest on restricted contracts,
# cooperative agents on premium, etc.
_TYPE_TIER_SCORES = {
    (AgentType.COOPERATIVE, "premium"): 1.0,
    (AgentType.COOPERATIVE, "standard"): 0.7,
    (AgentType.COOPERATIVE, "restricted"): 0.3,
    (AgentType.STATIC, "premium"): 0.6,
    (AgentType.STATIC, "standard"): 1.0,
    (AgentType.STATIC, "restricted"): 0.5,
    (AgentType.SELF_MODIFYING, "premium"): 0.4,
    (AgentType.SELF_MODIFYING, "standard"): 0.8,
    (AgentType.SELF_MODIFYING, "restricted"): 0.6,
    (AgentType.ADVERSARIAL, "premium"): 0.1,
    (AgentType.ADVERSARIAL, "standard"): 0.4,
    (AgentType.ADVERSARIAL, "restricted"): 1.0,
}


class ScreeningProtocol:
    """Evaluate agents against governance contracts and assign tiers.

    The screener implements a compatibility-scoring approach:

    1. For each (agent, contract) pair, compute a compatibility score.
    2. Assign the agent to the highest-scoring compatible contract.
    3. Reject agents that score below the minimum threshold on all
       contracts.

    Sorting measurement — the distribution of agent types across
    contract tiers — is tracked for post-hoc analysis.

    Example::

        screener = ScreeningProtocol(config)
        assignment = screener.evaluate(manifest, contracts)
        # assignment.tier == "restricted"
    """

    def __init__(
        self,
        config: OpenSandboxConfig,
        min_score: float = 0.3,
        max_records: int = 50_000,
    ) -> None:
        self._config = config
        self._min_score = min_score
        self._max_records = max_records
        self._total_records = 0
        # Sorting ledger: tier -> list of (agent_id, agent_type, score)
        self._sorting_ledger: Dict[str, List[Dict]] = {}
        self._events: List[OpenSandboxEvent] = []

    def evaluate(
        self,
        manifest: CapabilityManifest,
        contracts: Optional[Dict[str, GovernanceContract]] = None,
    ) -> ContractAssignment:
        """Screen an agent and assign it to a contract tier.

        Args:
            manifest: The agent's capability manifest.
            contracts: Available contracts (defaults to config contracts).

        Returns:
            A ContractAssignment with tier info and sandbox env vars.
        """
        available = contracts or self._config.contracts
        if not available:
            available = {"default": self._config.default_contract}

        best_contract: Optional[GovernanceContract] = None
        best_score = -1.0

        for contract in available.values():
            score = self._compatibility_score(manifest, contract)
            if score > best_score:
                best_score = score
                best_contract = contract

        if best_contract is None or best_score < self._min_score:
            assignment = ContractAssignment(
                agent_id=manifest.agent_id,
                contract_id="",
                tier="",
                rejected=True,
                rejection_reason=(
                    f"No compatible contract (best score={best_score:.3f}, "
                    f"threshold={self._min_score:.3f})"
                ),
                score=best_score,
            )
            self._record_event(
                OpenSandboxEventType.AGENT_DENIED,
                manifest.agent_id,
                contract_id=None,
                payload={
                    "reason": assignment.rejection_reason,
                    "score": best_score,
                    "agent_type": manifest.agent_type.value,
                },
            )
            logger.info(
                "Screening rejected agent %s: %s",
                manifest.agent_id,
                assignment.rejection_reason,
            )
            return assignment

        assignment = ContractAssignment(
            agent_id=manifest.agent_id,
            contract_id=best_contract.contract_id,
            tier=best_contract.tier,
            sandbox_env=best_contract.to_sandbox_env(),
            score=best_score,
            metadata={
                "agent_type": manifest.agent_type.value,
                "declared_intent": manifest.declared_intent,
            },
        )

        # Record sorting (M3: bounded growth)
        tier_key = best_contract.tier
        ledger = self._sorting_ledger.setdefault(tier_key, [])
        ledger.append({
            "agent_id": manifest.agent_id,
            "agent_type": manifest.agent_type.value,
            "score": best_score,
        })
        self._total_records += 1
        if self._total_records > self._max_records:
            # Trim oldest half from each tier
            for k in list(self._sorting_ledger.keys()):
                lst = self._sorting_ledger[k]
                drop = len(lst) // 2
                if drop > 0:
                    self._sorting_ledger[k] = lst[drop:]
                    self._total_records -= drop

        self._record_event(
            OpenSandboxEventType.AGENT_ADMITTED,
            manifest.agent_id,
            contract_id=best_contract.contract_id,
            payload={
                "tier": best_contract.tier,
                "score": best_score,
                "agent_type": manifest.agent_type.value,
            },
        )
        logger.info(
            "Screening admitted agent %s to tier %s (score=%.3f)",
            manifest.agent_id,
            best_contract.tier,
            best_score,
        )
        return assignment

    def get_sorting_ledger(self) -> Dict[str, List[Dict]]:
        """Return the sorting ledger for mechanism-design analysis."""
        return dict(self._sorting_ledger)

    def get_events(self) -> List[OpenSandboxEvent]:
        """Return all screening events."""
        return list(self._events)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compatibility_score(
        manifest: CapabilityManifest,
        contract: GovernanceContract,
    ) -> float:
        """Compute compatibility between a manifest and a contract.

        Scoring factors (each in [0, 1], equally weighted):
        - capability_coverage: fraction of requested caps that the
          contract permits.
        - network_match: 1.0 if the contract provides the needed
          network level, else 0.0.
        - resource_fit: how well resource limits match.
        - type_alignment: whether the agent type is appropriate for
          the contract tier.
        """
        # Capability coverage
        if manifest.capabilities:
            covered = sum(
                1 for cap in manifest.capabilities
                if contract.allows_capability(cap)
            )
            cap_score = covered / len(manifest.capabilities)
        else:
            cap_score = 1.0

        # Network match
        if manifest.requires_network:
            net_score = 0.0 if contract.network == NetworkPolicy.DENY_ALL else 1.0
        else:
            net_score = 1.0

        # Resource fit (1.0 if contract provides enough, degrades if not)
        if contract.max_memory_mb >= manifest.max_memory_mb:
            mem_score = 1.0
        else:
            mem_score = contract.max_memory_mb / max(manifest.max_memory_mb, 1)

        if contract.max_cpu_shares >= manifest.max_cpu_shares:
            cpu_score = 1.0
        else:
            cpu_score = contract.max_cpu_shares / max(manifest.max_cpu_shares, 1)

        resource_score = (mem_score + cpu_score) / 2.0

        # Type alignment: score depends on (agent_type, contract_tier)
        type_score = _TYPE_TIER_SCORES.get(
            (manifest.agent_type, contract.tier), 0.5,
        )

        return (cap_score + net_score + resource_score + type_score) / 4.0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record_event(
        self,
        event_type: OpenSandboxEventType,
        agent_id: str,
        contract_id: Optional[str] = None,
        payload: Optional[Dict] = None,
    ) -> None:
        self._events.append(
            OpenSandboxEvent(
                event_type=event_type,
                agent_id=agent_id,
                contract_id=contract_id,
                payload=payload or {},
            )
        )
