"""Maps AgentVeil trust decisions and attestations to SWARM observables and interactions.

Failure modes addressed:
  - D5 (p invariant): All observable fields clamped to [-1, +1]; SoftInteraction.p validated
  - C4 (tier amplification): Tier contribution capped at max_tier_v_hat_contribution
  - D1 (double-counting): Reputation write-back suppressed; tier influences p only via mapper
  - D2 (categorical→probabilistic): Tier mapped to ordinal then scaled; raw score preferred if available
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Protocol

from swarm.bridges.agentveil.config import AgentVeilConfig
from swarm.bridges.agentveil.events import (
    AttestationEvent,
    ReputationSnapshotEvent,
)
from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.models.interaction import InteractionType, SoftInteraction

logger = logging.getLogger(__name__)


class TrustDecisionProtocol(Protocol):
    """Protocol for trust decision objects to avoid circular imports.

    This protocol defines the interface AVPMapper expects from a trust decision,
    decoupling mapper from client.py.
    """

    allowed: bool
    tier: str
    confidence: float


class AVPMapper:
    """Maps AgentVeil trust decisions and attestations to SWARM probabilistic labels."""

    # Tier ordinal mapping: maps categorical tier to normalized progress signal
    _TIER_ORDINAL_MAP = {
        "newcomer": -0.5,
        "basic": 0.0,
        "trusted": 0.5,
        "elite": 1.0,
    }

    def __init__(self, config: AgentVeilConfig | None = None) -> None:
        """Initialize the mapper with configuration.

        Args:
            config: AgentVeilConfig instance. If None, uses defaults.
        """
        self._config = config or AgentVeilConfig()
        self._proxy = ProxyComputer(sigmoid_k=self._config.proxy_sigmoid_k)

    def trust_decision_to_observables(
        self,
        decision: TrustDecisionProtocol | Any,
        *,
        peer_id: str,
    ) -> ProxyObservables:
        """Convert a trust decision to ProxyObservables.

        Maps AVP's categorical tier and confidence score to SWARM observables:
        - Tier ordinal → task_progress_delta (clamped to max_tier_v_hat_contribution)
        - allowed=False → verifier_rejections=1 (admission gate rejection)
        - confidence → counterparty_engagement_delta ([0,1] already validated)

        Failure modes D5 (p∈[0,1]) and C4 (tier amplification) are mitigated:
        - C4: tier contribution clamped to ±max_tier_v_hat_contribution (default 0.6)
          prevents tier alone from saturating v_hat above 0.8.
        - D5: all observable fields clamped to [-1, +1] belt-and-suspenders.

        Args:
            decision: Trust decision with .allowed, .tier, .confidence fields
            peer_id: Identifier of the peer being evaluated (for logging)

        Returns:
            ProxyObservables ready for ProxyComputer.compute_labels()
        """
        # Map tier to ordinal signal
        tier = getattr(decision, "tier", "")
        tier_signal = self._TIER_ORDINAL_MAP.get(tier, 0.0)
        if tier not in self._TIER_ORDINAL_MAP:
            logger.warning(
                "Unknown tier '%s' for peer %s; defaulting to 0.0", tier, peer_id
            )

        # Apply C4 cap: clamp tier contribution to max_tier_v_hat_contribution
        max_cap = self._config.max_tier_v_hat_contribution
        task_progress_delta = max(-max_cap, min(max_cap, tier_signal))

        # D5 belt-and-suspenders: clamp to [-1, +1] before returning
        task_progress_delta = max(-1.0, min(1.0, task_progress_delta))

        # Rejection signal: 1 if decision.allowed is False, else 0
        allowed = getattr(decision, "allowed", False)
        verifier_rejections = 0 if allowed else 1

        # Engagement signal: confidence is already in [0, 1]
        confidence = getattr(decision, "confidence", 0.0)
        counterparty_engagement_delta = max(-1.0, min(1.0, confidence))

        return ProxyObservables(
            task_progress_delta=task_progress_delta,
            rework_count=0,
            verifier_rejections=verifier_rejections,
            tool_misuse_flags=0,
            counterparty_engagement_delta=counterparty_engagement_delta,
        )

    def attestation_received_to_interaction(
        self,
        attestation: AttestationEvent,
        *,
        initiator_id: str,
        counterparty_id: str,
        computer: ProxyComputer,
    ) -> SoftInteraction:
        """Convert an incoming attestation to a SoftInteraction.

        An incoming attestation from another agent is treated as a high-confidence
        external signal about the counterparty. Positive attestations (outcome_sign=+1)
        map to high task_progress_delta; negative attestations map to rejection.

        Failure mode D2 mitigation: continuous attestation outcome_sign (±1) is used
        rather than parsing categorical tiers, preserving signal fidelity.

        Failure mode D5 mitigation: The returned SoftInteraction's p is validated
        by SoftInteraction.p Pydantic validator, catching any violations.

        Args:
            attestation: AttestationEvent with outcome_sign ∈ {-1, 0, +1}
            initiator_id: ID of the agent receiving the attestation
            counterparty_id: ID of the agent being attested about
            computer: ProxyComputer to convert observables to (v_hat, p)

        Returns:
            SoftInteraction with p ∈ [0, 1] validated by Pydantic
        """
        # Attestation outcome_sign: +1 (positive), -1 (negative), 0 (suppressed/neutral)
        # High-confidence mapping: ±0.8 in task_progress_delta.
        #
        # Negative attestations must also populate the penalty fields. Leaving
        # rework/rejections/misuse at zero treats them as "no rework / no
        # rejections / no misuse" — *positive* signals in the proxy weighting —
        # which combined with task_progress=-0.8 still yields v_hat ≈ +0.08
        # and p ≈ 0.54 under default weights. A negative attestation would
        # therefore record as a beneficial interaction. We mirror the negative
        # signal across the penalty fields so v_hat tracks the attestation's
        # true sign.
        outcome_sign = getattr(attestation, "outcome_sign", 0)
        task_progress = 0.8 * outcome_sign if outcome_sign != 0 else 0.0
        if outcome_sign < 0:
            rework_count = 3
            verifier_rejections = 2
            tool_misuse_flags = 1
            engagement_delta = -0.5
        else:
            rework_count = 0
            verifier_rejections = 0
            tool_misuse_flags = 0
            engagement_delta = 0.0

        observables = ProxyObservables(
            task_progress_delta=task_progress,
            rework_count=rework_count,
            verifier_rejections=verifier_rejections,
            tool_misuse_flags=tool_misuse_flags,
            counterparty_engagement_delta=engagement_delta,
        )

        v_hat, p = computer.compute_labels(observables)

        # Use interaction_id from attestation if present, else generate fresh UUID
        interaction_id = getattr(attestation, "interaction_id", "")
        if not interaction_id:
            interaction_id = str(uuid.uuid4())

        return SoftInteraction(
            interaction_id=interaction_id,
            initiator=initiator_id,
            counterparty=counterparty_id,
            interaction_type=InteractionType.REPLY,
            accepted=outcome_sign > 0,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=observables.tool_misuse_flags,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            metadata={
                "bridge": "agentveil",
                "event_type": "attestation_received",
                "outcome_sign": outcome_sign,
                "evidence_hash": getattr(attestation, "evidence_hash", ""),
            },
        )

    def reputation_change_to_update(
        self,
        snapshot: ReputationSnapshotEvent,
        *,
        agent_id: str,
    ) -> dict[str, Any]:
        """Convert a reputation snapshot to a payoff reputation update.

        Failure mode D1 (double-counting) mitigation: This method returns
        r_delta = 0.0 by design. In v1, AVP tier influences p ONLY via the mapper
        (trust_decision_to_observables); it does NOT adjust payoff-side reputation
        (r_a / r_b in the SoftPayoffEngine).

        Rationale for D1 suppression:
          If tier raised p via mapper AND also boosted r_a/r_b in payoff,
          trusted agents would receive compounded reputation rewards, creating
          feedback amplification (C4). By choosing tier→p channel exclusively,
          we allow governance (via p's effect on transfer and costs) without
          distorting wealth distribution (r_a/r_b).

        Future (v2): If an explicit `enable_payoff_reputation_writeback` flag is added
        to AgentVeilConfig, this method can be extended to compute r_delta from AVP's
        Bayesian score. For now, this is hard-coded conservative.

        Args:
            snapshot: ReputationSnapshotEvent with score, tier, attestation_count
            agent_id: ID of the agent whose reputation changed

        Returns:
            Dict with keys: agent_id, r_delta (0.0), source ("agentveil")
        """
        logger.debug(
            "D1 mitigation: reputation update suppressed for agent %s; "
            "tier already influences p via mapper. "
            "To enable payoff-side r_delta, add AgentVeilConfig.enable_payoff_reputation_writeback in v2.",
            agent_id,
        )

        return {
            "agent_id": agent_id,
            "r_delta": 0.0,
            "source": "agentveil",
        }
