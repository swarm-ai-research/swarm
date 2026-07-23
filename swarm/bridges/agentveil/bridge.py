"""Main orchestrator for the SWARM–AgentVeil bridge (v1, mock-only).

Wires the four bridge components into a single per-interaction control loop:

    AVPClient   — mock AVP SDK (resolve_did / can_trust / submit_attestation / get_reputation)
    AVPMapper   — trust decision → ProxyObservables → (v_hat, p) → SoftInteraction
    AVPPolicy   — admission gate + write-back gate (C4/C5/C6) + circuit breaker (A1)
    EventLog    — SWARM's append-only JSONL log

Data flow per interaction (mirrors ``docs/bridges/agentveil.md`` step list):

    1. ``can_trust(counterparty)``                         → TrustDecision
    2. admission gate (tier / allowed)                     → ALLOW | DENY
    3. mapper: trust decision → ProxyObservables           → (v_hat, p)
    4. build terminal SoftInteraction                      → feeds metrics
    5. write-back gate (uncertain band, rate limits)       → ALLOW | DENY
    6. if ALLOW: ``submit_attestation(sign, evidence_hash)``

Every AVP-visible action is recorded twice: once as a typed
:class:`AgentVeilEvent` in an in-memory audit list (full fidelity), and —
when an :class:`EventLog` is provided — once as a core :class:`Event` so the
on-disk JSONL log stays replayable. This is the failure-mode **D4**
mitigation (no hidden attestation side-effects): the write-back path can
never submit an attestation without emitting an ``ATTESTATION_SUBMITTED``
event, and every suppression emits ``ATTESTATION_SUPPRESSED``.

Per failure-mode **E3**, the attestation carries only the outcome sign and
the canonical evidence hash ``SHA-256(interaction_id || outcome_sign)`` — the
raw ``p`` is never sent to the registry nor written into the attestation
event payload.
"""

from __future__ import annotations

import hashlib
import logging
from typing import List, Optional

from swarm.bridges.agentveil.client import (
    AttestationReceipt,
    AVPClient,
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
from swarm.bridges.agentveil.policy import AVPPolicy, PolicyDecision
from swarm.core.proxy import ProxyComputer
from swarm.logging.event_log import EventLog
from swarm.models.events import Event, EventType
from swarm.models.interaction import InteractionType, SoftInteraction

logger = logging.getLogger(__name__)


class AgentVeilBridge:
    """Connects AVP trust decisions and attestations to SWARM governance.

    The bridge is stateful: it tracks the current epoch (for the policy's
    per-epoch rate-limit counters), a circuit-breaker failure count, and
    bounded in-memory histories of events and interactions.
    """

    def __init__(
        self,
        config: AgentVeilConfig | None = None,
        *,
        client: AVPClient | None = None,
        event_log: EventLog | None = None,
    ) -> None:
        """Initialize the bridge.

        Args:
            config: Bridge configuration. If None, uses defaults.
            client: Pre-built AVPClient (e.g. with test fixtures). If None,
                a fresh mock-mode client is created from ``config.mock_mode``.
            event_log: Optional core EventLog. When provided, interactions and
                write-back attestations are mirrored as replayable core events.
        """
        self._config = config or AgentVeilConfig()
        self._client = client or AVPClient(mock_mode=self._config.mock_mode)
        self._mapper = AVPMapper(self._config)
        self._policy = AVPPolicy(self._config)
        self._proxy = ProxyComputer(sigmoid_k=self._config.proxy_sigmoid_k)
        self._event_log = event_log

        self._events: List[AgentVeilEvent] = []
        self._interactions: List[SoftInteraction] = []

        self._current_epoch: int = 0
        self._consecutive_failures: int = 0

        if self._config.mock_mode:
            logger.warning(
                "AgentVeilBridge running in mock_mode: trust checks are "
                "served from fixtures/defaults, not the live registry (E5)."
            )

    # Epoch / circuit-breaker lifecycle

    def start_epoch(self, epoch: int) -> None:
        """Advance to a new epoch and rotate the policy's per-epoch counters."""
        self._current_epoch = epoch
        self._policy.reset_epoch(epoch=epoch)

    def record_registry_failure(self) -> None:
        """Record a registry call failure for the circuit breaker (A1).

        v1 mock mode never fails, but live mode (v2) will call this when an
        AVP HTTP request errors so the breaker can trip per ``fail_mode``.
        """
        self._consecutive_failures += 1

    def record_registry_success(self) -> None:
        """Reset the circuit-breaker failure count after a healthy call."""
        self._consecutive_failures = 0

    @property
    def circuit_open(self) -> bool:
        """True when the breaker is tripped and configured to fail closed."""
        result = self._policy.circuit_breaker_state(
            consecutive_failures=self._consecutive_failures
        )
        return result.decision == PolicyDecision.DENY

    # Core per-interaction flow

    def process_interaction(
        self,
        *,
        initiator_did: str,
        counterparty_did: str,
        epoch: Optional[int] = None,
        step: int = 0,
    ) -> Optional[SoftInteraction]:
        """Run one trust-gated interaction and (optionally) write back.

        Returns the terminal :class:`SoftInteraction` if the counterparty
        cleared the admission gate, or ``None`` if the interaction was denied
        (registry denial, tier below ``min_tier``, or open circuit breaker).

        Args:
            initiator_did: DID of the agent initiating (the attestor).
            counterparty_did: DID of the agent being trusted/attested about.
            epoch: Epoch for per-epoch rate limits. Defaults to the current
                epoch; a changed value rotates the policy counters first.
            step: Step index recorded on emitted events.
        """
        if epoch is not None and epoch != self._current_epoch:
            self.start_epoch(epoch)
        current_epoch = self._current_epoch

        # Circuit breaker (A1): if the registry is down and we fail closed,
        # deny before making any call.
        if self.circuit_open:
            self._record_avp_event(
                AgentVeilEventType.CIRCUIT_BREAKER_TRIPPED,
                subject_did=counterparty_did,
                actor_did=initiator_did,
                step=step,
                payload={"consecutive_failures": self._consecutive_failures},
            )
            logger.warning(
                "Circuit breaker open; denying interaction with %s", counterparty_did
            )
            return None

        # 1. Trust check
        self._record_avp_event(
            AgentVeilEventType.TRUST_CHECK_REQUESTED,
            subject_did=counterparty_did,
            actor_did=initiator_did,
            step=step,
            payload={"min_tier": self._config.min_tier},
        )
        decision: TrustDecision = self._client.can_trust(
            counterparty_did, min_tier=self._config.min_tier
        )
        self.record_registry_success()

        # 2. Admission gate
        admission = self._policy.admission_decision(
            tier=decision.tier, allowed=decision.allowed
        )
        trust_payload = TrustDecisionEvent(
            allowed=decision.allowed,
            tier=decision.tier,
            risk_level=decision.risk_level,
            confidence=decision.confidence,
            reason=admission.reason,
        ).to_dict()

        if admission.decision == PolicyDecision.DENY:
            self._record_avp_event(
                AgentVeilEventType.TRUST_DENIED,
                subject_did=counterparty_did,
                actor_did=initiator_did,
                step=step,
                payload=trust_payload,
            )
            return None

        self._record_avp_event(
            AgentVeilEventType.TRUST_ALLOWED,
            subject_did=counterparty_did,
            actor_did=initiator_did,
            step=step,
            payload=trust_payload,
        )

        # 3. Map trust decision → observables → (v_hat, p)
        observables = self._mapper.trust_decision_to_observables(
            decision, peer_id=counterparty_did
        )
        v_hat, p = self._proxy.compute_labels(observables)

        # 4. Build the terminal interaction (p validated to [0, 1] by Pydantic — D5)
        interaction = SoftInteraction(
            initiator=initiator_did,
            counterparty=counterparty_did,
            interaction_type=InteractionType.REPLY,
            accepted=True,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=observables.tool_misuse_flags,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            metadata={
                "bridge": "agentveil",
                "tier": decision.tier,
                "risk_level": decision.risk_level,
                "avp_confidence": decision.confidence,
            },
        )
        self._record_interaction(interaction)
        self._log_interaction(interaction, step=step)

        # 5–6. Write-back gate + attestation
        self._maybe_write_back(
            interaction,
            initiator_did=initiator_did,
            counterparty_did=counterparty_did,
            epoch=current_epoch,
            step=step,
        )

        return interaction

    def _maybe_write_back(
        self,
        interaction: SoftInteraction,
        *,
        initiator_did: str,
        counterparty_did: str,
        epoch: int,
        step: int,
    ) -> Optional[AttestationReceipt]:
        """Decide whether to attest, and submit if the write-back gate allows.

        On suppression, emits ``ATTESTATION_SUPPRESSED`` (with the gate reason)
        and returns None. On submission, computes the canonical evidence hash
        (E3), calls the client, and emits ``ATTESTATION_SUBMITTED`` (D4).
        """
        result = self._policy.writeback_decision(
            terminal_p=interaction.p,
            initiator_did=initiator_did,
            counterparty_did=counterparty_did,
            current_epoch=epoch,
        )

        if result.decision == PolicyDecision.DENY:
            payload = AttestationEvent(
                interaction_id=interaction.interaction_id,
                outcome_sign=0,
                evidence_hash="",
                suppression_reason=result.reason,
            ).to_dict()
            self._record_avp_event(
                AgentVeilEventType.ATTESTATION_SUPPRESSED,
                subject_did=counterparty_did,
                actor_did=initiator_did,
                step=step,
                payload=payload,
            )
            self._log_attestation(
                EventType.ATTESTATION_SUPPRESSED,
                interaction_id=interaction.interaction_id,
                initiator_did=initiator_did,
                counterparty_did=counterparty_did,
                outcome_sign=0,
                evidence_hash="",
                suppression_reason=result.reason,
                step=step,
            )
            return None

        outcome_sign = int(result.metadata["outcome_sign"])
        evidence_hash = self._evidence_hash(interaction.interaction_id, outcome_sign)

        receipt = self._client.submit_attestation(
            counterparty_did, outcome_sign, evidence_hash
        )

        payload = AttestationEvent(
            interaction_id=interaction.interaction_id,
            outcome_sign=outcome_sign,
            evidence_hash=evidence_hash,
            suppression_reason=None,
        ).to_dict()
        payload["attestation_id"] = receipt.attestation_id
        self._record_avp_event(
            AgentVeilEventType.ATTESTATION_SUBMITTED,
            subject_did=counterparty_did,
            actor_did=initiator_did,
            step=step,
            payload=payload,
        )
        self._log_attestation(
            EventType.ATTESTATION_SUBMITTED,
            interaction_id=interaction.interaction_id,
            initiator_did=initiator_did,
            counterparty_did=counterparty_did,
            outcome_sign=outcome_sign,
            evidence_hash=evidence_hash,
            suppression_reason=None,
            step=step,
            attestation_id=receipt.attestation_id,
        )
        return receipt

    # Auxiliary AVP flows

    def ingest_attestation(
        self,
        attestation: AttestationEvent,
        *,
        initiator_id: str,
        counterparty_id: str,
    ) -> SoftInteraction:
        """Convert an *incoming* peer attestation to a SoftInteraction.

        Used when another agent attests about a counterparty and SWARM wants
        to fold that external signal into its own metrics stream.
        """
        interaction = self._mapper.attestation_received_to_interaction(
            attestation,
            initiator_id=initiator_id,
            counterparty_id=counterparty_id,
            computer=self._proxy,
        )
        self._record_interaction(interaction)
        self._log_interaction(interaction, step=0)
        return interaction

    def fetch_reputation(self, did: str, *, step: int = 0) -> ReputationSnapshot:
        """Fetch a DID's reputation and record a REPUTATION_FETCHED event."""
        snapshot = self._client.get_reputation(did)
        self.record_registry_success()
        self._record_avp_event(
            AgentVeilEventType.REPUTATION_FETCHED,
            subject_did=did,
            actor_did=self._config.orchestrator_did,
            step=step,
            payload=ReputationSnapshotEvent(
                score=snapshot.score,
                confidence=snapshot.confidence,
                tier=snapshot.tier,
                attestation_count=snapshot.attestation_count,
            ).to_dict(),
        )
        return snapshot

    # Event / interaction bookkeeping

    def _record_avp_event(
        self,
        event_type: AgentVeilEventType,
        *,
        subject_did: str,
        actor_did: str,
        step: int,
        payload: dict,
    ) -> None:
        event = AgentVeilEvent(
            event_type=event_type,
            subject_did=subject_did,
            actor_did=actor_did,
            step=step,
            payload=payload,
        )
        self._events.append(event)
        if len(self._events) > self._config.max_events:
            self._events = self._events[-self._config.max_events :]

    def _record_interaction(self, interaction: SoftInteraction) -> None:
        self._interactions.append(interaction)
        if len(self._interactions) > self._config.max_interactions:
            self._interactions = self._interactions[-self._config.max_interactions :]

    def _log_interaction(self, interaction: SoftInteraction, *, step: int) -> None:
        if self._event_log is None:
            return
        self._event_log.append(
            Event(
                event_type=EventType.INTERACTION_COMPLETED,
                interaction_id=interaction.interaction_id,
                initiator_id=interaction.initiator,
                counterparty_id=interaction.counterparty,
                step=step,
                epoch=self._current_epoch,
                payload={
                    "accepted": interaction.accepted,
                    "v_hat": interaction.v_hat,
                    "p": interaction.p,
                    "bridge": "agentveil",
                    "metadata": dict(interaction.metadata or {}),
                },
            )
        )

    def _log_attestation(
        self,
        event_type: EventType,
        *,
        interaction_id: str,
        initiator_did: str,
        counterparty_did: str,
        outcome_sign: int,
        evidence_hash: str,
        suppression_reason: Optional[str],
        step: int,
        attestation_id: Optional[str] = None,
    ) -> None:
        """Mirror a write-back decision into the replayable core event log (D4).

        Per E3, the payload carries only the sign + evidence hash, never the
        raw ``p``.
        """
        if self._event_log is None:
            return
        payload: dict = {
            "bridge": "agentveil",
            "subject_did": counterparty_did,
            "actor_did": initiator_did,
            "outcome_sign": outcome_sign,
            "evidence_hash": evidence_hash,
        }
        if suppression_reason is not None:
            payload["suppression_reason"] = suppression_reason
        if attestation_id is not None:
            payload["attestation_id"] = attestation_id
        self._event_log.append(
            Event(
                event_type=event_type,
                interaction_id=interaction_id,
                initiator_id=initiator_did,
                counterparty_id=counterparty_did,
                step=step,
                epoch=self._current_epoch,
                payload=payload,
            )
        )

    @staticmethod
    def _evidence_hash(interaction_id: str, outcome_sign: int) -> str:
        """Canonical attestation evidence hash: SHA-256(interaction_id || sign).

        The sign is rendered with an explicit ``+``/``-`` (``+1`` / ``-1``) so
        the hashed string matches the plan's canonical form and round-trips
        the client's ``did:key`` hex validation (E3).
        """
        combined = f"{interaction_id}||{outcome_sign:+d}".encode()
        return hashlib.sha256(combined).hexdigest()

    # Introspection

    def get_interactions(self) -> List[SoftInteraction]:
        return list(self._interactions)

    def get_events(self) -> List[AgentVeilEvent]:
        return list(self._events)

    def get_events_by_type(
        self, event_type: AgentVeilEventType
    ) -> List[AgentVeilEvent]:
        return [e for e in self._events if e.event_type == event_type]

    @property
    def policy(self) -> AVPPolicy:
        return self._policy

    @property
    def client(self) -> AVPClient:
        return self._client

    @property
    def current_epoch(self) -> int:
        return self._current_epoch
