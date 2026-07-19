"""Attestation middleware — intercepts events and attaches receipts.

Sits in the simulation pipeline between action execution and event logging.
Every event that passes through the middleware is sealed with a
``AdmissibilityReceipt`` and the receipt is stored in the relay for
downstream consumption.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from swarm.attestation.receipt import (
    AdmissibilityReceipt,
    ExecutionBounds,
    PolicyCompliance,
    ReceiptStatus,
)
from swarm.attestation.signer import ReceiptSigner
from swarm.models.events import Event, EventType

# Type alias for pluggable policy evaluators.
# A policy evaluator receives the event and returns a PolicyCompliance.
PolicyEvaluator = Callable[[Event], PolicyCompliance]


class AttestationMiddleware:
    """Intercepts events, evaluates policies, and seals receipts.

    Parameters
    ----------
    signer:
        The ``ReceiptSigner`` used to seal receipts.
    policy_evaluators:
        Optional list of callables that evaluate domain-specific policies
        against each event.  All evaluators run; the receipt records all
        results (pass *and* fail).
    default_bounds:
        Default execution bounds applied when no per-action bounds are
        supplied.
    event_callback:
        Optional callback invoked when a receipt is sealed, used to emit
        ``EventType.RECEIPT_SEALED`` events for replay.
    """

    def __init__(
        self,
        signer: ReceiptSigner,
        policy_evaluators: Optional[List[PolicyEvaluator]] = None,
        default_bounds: Optional[ExecutionBounds] = None,
        event_callback: Optional[Callable[[Event], None]] = None,
    ) -> None:
        self._signer = signer
        self._evaluators: List[PolicyEvaluator] = policy_evaluators or []
        self._default_bounds = default_bounds or ExecutionBounds()  # type: ignore[call-arg]
        self._receipts: Dict[str, AdmissibilityReceipt] = {}
        self._lock = threading.Lock()
        self._event_callback = event_callback

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def attest(
        self,
        event: Event,
        payload: Dict[str, Any],
        bounds: Optional[ExecutionBounds] = None,
        parent_receipt_ids: Optional[List[str]] = None,
    ) -> AdmissibilityReceipt:
        """Create, evaluate, seal, and store a receipt for *event*.

        Parameters
        ----------
        event:
            The simulation event being attested.
        payload:
            The full action payload (hashed into the receipt).
        bounds:
            Execution bounds for this action.  Falls back to
            ``default_bounds`` if *None*.
        parent_receipt_ids:
            Receipt IDs of upstream actions this event depends on.

        Returns
        -------
        AdmissibilityReceipt
            The sealed receipt (status ``SEALED``).  If any policy
            evaluator fails, the receipt is still sealed but
            ``is_admissible()`` will return ``False``.
        """
        now = datetime.utcnow()
        payload_hash = AdmissibilityReceipt.hash_payload(payload)

        receipt_id = AdmissibilityReceipt.generate_receipt_id(
            agent_id=event.agent_id or "system",
            action_type=event.event_type.value,
            payload_hash=payload_hash,
            parent_receipt_ids=parent_receipt_ids,
        )

        # Run policy evaluators
        policy_results: List[PolicyCompliance] = []
        for evaluator in self._evaluators:
            policy_results.append(evaluator(event))

        receipt = AdmissibilityReceipt(  # type: ignore[call-arg]
            receipt_id=receipt_id,
            timestamp=now,
            status=ReceiptStatus.PENDING,
            agent_id=event.agent_id or "system",
            action_type=event.event_type.value,
            event_id=event.event_id,
            parent_receipt_ids=parent_receipt_ids or [],
            payload_hash=payload_hash,
            policy_results=policy_results,
            bounds=bounds or self._default_bounds,
            scenario_id=event.scenario_id,
            epoch=event.epoch,
            step=event.step,
        )

        sealed = self._signer.seal(receipt)

        with self._lock:
            self._receipts[sealed.receipt_id] = sealed

        # Emit JSONL event for replay
        if self._event_callback:
            self._event_callback(
                Event(
                    event_type=EventType.RECEIPT_SEALED,
                    agent_id=sealed.agent_id,
                    scenario_id=sealed.scenario_id,
                    epoch=sealed.epoch,
                    step=sealed.step,
                    payload={
                        "receipt_id": sealed.receipt_id,
                        "action_type": sealed.action_type,
                        "payload_hash": sealed.payload_hash,
                        "admissible": sealed.is_admissible(),
                    },
                )
            )

        return sealed

    def get_receipt(self, receipt_id: str) -> Optional[AdmissibilityReceipt]:
        """Look up a receipt by ID."""
        with self._lock:
            return self._receipts.get(receipt_id)

    def get_receipts_for_event(self, event_id: str) -> List[AdmissibilityReceipt]:
        """Return all receipts associated with *event_id*."""
        with self._lock:
            return [
                r for r in self._receipts.values() if r.event_id == event_id
            ]

    def get_receipts_for_agent(self, agent_id: str) -> List[AdmissibilityReceipt]:
        """Return all receipts issued to *agent_id*."""
        with self._lock:
            return [
                r for r in self._receipts.values() if r.agent_id == agent_id
            ]

    @property
    def receipt_count(self) -> int:
        with self._lock:
            return len(self._receipts)

    def add_policy_evaluator(self, evaluator: PolicyEvaluator) -> None:
        """Register an additional policy evaluator."""
        self._evaluators.append(evaluator)
