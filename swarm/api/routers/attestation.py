"""Attestation & Relational Messaging API router.

Exposes endpoints for:
- Attesting agent actions (creating sealed receipts)
- Querying and verifying receipts
- Sending and receiving relay messages backed by receipts
"""

from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from swarm.api.models.attestation import (
    AttestActionRequest,
    ChainValidationResponse,
    MessageResponse,
    MessageSubmission,
    ReceiptResponse,
)
from swarm.attestation.middleware import AttestationMiddleware
from swarm.attestation.receipt import AdmissibilityReceipt, ExecutionBounds
from swarm.attestation.relay import ReceiptRelay, RelayMessage
from swarm.attestation.signer import ReceiptSigner, ReceiptVerifier
from swarm.models.events import Event, EventType

router = APIRouter()


# ------------------------------------------------------------------ #
# Dependency injection (replaces module-level singletons)
# ------------------------------------------------------------------ #


class _AttestationState:
    """Thread-safe container for the attestation subsystem singletons."""

    def __init__(self) -> None:
        self.signer = ReceiptSigner()
        # Verification needs no secret: the relay trusts exactly this
        # process's signing DID (bead jxyi).
        verifier = ReceiptVerifier(trusted_signers={self.signer.did})
        self.middleware = AttestationMiddleware(signer=self.signer)
        self.relay = ReceiptRelay(verifier=verifier)


@lru_cache(maxsize=1)
def _get_state() -> _AttestationState:
    return _AttestationState()


def _get_middleware(state: Annotated[_AttestationState, Depends(_get_state)]) -> AttestationMiddleware:
    return state.middleware


def _get_relay(state: Annotated[_AttestationState, Depends(_get_state)]) -> ReceiptRelay:
    return state.relay


# Annotated dependency types for endpoint signatures
MiddlewareDep = Annotated[AttestationMiddleware, Depends(_get_middleware)]
RelayDep = Annotated[ReceiptRelay, Depends(_get_relay)]


def _receipt_to_response(receipt: AdmissibilityReceipt) -> ReceiptResponse:
    return ReceiptResponse(
        receipt_id=receipt.receipt_id,
        timestamp=receipt.timestamp,
        status=receipt.status.value,
        agent_id=receipt.agent_id,
        action_type=receipt.action_type,
        event_id=receipt.event_id,
        parent_receipt_ids=receipt.parent_receipt_ids,
        payload_hash=receipt.payload_hash,
        policy_results=[pr.model_dump() for pr in receipt.policy_results],
        bounds=receipt.bounds.model_dump(),
        signature=receipt.signature,
        signer_id=receipt.signer_id,
        confidence=receipt.confidence,
        admissible=receipt.is_admissible(),
    )


# ------------------------------------------------------------------ #
# Receipt endpoints
# ------------------------------------------------------------------ #


@router.post("/attest", response_model=ReceiptResponse)
async def attest_action(
    req: AttestActionRequest,
    mw: MiddlewareDep,
    relay: RelayDep,
) -> ReceiptResponse:
    """Attest an agent action — creates and seals a receipt."""
    # Build a synthetic Event from the request
    event = Event(
        event_type=EventType(req.action_type)
        if req.action_type in {e.value for e in EventType}
        else EventType.AGENT_STATE_UPDATED,
        agent_id=req.agent_id,
        event_id=req.event_id or str(uuid.uuid4()),
        scenario_id=req.scenario_id,
        epoch=req.epoch,
        step=req.step,
        payload=req.payload,
    )

    bounds = ExecutionBounds(**req.bounds) if req.bounds else None

    receipt = mw.attest(
        event=event,
        payload=req.payload,
        bounds=bounds,
        parent_receipt_ids=req.parent_receipt_ids or None,
    )

    relay.ingest(receipt)
    return _receipt_to_response(receipt)


@router.get("/receipts/{receipt_id}", response_model=ReceiptResponse)
async def get_receipt(
    receipt_id: str,
    relay: RelayDep,
) -> ReceiptResponse:
    """Look up a single receipt by ID."""
    receipt = relay.get(receipt_id)
    if receipt is None:
        raise HTTPException(status_code=404, detail="Receipt not found")
    return _receipt_to_response(receipt)


@router.get("/receipts", response_model=List[ReceiptResponse])
async def list_receipts(
    relay: RelayDep,
    agent_id: Optional[str] = None,
    admissible_only: bool = False,
) -> List[ReceiptResponse]:
    """List receipts, optionally filtered by agent and admissibility."""
    if admissible_only:
        results = relay.query_admissible(agent_id=agent_id)
    elif agent_id:
        results = relay.query_by_agent(agent_id)
    else:
        results = relay.query_admissible()  # default: admissible only
    return [_receipt_to_response(r) for r in results]


@router.get(
    "/receipts/{receipt_id}/chain",
    response_model=ChainValidationResponse,
)
async def validate_chain(
    receipt_id: str,
    relay: RelayDep,
) -> ChainValidationResponse:
    """Validate the full parent chain of a receipt."""
    receipt = relay.get(receipt_id)
    if receipt is None:
        raise HTTPException(status_code=404, detail="Receipt not found")
    valid = relay.chain_valid(receipt_id)
    return ChainValidationResponse(receipt_id=receipt_id, chain_valid=valid)


# ------------------------------------------------------------------ #
# Relational messaging endpoints
# ------------------------------------------------------------------ #


@router.post("/messages", response_model=MessageResponse)
async def send_message(
    msg: MessageSubmission,
    relay: RelayDep,
) -> MessageResponse:
    """Send a relay message backed by receipt references."""
    message = RelayMessage(
        message_id=str(uuid.uuid4()),
        from_agent=msg.from_agent,
        to_agent=msg.to_agent,
        body=msg.body,
        receipt_ids=msg.receipt_ids,
        metadata=msg.metadata,
    )

    # Identify which receipt IDs are missing or inadmissible
    bad_ids = []
    for rid in msg.receipt_ids:
        r = relay.get(rid)
        if r is None or not r.is_admissible():
            bad_ids.append(rid)
    if bad_ids:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Message rejected: the following receipt IDs are missing or "
                f"inadmissible: {bad_ids}"
            ),
        )

    if not relay.send_message(message):
        raise HTTPException(
            status_code=422,
            detail="Message rejected: receipt validation failed",
        )

    return MessageResponse(
        message_id=message.message_id,
        timestamp=message.timestamp,
        from_agent=message.from_agent,
        to_agent=message.to_agent,
        body=message.body,
        receipt_ids=message.receipt_ids,
        metadata=message.metadata,
        acknowledged=message.acknowledged,
    )


@router.get("/messages/inbox/{agent_id}", response_model=List[MessageResponse])
async def get_inbox(
    agent_id: str,
    relay: RelayDep,
    unacknowledged_only: bool = True,
) -> List[MessageResponse]:
    """Retrieve messages for an agent's inbox."""
    msgs = relay.inbox(agent_id, unacknowledged_only=unacknowledged_only)
    return [
        MessageResponse(
            message_id=m.message_id,
            timestamp=m.timestamp,
            from_agent=m.from_agent,
            to_agent=m.to_agent,
            body=m.body,
            receipt_ids=m.receipt_ids,
            metadata=m.metadata,
            acknowledged=m.acknowledged,
        )
        for m in msgs
    ]


@router.post("/messages/{message_id}/ack")
async def acknowledge_message(
    message_id: str,
    relay: RelayDep,
) -> dict:
    """Acknowledge a relay message."""
    if not relay.acknowledge(message_id):
        raise HTTPException(status_code=404, detail="Message not found")
    return {"message_id": message_id, "acknowledged": True}
