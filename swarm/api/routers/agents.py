"""Agent registration and management endpoints."""

import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from swarm.api.middleware import DEFAULT_SCOPES, Scope, register_api_key, require_scope
from swarm.api.models.agent import (
    AgentRegistration,
    AgentResponse,
    AgentStatus,
    AgentUpdate,
)

router = APIRouter()

# In-memory storage for development (replace with database in production)
_registered_agents: dict[str, AgentResponse] = {}

# Maps agent_id -> raw API key (held until approval, then registered)
_pending_keys: dict[str, str] = {}

# Rate limiting for registration endpoint (per-IP).
# Maps IP -> list of registration timestamps within the window.
_registration_rate: dict[str, list[float]] = {}
_REGISTRATION_WINDOW = 60.0  # seconds
_REGISTRATION_LIMIT = 10  # max registrations per IP per window
MAX_REGISTERED_AGENTS = 10_000


@router.post("/register", response_model=AgentResponse)
async def register_agent(
    registration: AgentRegistration, request: Request
) -> AgentResponse:
    """Register a new agent to participate in SWARM simulations.

    Args:
        registration: Agent registration details.

    Returns:
        Registered agent info with API key.
    """
    # Per-IP rate limiting to prevent registration flooding (security fix 5.1)
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _registration_rate.get(client_ip, [])
    window = [t for t in window if now - t < _REGISTRATION_WINDOW]
    if len(window) >= _REGISTRATION_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Too many registrations. Try again later.",
        )
    window.append(now)
    _registration_rate[client_ip] = window

    # Cap total registered agents
    if len(_registered_agents) >= MAX_REGISTERED_AGENTS:
        raise HTTPException(
            status_code=429,
            detail="Registration capacity reached. Try again later.",
        )

    agent_id = str(uuid.uuid4())
    api_key = f"swarm_{uuid.uuid4().hex}"  # Simple key for now

    # Check auto-approve config (stashed on the app instance)
    auto_approve = getattr(request.app, "_swarm_auto_approve", True)

    if auto_approve:
        status = AgentStatus.APPROVED
    else:
        status = AgentStatus.PENDING_REVIEW

    agent = AgentResponse(
        agent_id=agent_id,
        api_key=api_key,
        name=registration.name,
        description=registration.description,
        capabilities=registration.capabilities,
        status=status,
        registered_at=datetime.now(timezone.utc),
    )

    _registered_agents[agent_id] = agent

    if auto_approve:
        # Register the API key immediately
        register_api_key(api_key, agent_id, scopes=DEFAULT_SCOPES)
    else:
        # Hold the key until approved
        _pending_keys[agent_id] = api_key

    return agent


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str) -> AgentResponse:
    """Get agent details by ID.

    Args:
        agent_id: The agent's unique identifier.

    Returns:
        Agent details.

    Raises:
        HTTPException: If agent not found.
    """
    if agent_id not in _registered_agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = _registered_agents[agent_id]
    # Don't return the API key on subsequent requests
    return _redacted_response(agent)


@router.get("/", response_model=list[AgentResponse])
async def list_agents(
    status: AgentStatus | None = None,
    capability: str | None = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[AgentResponse]:
    """List all registered agents with optional filtering and pagination.

    Args:
        status: Filter by agent status.
        capability: Filter agents that have this capability.
        limit: Maximum number of results to return.
        offset: Number of results to skip.

    Returns:
        List of registered agents (with redacted API keys).
    """
    agents = list(_registered_agents.values())

    if status is not None:
        agents = [a for a in agents if a.status == status]

    if capability is not None:
        agents = [a for a in agents if capability in a.capabilities]

    agents = agents[offset : offset + limit]

    return [_redacted_response(a) for a in agents]


def _redacted_response(agent: AgentResponse) -> AgentResponse:
    """Return a copy of the agent with the API key redacted."""
    return AgentResponse(
        agent_id=agent.agent_id,
        api_key="[REDACTED]",
        name=agent.name,
        description=agent.description,
        capabilities=agent.capabilities,
        status=agent.status,
        registered_at=agent.registered_at,
    )


@router.patch("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    update: AgentUpdate,
    caller_agent_id: str = Depends(require_scope(Scope.WRITE)),
) -> AgentResponse:
    """Update an agent's profile fields.

    Regular agents can only update their own profile.
    Requires WRITE scope.

    Args:
        agent_id: The agent to update.
        update: Fields to update.
        caller_agent_id: Authenticated caller's agent ID (injected).

    Returns:
        Updated agent details (API key redacted).
    """
    if agent_id not in _registered_agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Non-admin agents can only update their own profile
    if caller_agent_id != agent_id:
        raise HTTPException(
            status_code=403,
            detail="Cannot update another agent's profile",
        )

    agent = _registered_agents[agent_id]
    update_data = update.model_dump(exclude_unset=True)

    # Apply updates to create a new AgentResponse with changed fields
    updated = agent.model_copy(update=update_data)
    _registered_agents[agent_id] = updated

    return _redacted_response(updated)


@router.post("/{agent_id}/suspend", response_model=AgentResponse)
async def suspend_agent(
    agent_id: str,
    _admin_agent_id: str = Depends(require_scope(Scope.ADMIN)),
) -> AgentResponse:
    """Suspend an approved agent. Requires ADMIN scope.

    Valid transition: approved -> suspended.

    Args:
        agent_id: The agent to suspend.

    Returns:
        Updated agent details (API key redacted).
    """
    if agent_id not in _registered_agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = _registered_agents[agent_id]
    if agent.status != AgentStatus.APPROVED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot suspend agent with status '{agent.status.value}'. "
            f"Only approved agents can be suspended.",
        )

    updated = agent.model_copy(update={"status": AgentStatus.SUSPENDED})
    _registered_agents[agent_id] = updated

    return _redacted_response(updated)


@router.post("/{agent_id}/reactivate", response_model=AgentResponse)
async def reactivate_agent(
    agent_id: str,
    _admin_agent_id: str = Depends(require_scope(Scope.ADMIN)),
) -> AgentResponse:
    """Reactivate a suspended agent. Requires ADMIN scope.

    Valid transition: suspended -> approved.

    Args:
        agent_id: The agent to reactivate.

    Returns:
        Updated agent details (API key redacted).
    """
    if agent_id not in _registered_agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = _registered_agents[agent_id]
    if agent.status != AgentStatus.SUSPENDED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot reactivate agent with status '{agent.status.value}'. "
            f"Only suspended agents can be reactivated.",
        )

    updated = agent.model_copy(update={"status": AgentStatus.APPROVED})
    _registered_agents[agent_id] = updated

    return _redacted_response(updated)


@router.post("/{agent_id}/approve", response_model=AgentResponse)
async def approve_agent(
    agent_id: str,
    _admin_agent_id: str = Depends(require_scope(Scope.ADMIN)),
) -> AgentResponse:
    """Approve a pending agent. Requires ADMIN scope.

    Activates the agent's API key so it can make authenticated requests.
    Valid transition: pending_review -> approved.
    """
    if agent_id not in _registered_agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = _registered_agents[agent_id]
    if agent.status != AgentStatus.PENDING_REVIEW:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot approve agent with status '{agent.status.value}'. "
            f"Only pending agents can be approved.",
        )

    # Activate the held API key
    raw_key = _pending_keys.pop(agent_id, None)
    if raw_key is not None:
        register_api_key(raw_key, agent_id, scopes=DEFAULT_SCOPES)

    updated = agent.model_copy(update={"status": AgentStatus.APPROVED})
    _registered_agents[agent_id] = updated

    return _redacted_response(updated)


@router.post("/{agent_id}/reject", response_model=AgentResponse)
async def reject_agent(
    agent_id: str,
    _admin_agent_id: str = Depends(require_scope(Scope.ADMIN)),
) -> AgentResponse:
    """Reject a pending agent. Requires ADMIN scope.

    Discards the held API key. Valid transition: pending_review -> rejected.
    """
    if agent_id not in _registered_agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = _registered_agents[agent_id]
    if agent.status != AgentStatus.PENDING_REVIEW:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot reject agent with status '{agent.status.value}'. "
            f"Only pending agents can be rejected.",
        )

    # Discard the held key
    _pending_keys.pop(agent_id, None)

    updated = agent.model_copy(update={"status": AgentStatus.REJECTED})
    _registered_agents[agent_id] = updated

    return _redacted_response(updated)
