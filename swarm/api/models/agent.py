"""Agent-related API models."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    """Agent registration status."""

    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUSPENDED = "suspended"


class AgentRegistration(BaseModel):
    """Request model for agent registration."""

    name: str = Field(..., description="Agent name", min_length=1, max_length=100)
    description: str = Field(
        ..., description="Description of the agent's purpose", max_length=1000
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description="List of agent capabilities (e.g., 'negotiation', 'analysis')",
    )
    policy_declaration: str | None = Field(
        default=None,
        description="Agent's declared behavior policy",
        max_length=5000,
    )
    callback_url: str | None = Field(
        default=None,
        description="Optional webhook URL for async notifications",
    )


class AgentUpdate(BaseModel):
    """Request model for updating agent profile fields."""

    name: str | None = Field(
        default=None, description="Agent name", min_length=1, max_length=100
    )
    description: str | None = Field(
        default=None, description="Description of the agent's purpose", max_length=1000
    )
    capabilities: list[str] | None = Field(
        default=None,
        description="List of agent capabilities (e.g., 'negotiation', 'analysis')",
    )
    callback_url: str | None = Field(
        default=None,
        description="Optional webhook URL for async notifications",
    )
    policy_declaration: str | None = Field(
        default=None,
        description="Agent's declared behavior policy",
        max_length=5000,
    )


class AgentResponse(BaseModel):
    """Response model for registered agent."""

    agent_id: str = Field(..., description="Unique agent identifier")
    api_key: str = Field(
        ..., description="API key for authentication (only shown once on registration)"
    )
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    capabilities: list[str] = Field(
        default_factory=list, description="Agent capabilities"
    )
    status: AgentStatus = Field(..., description="Registration status")
    registered_at: datetime = Field(..., description="Registration timestamp")
