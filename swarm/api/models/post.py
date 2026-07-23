"""Post/feed card models for the Moltbook-style results feed."""

import json
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class PostCreate(BaseModel):
    """Request to publish a run result card to the feed."""

    run_id: str = Field(..., description="ID of the completed run")
    title: str = Field(
        ..., description="Card title", min_length=1, max_length=200
    )
    blurb: str = Field(
        ..., description="Short summary of results", max_length=2000
    )
    key_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Highlighted metrics to display on the card",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for filtering (e.g. 'baseline', 'sweep')",
    )

    @field_validator("key_metrics")
    @classmethod
    def validate_key_metrics(cls, v: dict) -> dict:
        """Bound the size of key_metrics to prevent memory bombs."""
        if len(v) > 50:
            raise ValueError("key_metrics must have at most 50 entries")
        # Reject deeply nested or oversized payloads by checking serialized size
        serialized = json.dumps(v, default=str)
        if len(serialized) > 10_000:
            raise ValueError("key_metrics serialized size must be under 10KB")
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: list) -> list:
        """Bound tags count and individual tag length."""
        if len(v) > 20:
            raise ValueError("At most 20 tags allowed")
        for tag in v:
            if not isinstance(tag, str):
                raise ValueError("Each tag must be a string")
            if len(tag) > 100:
                raise ValueError("Each tag must be at most 100 characters")
            if not tag.strip():
                raise ValueError("Tags must not be blank")
        return v


class PostResponse(BaseModel):
    """A published result card."""

    post_id: str = Field(..., description="Unique post identifier")
    run_id: str = Field(..., description="Associated run")
    agent_id: str = Field(..., description="Agent that published the card")
    title: str
    blurb: str
    key_metrics: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    published_at: datetime
    run_url: Optional[str] = Field(None, description="Link to full run results")
    upvotes: int = Field(0, description="Current upvote count")
    downvotes: int = Field(0, description="Current downvote count")


class FeedQuery(BaseModel):
    """Query parameters for the feed."""

    tags: Optional[list[str]] = Field(None, description="Filter by tags")
    agent_id: Optional[str] = Field(None, description="Filter by publishing agent")
    limit: int = Field(default=20, ge=1, le=100, description="Max results")
    offset: int = Field(default=0, ge=0, description="Pagination offset")
