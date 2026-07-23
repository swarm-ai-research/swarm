"""Configuration for the CrewAI bridge."""

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class CrewAIBridgeConfig(BaseModel):
    """Configuration for mapping CrewAI crew runs to SWARM interactions.

    Instances are mutable, allowing optional runtime configuration adjustments.

    Attributes:
        crew_id: Identifier for this crew in SWARM (used as initiator_id).
        proxy_sigmoid_k: Steepness of sigmoid for v_hat → p.
        max_delegation_depth: Delegation depth treated as maximum rework.
        engagement_max_chars: Output character count treated as full engagement.
        timeout_seconds: Seconds before a crew.kickoff() times out.
        enable_event_log: Whether to write interactions to EventLog.
        event_log_path: Path for the JSONL event log (optional).
    """

    crew_id: str = "crewai-crew"
    proxy_sigmoid_k: float = Field(default=2.0, gt=0)
    max_delegation_depth: int = Field(default=5, ge=1, le=50)
    engagement_max_chars: int = Field(default=3000, ge=1)
    timeout_seconds: float = Field(default=120.0, gt=0)
    enable_event_log: bool = True
    event_log_path: Optional[str] = None

    model_config = {"frozen": False}

    @model_validator(mode="after")
    def _validate(self) -> "CrewAIBridgeConfig":
        if not self.crew_id.strip():
            raise ValueError("crew_id cannot be empty")
        return self
