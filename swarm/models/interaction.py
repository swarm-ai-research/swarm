"""Soft interaction data model with probabilistic labels."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, cast

from pydantic import BaseModel, Field, field_validator


class InteractionType(Enum):
    """Types of interactions between agents."""

    REPLY = "reply"
    VOTE = "vote"
    TRADE = "trade"
    COLLABORATION = "collaboration"


class SoftInteraction(BaseModel):
    """
    An interaction between two agents with soft (probabilistic) labels.

    The soft label p represents P(v = +1), the probability that the
    interaction is beneficial rather than harmful. The value of p is
    constrained to the range [0, 1].
    """

    # Identity
    interaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)

    # Parties
    initiator: str = ""
    counterparty: str = ""

    # Interaction metadata
    interaction_type: InteractionType = InteractionType.REPLY
    accepted: bool = False

    # Proxy inputs (raw observables)
    task_progress_delta: float = 0.0
    rework_count: int = 0
    verifier_rejections: int = 0
    tool_misuse_flags: int = 0
    counterparty_engagement_delta: float = 0.0

    # Computed labels
    v_hat: float = 0.0  # raw proxy score [-1, +1]
    p: float = 0.5  # P(v = +1) in [0, 1]

    # Payoff components
    tau: float = 0.0  # transfer from initiator to counterparty
    c_a: float = 0.0  # governance cost (initiator)
    c_b: float = 0.0  # governance cost (counterparty)
    r_a: float = 0.0  # reputation change (initiator)
    r_b: float = 0.0  # reputation change (counterparty)

    # Optional ground truth for calibration testing
    ground_truth: Optional[int] = None  # +1 or -1 if known

    # Causal parent links for credit propagation DAG
    causal_parents: list[str] = Field(default_factory=list)

    # Optional metadata for domain-specific interactions
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('p')
    @classmethod
    def p_must_be_probability(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f'p must be in [0, 1], got {v}')
        return v

    @field_validator('v_hat')
    @classmethod
    def v_hat_must_be_bounded(cls, v: float) -> float:
        if not (-1.0 <= v <= 1.0):
            raise ValueError(f'v_hat must be in [-1, 1], got {v}')
        return v

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> "SoftInteraction":
        """Override to enforce p and v_hat invariants after copy.

        Pydantic v2's model_copy(update=...) skips field_validators,
        so we manually enforce safety-critical invariants here to ensure
        ALL callers — not just _validated_copy — get bounds checking.
        """
        result = cast(
            "SoftInteraction",
            super().model_copy(update=update, deep=deep),
        )
        if not (0.0 <= result.p <= 1.0):
            raise ValueError(f"p invariant violated after copy: {result.p}")
        if not (-1.0 <= result.v_hat <= 1.0):
            raise ValueError(f"v_hat invariant violated after copy: {result.v_hat}")
        return result

    def is_high_quality(self, threshold: float = 0.5) -> bool:
        """Check if interaction is above quality threshold."""
        return self.p >= threshold

    def is_uncertain(self, band: float = 0.2) -> bool:
        """Check if p is in the uncertain band around 0.5."""
        return 0.5 - band < self.p < 0.5 + band

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Delegates to Pydantic's ``model_dump(mode='json')`` which handles
        datetime -> ISO-8601 string and Enum -> value conversions automatically.
        """
        result: dict = self.model_dump(mode="json")
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "SoftInteraction":
        """Create from dictionary.

        Delegates to Pydantic's ``model_validate()`` which coerces ISO-8601
        strings back to datetime and enum value strings back to InteractionType.
        """
        instance: SoftInteraction = cls.model_validate(data)
        return instance
