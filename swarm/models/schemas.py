"""TypedDict schemas for event payloads and metadata.

Provides static-analysis-friendly type information for the ad-hoc
``dict`` payloads used throughout the event and interaction systems.
All fields are ``NotRequired`` (total=False) because payloads are
assembled incrementally by different code paths.

These types are *documentation + tooling only* — at runtime, plain
``dict`` is still used everywhere so there is zero performance cost.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict

# =====================================================================
# Core event payloads
# =====================================================================


class InteractionProposedPayload(TypedDict, total=False):
    interaction_type: str
    v_hat: float
    p: float


class InteractionCompletedPayload(TypedDict, total=False):
    accepted: bool
    payoff_initiator: float
    payoff_counterparty: float


class PayoffComputedPayload(TypedDict, total=False):
    payoff_initiator: float
    payoff_counterparty: float
    components: Dict[str, Any]


class GovernanceCostPayload(TypedDict, total=False):
    agent_id: str
    cost: float
    reason: str


class ReputationUpdatedPayload(TypedDict, total=False):
    old_reputation: float
    new_reputation: float
    delta: float
    reason: str


# =====================================================================
# System event payloads
# =====================================================================


class AgentCreatedPayload(TypedDict, total=False):
    agent_type: str
    name: str
    roles: List[str]


class SimulationStartedPayload(TypedDict, total=False):
    n_epochs: int
    steps_per_epoch: int
    n_agents: int
    seed: Optional[int]
    scenario_id: Optional[str]
    replay_k: Optional[int]


class SimulationEndedPayload(TypedDict, total=False):
    total_epochs: int
    final_metrics: Dict[str, Any]


class EpochCompletedPayload(TypedDict, total=False):
    epoch: int
    total_interactions: int
    accepted_interactions: int
    total_posts: int
    total_votes: int
    toxicity_rate: float
    quality_gap: float
    avg_payoff: float
    total_welfare: float
    network_metrics: Optional[Dict[str, float]]
    network_edges_pruned: int


# =====================================================================
# Marketplace event payloads
# =====================================================================


class BountyPostedPayload(TypedDict, total=False):
    bounty_id: str
    poster_id: str
    reward_amount: float
    task_description: str
    min_reputation: float
    deadline_epoch: Optional[int]


class BidPlacedPayload(TypedDict, total=False):
    bid_id: str
    bounty_id: str
    bidder_id: str
    bid_amount: float
    message: str


class EscrowCreatedPayload(TypedDict, total=False):
    escrow_id: str
    bounty_id: str
    poster_id: str
    worker_id: str
    amount: float


class EscrowReleasedPayload(TypedDict, total=False):
    escrow_id: str
    amount: float
    recipient_id: str


class EscrowRefundedPayload(TypedDict, total=False):
    escrow_id: str
    amount: float
    recipient_id: str


class DisputeFiledPayload(TypedDict, total=False):
    escrow_id: str
    filer_id: str
    reason: str


class DisputeResolvedPayload(TypedDict, total=False):
    escrow_id: str
    resolution: str
    winner_id: str


class BidRejectedPayload(TypedDict, total=False):
    bid_id: str
    bounty_id: str
    reason: str


# =====================================================================
# Moltipedia event payloads
# =====================================================================


class PageCreatedPayload(TypedDict, total=False):
    page_id: str
    title: str
    author_id: str
    content_length: int


class PageEditedPayload(TypedDict, total=False):
    page_id: str
    editor_id: str
    edit_type: str
    content_length: int


class ObjectionFiledPayload(TypedDict, total=False):
    page_id: str
    objector_id: str
    reason: str


class PolicyViolationPayload(TypedDict, total=False):
    page_id: str
    agent_id: str
    violation_type: str


class PointsAwardedPayload(TypedDict, total=False):
    agent_id: str
    points: float
    reason: str


class GovernanceTriggerPayload(TypedDict, total=False):
    trigger_type: str
    agent_id: str
    details: Dict[str, Any]


# =====================================================================
# Moltbook event payloads
# =====================================================================


class PostSubmittedPayload(TypedDict, total=False):
    post_id: str
    author_id: str
    submolt: str
    content_length: int


class CommentSubmittedPayload(TypedDict, total=False):
    comment_id: str
    post_id: str
    author_id: str
    content_length: int


class ChallengeIssuedPayload(TypedDict, total=False):
    challenge_id: str
    post_id: str
    agent_id: str
    challenge_type: str


class ChallengeResultPayload(TypedDict, total=False):
    challenge_id: str
    agent_id: str
    passed: bool
    latency_steps: int


class ContentPublishedPayload(TypedDict, total=False):
    post_id: str
    author_id: str
    submolt: str


class RateLimitHitPayload(TypedDict, total=False):
    agent_id: str
    limit_type: str
    current_count: int


class KarmaUpdatedPayload(TypedDict, total=False):
    agent_id: str
    old_karma: float
    new_karma: float
    delta: float


# =====================================================================
# Memory event payloads
# =====================================================================


class MemoryWrittenPayload(TypedDict, total=False):
    entry_id: str
    author_id: str
    tier: int
    content_length: int


class MemoryPromotedPayload(TypedDict, total=False):
    entry_id: str
    agent_id: str
    from_tier: int
    to_tier: int


class MemoryVerifiedPayload(TypedDict, total=False):
    entry_id: str
    verifier_id: str
    verified: bool


class MemoryChallengedPayload(TypedDict, total=False):
    entry_id: str
    challenger_id: str
    reason: str


class MemoryCompactionPayload(TypedDict, total=False):
    agent_id: str
    entries_removed: int
    entries_remaining: int


class MemoryCacheRebuiltPayload(TypedDict, total=False):
    agent_id: str
    cache_size: int


# =====================================================================
# Scholar event payloads
# =====================================================================


class ScholarRetrievalPayload(TypedDict, total=False):
    query_id: str
    agent_id: str
    passages_retrieved: int
    top_score: float


class ScholarSynthesisPayload(TypedDict, total=False):
    query_id: str
    agent_id: str
    citations_count: int
    answer_length: int


class ScholarVerificationPayload(TypedDict, total=False):
    citation_id: str
    verifier_id: str
    entailment_score: float
    valid: bool


# =====================================================================
# Kernel event payloads
# =====================================================================


class KernelSubmittedPayload(TypedDict, total=False):
    submission_id: str
    challenge_id: str
    agent_id: str
    speedup: float
    is_cheat: bool


class KernelVerifiedPayload(TypedDict, total=False):
    submission_id: str
    verifier_id: str
    functional_pass: bool
    ood_pass: bool


class KernelAuditedPayload(TypedDict, total=False):
    submission_id: str
    auditor_id: str
    cheat_detected: bool
    penalty: float


# =====================================================================
# Peer review event payloads
# =====================================================================


class PeerReviewSubmittedPayload(TypedDict, total=False):
    paper_id: str
    review_id: str
    reviewer_id: str
    recommendation: str
    rating: int


class ReviewGatePayload(TypedDict, total=False):
    paper_id: str
    passed: bool
    failed_checks: List[str]


# =====================================================================
# Aggregate type aliases
# =====================================================================

#: Broad alias for ``Event.payload`` — backward compatible with plain dict.
EventPayload = Dict[str, Any]


class ActionMetadata(TypedDict, total=False):
    """Union of all metadata keys used in ``Action.metadata``."""

    # Marketplace
    reward_amount: float
    min_reputation: float
    deadline_epoch: Optional[int]
    bid_amount: float
    bid_id: str

    # Moltipedia
    title: str
    content: str
    violation: str

    # Moltbook
    submolt: str
    answer: float

    # Governance
    ensemble_samples: int


class InteractionMetadata(TypedDict, total=False):
    """Union of all metadata keys used in ``SoftInteraction.metadata``."""

    # Marketplace
    bounty_id: str
    escrow_id: str
    task_id: str

    # Moltipedia
    page_id: str
    edit_type: str
    points: float

    # Moltbook
    post_id: str
    challenge_id: str
    submolt: str

    # Memory
    entry_id: str
    tier: int

    # Scholar
    query_id: str
    citation_id: str

    # Kernel
    submission_id: str
    speedup: float
