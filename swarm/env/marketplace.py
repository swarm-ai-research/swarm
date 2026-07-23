"""Marketplace with bounties, bids, escrow, and dispute resolution."""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, model_validator


class BountyStatus(Enum):
    """Status of a bounty."""

    OPEN = "open"
    AWARDED = "awarded"
    COMPLETED = "completed"
    FAILED = "failed"
    DISPUTED = "disputed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class BidStatus(Enum):
    """Status of a bid."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


class EscrowStatus(Enum):
    """Status of an escrow."""

    HELD = "held"
    RELEASED = "released"
    REFUNDED = "refunded"
    PARTIAL = "partial"


class DisputeStatus(Enum):
    """Status of a dispute."""

    OPEN = "open"
    UNDER_REVIEW = "under_review"
    RESOLVED_RELEASE = "resolved_release"
    RESOLVED_REFUND = "resolved_refund"
    RESOLVED_SPLIT = "resolved_split"


@dataclass
class Bounty:
    """A marketplace bounty posted by an agent."""

    bounty_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    poster_id: str = ""
    task_id: str = ""
    reward_amount: float = 0.0
    min_reputation: float = 0.0
    deadline_epoch: Optional[int] = None
    status: BountyStatus = BountyStatus.OPEN
    awarded_to: Optional[str] = None
    escrow_id: Optional[str] = None
    created_epoch: int = 0

    def to_dict(self) -> Dict:
        """Serialize bounty."""
        return {
            "bounty_id": self.bounty_id,
            "poster_id": self.poster_id,
            "task_id": self.task_id,
            "reward_amount": self.reward_amount,
            "min_reputation": self.min_reputation,
            "deadline_epoch": self.deadline_epoch,
            "status": self.status.value,
            "awarded_to": self.awarded_to,
            "escrow_id": self.escrow_id,
            "created_epoch": self.created_epoch,
        }


@dataclass
class Bid:
    """A bid on a bounty."""

    bid_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    bounty_id: str = ""
    bidder_id: str = ""
    bid_amount: float = 0.0
    status: BidStatus = BidStatus.PENDING
    message: str = ""

    def to_dict(self) -> Dict:
        """Serialize bid."""
        return {
            "bid_id": self.bid_id,
            "bounty_id": self.bounty_id,
            "bidder_id": self.bidder_id,
            "bid_amount": self.bid_amount,
            "status": self.status.value,
            "message": self.message,
        }


@dataclass
class Escrow:
    """Escrow holding funds for a bounty."""

    escrow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    bounty_id: str = ""
    poster_id: str = ""
    worker_id: str = ""
    amount: float = 0.0
    status: EscrowStatus = EscrowStatus.HELD
    released_amount: float = 0.0
    refunded_amount: float = 0.0

    def to_dict(self) -> Dict:
        """Serialize escrow."""
        return {
            "escrow_id": self.escrow_id,
            "bounty_id": self.bounty_id,
            "poster_id": self.poster_id,
            "worker_id": self.worker_id,
            "amount": self.amount,
            "status": self.status.value,
            "released_amount": self.released_amount,
            "refunded_amount": self.refunded_amount,
        }


@dataclass
class Dispute:
    """A dispute on an escrow."""

    dispute_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    escrow_id: str = ""
    filed_by: str = ""
    reason: str = ""
    status: DisputeStatus = DisputeStatus.OPEN
    worker_share: float = 0.5
    filed_epoch: int = 0

    def to_dict(self) -> Dict:
        """Serialize dispute."""
        return {
            "dispute_id": self.dispute_id,
            "escrow_id": self.escrow_id,
            "filed_by": self.filed_by,
            "reason": self.reason,
            "status": self.status.value,
            "worker_share": self.worker_share,
            "filed_epoch": self.filed_epoch,
        }


class MarketplaceConfig(BaseModel):
    """Configuration for the marketplace."""

    enabled: bool = True
    escrow_fee_rate: float = 0.02
    min_bounty_amount: float = 1.0
    max_bids_per_bounty: int = 10
    bid_deadline_epochs: int = 3
    dispute_resolution_epochs: int = 2
    auto_expire_bounties: bool = True
    dispute_default_split: float = 0.5

    @model_validator(mode="after")
    def _run_validation(self) -> "MarketplaceConfig":
        self._check_values()
        return self

    def _check_values(self) -> None:
        """Validate configuration."""
        if self.escrow_fee_rate < 0 or self.escrow_fee_rate > 1:
            raise ValueError("escrow_fee_rate must be in [0, 1]")
        if self.min_bounty_amount < 0:
            raise ValueError("min_bounty_amount must be non-negative")
        if self.max_bids_per_bounty < 1:
            raise ValueError("max_bids_per_bounty must be >= 1")
        if self.dispute_default_split < 0 or self.dispute_default_split > 1:
            raise ValueError("dispute_default_split must be in [0, 1]")


class Marketplace:
    """
    Marketplace engine managing bounties, bids, escrow, and disputes.

    Lifecycle:
    1. Agent posts bounty (funds deducted)
    2. Other agents place bids
    3. Poster accepts a bid -> escrow created
    4. Worker completes task -> escrow settled
    5. Disputes can be filed on held escrows
    """

    def __init__(self, config: Optional[MarketplaceConfig] = None) -> None:
        """Initialize marketplace."""
        self.config = MarketplaceConfig() if config is None else config

        self._bounties: Dict[str, Bounty] = {}
        self._bids: Dict[str, Bid] = {}
        self._escrows: Dict[str, Escrow] = {}
        self._disputes: Dict[str, Dispute] = {}

        # Indexes
        self._bounties_by_task: Dict[str, str] = {}  # task_id -> bounty_id
        self._bids_by_bounty: Dict[str, List[str]] = {}  # bounty_id -> [bid_id]
        self._bounties_by_agent: Dict[str, List[str]] = {}  # agent_id -> [bounty_id]
        self._bids_by_agent: Dict[str, List[str]] = {}  # agent_id -> [bid_id]
        self._escrows_by_agent: Dict[str, List[str]] = {}  # agent_id -> [escrow_id]

    def post_bounty(
        self,
        poster_id: str,
        task_id: str,
        reward_amount: float,
        min_reputation: float = 0.0,
        deadline_epoch: Optional[int] = None,
        current_epoch: int = 0,
    ) -> Bounty:
        """
        Post a new bounty.

        Args:
            poster_id: Agent posting the bounty
            task_id: Associated task ID
            reward_amount: Funds offered
            min_reputation: Minimum bidder reputation
            deadline_epoch: Bounty expiry epoch
            current_epoch: Current simulation epoch

        Returns:
            The created Bounty

        Raises:
            ValueError: If reward is below minimum
        """
        if reward_amount < self.config.min_bounty_amount:
            raise ValueError(
                f"Reward {reward_amount} below minimum {self.config.min_bounty_amount}"
            )

        bounty = Bounty(
            poster_id=poster_id,
            task_id=task_id,
            reward_amount=reward_amount,
            min_reputation=min_reputation,
            deadline_epoch=deadline_epoch,
            created_epoch=current_epoch,
        )

        self._bounties[bounty.bounty_id] = bounty
        self._bounties_by_task[task_id] = bounty.bounty_id
        self._bids_by_bounty[bounty.bounty_id] = []

        if poster_id not in self._bounties_by_agent:
            self._bounties_by_agent[poster_id] = []
        self._bounties_by_agent[poster_id].append(bounty.bounty_id)

        return bounty

    def place_bid(
        self,
        bounty_id: str,
        bidder_id: str,
        bid_amount: float,
        message: str = "",
    ) -> Optional[Bid]:
        """
        Place a bid on a bounty.

        Returns None if bid is invalid.
        """
        bounty = self._bounties.get(bounty_id)
        if not bounty:
            return None
        if bounty.status != BountyStatus.OPEN:
            return None
        if bid_amount > bounty.reward_amount:
            return None
        if bid_amount <= 0:
            return None
        if bidder_id == bounty.poster_id:
            return None

        # Check max bids
        existing_bids = self._bids_by_bounty.get(bounty_id, [])
        active_bids = [
            bid_id
            for bid_id in existing_bids
            if self._bids[bid_id].status == BidStatus.PENDING
        ]
        if len(active_bids) >= self.config.max_bids_per_bounty:
            return None

        # Check duplicate bid from same agent
        for bid_id in existing_bids:
            bid = self._bids[bid_id]
            if bid.bidder_id == bidder_id and bid.status == BidStatus.PENDING:
                return None

        bid = Bid(
            bounty_id=bounty_id,
            bidder_id=bidder_id,
            bid_amount=bid_amount,
            message=message,
        )

        self._bids[bid.bid_id] = bid
        self._bids_by_bounty[bounty_id].append(bid.bid_id)

        if bidder_id not in self._bids_by_agent:
            self._bids_by_agent[bidder_id] = []
        self._bids_by_agent[bidder_id].append(bid.bid_id)

        return bid

    def accept_bid(
        self, bounty_id: str, bid_id: str, poster_id: str
    ) -> Optional[Escrow]:
        """
        Accept a bid and create escrow.

        Only the bounty poster can accept. Escrow amount = bid amount.
        Difference between reward and bid goes back to poster implicitly
        (they only had reward_amount deducted at posting time).

        Returns None if invalid.
        """
        bounty = self._bounties.get(bounty_id)
        if not bounty:
            return None
        if bounty.poster_id != poster_id:
            return None
        if bounty.status != BountyStatus.OPEN:
            return None

        bid = self._bids.get(bid_id)
        if not bid:
            return None
        if bid.bounty_id != bounty_id:
            return None
        if bid.status != BidStatus.PENDING:
            return None

        # Create escrow
        escrow = Escrow(
            bounty_id=bounty_id,
            poster_id=poster_id,
            worker_id=bid.bidder_id,
            amount=bid.bid_amount,
        )

        self._escrows[escrow.escrow_id] = escrow

        # Update bounty
        bounty.status = BountyStatus.AWARDED
        bounty.awarded_to = bid.bidder_id
        bounty.escrow_id = escrow.escrow_id

        # Update bid
        bid.status = BidStatus.ACCEPTED

        # Reject all other pending bids
        for other_bid_id in self._bids_by_bounty.get(bounty_id, []):
            other_bid = self._bids[other_bid_id]
            if other_bid.bid_id != bid_id and other_bid.status == BidStatus.PENDING:
                other_bid.status = BidStatus.REJECTED

        # Index escrow
        for agent_id in (poster_id, bid.bidder_id):
            if agent_id not in self._escrows_by_agent:
                self._escrows_by_agent[agent_id] = []
            self._escrows_by_agent[agent_id].append(escrow.escrow_id)

        return escrow

    def reject_bid(self, bid_id: str, poster_id: str) -> bool:
        """Reject a bid. Only the bounty poster can reject."""
        bid = self._bids.get(bid_id)
        if not bid:
            return False
        if bid.status != BidStatus.PENDING:
            return False

        bounty = self._bounties.get(bid.bounty_id)
        if not bounty:
            return False
        if bounty.poster_id != poster_id:
            return False

        bid.status = BidStatus.REJECTED
        return True

    def withdraw_bid(self, bid_id: str, bidder_id: str) -> bool:
        """Withdraw a bid. Only the bidder can withdraw."""
        bid = self._bids.get(bid_id)
        if not bid:
            return False
        if bid.bidder_id != bidder_id:
            return False
        if bid.status != BidStatus.PENDING:
            return False

        bid.status = BidStatus.WITHDRAWN
        return True

    def settle_escrow(
        self, escrow_id: str, success: bool, quality_score: float = 1.0
    ) -> Dict:
        """
        Settle an escrow after task completion/failure.

        Args:
            escrow_id: Escrow to settle
            success: Whether task was completed successfully
            quality_score: Quality of work (0-1), affects partial release

        Returns:
            Settlement details dict, or empty dict if invalid
        """
        escrow = self._escrows.get(escrow_id)
        if not escrow:
            return {}
        if escrow.status != EscrowStatus.HELD:
            return {}

        bounty = self._bounties.get(escrow.bounty_id)

        fee = escrow.amount * self.config.escrow_fee_rate

        if success:
            released = escrow.amount - fee
            escrow.released_amount = released
            escrow.refunded_amount = 0.0
            escrow.status = EscrowStatus.RELEASED

            if bounty:
                bounty.status = BountyStatus.COMPLETED

            # If bid was less than reward, poster gets the difference back
            refund_to_poster = 0.0
            if bounty and bounty.reward_amount > escrow.amount:
                refund_to_poster = bounty.reward_amount - escrow.amount

            return {
                "escrow_id": escrow_id,
                "success": True,
                "released_to_worker": released,
                "fee": fee,
                "refund_to_poster": refund_to_poster,
                "worker_id": escrow.worker_id,
                "poster_id": escrow.poster_id,
                "quality_score": quality_score,
            }
        else:
            refunded = escrow.amount - fee
            escrow.refunded_amount = refunded
            escrow.released_amount = 0.0
            escrow.status = EscrowStatus.REFUNDED

            if bounty:
                bounty.status = BountyStatus.FAILED

            return {
                "escrow_id": escrow_id,
                "success": False,
                "refunded_to_poster": refunded,
                "fee": fee,
                "worker_id": escrow.worker_id,
                "poster_id": escrow.poster_id,
                "quality_score": quality_score,
            }

    def file_dispute(
        self,
        escrow_id: str,
        filed_by: str,
        reason: str,
        current_epoch: int = 0,
    ) -> Optional[Dispute]:
        """
        File a dispute on a held escrow.

        Only poster or worker can file. Cannot dispute released/refunded escrows.
        """
        escrow = self._escrows.get(escrow_id)
        if not escrow:
            return None
        if escrow.status != EscrowStatus.HELD:
            return None
        if filed_by not in (escrow.poster_id, escrow.worker_id):
            return None

        # Check if dispute already exists for this escrow
        for dispute in self._disputes.values():
            if dispute.escrow_id == escrow_id and dispute.status in (
                DisputeStatus.OPEN,
                DisputeStatus.UNDER_REVIEW,
            ):
                return None

        dispute = Dispute(
            escrow_id=escrow_id,
            filed_by=filed_by,
            reason=reason,
            filed_epoch=current_epoch,
        )

        self._disputes[dispute.dispute_id] = dispute

        # Update bounty status
        bounty = self._bounties.get(escrow.bounty_id)
        if bounty:
            bounty.status = BountyStatus.DISPUTED

        return dispute

    def resolve_dispute(self, dispute_id: str, worker_share: float) -> Dict:
        """
        Resolve a dispute by splitting escrow funds.

        Args:
            dispute_id: Dispute to resolve
            worker_share: Fraction [0, 1] of escrow to give to worker

        Returns:
            Resolution details dict
        """
        dispute = self._disputes.get(dispute_id)
        if not dispute:
            return {}
        if dispute.status not in (DisputeStatus.OPEN, DisputeStatus.UNDER_REVIEW):
            return {}

        worker_share = max(0.0, min(1.0, worker_share))

        escrow = self._escrows.get(dispute.escrow_id)
        if not escrow:
            return {}

        fee = escrow.amount * self.config.escrow_fee_rate
        distributable = escrow.amount - fee

        worker_amount = distributable * worker_share
        poster_amount = distributable * (1.0 - worker_share)

        escrow.released_amount = worker_amount
        escrow.refunded_amount = poster_amount

        if worker_share >= 0.99:
            escrow.status = EscrowStatus.RELEASED
            dispute.status = DisputeStatus.RESOLVED_RELEASE
        elif worker_share <= 0.01:
            escrow.status = EscrowStatus.REFUNDED
            dispute.status = DisputeStatus.RESOLVED_REFUND
        else:
            escrow.status = EscrowStatus.PARTIAL
            dispute.status = DisputeStatus.RESOLVED_SPLIT

        dispute.worker_share = worker_share

        bounty = self._bounties.get(escrow.bounty_id)
        if bounty:
            bounty.status = (
                BountyStatus.COMPLETED if worker_share > 0.5 else BountyStatus.FAILED
            )

        return {
            "dispute_id": dispute_id,
            "escrow_id": escrow.escrow_id,
            "worker_share": worker_share,
            "worker_amount": worker_amount,
            "poster_amount": poster_amount,
            "fee": fee,
            "worker_id": escrow.worker_id,
            "poster_id": escrow.poster_id,
        }

    def get_open_bounties(
        self,
        current_epoch: int = 0,
        min_reputation: Optional[float] = None,
    ) -> List[Bounty]:
        """Get open bounties, optionally filtered by reputation requirement."""
        results = []
        for bounty in self._bounties.values():
            if bounty.status != BountyStatus.OPEN:
                continue
            if bounty.deadline_epoch and current_epoch >= bounty.deadline_epoch:
                continue
            if min_reputation is not None and bounty.min_reputation > min_reputation:
                continue
            results.append(bounty)
        return results

    def get_bids_for_bounty(self, bounty_id: str) -> List[Bid]:
        """Get all bids for a bounty."""
        bid_ids = self._bids_by_bounty.get(bounty_id, [])
        return [self._bids[bid_id] for bid_id in bid_ids if bid_id in self._bids]

    def get_bounty_for_task(self, task_id: str) -> Optional[Bounty]:
        """Get the bounty associated with a task."""
        bounty_id = self._bounties_by_task.get(task_id)
        if bounty_id:
            return self._bounties.get(bounty_id)
        return None

    def get_agent_bounties(self, agent_id: str) -> List[Bounty]:
        """Get all bounties posted by an agent."""
        bounty_ids = self._bounties_by_agent.get(agent_id, [])
        return [
            self._bounties[bounty_id]
            for bounty_id in bounty_ids
            if bounty_id in self._bounties
        ]

    def get_agent_bids(self, agent_id: str) -> List[Bid]:
        """Get all bids placed by an agent."""
        bid_ids = self._bids_by_agent.get(agent_id, [])
        return [self._bids[bid_id] for bid_id in bid_ids if bid_id in self._bids]

    def get_agent_escrows(self, agent_id: str) -> List[Escrow]:
        """Get all escrows involving an agent."""
        escrow_ids = self._escrows_by_agent.get(agent_id, [])
        return [self._escrows[eid] for eid in escrow_ids if eid in self._escrows]

    def expire_bounties(self, current_epoch: int) -> List[str]:
        """
        Expire bounties past their deadline. Returns list of expired bounty IDs.

        Expired bounties have their funds returned to posters.
        """
        if not self.config.auto_expire_bounties:
            return []

        expired = []
        for bounty in self._bounties.values():
            if bounty.status != BountyStatus.OPEN:
                continue
            if bounty.deadline_epoch and current_epoch >= bounty.deadline_epoch:
                bounty.status = BountyStatus.EXPIRED
                expired.append(bounty.bounty_id)

                # Reject all pending bids
                for bid_id in self._bids_by_bounty.get(bounty.bounty_id, []):
                    bid = self._bids.get(bid_id)
                    if bid and bid.status == BidStatus.PENDING:
                        bid.status = BidStatus.REJECTED

        return expired

    def auto_resolve_disputes(self, current_epoch: int) -> List[str]:
        """
        Auto-resolve disputes that have timed out.

        Uses default split from config. Returns list of resolved dispute IDs.
        """
        resolved = []
        for dispute in self._disputes.values():
            if dispute.status not in (DisputeStatus.OPEN, DisputeStatus.UNDER_REVIEW):
                continue
            if (
                current_epoch - dispute.filed_epoch
                >= self.config.dispute_resolution_epochs
            ):
                self.resolve_dispute(
                    dispute.dispute_id, self.config.dispute_default_split
                )
                resolved.append(dispute.dispute_id)
        return resolved

    def get_bounty(self, bounty_id: str) -> Optional[Bounty]:
        """Get a bounty by ID."""
        return self._bounties.get(bounty_id)

    def get_escrow(self, escrow_id: str) -> Optional[Escrow]:
        """Get an escrow by ID."""
        return self._escrows.get(escrow_id)

    def get_dispute(self, dispute_id: str) -> Optional[Dispute]:
        """Get a dispute by ID."""
        return self._disputes.get(dispute_id)

    def get_stats(self) -> Dict:
        """Get marketplace statistics."""
        bounty_statuses: Dict[str, int] = {}
        for b in self._bounties.values():
            s = b.status.value
            bounty_statuses[s] = bounty_statuses.get(s, 0) + 1

        total_reward = sum(b.reward_amount for b in self._bounties.values())
        total_escrowed = sum(
            e.amount for e in self._escrows.values() if e.status == EscrowStatus.HELD
        )
        total_released = sum(e.released_amount for e in self._escrows.values())
        total_refunded = sum(e.refunded_amount for e in self._escrows.values())

        return {
            "total_bounties": len(self._bounties),
            "total_bids": len(self._bids),
            "total_escrows": len(self._escrows),
            "total_disputes": len(self._disputes),
            "bounty_statuses": bounty_statuses,
            "total_reward_posted": total_reward,
            "total_escrowed": total_escrowed,
            "total_released": total_released,
            "total_refunded": total_refunded,
        }
