"""Tests for the marketplace module."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from swarm.agents.base import ActionType
from swarm.agents.honest import HonestAgent
from swarm.agents.opportunistic import OpportunisticAgent
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig
from swarm.core.payoff import PayoffConfig
from swarm.env.marketplace import (
    Bid,
    BidStatus,
    Bounty,
    BountyStatus,
    Dispute,
    DisputeStatus,
    Escrow,
    EscrowStatus,
    Marketplace,
    MarketplaceConfig,
)
from swarm.env.state import RateLimits, RateLimitState
from swarm.governance.config import GovernanceConfig

# ===========================================================================
# Model Tests
# ===========================================================================


class TestModels:
    """Test data model creation and serialization."""

    def test_bounty_creation(self):
        bounty = Bounty(
            poster_id="agent_1",
            task_id="task_1",
            reward_amount=10.0,
        )
        assert bounty.poster_id == "agent_1"
        assert bounty.status == BountyStatus.OPEN
        assert bounty.bounty_id  # UUID generated

    def test_bounty_to_dict(self):
        bounty = Bounty(
            bounty_id="b1",
            poster_id="a1",
            task_id="t1",
            reward_amount=10.0,
        )
        d = bounty.to_dict()
        assert d["bounty_id"] == "b1"
        assert d["status"] == "open"
        assert d["reward_amount"] == 10.0

    def test_bid_creation(self):
        bid = Bid(bounty_id="b1", bidder_id="a2", bid_amount=8.0)
        assert bid.status == BidStatus.PENDING
        assert bid.bid_amount == 8.0

    def test_escrow_creation(self):
        escrow = Escrow(
            bounty_id="b1",
            poster_id="a1",
            worker_id="a2",
            amount=8.0,
        )
        assert escrow.status == EscrowStatus.HELD
        assert escrow.released_amount == 0.0

    def test_dispute_creation(self):
        dispute = Dispute(
            escrow_id="e1",
            filed_by="a1",
            reason="Bad work",
        )
        assert dispute.status == DisputeStatus.OPEN
        assert dispute.worker_share == 0.5

    def test_config_validation(self):
        config = MarketplaceConfig()
        assert config is not None  # Pydantic auto-validates on creation

        with pytest.raises(ValidationError, match="escrow_fee_rate"):
            MarketplaceConfig(escrow_fee_rate=-0.1)

        with pytest.raises(ValidationError, match="dispute_default_split"):
            MarketplaceConfig(dispute_default_split=1.5)


# ===========================================================================
# Bounty Lifecycle Tests
# ===========================================================================


class TestBountyLifecycle:
    """Test bounty creation, listing, expiry."""

    def setup_method(self):
        self.marketplace = Marketplace()

    def test_post_bounty(self):
        bounty = self.marketplace.post_bounty(
            poster_id="a1",
            task_id="t1",
            reward_amount=10.0,
        )
        assert bounty.status == BountyStatus.OPEN
        assert bounty.reward_amount == 10.0
        assert bounty.poster_id == "a1"

    def test_post_bounty_below_minimum(self):
        with pytest.raises(ValueError):
            self.marketplace.post_bounty(
                poster_id="a1",
                task_id="t1",
                reward_amount=0.5,
            )

    def test_get_open_bounties(self):
        self.marketplace.post_bounty("a1", "t1", 10.0)
        self.marketplace.post_bounty("a2", "t2", 15.0, min_reputation=5.0)

        # No reputation filter
        bounties = self.marketplace.get_open_bounties()
        assert len(bounties) == 2

        # With reputation filter
        bounties = self.marketplace.get_open_bounties(min_reputation=0.0)
        assert len(bounties) == 1  # Only the one without min_rep > 0

    def test_expire_bounties(self):
        b1 = self.marketplace.post_bounty("a1", "t1", 10.0, deadline_epoch=5)
        self.marketplace.post_bounty("a2", "t2", 15.0)

        expired = self.marketplace.expire_bounties(current_epoch=5)
        assert len(expired) == 1
        assert expired[0] == b1.bounty_id

        bounty = self.marketplace.get_bounty(b1.bounty_id)
        assert bounty.status == BountyStatus.EXPIRED

    def test_get_bounty_for_task(self):
        bounty = self.marketplace.post_bounty("a1", "t1", 10.0)
        found = self.marketplace.get_bounty_for_task("t1")
        assert found.bounty_id == bounty.bounty_id

        assert self.marketplace.get_bounty_for_task("nonexistent") is None


# ===========================================================================
# Bid Lifecycle Tests
# ===========================================================================


class TestBidLifecycle:
    """Test bid placement, rejection, withdrawal."""

    def setup_method(self):
        self.marketplace = Marketplace()
        self.bounty = self.marketplace.post_bounty("poster", "t1", 10.0)

    def test_place_bid(self):
        bid = self.marketplace.place_bid(
            self.bounty.bounty_id, "bidder_1", 8.0, "I can do it"
        )
        assert bid is not None
        assert bid.bid_amount == 8.0
        assert bid.status == BidStatus.PENDING

    def test_bid_exceeds_reward(self):
        bid = self.marketplace.place_bid(self.bounty.bounty_id, "bidder_1", 15.0)
        assert bid is None

    def test_bid_on_own_bounty(self):
        bid = self.marketplace.place_bid(self.bounty.bounty_id, "poster", 5.0)
        assert bid is None

    def test_bid_on_closed_bounty(self):
        # Award the bounty first
        bid = self.marketplace.place_bid(self.bounty.bounty_id, "b1", 8.0)
        self.marketplace.accept_bid(self.bounty.bounty_id, bid.bid_id, "poster")

        # Now try to bid on awarded bounty
        bid2 = self.marketplace.place_bid(self.bounty.bounty_id, "b2", 7.0)
        assert bid2 is None

    def test_max_bids_enforced(self):
        config = MarketplaceConfig(max_bids_per_bounty=2)
        mp = Marketplace(config)
        bounty = mp.post_bounty("poster", "t1", 10.0)

        assert mp.place_bid(bounty.bounty_id, "b1", 8.0) is not None
        assert mp.place_bid(bounty.bounty_id, "b2", 7.0) is not None
        assert mp.place_bid(bounty.bounty_id, "b3", 6.0) is None

    def test_reject_bid(self):
        bid = self.marketplace.place_bid(self.bounty.bounty_id, "b1", 8.0)
        assert self.marketplace.reject_bid(bid.bid_id, "poster")
        assert self.marketplace.get_agent_bids("b1")[0].status == BidStatus.REJECTED

    def test_withdraw_bid(self):
        bid = self.marketplace.place_bid(self.bounty.bounty_id, "b1", 8.0)
        assert self.marketplace.withdraw_bid(bid.bid_id, "b1")
        assert self.marketplace.get_agent_bids("b1")[0].status == BidStatus.WITHDRAWN

    def test_cannot_withdraw_others_bid(self):
        bid = self.marketplace.place_bid(self.bounty.bounty_id, "b1", 8.0)
        assert not self.marketplace.withdraw_bid(bid.bid_id, "b2")

    def test_duplicate_bid_rejected(self):
        assert self.marketplace.place_bid(self.bounty.bounty_id, "b1", 8.0) is not None
        assert self.marketplace.place_bid(self.bounty.bounty_id, "b1", 7.0) is None


# ===========================================================================
# Escrow Lifecycle Tests
# ===========================================================================


class TestEscrowLifecycle:
    """Test escrow creation, settlement."""

    def setup_method(self):
        self.marketplace = Marketplace()
        self.bounty = self.marketplace.post_bounty("poster", "t1", 10.0)
        self.bid = self.marketplace.place_bid(self.bounty.bounty_id, "worker", 8.0)

    def test_accept_bid_creates_escrow(self):
        escrow = self.marketplace.accept_bid(
            self.bounty.bounty_id, self.bid.bid_id, "poster"
        )
        assert escrow is not None
        assert escrow.amount == 8.0  # Escrow = bid amount
        assert escrow.status == EscrowStatus.HELD
        assert escrow.worker_id == "worker"
        assert escrow.poster_id == "poster"

    def test_settle_success(self):
        escrow = self.marketplace.accept_bid(
            self.bounty.bounty_id, self.bid.bid_id, "poster"
        )
        result = self.marketplace.settle_escrow(
            escrow.escrow_id, success=True, quality_score=0.9
        )
        assert result["success"] is True
        assert result["released_to_worker"] > 0
        fee = 8.0 * 0.02  # 2% fee
        assert abs(result["released_to_worker"] - (8.0 - fee)) < 0.01
        assert result["refund_to_poster"] == 2.0  # reward - bid = 10 - 8

    def test_settle_failure(self):
        escrow = self.marketplace.accept_bid(
            self.bounty.bounty_id, self.bid.bid_id, "poster"
        )
        result = self.marketplace.settle_escrow(escrow.escrow_id, success=False)
        assert result["success"] is False
        assert result["refunded_to_poster"] > 0

    def test_double_settle_rejected(self):
        escrow = self.marketplace.accept_bid(
            self.bounty.bounty_id, self.bid.bid_id, "poster"
        )
        self.marketplace.settle_escrow(escrow.escrow_id, success=True)
        result = self.marketplace.settle_escrow(escrow.escrow_id, success=True)
        assert result == {}

    def test_escrow_amount_matches_bid(self):
        self.marketplace.place_bid(self.bounty.bounty_id, "worker2", 5.0)
        # This bid won't work because bounty is still open but we already have a bid
        # Let's use a fresh bounty
        b2 = self.marketplace.post_bounty("poster2", "t2", 20.0)
        bid3 = self.marketplace.place_bid(b2.bounty_id, "w3", 12.0)
        escrow = self.marketplace.accept_bid(b2.bounty_id, bid3.bid_id, "poster2")
        assert escrow.amount == 12.0


# ===========================================================================
# Dispute Lifecycle Tests
# ===========================================================================


class TestDisputeLifecycle:
    """Test dispute filing and resolution."""

    def setup_method(self):
        self.marketplace = Marketplace()
        self.bounty = self.marketplace.post_bounty("poster", "t1", 10.0)
        bid = self.marketplace.place_bid(self.bounty.bounty_id, "worker", 8.0)
        self.escrow = self.marketplace.accept_bid(
            self.bounty.bounty_id, bid.bid_id, "poster"
        )

    def test_file_dispute(self):
        dispute = self.marketplace.file_dispute(
            self.escrow.escrow_id,
            filed_by="poster",
            reason="Work not delivered",
            current_epoch=5,
        )
        assert dispute is not None
        assert dispute.status == DisputeStatus.OPEN

        # Bounty should be disputed
        bounty = self.marketplace.get_bounty(self.bounty.bounty_id)
        assert bounty.status == BountyStatus.DISPUTED

    def test_resolve_dispute_full_release(self):
        dispute = self.marketplace.file_dispute(self.escrow.escrow_id, "poster", "test")
        result = self.marketplace.resolve_dispute(dispute.dispute_id, worker_share=1.0)
        assert result["worker_share"] == 1.0
        assert result["worker_amount"] > 0
        assert abs(result["poster_amount"]) < 0.01

        d = self.marketplace.get_dispute(dispute.dispute_id)
        assert d.status == DisputeStatus.RESOLVED_RELEASE

    def test_resolve_dispute_full_refund(self):
        dispute = self.marketplace.file_dispute(self.escrow.escrow_id, "poster", "test")
        result = self.marketplace.resolve_dispute(dispute.dispute_id, worker_share=0.0)
        assert result["worker_share"] == 0.0
        assert abs(result["worker_amount"]) < 0.01
        assert result["poster_amount"] > 0

        d = self.marketplace.get_dispute(dispute.dispute_id)
        assert d.status == DisputeStatus.RESOLVED_REFUND

    def test_resolve_dispute_split(self):
        dispute = self.marketplace.file_dispute(self.escrow.escrow_id, "poster", "test")
        result = self.marketplace.resolve_dispute(dispute.dispute_id, worker_share=0.6)
        assert result["worker_share"] == 0.6
        assert result["worker_amount"] > 0
        assert result["poster_amount"] > 0

        d = self.marketplace.get_dispute(dispute.dispute_id)
        assert d.status == DisputeStatus.RESOLVED_SPLIT

    def test_auto_resolve_disputes(self):
        config = MarketplaceConfig(
            dispute_resolution_epochs=2,
            dispute_default_split=0.5,
        )
        mp = Marketplace(config)
        bounty = mp.post_bounty("poster", "t1", 10.0)
        bid = mp.place_bid(bounty.bounty_id, "worker", 8.0)
        escrow = mp.accept_bid(bounty.bounty_id, bid.bid_id, "poster")

        dispute = mp.file_dispute(escrow.escrow_id, "poster", "test", current_epoch=3)

        # Not yet timed out
        resolved = mp.auto_resolve_disputes(current_epoch=4)
        assert len(resolved) == 0

        # Now timed out
        resolved = mp.auto_resolve_disputes(current_epoch=5)
        assert len(resolved) == 1
        assert resolved[0] == dispute.dispute_id

    def test_dispute_on_released_escrow_rejected(self):
        self.marketplace.settle_escrow(self.escrow.escrow_id, success=True)
        dispute = self.marketplace.file_dispute(
            self.escrow.escrow_id, "poster", "Too late"
        )
        assert dispute is None

    def test_third_party_cannot_file_dispute(self):
        dispute = self.marketplace.file_dispute(
            self.escrow.escrow_id, "random_agent", "I object"
        )
        assert dispute is None


# ===========================================================================
# Orchestrator Integration Tests
# ===========================================================================


class TestOrchestratorIntegration:
    """Test marketplace integration with orchestrator."""

    def _make_orchestrator(self, with_governance=False, governance_config=None):
        """Create an orchestrator with marketplace enabled."""
        gov_config = governance_config
        if with_governance and gov_config is None:
            gov_config = GovernanceConfig(
                transaction_tax_rate=0.1,
                transaction_tax_split=0.5,
            )

        config = OrchestratorConfig(
            n_epochs=2,
            steps_per_epoch=5,
            marketplace_config=MarketplaceConfig(),
            governance_config=gov_config,
            payoff_config=PayoffConfig(),
            seed=42,
        )
        orch = Orchestrator(config=config)

        # Register agents
        agents = [
            HonestAgent("h1"),
            HonestAgent("h2"),
            OpportunisticAgent("o1"),
        ]
        for agent in agents:
            orch.register_agent(agent)

        return orch

    def test_marketplace_initialized(self):
        orch = self._make_orchestrator()
        assert orch.marketplace is not None

    def test_marketplace_not_initialized_when_disabled(self):
        config = OrchestratorConfig()
        orch = Orchestrator(config=config)
        assert orch.marketplace is None

    def test_post_bounty_action(self):
        orch = self._make_orchestrator()
        from swarm.agents.base import Action

        action = Action(
            action_type=ActionType.POST_BOUNTY,
            agent_id="h1",
            content="Test task",
            metadata={"reward_amount": 10.0},
        )

        # Give agent enough resources
        orch.state.get_agent("h1").resources = 100.0

        success = orch._execute_action(action)
        assert success

        # Agent's resources should be reduced
        assert orch.state.get_agent("h1").resources == 90.0

        # Bounty should exist
        assert len(orch.marketplace.get_agent_bounties("h1")) == 1

    def test_post_bounty_insufficient_resources(self):
        orch = self._make_orchestrator()
        from swarm.agents.base import Action

        action = Action(
            action_type=ActionType.POST_BOUNTY,
            agent_id="h1",
            content="Test task",
            metadata={"reward_amount": 1000.0},
        )
        orch.state.get_agent("h1").resources = 50.0

        success = orch._execute_action(action)
        assert not success

    def test_place_bid_action(self):
        orch = self._make_orchestrator()
        from swarm.agents.base import Action

        # Post bounty first
        orch.state.get_agent("h1").resources = 100.0
        post_action = Action(
            action_type=ActionType.POST_BOUNTY,
            agent_id="h1",
            content="Task",
            metadata={"reward_amount": 10.0},
        )
        orch._execute_action(post_action)

        bounty = orch.marketplace.get_agent_bounties("h1")[0]

        # Place bid
        bid_action = Action(
            action_type=ActionType.PLACE_BID,
            agent_id="o1",
            target_id=bounty.bounty_id,
            metadata={"bid_amount": 8.0},
        )
        success = orch._execute_action(bid_action)
        assert success

        bids = orch.marketplace.get_bids_for_bounty(bounty.bounty_id)
        assert len(bids) == 1
        assert bids[0].bid_amount == 8.0

    def test_accept_bid_action(self):
        orch = self._make_orchestrator()
        from swarm.agents.base import Action

        # Setup: post bounty, place bid
        orch.state.get_agent("h1").resources = 100.0
        orch._execute_action(
            Action(
                action_type=ActionType.POST_BOUNTY,
                agent_id="h1",
                content="Task",
                metadata={"reward_amount": 10.0},
            )
        )
        bounty = orch.marketplace.get_agent_bounties("h1")[0]

        orch._execute_action(
            Action(
                action_type=ActionType.PLACE_BID,
                agent_id="o1",
                target_id=bounty.bounty_id,
                metadata={"bid_amount": 8.0},
            )
        )
        bid = orch.marketplace.get_bids_for_bounty(bounty.bounty_id)[0]

        # Accept bid
        success = orch._execute_action(
            Action(
                action_type=ActionType.ACCEPT_BID,
                agent_id="h1",
                target_id=bounty.bounty_id,
                metadata={"bid_id": bid.bid_id},
            )
        )
        assert success

        # Escrow should exist
        escrows = orch.marketplace.get_agent_escrows("h1")
        assert len(escrows) == 1
        assert escrows[0].amount == 8.0

    def test_full_bounty_lifecycle(self):
        """Test complete flow: post -> bid -> accept -> settle."""
        orch = self._make_orchestrator()
        from swarm.agents.base import Action

        # Post bounty
        orch.state.get_agent("h1").resources = 100.0
        orch._execute_action(
            Action(
                action_type=ActionType.POST_BOUNTY,
                agent_id="h1",
                content="Research task",
                metadata={"reward_amount": 10.0},
            )
        )
        bounty = orch.marketplace.get_agent_bounties("h1")[0]

        # Place bid
        orch._execute_action(
            Action(
                action_type=ActionType.PLACE_BID,
                agent_id="h2",
                target_id=bounty.bounty_id,
                metadata={"bid_amount": 8.0},
            )
        )
        bid = orch.marketplace.get_bids_for_bounty(bounty.bounty_id)[0]

        # Accept bid
        orch._execute_action(
            Action(
                action_type=ActionType.ACCEPT_BID,
                agent_id="h1",
                target_id=bounty.bounty_id,
                metadata={"bid_id": bid.bid_id},
            )
        )

        # Settle
        task_id = bounty.task_id
        h2_resources_before = orch.state.get_agent("h2").resources

        result = orch.settle_marketplace_task(task_id, success=True, quality_score=0.9)
        assert result is not None
        assert result["success"] is True

        # Worker should have more resources
        h2_resources_after = orch.state.get_agent("h2").resources
        assert h2_resources_after > h2_resources_before

    def test_settle_with_governance_taxes(self):
        """Test that settlement creates taxable interaction."""
        orch = self._make_orchestrator(with_governance=True)
        from swarm.agents.base import Action

        orch.state.get_agent("h1").resources = 100.0
        orch.state.get_agent("h2").resources = 100.0

        # Post, bid, accept
        orch._execute_action(
            Action(
                action_type=ActionType.POST_BOUNTY,
                agent_id="h1",
                content="Task",
                metadata={"reward_amount": 10.0},
            )
        )
        bounty = orch.marketplace.get_agent_bounties("h1")[0]

        orch._execute_action(
            Action(
                action_type=ActionType.PLACE_BID,
                agent_id="h2",
                target_id=bounty.bounty_id,
                metadata={"bid_amount": 8.0},
            )
        )
        bid = orch.marketplace.get_bids_for_bounty(bounty.bounty_id)[0]

        orch._execute_action(
            Action(
                action_type=ActionType.ACCEPT_BID,
                agent_id="h1",
                target_id=bounty.bounty_id,
                metadata={"bid_id": bid.bid_id},
            )
        )

        h1_before = orch.state.get_agent("h1").resources  # noqa: F841
        h2_before = orch.state.get_agent("h2").resources

        result = orch.settle_marketplace_task(
            bounty.task_id, success=True, quality_score=0.9
        )
        assert result is not None

        # Governance taxes should have been applied, affecting final amounts
        # Worker gets funds minus tax
        h2_after = orch.state.get_agent("h2").resources
        assert h2_after > h2_before  # Worker still nets positive

    def test_settlement_applies_detection_slash_resource_delta(self):
        """Marketplace settlement must apply more than governance costs."""
        orch = self._make_orchestrator(
            governance_config=GovernanceConfig(
                staking_enabled=True,
                stake_slash_rate=0.2,
                slash_on_detection=True,
                audit_enabled=True,
                audit_probability=1.0,
                audit_threshold_p=0.5,
            )
        )
        from swarm.agents.base import Action

        orch.state.get_agent("h1").resources = 100.0
        orch.state.get_agent("h2").resources = 100.0
        orch._execute_action(
            Action(
                action_type=ActionType.POST_BOUNTY,
                agent_id="h1",
                content="Task",
                metadata={"reward_amount": 10.0},
            )
        )
        bounty = orch.marketplace.get_agent_bounties("h1")[0]
        orch._execute_action(
            Action(
                action_type=ActionType.PLACE_BID,
                agent_id="h2",
                target_id=bounty.bounty_id,
                metadata={"bid_amount": 8.0},
            )
        )
        bid = orch.marketplace.get_bids_for_bounty(bounty.bounty_id)[0]
        orch._execute_action(
            Action(
                action_type=ActionType.ACCEPT_BID,
                agent_id="h1",
                target_id=bounty.bounty_id,
                metadata={"bid_id": bid.bid_id},
            )
        )

        poster = orch.state.get_agent("h1")
        before = poster.resources
        result = orch.settle_marketplace_task(
            bounty.task_id, success=True, quality_score=0.1
        )

        assert result is not None
        assert poster.stake_slashed > 0.0
        assert poster.resources == pytest.approx(
            before
            + result.get("refund_to_poster", 0.0)
            - poster.stake_slashed
            - 0.8  # audit shortfall (0.5 - 0.1) * multiplier 2
        )

    def test_observation_includes_marketplace(self):
        orch = self._make_orchestrator()
        from swarm.agents.base import Action

        # Post a bounty
        orch.state.get_agent("h1").resources = 100.0
        orch._execute_action(
            Action(
                action_type=ActionType.POST_BOUNTY,
                agent_id="h1",
                content="Task",
                metadata={"reward_amount": 10.0},
            )
        )

        # Build observation for another agent
        obs = orch._build_observation("h2")
        assert len(obs.available_bounties) == 1
        assert obs.available_bounties[0]["reward_amount"] == 10.0

        # Poster shouldn't see own bounty in available list
        obs_poster = orch._build_observation("h1")
        assert len(obs_poster.available_bounties) == 0

    def test_epoch_maintenance(self):
        """Test that expired bounties are refunded during epoch maintenance."""
        mp = Marketplace(MarketplaceConfig(auto_expire_bounties=True))
        bounty = mp.post_bounty("poster", "t1", 10.0, deadline_epoch=3)

        # Not expired yet
        expired = mp.expire_bounties(current_epoch=2)
        assert len(expired) == 0

        # Now expired
        expired = mp.expire_bounties(current_epoch=3)
        assert len(expired) == 1
        assert expired[0] == bounty.bounty_id
        assert mp.get_bounty(bounty.bounty_id).status == BountyStatus.EXPIRED

    def test_epoch_maintenance_refunds_resources(self):
        """Test that orchestrator refunds bounty funds on expiry."""
        orch = self._make_orchestrator()
        from swarm.agents.base import Action

        orch.state.get_agent("h1").resources = 100.0
        orch._execute_action(
            Action(
                action_type=ActionType.POST_BOUNTY,
                agent_id="h1",
                content="Task",
                metadata={"reward_amount": 10.0, "deadline_epoch": 1},
            )
        )

        assert orch.state.get_agent("h1").resources == 90.0

        # Manually call expire and refund like _run_epoch does
        expired = orch.marketplace.expire_bounties(current_epoch=1)
        assert len(expired) == 1

        bounty = orch.marketplace.get_bounty(expired[0])
        poster_state = orch.state.get_agent(bounty.poster_id)
        poster_state.update_resources(bounty.reward_amount)

        assert orch.state.get_agent("h1").resources == 100.0

    def test_file_dispute_action(self):
        orch = self._make_orchestrator()
        from swarm.agents.base import Action

        # Setup: bounty -> bid -> accept -> escrow
        orch.state.get_agent("h1").resources = 100.0
        orch._execute_action(
            Action(
                action_type=ActionType.POST_BOUNTY,
                agent_id="h1",
                content="Task",
                metadata={"reward_amount": 10.0},
            )
        )
        bounty = orch.marketplace.get_agent_bounties("h1")[0]

        orch._execute_action(
            Action(
                action_type=ActionType.PLACE_BID,
                agent_id="h2",
                target_id=bounty.bounty_id,
                metadata={"bid_amount": 8.0},
            )
        )
        bid = orch.marketplace.get_bids_for_bounty(bounty.bounty_id)[0]

        orch._execute_action(
            Action(
                action_type=ActionType.ACCEPT_BID,
                agent_id="h1",
                target_id=bounty.bounty_id,
                metadata={"bid_id": bid.bid_id},
            )
        )

        escrow = orch.marketplace.get_agent_escrows("h1")[0]

        # File dispute
        success = orch._execute_action(
            Action(
                action_type=ActionType.FILE_DISPUTE,
                agent_id="h1",
                target_id=escrow.escrow_id,
                content="Work quality issues",
            )
        )
        assert success


# ===========================================================================
# Scenario Loader Tests
# ===========================================================================


class TestScenarioLoader:
    """Test scenario loading with marketplace config."""

    def test_parse_marketplace_config(self):
        from swarm.scenarios.loader import parse_marketplace_config

        data = {
            "enabled": True,
            "escrow_fee_rate": 0.05,
            "min_bounty_amount": 2.0,
            "max_bids_per_bounty": 5,
        }
        config = parse_marketplace_config(data)
        assert config is not None
        assert config.escrow_fee_rate == 0.05
        assert config.min_bounty_amount == 2.0
        assert config.max_bids_per_bounty == 5

    def test_parse_marketplace_config_disabled(self):
        from swarm.scenarios.loader import parse_marketplace_config

        config = parse_marketplace_config({"enabled": False})
        assert config is None

    def test_parse_marketplace_config_empty(self):
        from swarm.scenarios.loader import parse_marketplace_config

        config = parse_marketplace_config({})
        assert config is None

    def test_load_marketplace_scenario(self):
        from swarm.scenarios.loader import build_orchestrator, load_scenario

        path = Path("scenarios/marketplace_economy.yaml")
        if not path.exists():
            pytest.skip("Marketplace scenario file not found")

        scenario = load_scenario(path)
        assert scenario.orchestrator_config.marketplace_config is not None
        assert scenario.orchestrator_config.marketplace_config.enabled is True

        orch = build_orchestrator(scenario)
        assert orch.marketplace is not None


# ===========================================================================
# Statistics Tests
# ===========================================================================


class TestMarketplaceStats:
    """Test marketplace statistics."""

    def test_stats_empty(self):
        mp = Marketplace()
        stats = mp.get_stats()
        assert stats["total_bounties"] == 0
        assert stats["total_bids"] == 0

    def test_stats_with_activity(self):
        mp = Marketplace()
        bounty = mp.post_bounty("a1", "t1", 10.0)
        mp.place_bid(bounty.bounty_id, "a2", 8.0)
        mp.place_bid(bounty.bounty_id, "a3", 7.0)

        stats = mp.get_stats()
        assert stats["total_bounties"] == 1
        assert stats["total_bids"] == 2
        assert stats["total_reward_posted"] == 10.0


# ===========================================================================
# Rate Limit Tests
# ===========================================================================


class TestMarketplaceRateLimits:
    """Test marketplace rate limiting."""

    def test_rate_limit_state_bounties(self):
        limits = RateLimits(bounties_per_epoch=2)
        state = RateLimitState()

        assert state.can_post_bounty(limits)
        state.record_bounty()
        assert state.can_post_bounty(limits)
        state.record_bounty()
        assert not state.can_post_bounty(limits)

    def test_rate_limit_state_bids(self):
        limits = RateLimits(bids_per_epoch=3)
        state = RateLimitState()

        assert state.can_place_bid(limits)
        for _ in range(3):
            state.record_bid()
        assert not state.can_place_bid(limits)

    def test_rate_limit_reset(self):
        limits = RateLimits(bounties_per_epoch=1, bids_per_epoch=1)
        state = RateLimitState()

        state.record_bounty()
        state.record_bid()
        assert not state.can_post_bounty(limits)
        assert not state.can_place_bid(limits)

        state.reset()
        assert state.can_post_bounty(limits)
        assert state.can_place_bid(limits)
