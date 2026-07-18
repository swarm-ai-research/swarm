"""Marketplace action handling extracted from the orchestrator.

Encapsulates all marketplace-related logic: bounty posting, bid
management, escrow settlement, dispute handling, and epoch
maintenance.
"""

from typing import Any, Dict, List, Optional

from swarm.agents.base import Action, ActionType
from swarm.core.handler import Handler, HandlerActionResult
from swarm.env.marketplace import EscrowStatus, Marketplace
from swarm.env.state import EnvState
from swarm.env.tasks import TaskPool
from swarm.governance.engine import GovernanceEngine
from swarm.logging.event_bus import EventBus
from swarm.models.events import Event, EventType
from swarm.models.interaction import InteractionType, SoftInteraction


class MarketplaceHandler(Handler):
    """Handles all marketplace actions and lifecycle events.

    Operates on shared state objects owned by the orchestrator.
    The orchestrator delegates marketplace-related actions and epoch
    maintenance to this handler.
    """

    @staticmethod
    def handled_action_types() -> frozenset:
        return frozenset({
            ActionType.POST_BOUNTY,
            ActionType.PLACE_BID,
            ActionType.ACCEPT_BID,
            ActionType.REJECT_BID,
            ActionType.WITHDRAW_BID,
            ActionType.FILE_DISPUTE,
        })

    def __init__(
        self,
        marketplace: Marketplace,
        task_pool: TaskPool,
        *,
        event_bus: EventBus,
        enable_rate_limits: bool = True,
    ):
        super().__init__(event_bus=event_bus)
        self.marketplace = marketplace
        self.task_pool = task_pool
        self._enable_rate_limits = enable_rate_limits

    def handle_action(self, action: Action, state: Any) -> HandlerActionResult:
        """Dispatch a marketplace action and return a unified result.

        Marketplace actions don't generate interactions/observables, so the
        result always has ``observables=None``.
        """
        dispatch = {
            ActionType.POST_BOUNTY: lambda: self.handle_post_bounty(
                action, state, enable_rate_limits=self._enable_rate_limits
            ),
            ActionType.PLACE_BID: lambda: self.handle_place_bid(
                action, state, enable_rate_limits=self._enable_rate_limits
            ),
            ActionType.ACCEPT_BID: lambda: self.handle_accept_bid(action, state),
            ActionType.REJECT_BID: lambda: self.handle_reject_bid(action, state),
            ActionType.WITHDRAW_BID: lambda: self.handle_withdraw_bid(action),
            ActionType.FILE_DISPUTE: lambda: self.handle_file_dispute(action, state),
        }
        handler_fn = dispatch.get(action.action_type)
        if handler_fn is None:
            return HandlerActionResult(success=False)
        success = handler_fn()
        return HandlerActionResult(success=success)

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def handle_post_bounty(
        self,
        action: Action,
        state: EnvState,
        *,
        enable_rate_limits: bool = True,
    ) -> bool:
        """Handle POST_BOUNTY action."""
        agent_id = action.agent_id
        rate_limit = state.get_rate_limit_state(agent_id)

        if enable_rate_limits and not rate_limit.can_post_bounty(state.rate_limits):
            return False

        reward_amount = action.metadata.get("reward_amount", 0)
        min_reputation = action.metadata.get("min_reputation", 0.0)
        deadline_epoch = action.metadata.get("deadline_epoch")

        agent_state = state.get_agent(agent_id)
        if not agent_state or agent_state.resources < reward_amount:
            return False

        try:
            task = self.task_pool.create_task(
                prompt=action.content or "Marketplace bounty task",
                description=action.content or "Marketplace bounty task",
                bounty=reward_amount,
                min_reputation=min_reputation,
                deadline_epoch=deadline_epoch,
            )

            bounty = self.marketplace.post_bounty(
                poster_id=agent_id,
                task_id=task.task_id,
                reward_amount=reward_amount,
                min_reputation=min_reputation,
                deadline_epoch=deadline_epoch,
                current_epoch=state.current_epoch,
            )
        except ValueError:
            return False

        agent_state.update_resources(-reward_amount)
        rate_limit.record_bounty()

        self._emit_event(
            Event(
                event_type=EventType.BOUNTY_POSTED,
                agent_id=agent_id,
                payload={
                    "bounty_id": bounty.bounty_id,
                    "task_id": task.task_id,
                    "reward_amount": reward_amount,
                },
                epoch=state.current_epoch,
                step=state.current_step,
            )
        )

        return True

    def handle_place_bid(
        self,
        action: Action,
        state: EnvState,
        *,
        enable_rate_limits: bool = True,
    ) -> bool:
        """Handle PLACE_BID action."""
        agent_id = action.agent_id
        rate_limit = state.get_rate_limit_state(agent_id)

        if enable_rate_limits and not rate_limit.can_place_bid(state.rate_limits):
            return False

        bounty_id = action.target_id
        bid_amount = action.metadata.get("bid_amount", 0)

        bid = self.marketplace.place_bid(
            bounty_id=bounty_id,
            bidder_id=agent_id,
            bid_amount=bid_amount,
            message=action.content,
        )

        if bid is None:
            return False

        rate_limit.record_bid()

        self._emit_event(
            Event(
                event_type=EventType.BID_PLACED,
                agent_id=agent_id,
                payload={
                    "bid_id": bid.bid_id,
                    "bounty_id": bounty_id,
                    "bid_amount": bid_amount,
                },
                epoch=state.current_epoch,
                step=state.current_step,
            )
        )

        return True

    def handle_accept_bid(self, action: Action, state: EnvState) -> bool:
        """Handle ACCEPT_BID action."""
        agent_id = action.agent_id
        bounty_id = action.target_id
        bid_id = action.metadata.get("bid_id", "")

        escrow = self.marketplace.accept_bid(
            bounty_id=bounty_id,
            bid_id=bid_id,
            poster_id=agent_id,
        )

        if escrow is None:
            return False

        bounty = self.marketplace.get_bounty(bounty_id)
        if bounty:
            worker_state = state.get_agent(escrow.worker_id)
            if worker_state:
                self.task_pool.claim_task(
                    task_id=bounty.task_id,
                    agent_id=escrow.worker_id,
                    agent_reputation=worker_state.reputation,
                )

        self._emit_event(
            Event(
                event_type=EventType.ESCROW_CREATED,
                agent_id=agent_id,
                payload={
                    "escrow_id": escrow.escrow_id,
                    "bounty_id": bounty_id,
                    "bid_id": bid_id,
                    "worker_id": escrow.worker_id,
                    "amount": escrow.amount,
                },
                epoch=state.current_epoch,
                step=state.current_step,
            )
        )

        return True

    def handle_reject_bid(self, action: Action, state: EnvState) -> bool:
        """Handle REJECT_BID action."""
        success: bool = self.marketplace.reject_bid(
            bid_id=action.target_id,
            poster_id=action.agent_id,
        )

        if success:
            self._emit_event(
                Event(
                    event_type=EventType.BID_REJECTED,
                    agent_id=action.agent_id,
                    payload={"bid_id": action.target_id},
                    epoch=state.current_epoch,
                    step=state.current_step,
                )
            )

        return success

    def handle_withdraw_bid(self, action: Action) -> bool:
        """Handle WITHDRAW_BID action."""
        return bool(self.marketplace.withdraw_bid(
            bid_id=action.target_id,
            bidder_id=action.agent_id,
        ))

    def handle_file_dispute(self, action: Action, state: EnvState) -> bool:
        """Handle FILE_DISPUTE action."""
        dispute = self.marketplace.file_dispute(
            escrow_id=action.target_id,
            filed_by=action.agent_id,
            reason=action.content,
            current_epoch=state.current_epoch,
        )

        if dispute is None:
            return False

        self._emit_event(
            Event(
                event_type=EventType.DISPUTE_FILED,
                agent_id=action.agent_id,
                payload={
                    "dispute_id": dispute.dispute_id,
                    "escrow_id": action.target_id,
                    "reason": action.content,
                },
                epoch=state.current_epoch,
                step=state.current_step,
            )
        )

        return True

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def settle_task(
        self,
        task_id: str,
        success: bool,
        state: EnvState,
        governance_engine: Optional[GovernanceEngine] = None,
        quality_score: float = 1.0,
    ) -> Optional[Dict]:
        """Settle a marketplace bounty/escrow after task completion.

        Called after VERIFY_OUTPUT succeeds. Checks if the task has
        an associated bounty/escrow and settles it.

        Args:
            task_id: The completed task ID.
            success: Whether the task was completed successfully.
            state: Current environment state.
            governance_engine: Optional governance engine for taxes.
            quality_score: Quality score from verifier.

        Returns:
            Settlement details, or None if no marketplace bounty.
        """
        bounty = self.marketplace.get_bounty_for_task(task_id)
        if not bounty or not bounty.escrow_id:
            return None

        settlement: Optional[Dict] = self.marketplace.settle_escrow(
            escrow_id=bounty.escrow_id,
            success=success,
            quality_score=quality_score,
        )

        if not settlement:
            return None

        if success:
            worker_id = settlement["worker_id"]
            poster_id = settlement["poster_id"]
            released = settlement["released_to_worker"]
            refund_to_poster = settlement.get("refund_to_poster", 0.0)

            worker_state = state.get_agent(worker_id)
            poster_state = state.get_agent(poster_id)

            if worker_state:
                worker_state.update_resources(released)
            if poster_state and refund_to_poster > 0:
                poster_state.update_resources(refund_to_poster)

            # Apply governance taxes if engine provided
            if governance_engine:
                interaction = SoftInteraction(
                    initiator=poster_id,
                    counterparty=worker_id,
                    interaction_type=InteractionType.TRADE,
                    accepted=True,
                    task_progress_delta=quality_score,
                    rework_count=0,
                    verifier_rejections=0,
                    tool_misuse_flags=0,
                    counterparty_engagement_delta=quality_score * 0.5,
                    v_hat=quality_score * 2 - 1,
                    p=quality_score,
                    tau=released,
                )
                gov_effect = governance_engine.apply_interaction(interaction, state)
                if gov_effect.cost_a > 0 and poster_state:
                    poster_state.update_resources(-gov_effect.cost_a)
                if gov_effect.cost_b > 0 and worker_state:
                    worker_state.update_resources(-gov_effect.cost_b)

            self._emit_event(
                Event(
                    event_type=EventType.ESCROW_RELEASED,
                    payload={
                        "escrow_id": bounty.escrow_id,
                        "worker_id": worker_id,
                        "amount": released,
                        "quality_score": quality_score,
                    },
                    epoch=state.current_epoch,
                    step=state.current_step,
                )
            )
        else:
            poster_id = settlement["poster_id"]
            refunded = settlement["refunded_to_poster"]
            poster_state = state.get_agent(poster_id)
            if poster_state:
                poster_state.update_resources(refunded)

            self._emit_event(
                Event(
                    event_type=EventType.ESCROW_REFUNDED,
                    payload={
                        "escrow_id": bounty.escrow_id,
                        "poster_id": poster_id,
                        "amount": refunded,
                    },
                    epoch=state.current_epoch,
                    step=state.current_step,
                )
            )

        return settlement

    # ------------------------------------------------------------------
    # Epoch maintenance
    # ------------------------------------------------------------------

    def on_epoch_end(self, state: EnvState) -> None:
        """Run marketplace end-of-epoch maintenance.

        Expires bounties (refunding posters) and auto-resolves disputes.
        """
        expired_bounties = self.marketplace.expire_bounties(state.current_epoch)
        for bounty_id in expired_bounties:
            bounty = self.marketplace.get_bounty(bounty_id)
            if bounty:
                poster_state = state.get_agent(bounty.poster_id)
                if poster_state:
                    poster_state.update_resources(bounty.reward_amount)

        resolved_disputes = self.marketplace.auto_resolve_disputes(state.current_epoch)
        for dispute_id in resolved_disputes:
            dispute = self.marketplace.get_dispute(dispute_id)
            if dispute:
                escrow = self.marketplace.get_escrow(dispute.escrow_id)
                if escrow:
                    worker_state = state.get_agent(escrow.worker_id)
                    poster_state = state.get_agent(escrow.poster_id)
                    if worker_state:
                        worker_state.update_resources(escrow.released_amount)
                    if poster_state:
                        poster_state.update_resources(escrow.refunded_amount)
                    self._emit_event(
                        Event(
                            event_type=EventType.DISPUTE_RESOLVED,
                            payload={
                                "dispute_id": dispute_id,
                                "escrow_id": escrow.escrow_id,
                                "worker_share": dispute.worker_share,
                                "auto_resolved": True,
                            },
                            epoch=state.current_epoch,
                            step=state.current_step,
                        )
                    )

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def build_observation_fields(
        self,
        agent_id: str,
        state: EnvState,
    ) -> Dict[str, Any]:
        """Build marketplace-related observation fields for an agent.

        Returns a dict with keys: available_bounties, active_bids,
        active_escrows, pending_bid_decisions.
        """
        agent_state = state.get_agent(agent_id)
        agent_rep = agent_state.reputation if agent_state else 0

        available_bounties = [
            b.to_dict()
            for b in self.marketplace.get_open_bounties(
                current_epoch=state.current_epoch,
                min_reputation=agent_rep,
            )
            if b.poster_id != agent_id
        ]

        active_bids = [
            b.to_dict()
            for b in self.marketplace.get_agent_bids(agent_id)
            if b.status.value == "pending"
        ]

        active_escrows = [
            e.to_dict()
            for e in self.marketplace.get_agent_escrows(agent_id)
            if e.status == EscrowStatus.HELD
        ]

        pending_bid_decisions: List[Dict] = []
        for bounty in self.marketplace.get_agent_bounties(agent_id):
            if bounty.status.value == "open":
                for bid in self.marketplace.get_bids_for_bounty(bounty.bounty_id):
                    if bid.status.value == "pending":
                        bid_dict = bid.to_dict()
                        bid_dict["bounty_reward"] = bounty.reward_amount
                        pending_bid_decisions.append(bid_dict)

        return {
            "available_bounties": available_bounties,
            "active_bids": active_bids,
            "active_escrows": active_escrows,
            "pending_bid_decisions": pending_bid_decisions,
        }
