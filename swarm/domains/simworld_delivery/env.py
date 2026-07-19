"""SimWorld Delivery environment: urban delivery economy simulation.

Models a multi-agent delivery economy where agents bid on orders,
invest in tools, and navigate an urban grid. Inspired by SimWorld's
Case 2 benchmark (NeurIPS 2025).
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any

from swarm.domains.simworld_delivery.config import DeliveryConfig
from swarm.domains.simworld_delivery.entities import (
    AgentState,
    Bid,
    DeliveryAction,
    DeliveryActionType,
    DeliveryEvent,
    DeliveryOrder,
    OrderStatus,
    PersonaType,
)

logger = logging.getLogger(__name__)

# Distance threshold for considering an agent "at" a destination.
ARRIVAL_DISTANCE_THRESHOLD = 0.1
_DEFAULT_SIGMOID_K = 5.0


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class DeliveryEnvironment:
    """SimWorld-inspired delivery economy environment.

    Provides:
      - Order generation and lifecycle
      - Bidding and assignment mechanics
      - Movement and delivery simulation
      - Reputation tracking
      - Event logging
    """

    def __init__(self, config: DeliveryConfig) -> None:
        self._config = config
        self._rng = random.Random(config.seed)

        # Agents
        self._agents: dict[str, AgentState] = {}

        # Orders
        self._orders: dict[str, DeliveryOrder] = {}
        self._order_counter = 0

        # Pending bids for current step
        self._pending_bids: list[Bid] = []

        # Events
        self._events: list[DeliveryEvent] = []

        # Counters
        self._current_step = 0
        self._current_epoch = 0

        # Shared orders tracking
        self._shared_orders_map: dict[str, str] = {}

        # Generate depot locations
        self._depots: list[tuple[float, float]] = []
        for _ in range(config.city.num_depots):
            self._depots.append((
                self._rng.uniform(0, config.city.width),
                self._rng.uniform(0, config.city.height),
            ))

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------

    def add_agent(
        self,
        agent_id: str,
        persona: PersonaType = PersonaType.CONSCIENTIOUS,
    ) -> AgentState:
        """Register a delivery agent."""
        pos = (
            self._rng.uniform(0, self._config.city.width),
            self._rng.uniform(0, self._config.city.height),
        )
        agent = AgentState(
            agent_id=agent_id,
            persona=persona,
            position=pos,
            budget=self._config.economy.starting_budget,
            speed=1.0,
        )
        self._agents[agent_id] = agent
        return agent

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def obs(self, agent_id: str) -> dict[str, Any]:
        """Build observation for an agent."""
        agent = self._agents[agent_id]
        available_orders = [
            {
                "order_id": o.order_id,
                "origin": o.origin,
                "destination": o.destination,
                "value": o.value,
                "deadline_steps": o.deadline_steps,
                "weight": o.weight,
                "distance": _distance(o.origin, o.destination),
                "pickup_distance": _distance(agent.position, o.origin),
                "steps_remaining": (
                    o.deadline_steps - (self._current_step - o.created_step)
                ),
            }
            for o in self._orders.values()
            if o.status == OrderStatus.AVAILABLE
        ]

        current_order_info = None
        if agent.current_order and agent.current_order in self._orders:
            o = self._orders[agent.current_order]
            current_order_info = {
                "order_id": o.order_id,
                "origin": o.origin,
                "destination": o.destination,
                "value": o.value,
                "carrying": agent.carrying,
                "distance_to_target": _distance(
                    agent.position,
                    o.destination if agent.carrying else o.origin,
                ),
            }

        other_agents = [
            {
                "agent_id": aid,
                "position": a.position,
                "reputation": a.reputation,
                "has_scooter": a.has_scooter,
                "busy": a.current_order is not None,
            }
            for aid, a in self._agents.items()
            if aid != agent_id
        ]

        return {
            "agent_id": agent_id,
            "position": agent.position,
            "budget": agent.budget,
            "speed": agent.speed * (
                self._config.economy.scooter_speed_multiplier
                if agent.has_scooter else 1.0
            ),
            "has_scooter": agent.has_scooter,
            "reputation": agent.reputation,
            "current_order": current_order_info,
            "available_orders": available_orders,
            "other_agents": other_agents,
            "deliveries_completed": agent.deliveries_completed,
            "total_earnings": agent.total_earnings,
            "epoch": self._current_epoch,
            "step": self._current_step,
        }

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    def apply_actions(
        self, actions: dict[str, DeliveryAction],
    ) -> list[DeliveryEvent]:
        """Apply all agent actions for one step."""
        step_events: list[DeliveryEvent] = []

        # Process actions by type
        for agent_id, action in actions.items():
            agent = self._agents.get(agent_id)
            if agent is None:
                continue

            if action.action_type == DeliveryActionType.BID:
                evt = self._handle_bid(agent, action)
            elif action.action_type == DeliveryActionType.DELIVER:
                evt = self._handle_deliver(agent, action)
            elif action.action_type == DeliveryActionType.BUY_SCOOTER:
                evt = self._handle_buy_scooter(agent)
            elif action.action_type == DeliveryActionType.SHARE_ORDER:
                evt = self._handle_share_order(agent, action)
            elif action.action_type == DeliveryActionType.WAIT:
                agent.idle_steps += 1
                evt = DeliveryEvent(
                    event_type="wait",
                    step=self._current_step,
                    epoch=self._current_epoch,
                    agent_id=agent_id,
                )
            else:
                # Unknown action types are recorded as waits (without the
                # idle_steps accounting a deliberate WAIT gets) — warn so
                # a typo'd or unhandled action type is visible in logs.
                logger.warning(
                    "Unknown delivery action type %r from %s; treating as wait",
                    action.action_type,
                    agent_id,
                )
                evt = DeliveryEvent(
                    event_type="wait",
                    step=self._current_step,
                    epoch=self._current_epoch,
                    agent_id=agent_id,
                )

            if evt:
                step_events.append(evt)

        # Resolve pending bids
        bid_events = self._resolve_bids()
        step_events.extend(bid_events)

        # Move agents toward their delivery targets
        move_events = self._move_agents()
        step_events.extend(move_events)

        # Check for completed/failed deliveries
        delivery_events = self._check_deliveries()
        step_events.extend(delivery_events)

        # Expire old orders
        expire_events = self._expire_orders()
        step_events.extend(expire_events)

        self._events.extend(step_events)
        self._current_step += 1
        return step_events

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_bid(
        self, agent: AgentState, action: DeliveryAction,
    ) -> DeliveryEvent:
        gov = self._config.governance
        econ = self._config.economy

        # Check reputation threshold
        if agent.reputation < gov.min_reputation_to_bid:
            return DeliveryEvent(
                event_type="bid_fail",
                step=self._current_step,
                epoch=self._current_epoch,
                agent_id=agent.agent_id,
                details={"reason": "low_reputation",
                         "reputation": agent.reputation},
            )

        # Check agent not already delivering
        if agent.current_order is not None:
            return DeliveryEvent(
                event_type="bid_fail",
                step=self._current_step,
                epoch=self._current_epoch,
                agent_id=agent.agent_id,
                details={"reason": "already_delivering"},
            )

        order = self._orders.get(action.order_id)
        if order is None or order.status != OrderStatus.AVAILABLE:
            return DeliveryEvent(
                event_type="bid_fail",
                step=self._current_step,
                epoch=self._current_epoch,
                agent_id=agent.agent_id,
                details={"reason": "order_unavailable"},
            )

        # Clamp bid
        bid_amount = max(econ.bid_floor, min(action.bid_amount, econ.bid_ceiling))

        # Track overbids (bidding more than order value)
        if bid_amount > order.value:
            agent.overbids += 1

        agent.total_bids += 1
        self._pending_bids.append(Bid(
            agent_id=agent.agent_id,
            order_id=action.order_id,
            amount=bid_amount,
            step=self._current_step,
        ))

        return DeliveryEvent(
            event_type="bid_placed",
            step=self._current_step,
            epoch=self._current_epoch,
            agent_id=agent.agent_id,
            details={
                "order_id": action.order_id,
                "bid_amount": bid_amount,
                "order_value": order.value,
            },
        )

    def _handle_deliver(
        self, agent: AgentState, action: DeliveryAction,
    ) -> DeliveryEvent:
        """Agent explicitly acts to deliver (pickup or dropoff)."""
        if agent.current_order is None:
            return DeliveryEvent(
                event_type="deliver_fail",
                step=self._current_step,
                epoch=self._current_epoch,
                agent_id=agent.agent_id,
                details={"reason": "no_order"},
            )
        return DeliveryEvent(
            event_type="deliver_continue",
            step=self._current_step,
            epoch=self._current_epoch,
            agent_id=agent.agent_id,
            details={"order_id": agent.current_order},
        )

    def _handle_buy_scooter(self, agent: AgentState) -> DeliveryEvent:
        cost = self._config.economy.scooter_cost
        if agent.has_scooter:
            return DeliveryEvent(
                event_type="buy_fail",
                step=self._current_step,
                epoch=self._current_epoch,
                agent_id=agent.agent_id,
                details={"reason": "already_has_scooter"},
            )
        if agent.budget < cost:
            return DeliveryEvent(
                event_type="buy_fail",
                step=self._current_step,
                epoch=self._current_epoch,
                agent_id=agent.agent_id,
                details={"reason": "insufficient_budget",
                         "budget": agent.budget, "cost": cost},
            )
        agent.budget -= cost
        agent.has_scooter = True
        return DeliveryEvent(
            event_type="buy_scooter",
            step=self._current_step,
            epoch=self._current_epoch,
            agent_id=agent.agent_id,
            details={"cost": cost, "remaining_budget": agent.budget},
        )

    def _handle_share_order(
        self, agent: AgentState, action: DeliveryAction,
    ) -> DeliveryEvent:
        """Share an order with another agent (cooperation)."""
        partner = self._agents.get(action.share_with)
        if partner is None or partner.current_order is not None:
            return DeliveryEvent(
                event_type="share_fail",
                step=self._current_step,
                epoch=self._current_epoch,
                agent_id=agent.agent_id,
                details={"reason": "partner_unavailable"},
            )
        order = self._orders.get(action.order_id)
        if order is None or order.status != OrderStatus.AVAILABLE:
            return DeliveryEvent(
                event_type="share_fail",
                step=self._current_step,
                epoch=self._current_epoch,
                agent_id=agent.agent_id,
                details={"reason": "order_unavailable"},
            )
        # Assign to partner; sharer gets a sharing bonus on completion
        order.status = OrderStatus.ASSIGNED
        order.assigned_agent = partner.agent_id
        partner.current_order = order.order_id
        agent.orders_shared += 1
        self._shared_orders[order.order_id] = agent.agent_id

        return DeliveryEvent(
            event_type="order_shared",
            step=self._current_step,
            epoch=self._current_epoch,
            agent_id=agent.agent_id,
            details={
                "order_id": order.order_id,
                "shared_with": partner.agent_id,
            },
        )

    # ------------------------------------------------------------------
    # Bid resolution
    # ------------------------------------------------------------------

    def _resolve_bids(self) -> list[DeliveryEvent]:
        """Resolve pending bids via reverse auction (lowest bid wins).

        This models a procurement-style auction where the platform
        wants the cheapest delivery. Aggressive agents that overbid
        (bid > order value) will systematically lose because their
        bids are higher. This is intentional: overbidding is a
        resource-wasting strategy that the auction penalizes.
        """
        events: list[DeliveryEvent] = []

        # Group bids by order
        bids_by_order: dict[str, list[Bid]] = {}
        for bid in self._pending_bids:
            bids_by_order.setdefault(bid.order_id, []).append(bid)

        for order_id, bids in bids_by_order.items():
            order = self._orders.get(order_id)
            if order is None or order.status != OrderStatus.AVAILABLE:
                continue

            # Lowest bid wins (cheapest delivery); fall through to
            # next-best bidder if the winner is unavailable.
            bids.sort(key=lambda b: b.amount)
            winner = None
            winner_agent = None
            fee_rate = self._config.governance.delivery_fee_rate
            for bid in bids:
                candidate = self._agents.get(bid.agent_id)
                if candidate is None or candidate.current_order is not None:
                    continue
                if candidate.budget < bid.amount * fee_rate:
                    continue  # Can't afford the fee
                winner = bid
                winner_agent = candidate
                break
            if winner is None or winner_agent is None:
                continue

            # Assign order
            order.status = OrderStatus.ASSIGNED
            order.assigned_agent = winner.agent_id
            winner_agent.current_order = order.order_id

            # Deduct delivery fee
            fee = winner.amount * self._config.governance.delivery_fee_rate
            winner_agent.budget -= fee

            events.append(DeliveryEvent(
                event_type="bid_won",
                step=self._current_step,
                epoch=self._current_epoch,
                agent_id=winner.agent_id,
                details={
                    "order_id": order_id,
                    "bid_amount": winner.amount,
                    "fee": fee,
                    "num_bidders": len(bids),
                },
            ))

            # Notify losers
            for bid in bids[1:]:
                events.append(DeliveryEvent(
                    event_type="bid_lost",
                    step=self._current_step,
                    epoch=self._current_epoch,
                    agent_id=bid.agent_id,
                    details={"order_id": order_id},
                ))

        self._pending_bids.clear()
        return events

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def _move_agents(self) -> list[DeliveryEvent]:
        """Move agents toward their current delivery target."""
        events: list[DeliveryEvent] = []
        econ = self._config.economy

        for agent in self._agents.values():
            if agent.current_order is None:
                continue

            order = self._orders.get(agent.current_order)
            if order is None:
                agent.current_order = None
                continue

            # Determine target: origin (pickup) or destination (dropoff)
            target = order.destination if agent.carrying else order.origin
            dist = _distance(agent.position, target)
            speed = agent.speed * (
                econ.scooter_speed_multiplier if agent.has_scooter else 1.0
            )

            if dist <= speed:
                # Arrived at target
                agent.position = target
                agent.total_distance += dist
                if not agent.carrying:
                    # Picked up package
                    agent.carrying = True
                    order.status = OrderStatus.IN_TRANSIT
                    events.append(DeliveryEvent(
                        event_type="pickup",
                        step=self._current_step,
                        epoch=self._current_epoch,
                        agent_id=agent.agent_id,
                        details={"order_id": order.order_id},
                    ))
            else:
                # Move toward target
                dx = target[0] - agent.position[0]
                dy = target[1] - agent.position[1]
                ratio = speed / dist
                agent.position = (
                    agent.position[0] + dx * ratio,
                    agent.position[1] + dy * ratio,
                )
                agent.total_distance += speed

        return events

    # ------------------------------------------------------------------
    # Delivery completion/failure checks
    # ------------------------------------------------------------------

    def _check_deliveries(self) -> list[DeliveryEvent]:
        """Check for completed and failed deliveries."""
        events: list[DeliveryEvent] = []
        gov = self._config.governance
        econ = self._config.economy

        for agent in self._agents.values():
            if agent.current_order is None or not agent.carrying:
                continue

            order = self._orders.get(agent.current_order)
            if order is None:
                agent.current_order = None
                agent.carrying = False
                continue

            # Check if at destination
            dist_to_dest = _distance(agent.position, order.destination)
            if dist_to_dest > ARRIVAL_DISTANCE_THRESHOLD:
                # Check for deadline failure
                elapsed = self._current_step - order.created_step
                if elapsed > order.deadline_steps:
                    self._fail_delivery(agent, order, events)
                continue

            # Delivery complete
            elapsed = self._current_step - order.created_step
            on_time = elapsed <= order.deadline_steps

            # Calculate payout
            payout = order.value
            if not on_time:
                payout *= (1.0 - econ.late_penalty_fraction)
                agent.late_deliveries += 1
                agent.reputation = max(
                    0.0, agent.reputation - gov.reputation_penalty_late,
                )
            else:
                agent.on_time_deliveries += 1
                agent.reputation = min(
                    1.0, agent.reputation + gov.reputation_bonus_on_time,
                )

            agent.budget += payout
            agent.total_earnings += payout
            agent.deliveries_completed += 1
            agent.current_order = None
            agent.carrying = False
            order.status = OrderStatus.DELIVERED
            order.delivered_step = self._current_step

            # Compute soft label
            v_hat, p = self.compute_soft_label(agent, on_time, elapsed, order.deadline_steps)

            events.append(DeliveryEvent(
                event_type="delivery_complete",
                step=self._current_step,
                epoch=self._current_epoch,
                agent_id=agent.agent_id,
                details={
                    "order_id": order.order_id,
                    "payout": payout,
                    "on_time": on_time,
                    "elapsed_steps": elapsed,
                    "v_hat": round(v_hat, 4),
                    "p": round(p, 4),
                },
            ))

            # Sharing bonus if order was shared
            if order.order_id in self._shared_orders:
                sharer_id = self._shared_orders[order.order_id]
                sharer = self._agents.get(sharer_id)
                if sharer is not None:
                    bonus = payout * econ.sharing_bonus_fraction
                    sharer.budget += bonus
                    sharer.total_earnings += bonus
                    events.append(DeliveryEvent(
                        event_type="sharing_bonus",
                        step=self._current_step,
                        epoch=self._current_epoch,
                        agent_id=sharer_id,
                        details={
                            "order_id": order.order_id,
                            "bonus": bonus,
                        },
                    ))

        return events

    def _fail_delivery(
        self,
        agent: AgentState,
        order: DeliveryOrder,
        events: list[DeliveryEvent],
    ) -> None:
        """Handle a failed delivery (deadline exceeded)."""
        econ = self._config.economy
        gov = self._config.governance

        penalty = econ.failed_delivery_penalty
        agent.budget -= penalty
        agent.deliveries_failed += 1
        agent.customer_complaints += 1
        agent.reputation = max(
            0.0, agent.reputation - gov.reputation_penalty_fail,
        )
        agent.current_order = None
        agent.carrying = False
        order.status = OrderStatus.FAILED

        events.append(DeliveryEvent(
            event_type="delivery_failed",
            step=self._current_step,
            epoch=self._current_epoch,
            agent_id=agent.agent_id,
            details={
                "order_id": order.order_id,
                "penalty": penalty,
            },
        ))

    # ------------------------------------------------------------------
    # Order lifecycle
    # ------------------------------------------------------------------

    def generate_orders(self, count: int | None = None) -> list[DeliveryEvent]:
        """Generate new delivery orders for the current step."""
        events: list[DeliveryEvent] = []
        n = count if count is not None else self._config.orders.orders_per_epoch
        cfg = self._config.orders
        city = self._config.city

        for _ in range(n):
            self._order_counter += 1
            order_id = f"order_{self._order_counter}"
            origin = (
                self._rng.uniform(0, city.width),
                self._rng.uniform(0, city.height),
            )
            destination = (
                self._rng.uniform(0, city.width),
                self._rng.uniform(0, city.height),
            )
            order = DeliveryOrder(
                order_id=order_id,
                origin=origin,
                destination=destination,
                value=self._rng.uniform(cfg.min_value, cfg.max_value),
                deadline_steps=self._rng.randint(
                    cfg.min_deadline_steps, cfg.max_deadline_steps,
                ),
                weight=self._rng.uniform(cfg.min_weight, cfg.max_weight),
                status=OrderStatus.AVAILABLE,
                created_step=self._current_step,
            )
            self._orders[order_id] = order
            events.append(DeliveryEvent(
                event_type="order_created",
                step=self._current_step,
                epoch=self._current_epoch,
                agent_id="",
                details={
                    "order_id": order_id,
                    "value": order.value,
                    "distance": _distance(origin, destination),
                },
            ))
        return events

    def _expire_orders(self) -> list[DeliveryEvent]:
        """Expire orders that have been available too long."""
        events: list[DeliveryEvent] = []
        expiry = self._config.orders.expiry_steps

        for order in self._orders.values():
            if order.status != OrderStatus.AVAILABLE:
                continue
            if self._current_step - order.created_step > expiry:
                order.status = OrderStatus.EXPIRED
                events.append(DeliveryEvent(
                    event_type="order_expired",
                    step=self._current_step,
                    epoch=self._current_epoch,
                    details={"order_id": order.order_id},
                ))
        return events

    # ------------------------------------------------------------------
    # Epoch boundary
    # ------------------------------------------------------------------

    def end_epoch(self) -> list[DeliveryEvent]:
        """Process epoch boundary: reputation decay, cleanup."""
        events: list[DeliveryEvent] = []
        gov = self._config.governance

        # Reputation decay
        for agent in self._agents.values():
            agent.reputation = max(
                0.0, agent.reputation - gov.reputation_decay,
            )

        # Cancel in-progress deliveries for deadline tracking
        for order in self._orders.values():
            if order.status in (OrderStatus.ASSIGNED, OrderStatus.IN_TRANSIT):
                elapsed = self._current_step - order.created_step
                if elapsed > order.deadline_steps:
                    assigned = order.assigned_agent
                    if assigned and assigned in self._agents:
                        self._fail_delivery(
                            self._agents[assigned], order, events,
                        )

        self._current_epoch += 1
        # Note: _current_step is NOT reset — it stays monotonic so that
        # elapsed-time calculations (current_step - created_step) remain
        # correct for orders that survive across epoch boundaries.

        # Clear completed/failed/expired orders
        self._orders = {
            oid: o for oid, o in self._orders.items()
            if o.status in (OrderStatus.AVAILABLE, OrderStatus.ASSIGNED,
                            OrderStatus.IN_TRANSIT)
        }

        return events

    # ------------------------------------------------------------------
    # Soft label computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_soft_label(
        agent: AgentState,
        on_time: bool,
        elapsed: int,
        deadline_steps: int,
    ) -> tuple[float, float]:
        """Compute soft label (v_hat, p) from delivery signals.

        Signals:
        - on_time_signal: 1.0 if on_time, 0.0 otherwise (weight 0.4)
        - time_margin: (deadline - elapsed) / deadline, clamped to [0, 1] (weight 0.35)
        - reputation: agent reputation, already in [0, 1] (weight 0.25)

        Returns:
            (v_hat, p) where v_hat is in [-1, 1] and p is sigmoid(k * v_hat) in [0, 1]
        """
        on_time_signal = 1.0 if on_time else 0.0
        time_margin = max(0.0, min(1.0, (deadline_steps - elapsed) / max(deadline_steps, 1)))
        reputation = agent.reputation

        # Weighted combination: rescale to [-1, 1] (treat each as 0.5-centered)
        v_hat = (
            0.4 * (2 * on_time_signal - 1.0)
            + 0.35 * (2 * time_margin - 1.0)
            + 0.25 * (2 * reputation - 1.0)
        )

        # Clamp v_hat to [-1, 1]
        v_hat = max(-1.0, min(1.0, v_hat))

        # Apply sigmoid: p = 1 / (1 + exp(-k * v_hat))
        try:
            p = 1.0 / (1.0 + math.exp(-_DEFAULT_SIGMOID_K * v_hat))
        except OverflowError:
            # Extreme values: approximate
            p = 1.0 if v_hat > 0 else 0.0

        return (v_hat, p)

    # ------------------------------------------------------------------
    # Shared orders tracking
    # ------------------------------------------------------------------

    @property
    def _shared_orders(self) -> dict[str, str]:
        """Track which orders were shared and by whom."""
        return self._shared_orders_map

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def agents(self) -> dict[str, AgentState]:
        return dict(self._agents)

    @property
    def orders(self) -> dict[str, DeliveryOrder]:
        return dict(self._orders)

    @property
    def events(self) -> list[DeliveryEvent]:
        return list(self._events)

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def config(self) -> DeliveryConfig:
        return self._config
