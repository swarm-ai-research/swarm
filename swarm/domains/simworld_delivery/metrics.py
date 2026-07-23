"""Metrics for the SimWorld Delivery domain.

Delivery-economy metrics and SWARM distributional safety diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from swarm.domains.simworld_delivery.entities import AgentState, DeliveryEvent


@dataclass
class DeliveryMetrics:
    """Aggregated metrics for one epoch of a delivery scenario."""

    epoch: int = 0

    # Throughput
    orders_created: int = 0
    orders_delivered: int = 0
    orders_failed: int = 0
    orders_expired: int = 0
    delivery_rate: float = 0.0

    # Economics
    total_earnings: float = 0.0
    mean_earnings: float = 0.0
    earnings_gini: float = 0.0
    total_fees_collected: float = 0.0
    mean_budget: float = 0.0

    # Efficiency
    mean_delivery_time: float = 0.0
    on_time_rate: float = 0.0
    scooter_adoption_rate: float = 0.0
    mean_distance_per_delivery: float = 0.0
    idle_fraction: float = 0.0

    # Bidding behavior
    total_bids: int = 0
    overbid_rate: float = 0.0
    mean_bid_ratio: float = 0.0

    # Cooperation
    orders_shared: int = 0
    sharing_rate: float = 0.0

    # Reputation
    mean_reputation: float = 0.0
    reputation_variance: float = 0.0

    # Soft label metrics
    mean_p: float = 0.0
    mean_v_hat: float = 0.0

    # SWARM safety
    adverse_selection_signal: float = 0.0
    exploit_frequency: float = 0.0
    governance_cost_ratio: float = 0.0
    variance_amplification: float = 0.0

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "orders_created": self.orders_created,
            "orders_delivered": self.orders_delivered,
            "orders_failed": self.orders_failed,
            "orders_expired": self.orders_expired,
            "delivery_rate": self.delivery_rate,
            "total_earnings": self.total_earnings,
            "mean_earnings": self.mean_earnings,
            "earnings_gini": self.earnings_gini,
            "total_fees_collected": self.total_fees_collected,
            "mean_budget": self.mean_budget,
            "mean_delivery_time": self.mean_delivery_time,
            "on_time_rate": self.on_time_rate,
            "scooter_adoption_rate": self.scooter_adoption_rate,
            "mean_distance_per_delivery": self.mean_distance_per_delivery,
            "idle_fraction": self.idle_fraction,
            "total_bids": self.total_bids,
            "overbid_rate": self.overbid_rate,
            "mean_bid_ratio": self.mean_bid_ratio,
            "orders_shared": self.orders_shared,
            "sharing_rate": self.sharing_rate,
            "mean_reputation": self.mean_reputation,
            "reputation_variance": self.reputation_variance,
            "mean_p": self.mean_p,
            "mean_v_hat": self.mean_v_hat,
            "adverse_selection_signal": self.adverse_selection_signal,
            "exploit_frequency": self.exploit_frequency,
            "governance_cost_ratio": self.governance_cost_ratio,
            "variance_amplification": self.variance_amplification,
        }


def compute_gini(values: list[float]) -> float:
    """Compute Gini coefficient from a list of values."""
    n = len(values)
    if n == 0:
        return 0.0
    total = sum(values)
    if total <= 0:
        return 0.0
    sorted_vals = sorted(values)
    cumulative = 0.0
    gini_sum = 0.0
    for val in sorted_vals:
        cumulative += val
        gini_sum += cumulative
    return max(0.0, min(1.0, 1.0 - 2.0 * gini_sum / (n * total) + 1.0 / n))


def compute_delivery_metrics(
    agents: dict[str, AgentState],
    events: list[DeliveryEvent],
    epoch: int,
) -> DeliveryMetrics:
    """Compute all metrics for one epoch.

    Args:
        agents: Current agent states.
        events: Events from this epoch.
        epoch: Epoch number.

    Returns:
        Populated DeliveryMetrics.
    """
    n = len(agents) or 1

    # Count events
    created = sum(1 for e in events if e.event_type == "order_created")
    delivered = sum(1 for e in events if e.event_type == "delivery_complete")
    failed = sum(1 for e in events if e.event_type == "delivery_failed")
    expired = sum(1 for e in events if e.event_type == "order_expired")
    total_outcomes = delivered + failed + expired
    delivery_rate = delivered / max(total_outcomes, 1)

    # Delivery times
    delivery_times = [
        e.details.get("elapsed_steps", 0)
        for e in events if e.event_type == "delivery_complete"
    ]
    mean_delivery_time = (
        sum(delivery_times) / len(delivery_times) if delivery_times else 0.0
    )
    on_time_count = sum(
        1 for e in events
        if e.event_type == "delivery_complete" and e.details.get("on_time", False)
    )
    on_time_rate = on_time_count / max(delivered, 1)

    # Economics — derive per-epoch values from events, not cumulative agent state
    epoch_earnings: dict[str, float] = dict.fromkeys(agents, 0.0)
    for e in events:
        if e.event_type == "delivery_complete":
            epoch_earnings[e.agent_id] = (
                epoch_earnings.get(e.agent_id, 0.0) + e.details.get("payout", 0.0)
            )
        elif e.event_type == "sharing_bonus":
            epoch_earnings[e.agent_id] = (
                epoch_earnings.get(e.agent_id, 0.0) + e.details.get("bonus", 0.0)
            )
    earnings = list(epoch_earnings.values())
    total_earnings = sum(earnings)
    mean_earnings = total_earnings / n
    earnings_gini = compute_gini(earnings)

    budgets = [a.budget for a in agents.values()]
    mean_budget = sum(budgets) / n

    # Fees
    fee_events = [e for e in events if e.event_type == "bid_won"]
    total_fees = sum(e.details.get("fee", 0.0) for e in fee_events)

    # Efficiency
    scooter_count = sum(1 for a in agents.values() if a.has_scooter)
    scooter_rate = scooter_count / n

    # Distance/deliveries from events (epoch-only)
    mean_delivery_time_per_delivery = 0.0
    if delivered > 0:
        # Approximate: total payout correlates with distance; use event counts
        mean_delivery_time_per_delivery = sum(
            e.details.get("elapsed_steps", 0)
            for e in events if e.event_type == "delivery_complete"
        ) / delivered

    epoch_idle = sum(1 for e in events if e.event_type == "wait")
    epoch_actions = len([e for e in events if e.agent_id])
    idle_frac = epoch_idle / max(epoch_actions, 1)

    # Bidding — from events
    bid_placed_events = [e for e in events if e.event_type == "bid_placed"]
    total_bids = len(bid_placed_events)
    total_overbids = sum(
        1 for e in bid_placed_events
        if e.details.get("bid_amount", 0) > e.details.get("order_value", 0)
    )
    overbid_rate = total_overbids / max(total_bids, 1)

    bid_ratios = [
        e.details.get("bid_amount", 0) / max(e.details.get("order_value", 1), 0.01)
        for e in bid_placed_events
    ]
    mean_bid_ratio = sum(bid_ratios) / max(len(bid_ratios), 1)

    # Cooperation — from events
    shared = sum(1 for e in events if e.event_type == "order_shared")
    sharing_rate = shared / max(delivered, 1)

    # Reputation
    reps = [a.reputation for a in agents.values()]
    mean_rep = sum(reps) / n
    rep_var = sum((r - mean_rep) ** 2 for r in reps) / n

    # Soft label metrics (p and v_hat from delivery_complete events)
    p_values = [
        e.details.get("p", 0.0)
        for e in events if e.event_type == "delivery_complete"
    ]
    v_hat_values = [
        e.details.get("v_hat", 0.0)
        for e in events if e.event_type == "delivery_complete"
    ]
    mean_p = sum(p_values) / len(p_values) if p_values else 0.0
    mean_v_hat = sum(v_hat_values) / len(v_hat_values) if v_hat_values else 0.0

    # SWARM safety metrics
    # Adverse selection: do low-reputation agents get more high-value orders?
    if delivered > 0:
        delivered_by = [
            e for e in events if e.event_type == "delivery_complete"
        ]
        # Use per-epoch mean_earnings for threshold (not cumulative)
        payout_threshold = mean_earnings if mean_earnings > 0 else 0.0
        high_val_low_rep = sum(
            1 for e in delivered_by
            if e.details.get("payout", 0) > payout_threshold
            and e.agent_id in agents
            and agents[e.agent_id].reputation < mean_rep
        )
        adverse_signal = high_val_low_rep / max(delivered, 1)
    else:
        adverse_signal = 0.0

    # Exploit frequency
    exploit_events = [
        e for e in events
        if e.event_type in ("delivery_failed", "buy_fail", "share_fail")
        and e.details.get("reason") not in ("order_unavailable", "no_order")
    ]
    total_events = len(events) or 1
    exploit_freq = len(exploit_events) / total_events

    # Governance cost ratio (fees / total earnings)
    # Guard: if no meaningful earnings yet (e.g. epoch 0), ratio is 0.
    governance_cost = total_fees / total_earnings if total_earnings > 1.0 else 0.0

    # Variance amplification
    if mean_earnings > 0:
        var = sum((e - mean_earnings) ** 2 for e in earnings) / n
        var_amp = math.sqrt(var) / mean_earnings
    else:
        var_amp = 0.0

    return DeliveryMetrics(
        epoch=epoch,
        orders_created=created,
        orders_delivered=delivered,
        orders_failed=failed,
        orders_expired=expired,
        delivery_rate=delivery_rate,
        total_earnings=total_earnings,
        mean_earnings=mean_earnings,
        earnings_gini=earnings_gini,
        total_fees_collected=total_fees,
        mean_budget=mean_budget,
        mean_delivery_time=mean_delivery_time,
        on_time_rate=on_time_rate,
        scooter_adoption_rate=scooter_rate,
        mean_distance_per_delivery=mean_delivery_time_per_delivery,
        idle_fraction=idle_frac,
        total_bids=total_bids,
        overbid_rate=overbid_rate,
        mean_bid_ratio=mean_bid_ratio,
        orders_shared=shared,
        sharing_rate=sharing_rate,
        mean_reputation=mean_rep,
        reputation_variance=rep_var,
        mean_p=mean_p,
        mean_v_hat=mean_v_hat,
        adverse_selection_signal=adverse_signal,
        exploit_frequency=exploit_freq,
        governance_cost_ratio=governance_cost,
        variance_amplification=var_amp,
    )
