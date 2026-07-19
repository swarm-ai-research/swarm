"""Metrics for the AI Economist GTB scenario.

Provides economist-style welfare metrics and SWARM systemic-risk diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

from swarm.domains.gather_trade_build.entities import (
    GTBEvent,
    WorkerState,
)


@dataclass
class GTBMetrics:
    """Aggregated metrics for one epoch of a GTB scenario."""

    epoch: int = 0

    # Productivity
    total_production: float = 0.0  # total gross income
    mean_production: float = 0.0
    total_houses_built: int = 0

    # Tax revenue
    total_tax_revenue: float = 0.0
    mean_effective_tax_rate: float = 0.0

    # Inequality
    gini_coefficient: float = 0.0
    atkinson_index: float = 0.0  # with epsilon=0.5

    # Welfare
    welfare: float = 0.0  # prod_weight * prod - ineq_weight * ineq

    # Enforcement
    total_audits: int = 0
    total_catches: int = 0
    total_fines: float = 0.0
    undetected_evasion_rate: float = 0.0  # fraction of evasion not caught
    enforcement_cost: float = 0.0  # proxy: audit_count * per_audit_cost

    # Bunching
    bunching_intensity: float = 0.0  # fraction of incomes within bin_width of thresholds

    # Collusion
    collusion_events_detected: int = 0
    collusion_suspicion_mean: float = 0.0

    # SWARM systemic-risk
    exploit_frequency: float = 0.0  # misreport + collusion events / total events
    governance_backfire_events: int = 0  # fines on honest agents (false positives)
    variance_amplification: float = 0.0  # std(income) / mean(income)

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "total_production": self.total_production,
            "mean_production": self.mean_production,
            "total_houses_built": self.total_houses_built,
            "total_tax_revenue": self.total_tax_revenue,
            "mean_effective_tax_rate": self.mean_effective_tax_rate,
            "gini_coefficient": self.gini_coefficient,
            "atkinson_index": self.atkinson_index,
            "welfare": self.welfare,
            "total_audits": self.total_audits,
            "total_catches": self.total_catches,
            "total_fines": self.total_fines,
            "undetected_evasion_rate": self.undetected_evasion_rate,
            "enforcement_cost": self.enforcement_cost,
            "bunching_intensity": self.bunching_intensity,
            "collusion_events_detected": self.collusion_events_detected,
            "collusion_suspicion_mean": self.collusion_suspicion_mean,
            "exploit_frequency": self.exploit_frequency,
            "governance_backfire_events": self.governance_backfire_events,
            "variance_amplification": self.variance_amplification,
        }


def compute_gini(incomes: List[float]) -> float:
    """Compute Gini coefficient from a list of incomes."""
    n = len(incomes)
    if n == 0:
        return 0.0
    total = sum(incomes)
    if total <= 0:
        return 0.0
    sorted_inc = sorted(incomes)
    cumulative = 0.0
    gini_sum = 0.0
    for inc in sorted_inc:
        cumulative += inc
        gini_sum += cumulative
    return max(0.0, min(1.0, 1.0 - 2.0 * gini_sum / (n * total) + 1.0 / n))


def compute_atkinson(incomes: List[float], epsilon: float = 0.5) -> float:
    """Compute Atkinson index with given inequality aversion parameter."""
    n = len(incomes)
    if n == 0:
        return 0.0
    mean_inc = sum(incomes) / n
    if mean_inc <= 0:
        return 0.0

    if abs(epsilon - 1.0) < 1e-9:
        # Logarithmic case
        log_sum = sum(math.log(max(inc, 1e-9)) for inc in incomes) / n
        return 1.0 - math.exp(log_sum) / mean_inc
    else:
        power = 1.0 - epsilon
        powered_sum = sum(max(inc, 1e-9) ** power for inc in incomes) / n
        return float(max(0.0, 1.0 - (powered_sum ** (1.0 / power)) / mean_inc))


def compute_bunching_intensity(
    incomes: List[float],
    thresholds: List[float],
    bin_width: float = 1.0,
) -> float:
    """Compute fraction of incomes bunched near bracket thresholds.

    An income is "bunched" if it falls within [threshold - bin_width, threshold].
    """
    if not incomes or not thresholds:
        return 0.0

    n_bunched = 0
    for inc in incomes:
        for thr in thresholds:
            if thr - bin_width <= inc <= thr:
                n_bunched += 1
                break  # only count once per income

    return n_bunched / len(incomes)


def compute_gtb_metrics(
    workers: Dict[str, WorkerState],
    events: List[GTBEvent],
    epoch: int,
    bracket_thresholds: List[float],
    *,
    prod_weight: float = 1.0,
    ineq_weight: float = 0.5,
    per_audit_cost: float = 0.5,
    bin_width: float = 1.0,
) -> GTBMetrics:
    """Compute all metrics for one epoch.

    Args:
        workers: Current worker states.
        events: Events from this epoch.
        epoch: Epoch number.
        bracket_thresholds: Tax bracket thresholds for bunching analysis.
        prod_weight: Weight on productivity in welfare.
        ineq_weight: Weight on inequality in welfare.
        per_audit_cost: Cost per audit for enforcement_cost metric.
        bin_width: Bin width for bunching detection.

    Returns:
        Populated GTBMetrics.
    """
    incomes = [w.gross_income_this_epoch for w in workers.values()]
    n = len(incomes) or 1

    total_prod = sum(incomes)
    mean_prod = total_prod / n

    gini = compute_gini(incomes)
    atkinson = compute_atkinson(incomes)

    welfare = prod_weight * mean_prod - ineq_weight * gini

    # Tax
    total_tax = sum(w.tax_paid_this_epoch for w in workers.values())
    # Mean-of-means effective rate: average of individual per-worker rates,
    # NOT the aggregate rate (total_tax / total_reported_income).
    # These diverge when worker incomes are heterogeneous.
    mean_eff_rate = (
        sum(
            w.tax_paid_this_epoch / max(w.reported_income_this_epoch, 1e-9)
            for w in workers.values()
        ) / n
    )

    # Enforcement
    audit_events = [e for e in events if e.event_type in ("audit_caught", "audit_miss", "audit_false_positive")]
    catches = [e for e in events if e.event_type == "audit_caught"]
    misses = [e for e in events if e.event_type == "audit_miss"]
    total_evasion_attempts = len(catches) + len(misses)
    undetected_rate = len(misses) / max(total_evasion_attempts, 1)
    total_fines = sum(e.details.get("fine", 0.0) for e in catches)

    # Bunching
    bunching = compute_bunching_intensity(incomes, bracket_thresholds, bin_width)

    # Collusion
    collusion_events = [e for e in events if e.event_type == "collusion_detected"]
    suspicion_scores = [
        e.details.get("suspicion_score", 0.0) for e in collusion_events
    ]
    mean_suspicion = (
        sum(suspicion_scores) / len(suspicion_scores) if suspicion_scores else 0.0
    )

    # SWARM systemic-risk metrics
    exploit_events = [
        e for e in events
        if e.event_type in ("misreport", "collusion_detected", "shift_income")
    ]
    total_events = len(events) or 1
    exploit_freq = len(exploit_events) / total_events

    # Variance amplification
    if mean_prod > 0:
        var = sum((inc - mean_prod) ** 2 for inc in incomes) / n
        variance_amp = math.sqrt(var) / mean_prod
    else:
        variance_amp = 0.0

    total_houses = sum(w.houses_built for w in workers.values())

    return GTBMetrics(
        epoch=epoch,
        total_production=total_prod,
        mean_production=mean_prod,
        total_houses_built=total_houses,
        total_tax_revenue=total_tax,
        mean_effective_tax_rate=mean_eff_rate,
        gini_coefficient=gini,
        atkinson_index=atkinson,
        welfare=welfare,
        total_audits=len(audit_events),
        total_catches=len(catches),
        total_fines=total_fines,
        undetected_evasion_rate=undetected_rate,
        enforcement_cost=len(audit_events) * per_audit_cost,
        bunching_intensity=bunching,
        collusion_events_detected=len(collusion_events),
        collusion_suspicion_mean=mean_suspicion,
        exploit_frequency=exploit_freq,
        governance_backfire_events=sum(
            1 for e in events if e.event_type == "audit_false_positive"
        ),
        variance_amplification=variance_amp,
    )
