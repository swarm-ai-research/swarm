"""Synthetic finance task catalog for the Team-of-Rivals scenario.

Each task has a deterministic ground truth and a list of traps that
critics should catch. Tasks span four types: reconciliation, trend
analysis, KPI dashboard, and anomaly triage.
"""

from typing import Any, Dict, List

from pydantic import BaseModel


class RivalsTask(BaseModel):
    """A single task in the rivals pipeline."""

    task_id: str
    task_type: str  # reconciliation, trend, kpi, anomaly
    description: str
    ground_truth: Dict[str, Any]
    traps: List[str]
    difficulty: float = 0.5  # 0.0 (easy) to 1.0 (hard)


# ---------------------------------------------------------------------------
# Task catalog (~16 tasks across 4 types)
# ---------------------------------------------------------------------------

TASK_CATALOG: List[RivalsTask] = [
    # --- Reconciliation tasks ---
    RivalsTask(
        task_id="recon_01",
        task_type="reconciliation",
        description="Reconcile Q3 ledger: EUR invoices vs USD payments",
        ground_truth={"mismatch_count": 3, "currency": "EUR", "net_delta": -1200.50},
        traps=["unit_conversion_eur_usd"],
        difficulty=0.4,
    ),
    RivalsTask(
        task_id="recon_02",
        task_type="reconciliation",
        description="Reconcile intercompany transfers across 3 subsidiaries",
        ground_truth={"mismatch_count": 5, "currency": "USD", "net_delta": 450.00},
        traps=["unit_conversion_eur_usd", "rounding_error"],
        difficulty=0.6,
    ),
    RivalsTask(
        task_id="recon_03",
        task_type="reconciliation",
        description="Reconcile payroll deductions against benefit enrollments",
        ground_truth={"mismatch_count": 2, "currency": "USD", "net_delta": -89.30},
        traps=["rounding_error"],
        difficulty=0.3,
    ),
    RivalsTask(
        task_id="recon_04",
        task_type="reconciliation",
        description="Reconcile trade settlement dates across time zones",
        ground_truth={"mismatch_count": 4, "currency": "USD", "net_delta": 0.0},
        traps=["timezone_off_by_one", "unit_conversion_eur_usd"],
        difficulty=0.7,
    ),
    # --- Trend analysis tasks ---
    RivalsTask(
        task_id="trend_01",
        task_type="trend",
        description="Identify revenue trend direction for Q1-Q4",
        ground_truth={"direction": "up", "magnitude": 0.12, "inflection_q": "Q3"},
        traps=["sign_error"],
        difficulty=0.3,
    ),
    RivalsTask(
        task_id="trend_02",
        task_type="trend",
        description="Analyze monthly churn rate trajectory",
        ground_truth={"direction": "down", "magnitude": 0.05, "inflection_q": "Q2"},
        traps=["sign_error", "base_rate_confusion"],
        difficulty=0.5,
    ),
    RivalsTask(
        task_id="trend_03",
        task_type="trend",
        description="Project operating margin trend with seasonal adjustment",
        ground_truth={"direction": "up", "magnitude": 0.08, "inflection_q": "Q4"},
        traps=["sign_error", "seasonal_artifact"],
        difficulty=0.7,
    ),
    RivalsTask(
        task_id="trend_04",
        task_type="trend",
        description="Detect regime change in daily transaction volumes",
        ground_truth={"direction": "up", "magnitude": 0.22, "inflection_q": "Q2"},
        traps=["base_rate_confusion"],
        difficulty=0.6,
    ),
    # --- KPI dashboard tasks ---
    RivalsTask(
        task_id="kpi_01",
        task_type="kpi",
        description="Build executive KPI dashboard with revenue and margin",
        ground_truth={"revenue": 4200000, "margin": 0.23, "chart_type": "bar"},
        traps=["misleading_axis"],
        difficulty=0.4,
    ),
    RivalsTask(
        task_id="kpi_02",
        task_type="kpi",
        description="Create customer satisfaction scorecard with NPS trend",
        ground_truth={"nps": 42, "csat": 0.87, "chart_type": "line"},
        traps=["misleading_axis", "truncated_y_axis"],
        difficulty=0.5,
    ),
    RivalsTask(
        task_id="kpi_03",
        task_type="kpi",
        description="Design operational efficiency dashboard",
        ground_truth={"utilization": 0.78, "throughput": 1200, "chart_type": "combo"},
        traps=["misleading_axis"],
        difficulty=0.5,
    ),
    RivalsTask(
        task_id="kpi_04",
        task_type="kpi",
        description="Build risk exposure dashboard with VaR metrics",
        ground_truth={"var_95": 150000, "var_99": 280000, "chart_type": "histogram"},
        traps=["truncated_y_axis", "misleading_axis"],
        difficulty=0.7,
    ),
    # --- Anomaly triage tasks ---
    RivalsTask(
        task_id="anomaly_01",
        task_type="anomaly",
        description="Triage unusual spikes in wire transfer volume",
        ground_truth={"anomaly_count": 3, "severity": "high", "root_cause": "batch_error"},
        traps=["compound_false_positive"],
        difficulty=0.5,
    ),
    RivalsTask(
        task_id="anomaly_02",
        task_type="anomaly",
        description="Investigate after-hours trading activity anomalies",
        ground_truth={"anomaly_count": 2, "severity": "medium", "root_cause": "timezone"},
        traps=["compound_false_positive", "benign_outlier"],
        difficulty=0.6,
    ),
    RivalsTask(
        task_id="anomaly_03",
        task_type="anomaly",
        description="Detect anomalous patterns in expense report submissions",
        ground_truth={"anomaly_count": 5, "severity": "low", "root_cause": "policy_change"},
        traps=["benign_outlier"],
        difficulty=0.4,
    ),
    RivalsTask(
        task_id="anomaly_04",
        task_type="anomaly",
        description="Triage credit limit breach alerts across portfolios",
        ground_truth={"anomaly_count": 1, "severity": "critical", "root_cause": "data_feed"},
        traps=["compound_false_positive", "benign_outlier"],
        difficulty=0.8,
    ),
]

# Index by task_id for quick lookup
TASK_INDEX: Dict[str, RivalsTask] = {t.task_id: t for t in TASK_CATALOG}


def sample_tasks(
    rng: Any,
    count: int,
) -> List[RivalsTask]:
    """Sample tasks from the catalog.

    Args:
        rng: Random instance for reproducibility.
        count: Number of tasks to sample.

    Returns:
        List of sampled tasks (with replacement if count > catalog size).
    """
    return [rng.choice(TASK_CATALOG) for _ in range(count)]
