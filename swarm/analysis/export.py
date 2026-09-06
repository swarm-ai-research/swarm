"""Export utilities for simulation results.

Supports export to:
- CSV (pandas DataFrame)
- Parquet (columnar format)
- JSON (full history)
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from swarm.analysis.aggregation import (
    AgentSnapshot,
    EpochSnapshot,
    SimulationHistory,
)


def history_to_epoch_records(
    history: SimulationHistory,
) -> List[Dict[str, Any]]:
    """
    Convert history to list of epoch records for tabular export.

    Args:
        history: Simulation history

    Returns:
        List of dicts, one per epoch
    """
    records = []

    for snapshot in history.epoch_snapshots:
        record = {
            "simulation_id": history.simulation_id,
            "epoch": snapshot.epoch,
            "timestamp": snapshot.timestamp.isoformat() if snapshot.timestamp else None,
            "total_interactions": snapshot.total_interactions,
            "accepted_interactions": snapshot.accepted_interactions,
            "rejected_interactions": snapshot.rejected_interactions,
            "toxicity_rate": snapshot.toxicity_rate,
            "quality_gap": snapshot.quality_gap,
            "avg_p": snapshot.avg_p,
            "total_welfare": snapshot.total_welfare,
            "avg_payoff": snapshot.avg_payoff,
            "payoff_std": snapshot.payoff_std,
            "gini_coefficient": snapshot.gini_coefficient,
            "total_posts": snapshot.total_posts,
            "total_votes": snapshot.total_votes,
            "total_tasks_completed": snapshot.total_tasks_completed,
            "n_agents": snapshot.n_agents,
            "n_frozen": snapshot.n_frozen,
            "n_quarantined": snapshot.n_quarantined,
            "avg_reputation": snapshot.avg_reputation,
            "reputation_std": snapshot.reputation_std,
            "n_edges": snapshot.n_edges,
            "avg_degree": snapshot.avg_degree,
            "avg_clustering": snapshot.avg_clustering,
            "n_components": snapshot.n_components,
            "ecosystem_threat_level": snapshot.ecosystem_threat_level,
            "active_threats": snapshot.active_threats,
            "contagion_depth": snapshot.contagion_depth,
            "ecosystem_collusion_risk": snapshot.ecosystem_collusion_risk,
            "n_flagged_pairs": snapshot.n_flagged_pairs,
            "avg_coordination_score": snapshot.avg_coordination_score,
            "avg_synergy_score": snapshot.avg_synergy_score,
            "tasks_completed": snapshot.tasks_completed,
            "followup_side_write_fraction": snapshot.followup_side_write_fraction,
        }
        records.append(record)

    return records


def history_to_agent_records(
    history: SimulationHistory,
) -> List[Dict[str, Any]]:
    """
    Convert history to list of agent-epoch records for tabular export.

    Args:
        history: Simulation history

    Returns:
        List of dicts, one per agent per epoch
    """
    records = []

    for agent_id, snapshots in history.agent_snapshots.items():
        for snapshot in snapshots:
            record = {
                "simulation_id": history.simulation_id,
                "agent_id": agent_id,
                "name": snapshot.name,
                "agent_type": snapshot.agent_type,
                "epoch": snapshot.epoch,
                "reputation": snapshot.reputation,
                "resources": snapshot.resources,
                "interactions_initiated": snapshot.interactions_initiated,
                "interactions_received": snapshot.interactions_received,
                "avg_p_initiated": snapshot.avg_p_initiated,
                "avg_p_received": snapshot.avg_p_received,
                "total_payoff": snapshot.total_payoff,
                "is_frozen": snapshot.is_frozen,
                "is_quarantined": snapshot.is_quarantined,
            }
            records.append(record)

    return records


def history_to_interaction_records(
    history: SimulationHistory,
) -> List[Dict[str, Any]]:
    """Convert interaction history to list of records for export."""
    records = []
    for interaction in history.interactions:
        record = {
            "event_type": "interaction",
            "interaction_id": interaction.interaction_id,
            "timestamp": interaction.timestamp.isoformat(),
            "epoch": interaction.metadata.get('epoch', 0),
            "step": interaction.metadata.get('step', 0),
            "initiator": interaction.initiator,
            "counterparty": interaction.counterparty,
            "interaction_type": interaction.interaction_type.value,
            "accepted": interaction.accepted,
            "p": interaction.p,
            "v_hat": interaction.v_hat,
            # None for ordinary runs; ±1 in calibration-injection scenarios
            # (arm A promoted, beads c89o) — consumed by
            # experiments/calibration_fidelity.py --from-run.
            "ground_truth": interaction.ground_truth,
        }
        records.append(record)
    return records


def export_to_csv(
    history: SimulationHistory,
    output_dir: Union[str, Path],
    prefix: str = "simulation",
) -> Dict[str, Path]:
    """
    Export simulation history to CSV files.

    Creates two files:
    - {prefix}_epochs.csv: Epoch-level metrics
    - {prefix}_agents.csv: Agent-level metrics

    Args:
        history: Simulation history
        output_dir: Output directory
        prefix: Filename prefix

    Returns:
        Dict mapping file type to Path
    """
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError:
        pd = None  # type: ignore[assignment]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Export epoch data
    epoch_records = history_to_epoch_records(history)
    if epoch_records:
        epochs_path = output_dir / f"{prefix}_epochs.csv"
        if pd is not None:
            pd.DataFrame(epoch_records).to_csv(epochs_path, index=False)
        else:
            _write_csv_records(epoch_records, epochs_path)
        paths["epochs"] = epochs_path

    # Export agent data
    agent_records = history_to_agent_records(history)
    if agent_records:
        agents_path = output_dir / f"{prefix}_agents.csv"
        if pd is not None:
            pd.DataFrame(agent_records).to_csv(agents_path, index=False)
        else:
            _write_csv_records(agent_records, agents_path)
        paths["agents"] = agents_path

    return paths


def _write_csv_records(records: List[Dict[str, Any]], path: Path) -> None:
    """Write records to CSV using the stdlib writer."""
    if not records:
        return

    fieldnames: List[str] = []
    seen = set()
    for record in records:
        for key in record.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def export_to_parquet(
    history: SimulationHistory,
    output_dir: Union[str, Path],
    prefix: str = "simulation",
) -> Dict[str, Path]:
    """
    Export simulation history to Parquet files.

    Args:
        history: Simulation history
        output_dir: Output directory
        prefix: Filename prefix

    Returns:
        Dict mapping file type to Path
    """
    try:
        import pandas as pd
    except ImportError as err:
        raise ImportError(
            "pandas required for Parquet export. Install with: python -m pip install pandas pyarrow"
        ) from err

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Export epoch data
    epoch_records = history_to_epoch_records(history)
    if epoch_records:
        epochs_path = output_dir / f"{prefix}_epochs.parquet"
        pd.DataFrame(epoch_records).to_parquet(epochs_path, index=False)
        paths["epochs"] = epochs_path

    # Export agent data
    agent_records = history_to_agent_records(history)
    if agent_records:
        agents_path = output_dir / f"{prefix}_agents.parquet"
        pd.DataFrame(agent_records).to_parquet(agents_path, index=False)
        paths["agents"] = agents_path

    return paths


def export_to_json(
    history: SimulationHistory,
    output_path: Union[str, Path],
    indent: int = 2,
    include_events: bool = False,
) -> Path:
    """
    Export full simulation history to JSON.

    Args:
        history: Simulation history
        output_path: Output file path
        indent: JSON indentation
        include_events: If True, include per-interaction events under the
            ``interaction_events`` key

    Returns:
        Output path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "simulation_id": history.simulation_id,
        "started_at": history.started_at.isoformat() if history.started_at else None,
        "ended_at": history.ended_at.isoformat() if history.ended_at else None,
        "n_epochs": history.n_epochs,
        "steps_per_epoch": history.steps_per_epoch,
        "n_agents": history.n_agents,
        "seed": history.seed,
        "epoch_snapshots": history_to_epoch_records(history),
        "agent_snapshots": history_to_agent_records(history),
    }

    if include_events and history.interactions:
        data["events"] = history_to_interaction_records(history)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=indent, default=str)

    return output_path


def load_from_json(
    input_path: Union[str, Path],
) -> SimulationHistory:
    """
    Load simulation history from JSON.

    Args:
        input_path: Input file path

    Returns:
        SimulationHistory object
    """
    input_path = Path(input_path)

    with open(input_path) as f:
        data = json.load(f)

    history = SimulationHistory(
        simulation_id=data.get("simulation_id", ""),
        started_at=datetime.fromisoformat(data["started_at"])
        if data.get("started_at")
        else None,
        ended_at=datetime.fromisoformat(data["ended_at"])
        if data.get("ended_at")
        else None,
        n_epochs=data.get("n_epochs", 0),
        steps_per_epoch=data.get("steps_per_epoch", 0),
        n_agents=data.get("n_agents", 0),
        seed=data.get("seed"),
    )

    # Load epoch snapshots
    for record in data.get("epoch_snapshots", []):
        snapshot = EpochSnapshot(
            epoch=record["epoch"],
            timestamp=datetime.fromisoformat(record["timestamp"])
            if record.get("timestamp")
            else datetime.now(),
            total_interactions=record.get("total_interactions", 0),
            accepted_interactions=record.get("accepted_interactions", 0),
            rejected_interactions=record.get("rejected_interactions", 0),
            toxicity_rate=record.get("toxicity_rate", 0.0),
            quality_gap=record.get("quality_gap", 0.0),
            avg_p=record.get("avg_p", 0.5),
            total_welfare=record.get("total_welfare", 0.0),
            avg_payoff=record.get("avg_payoff", 0.0),
            payoff_std=record.get("payoff_std", 0.0),
            gini_coefficient=record.get("gini_coefficient", 0.0),
            total_posts=record.get("total_posts", 0),
            total_votes=record.get("total_votes", 0),
            total_tasks_completed=record.get("total_tasks_completed", 0),
            n_agents=record.get("n_agents", 0),
            n_frozen=record.get("n_frozen", 0),
            n_quarantined=record.get("n_quarantined", 0),
            avg_reputation=record.get("avg_reputation", 0.0),
            reputation_std=record.get("reputation_std", 0.0),
            n_edges=record.get("n_edges", 0),
            avg_degree=record.get("avg_degree", 0.0),
            avg_clustering=record.get("avg_clustering", 0.0),
            n_components=record.get("n_components", 1),
            ecosystem_threat_level=record.get("ecosystem_threat_level", 0.0),
            active_threats=record.get("active_threats", 0),
            contagion_depth=record.get("contagion_depth", 0),
            ecosystem_collusion_risk=record.get("ecosystem_collusion_risk", 0.0),
            n_flagged_pairs=record.get("n_flagged_pairs", 0),
            avg_coordination_score=record.get("avg_coordination_score", 0.0),
            avg_synergy_score=record.get("avg_synergy_score", 0.0),
            tasks_completed=record.get("tasks_completed", 0),
        )
        history.add_epoch_snapshot(snapshot)

    # Load agent snapshots
    for record in data.get("agent_snapshots", []):
        agent_snap = AgentSnapshot(
            agent_id=record["agent_id"],
            epoch=record["epoch"],
            name=record.get("name"),
            agent_type=record.get("agent_type"),
            reputation=record.get("reputation", 0.0),
            resources=record.get("resources", 100.0),
            interactions_initiated=record.get("interactions_initiated", 0),
            interactions_received=record.get("interactions_received", 0),
            avg_p_initiated=record.get("avg_p_initiated", 0.5),
            avg_p_received=record.get("avg_p_received", 0.5),
            total_payoff=record.get("total_payoff", 0.0),
            is_frozen=record.get("is_frozen", False),
            is_quarantined=record.get("is_quarantined", False),
        )
        history.add_agent_snapshot(agent_snap)

    return history


def load_from_csv(
    epochs_path: Union[str, Path],
    agents_path: Optional[Union[str, Path]] = None,
) -> SimulationHistory:
    """
    Load simulation history from CSV files.

    Args:
        epochs_path: Path to epochs CSV
        agents_path: Optional path to agents CSV

    Returns:
        SimulationHistory object
    """
    try:
        import pandas as pd
    except ImportError as err:
        raise ImportError(
            "pandas required for CSV import. Install with: python -m pip install pandas"
        ) from err

    epochs_path = Path(epochs_path)
    epochs_df = pd.read_csv(epochs_path)

    history = SimulationHistory()

    # Infer metadata from data
    if len(epochs_df) > 0:
        history.simulation_id = (
            epochs_df["simulation_id"].iloc[0]
            if "simulation_id" in epochs_df.columns
            else ""
        )
        history.n_epochs = len(epochs_df)

    # Load epoch snapshots
    for _, row in epochs_df.iterrows():
        snapshot = EpochSnapshot(
            epoch=int(row.get("epoch", 0)),
            total_interactions=int(row.get("total_interactions", 0)),
            accepted_interactions=int(row.get("accepted_interactions", 0)),
            rejected_interactions=int(row.get("rejected_interactions", 0)),
            toxicity_rate=float(row.get("toxicity_rate", 0.0)),
            quality_gap=float(row.get("quality_gap", 0.0)),
            avg_p=float(row.get("avg_p", 0.5)),
            total_welfare=float(row.get("total_welfare", 0.0)),
            avg_payoff=float(row.get("avg_payoff", 0.0)),
            n_agents=int(row.get("n_agents", 0)),
            avg_reputation=float(row.get("avg_reputation", 0.0)),
        )
        history.add_epoch_snapshot(snapshot)

    # Load agent snapshots if provided
    if agents_path:
        agents_path = Path(agents_path)
        agents_df = pd.read_csv(agents_path)

        for _, row in agents_df.iterrows():
            name = row.get("name") if "name" in row else None
            if name is not None and pd.isna(name):
                name = None
            agent_snap = AgentSnapshot(
                agent_id=str(row["agent_id"]),
                epoch=int(row.get("epoch", 0)),
                name=str(name) if name is not None else None,
                agent_type=str(row.get("agent_type")) if "agent_type" in row and not pd.isna(row.get("agent_type")) else None,
                reputation=float(row.get("reputation", 0.0)),
                resources=float(row.get("resources", 100.0)),
                total_payoff=float(row.get("total_payoff", 0.0)),
            )
            history.add_agent_snapshot(agent_snap)

    return history


def generate_summary_report(
    history: SimulationHistory,
) -> str:
    """
    Generate a text summary report of the simulation.

    Args:
        history: Simulation history

    Returns:
        Formatted text report
    """
    lines = [
        "=" * 60,
        "SIMULATION SUMMARY REPORT",
        "=" * 60,
        "",
        f"Simulation ID: {history.simulation_id}",
        f"Started: {history.started_at}",
        f"Ended: {history.ended_at}",
        f"Duration: {(history.ended_at - history.started_at) if history.started_at and history.ended_at else 'N/A'}",
        "",
        "Configuration:",
        f"  Epochs: {history.n_epochs}",
        f"  Steps per epoch: {history.steps_per_epoch}",
        f"  Agents: {history.n_agents}",
        f"  Seed: {history.seed}",
        "",
    ]

    if history.epoch_snapshots:
        final = history.epoch_snapshots[-1]
        first = history.epoch_snapshots[0]

        lines.extend(
            [
                "Final Metrics:",
                f"  Toxicity Rate: {final.toxicity_rate:.4f}",
                f"  Quality Gap: {final.quality_gap:.4f}",
                f"  Total Welfare: {final.total_welfare:.2f}",
                f"  Avg Payoff: {final.avg_payoff:.2f}",
                f"  Gini Coefficient: {final.gini_coefficient:.4f}",
                f"  Ecosystem Threat Level: {final.ecosystem_threat_level:.4f}",
                f"  Collusion Risk: {final.ecosystem_collusion_risk:.4f}",
                "",
                "Changes from Start to End:",
                f"  Toxicity: {first.toxicity_rate:.4f} -> {final.toxicity_rate:.4f}",
                f"  Welfare: {first.total_welfare:.2f} -> {final.total_welfare:.2f}",
                f"  Avg Reputation: {first.avg_reputation:.2f} -> {final.avg_reputation:.2f}",
                "",
            ]
        )

        # Aggregate stats
        total_interactions = sum(s.total_interactions for s in history.epoch_snapshots)
        total_accepted = sum(s.accepted_interactions for s in history.epoch_snapshots)
        acceptance_rate = (
            total_accepted / total_interactions if total_interactions > 0 else 0
        )

        toxicity_values = [s.toxicity_rate for s in history.epoch_snapshots]
        welfare_values = [s.total_welfare for s in history.epoch_snapshots]

        lines.extend(
            [
                "Aggregate Statistics:",
                f"  Total Interactions: {total_interactions}",
                f"  Acceptance Rate: {acceptance_rate:.2%}",
                f"  Avg Toxicity: {np.mean(toxicity_values):.4f} (std: {np.std(toxicity_values):.4f})",
                f"  Avg Welfare: {np.mean(welfare_values):.2f} (std: {np.std(welfare_values):.2f})",
                "",
            ]
        )

    # Agent summary
    final_states = history.get_final_agent_states()
    if final_states:
        lines.append("Agent Summary:")
        for agent_id, state in sorted(final_states.items()):
            status = (
                "FROZEN"
                if state.is_frozen
                else ("QUARANTINED" if state.is_quarantined else "active")
            )
            lines.append(
                f"  {agent_id}: rep={state.reputation:.2f}, "
                f"payoff={state.total_payoff:.2f}, "
                f"status={status}"
            )
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)
