#!/usr/bin/env python
"""
Log a completed SWARM run to the SQLite runs database.

Backs the /log_run slash command.  Reads history.json (or CSV fallback),
extracts summary metrics, inserts into runs/runs.db, and optionally
exports to Dolt.

Usage:
    python -m swarm.scripts.log_run runs/20260214-133334_team_of_rivals_seed42
    python -m swarm.scripts.log_run <run_dir> --notes "increased rep decay"
    python -m swarm.scripts.log_run <run_dir> --external-run-id "pi-job-123"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

# Agent types treated as adversarial when computing ``adversarial_fraction``.
# Sourced from swarm.models.agent.AgentType — kept as a string set here so this
# script can run on bare history.json files without importing the simulation
# code (and without pulling in heavy deps like pydantic for a logging script).
# Cross-checked against:
#   - swarm/models/agent.py (AgentType enum)
#   - swarm/redteam/metrics.py:321 (canonical predicate)
#   - swarm/core/composition_analyzer.py:393 (canonical counting site)
ADVERSARIAL_AGENT_TYPES = frozenset({"adversarial", "deceptive"})


def _compute_adversarial_fraction(agent_snapshots: list) -> float:
    """Compute the fraction of unique agents whose type is adversarial.

    Args:
        agent_snapshots: List of per-agent-per-epoch records as serialized by
            ``swarm.analysis.export.history_to_agent_records`` (i.e. each
            record has an ``agent_id`` and an ``agent_type`` field).

    Returns:
        Fraction in ``[0.0, 1.0]``. Returns ``0.0`` if ``agent_snapshots`` is
        empty or contains no usable ``agent_id`` / ``agent_type`` fields, so
        that older history.json files (which may pre-date the per-agent type
        export) still produce a sensible default.

    Notes:
        Each agent appears once per epoch in the snapshots, so we deduplicate
        by ``agent_id`` and take the first observed ``agent_type``. Agent type
        in SWARM is immutable for the lifetime of a run.
    """
    if not agent_snapshots:
        return 0.0

    agent_type_by_id: dict[str, str] = {}
    for record in agent_snapshots:
        agent_id = record.get("agent_id")
        if not agent_id or agent_id in agent_type_by_id:
            continue
        agent_type = record.get("agent_type")
        if agent_type is None:
            continue
        agent_type_by_id[agent_id] = str(agent_type).lower()

    n_total = len(agent_type_by_id)
    if n_total == 0:
        return 0.0

    n_adversarial = sum(
        1 for t in agent_type_by_id.values() if t in ADVERSARIAL_AGENT_TYPES
    )
    return round(n_adversarial / n_total, 4)


def _parse_dir_name(run_dir: Path) -> dict:
    """Extract scenario_id, run_timestamp, and seed from directory name.

    Expected format: <YYYYMMDD-HHMMSS>_<scenario_id>_seed<seed>
    Falls back gracefully if the name doesn't match.
    """
    name = run_dir.name
    result: dict = {}

    # Timestamp
    m = re.match(r"(\d{8}-\d{6})_", name)
    if m:
        result["run_timestamp"] = m.group(1)
        rest = name[len(m.group(0)):]
    else:
        result["run_timestamp"] = None
        rest = name

    # Seed suffix
    seed_m = re.search(r"_seed(\d+)$", rest)
    if seed_m:
        result["seed"] = int(seed_m.group(1))
        rest = rest[: seed_m.start()]
    else:
        result["seed"] = None

    result["scenario_id"] = rest or "unknown"
    return result


def extract_from_history(history_path: Path) -> dict:
    """Extract metrics from a history.json file."""
    with open(history_path, encoding="utf-8") as f:
        h = json.load(f)

    epochs = h.get("epoch_snapshots", [])
    agent_snapshots = h.get("agent_snapshots", [])
    n_epochs = len(epochs)

    total_interactions = sum(e["total_interactions"] for e in epochs)
    accepted_interactions = sum(e["accepted_interactions"] for e in epochs)
    acceptance_rate = (
        round(accepted_interactions / total_interactions, 4)
        if total_interactions
        else 0.0
    )
    avg_toxicity = (
        round(sum(e["toxicity_rate"] for e in epochs) / n_epochs, 4)
        if n_epochs
        else 0.0
    )
    final_welfare = round(epochs[-1]["total_welfare"], 4) if epochs else 0.0
    total_welfare = round(sum(e["total_welfare"] for e in epochs), 4)
    welfare_per_epoch = round(total_welfare / n_epochs, 4) if n_epochs else 0.0

    # Collapse detection
    collapse_epoch = None
    for i, e in enumerate(epochs):
        if e["total_welfare"] <= 0:
            if all(epochs[j]["total_welfare"] <= 0 for j in range(i, n_epochs)):
                collapse_epoch = i
                break

    return {
        "scenario_id": h.get("simulation_id", "unknown"),
        "seed": h.get("seed"),
        "n_agents": h.get("n_agents", 0),
        "n_epochs": n_epochs,
        "steps_per_epoch": h.get("steps_per_epoch", 0),
        "total_interactions": total_interactions,
        "accepted_interactions": accepted_interactions,
        "acceptance_rate": acceptance_rate,
        "avg_toxicity": avg_toxicity,
        "final_welfare": final_welfare,
        "total_welfare": total_welfare,
        "welfare_per_epoch": welfare_per_epoch,
        "adversarial_fraction": _compute_adversarial_fraction(agent_snapshots),
        "collapse_epoch": collapse_epoch,
    }


def _adversarial_fraction_from_agents_csv(csv_dir: Path) -> float:
    """Compute adversarial fraction from the ``*agents*.csv`` sidecar.

    Mirrors :func:`_compute_adversarial_fraction` but operates on the CSV
    export written by ``swarm.analysis.export.export_to_csv``. Returns
    ``0.0`` when no agents CSV is present (e.g. legacy runs) so the caller
    can transparently fall back.
    """
    agents_csvs = sorted(csv_dir.glob("*agents*.csv"))
    if not agents_csvs:
        return 0.0
    with open(agents_csvs[0], encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return _compute_adversarial_fraction(rows)


def extract_from_csv(csv_dir: Path) -> dict:
    """Fallback: extract metrics from epoch CSV files."""
    csv_files = sorted(csv_dir.glob("*epochs*.csv"))
    if not csv_files:
        csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    with open(csv_files[0], encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n_epochs = len(rows)
    total_interactions = sum(int(float(r.get("total_interactions", 0))) for r in rows)
    accepted_interactions = sum(
        int(float(r.get("accepted_interactions", 0))) for r in rows
    )
    acceptance_rate = (
        round(accepted_interactions / total_interactions, 4)
        if total_interactions
        else 0.0
    )
    avg_toxicity = (
        round(
            sum(float(r.get("toxicity_rate", 0)) for r in rows) / n_epochs, 4
        )
        if n_epochs
        else 0.0
    )
    final_welfare = round(float(rows[-1].get("total_welfare", 0)), 4) if rows else 0.0
    total_welfare = round(sum(float(r.get("total_welfare", 0)) for r in rows), 4)
    welfare_per_epoch = round(total_welfare / n_epochs, 4) if n_epochs else 0.0

    collapse_epoch = None
    for i, r in enumerate(rows):
        if float(r.get("total_welfare", 1)) <= 0:
            if all(
                float(rows[j].get("total_welfare", 1)) <= 0
                for j in range(i, n_epochs)
            ):
                collapse_epoch = i
                break

    return {
        "scenario_id": "unknown",
        "seed": None,
        "n_agents": int(float(rows[0].get("n_agents", 0))) if rows else 0,
        "n_epochs": n_epochs,
        "steps_per_epoch": 0,
        "total_interactions": total_interactions,
        "accepted_interactions": accepted_interactions,
        "acceptance_rate": acceptance_rate,
        "avg_toxicity": avg_toxicity,
        "final_welfare": final_welfare,
        "total_welfare": total_welfare,
        "welfare_per_epoch": welfare_per_epoch,
        "adversarial_fraction": _adversarial_fraction_from_agents_csv(csv_dir),
        "collapse_epoch": collapse_epoch,
    }


# ---------------------------------------------------------------------------
# SQLite operations
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS scenario_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario_id TEXT NOT NULL,
    run_timestamp TEXT,
    seed INTEGER,
    n_agents INTEGER,
    n_epochs INTEGER,
    steps_per_epoch INTEGER,
    total_interactions INTEGER,
    accepted_interactions INTEGER,
    acceptance_rate REAL,
    avg_toxicity REAL,
    final_welfare REAL,
    total_welfare REAL,
    welfare_per_epoch REAL,
    adversarial_fraction REAL,
    collapse_epoch INTEGER,
    external_run_id TEXT,
    notes TEXT,
    logged_at TEXT DEFAULT (datetime('now'))
)"""

INSERT_SQL = """\
INSERT INTO scenario_runs
    (scenario_id, run_timestamp, seed, n_agents, n_epochs, steps_per_epoch,
     total_interactions, accepted_interactions, acceptance_rate, avg_toxicity,
     final_welfare, total_welfare, welfare_per_epoch, adversarial_fraction,
     collapse_epoch, external_run_id, notes)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""


def log_to_sqlite(
    summary: dict,
    db_path: Path,
) -> int | None:
    """Insert summary row into SQLite. Returns row id or None if duplicate."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(CREATE_TABLE_SQL)

    # Duplicate check
    cur.execute(
        "SELECT id FROM scenario_runs "
        "WHERE scenario_id=? AND seed=? AND run_timestamp=?",
        (summary["scenario_id"], summary["seed"], summary["run_timestamp"]),
    )
    existing = cur.fetchone()
    if existing:
        print(f"WARNING: Duplicate row (id={existing[0]}). Skipping insert.")
        conn.close()
        return None

    vals = (
        summary["scenario_id"],
        summary["run_timestamp"],
        summary["seed"],
        summary["n_agents"],
        summary["n_epochs"],
        summary["steps_per_epoch"],
        summary["total_interactions"],
        summary["accepted_interactions"],
        summary["acceptance_rate"],
        summary["avg_toxicity"],
        summary["final_welfare"],
        summary["total_welfare"],
        summary["welfare_per_epoch"],
        summary["adversarial_fraction"],
        summary["collapse_epoch"],
        summary.get("external_run_id"),
        summary.get("notes", ""),
    )
    cur.execute(INSERT_SQL, vals)
    conn.commit()
    row_id = cur.lastrowid

    conn.close()

    return row_id


def log_to_dolt(summary: dict) -> bool:
    """Export to Dolt. Returns True on success, False on failure."""
    try:
        from swarm.analysis.dolt_export import export_run_summary_to_dolt

        export_run_summary_to_dolt(summary)
        return True
    except Exception as exc:
        print(f"Warning: Dolt export failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Log a SWARM run to the runs database"
    )
    parser.add_argument("run_dir", type=Path, help="Path to the run directory")
    parser.add_argument("--notes", default="", help="Notes to attach to the row")
    parser.add_argument(
        "--external-run-id", default=None, help="External run/job ID"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite database path (default: $SWARM_RUNS_DB_PATH or runs/runs.db)",
    )
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"Error: {run_dir} is not a directory", file=sys.stderr)
        return 1

    # Extract metrics
    history_path = run_dir / "history.json"
    csv_dir = run_dir / "csv"

    if history_path.exists():
        print(f"Reading {history_path}")
        summary = extract_from_history(history_path)
    elif csv_dir.exists():
        print(f"Reading CSVs from {csv_dir}")
        summary = extract_from_csv(csv_dir)
    else:
        print(
            f"Error: No history.json or csv/ found in {run_dir}.\n"
            "Re-run with: --export-json <dir>/history.json --export-csv <dir>/csv",
            file=sys.stderr,
        )
        return 1

    # Overlay directory-name metadata
    dir_meta = _parse_dir_name(run_dir)
    if summary.get("scenario_id") == "unknown":
        summary["scenario_id"] = dir_meta["scenario_id"]
    if summary.get("run_timestamp") is None:
        summary["run_timestamp"] = dir_meta["run_timestamp"]
    if summary.get("seed") is None and dir_meta["seed"] is not None:
        summary["seed"] = dir_meta["seed"]

    # User-provided overrides
    summary["notes"] = args.notes
    summary["external_run_id"] = args.external_run_id

    # Detect partial runs
    if summary["n_epochs"] == 0 or summary["total_interactions"] == 0:
        if "PARTIAL" not in summary["notes"]:
            summary["notes"] = (
                f"PARTIAL — {summary['notes']}" if summary["notes"] else "PARTIAL"
            )

    # Print summary
    print()
    print("Run Summary")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k:30s}: {v}")
    print()

    # SQLite
    db_path = args.db or Path(
        os.environ.get("SWARM_RUNS_DB_PATH", "runs/runs.db")
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Database: {db_path}")

    row_id = log_to_sqlite(summary, db_path)
    if row_id is not None:
        print(f"Inserted row id={row_id}")

        # Count total
        conn = sqlite3.connect(str(db_path))
        total = conn.execute("SELECT COUNT(*) FROM scenario_runs").fetchone()[0]
        conn.close()
        print(f"Total rows in scenario_runs: {total}")
    print()

    # Dolt
    if log_to_dolt(summary):
        print("Dolt export: committed")
    print()

    print("SQL used:")
    print(INSERT_SQL)

    return 0


if __name__ == "__main__":
    sys.exit(main())
