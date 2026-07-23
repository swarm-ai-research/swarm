"""Export simulation results to a Dolt database.

Shells out to the ``dolt`` CLI so no Python MySQL driver is required.
All functions degrade gracefully — if ``dolt`` is not installed or the
repo directory is missing, a warning is printed and the run continues.
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from swarm.analysis.aggregation import SimulationHistory
from swarm.analysis.export import history_to_agent_records, history_to_epoch_records

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dolt_exec(cmd: List[str], cwd: str) -> subprocess.CompletedProcess:
    """Run a dolt CLI command and return the result."""
    return subprocess.run(
        ["dolt"] + cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _escape_sql(value: Any) -> str:
    """Return a SQL-safe literal representation of *value*."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    # String — escape single quotes
    return "'" + str(value).replace("'", "''") + "'"


def _build_insert_ignore(table: str, records: List[Dict[str, Any]]) -> str:
    """Build an INSERT IGNORE statement for a batch of records."""
    if not records:
        return ""
    columns = list(records[0].keys())
    col_list = ", ".join(f"`{c}`" for c in columns)
    rows = []
    for rec in records:
        vals = ", ".join(_escape_sql(rec.get(c)) for c in columns)
        rows.append(f"({vals})")
    values = ",\n".join(rows)
    return f"INSERT IGNORE INTO `{table}` ({col_list}) VALUES\n{values};\n"


def _run_dolt_preflight_checks(dolt_dir: str, context: str) -> bool:
    """Check if dolt is available and the repository exists.

    Parameters
    ----------
    dolt_dir:
        Path to the Dolt repository on disk.
    context:
        Context string for log/warning messages (e.g., "Dolt export" or "Dolt summary export").

    Returns
    -------
    True if preflight checks pass, False otherwise.
    """
    if not shutil.which("dolt"):
        logger.warning("dolt binary not found on PATH — skipping %s", context)
        print(f"Warning: dolt not found, skipping {context}")
        return False

    dolt_path = Path(dolt_dir)
    if not (dolt_path / ".dolt").is_dir():
        logger.warning("No Dolt repo at %s — skipping %s", dolt_dir, context)
        print(f"Warning: no Dolt repo at {dolt_dir}, skipping {context}")
        return False

    return True


def _execute_dolt_sql(
    dolt_path: Path, sql: str, context_error: str
) -> bool:
    """Execute SQL against a Dolt repository.

    Parameters
    ----------
    dolt_path:
        Path to the Dolt repository.
    sql:
        SQL text to execute.
    context_error:
        Context string for error messages (e.g., "Dolt SQL import failed").

    Returns
    -------
    True on success, False on failure.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sql", delete=False
    ) as tmp:
        tmp.write(sql)
        tmp_path = tmp.name

    try:
        result = _dolt_exec(["sql", "--file", tmp_path], cwd=str(dolt_path))
        if result.returncode != 0:
            logger.error("dolt sql failed: %s", result.stderr)
            print(f"Warning: {context_error}: {result.stderr.strip()}")
            return False
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return True


def _parse_commit_hash(commit_result: "subprocess.CompletedProcess[str]") -> Optional[str]:
    """Extract commit hash from dolt commit output.

    Parameters
    ----------
    commit_result:
        Result from running dolt commit.

    Returns
    -------
    The commit hash if found, None otherwise.
    """
    for line in commit_result.stdout.splitlines():
        if line.strip().startswith("commit"):
            parts = line.strip().split()
            if len(parts) >= 2:
                return parts[1]
    return None


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

_AGENTS_DDL = """\
CREATE TABLE IF NOT EXISTS `agents` (
  `simulation_id` VARCHAR(128) NOT NULL,
  `run_dir` VARCHAR(256) NOT NULL,
  `agent_id` VARCHAR(128) NOT NULL,
  `name` VARCHAR(256),
  `epoch` INT NOT NULL,
  `reputation` DOUBLE,
  `resources` DOUBLE,
  `interactions_initiated` INT,
  `interactions_received` INT,
  `avg_p_initiated` DOUBLE,
  `avg_p_received` DOUBLE,
  `total_payoff` DOUBLE,
  `is_frozen` TINYINT(1),
  `is_quarantined` TINYINT(1),
  PRIMARY KEY (`simulation_id`, `run_dir`, `agent_id`, `epoch`)
);
"""


def _ensure_agents_table(dolt_dir: str) -> None:
    """Create the agents table if it doesn't already exist."""
    _dolt_exec(["sql", "-q", _AGENTS_DDL], cwd=dolt_dir)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_to_dolt(
    history: SimulationHistory,
    run_dir: str,
    dolt_dir: str = "runs/dolt_runs",
) -> Optional[str]:
    """Export a simulation run to the Dolt database.

    Parameters
    ----------
    history:
        Completed simulation history.
    run_dir:
        Name / relative path of the run directory (used as ``run_dir`` column).
    dolt_dir:
        Path to the Dolt repository on disk.

    Returns
    -------
    The Dolt commit hash on success, or ``None`` on failure.
    """
    # --- Pre-flight checks ---------------------------------------------------
    if not _run_dolt_preflight_checks(dolt_dir, "Dolt export"):
        return None

    dolt_path = Path(dolt_dir)

    # --- Ensure agents table exists ------------------------------------------
    _ensure_agents_table(str(dolt_path))

    # --- Build epoch records -------------------------------------------------
    epoch_records = history_to_epoch_records(history)
    # Add run_dir column required by the Dolt epochs table
    for rec in epoch_records:
        rec["run_dir"] = run_dir
        # Drop columns not in the Dolt epochs schema
        rec.pop("total_posts", None)
        rec.pop("total_votes", None)
        rec.pop("total_tasks_completed", None)
        rec.pop("n_edges", None)
        rec.pop("avg_degree", None)
        rec.pop("avg_clustering", None)
        rec.pop("n_components", None)
        rec.pop("n_flagged_pairs", None)
        rec.pop("avg_coordination_score", None)
        rec.pop("avg_synergy_score", None)

    # --- Build agent records -------------------------------------------------
    agent_records = history_to_agent_records(history)
    for rec in agent_records:
        rec["run_dir"] = run_dir

    # --- Generate SQL --------------------------------------------------------
    sql_parts: List[str] = []
    if epoch_records:
        sql_parts.append(_build_insert_ignore("epochs", epoch_records))
    if agent_records:
        sql_parts.append(_build_insert_ignore("agents", agent_records))

    if not sql_parts:
        logger.info("No records to export to Dolt")
        return None

    sql = "\n".join(sql_parts)

    # --- Execute SQL ---------------------------------------------------------
    if not _execute_dolt_sql(dolt_path, sql, "Dolt SQL import failed"):
        return None

    # --- Commit --------------------------------------------------------------
    # Compute summary metrics for commit message
    avg_tox = 0.0
    total_welfare = 0.0
    if history.epoch_snapshots:
        avg_tox = sum(s.toxicity_rate for s in history.epoch_snapshots) / len(
            history.epoch_snapshots
        )
        total_welfare = sum(s.total_welfare for s in history.epoch_snapshots)

    n_epochs = len(epoch_records)
    scenario_id = history.simulation_id or "unknown"
    seed = history.seed if history.seed is not None else "?"
    commit_msg = (
        f"Run {scenario_id} seed={seed}: "
        f"{n_epochs}ep, tox={avg_tox:.3f}, welfare={total_welfare:.1f}"
    )

    _dolt_exec(["add", "."], cwd=str(dolt_path))
    commit_result = _dolt_exec(
        ["commit", "-m", commit_msg],
        cwd=str(dolt_path),
    )

    if commit_result.returncode != 0:
        # May fail if there are no changes (duplicate insert)
        stderr = commit_result.stderr.strip()
        if "nothing to commit" in stderr.lower() or "no changes added" in stderr.lower():
            print("Dolt: no new data to commit (duplicate run)")
            return None
        logger.error("dolt commit failed: %s", stderr)
        print(f"Warning: Dolt commit failed: {stderr}")
        return None

    # Extract commit hash from output
    commit_hash = _parse_commit_hash(commit_result)

    print(f"Dolt: committed {n_epochs} epoch rows + {len(agent_records)} agent rows")
    if commit_hash:
        print(f"Dolt commit: {commit_hash}")

    return commit_hash


def export_run_summary_to_dolt(
    summary: Dict[str, Any],
    dolt_dir: str = "runs/dolt_runs",
) -> Optional[str]:
    """Write a single summary row to the ``scenario_runs`` table in Dolt.

    Parameters
    ----------
    summary:
        Dict with keys matching the ``scenario_runs`` columns (same schema
        as the SQLite ``/log_run`` command produces).
    dolt_dir:
        Path to the Dolt repository on disk.

    Returns
    -------
    The Dolt commit hash on success, or ``None`` on failure.
    """
    # --- Pre-flight checks ---------------------------------------------------
    if not _run_dolt_preflight_checks(dolt_dir, "Dolt summary export"):
        return None

    dolt_path = Path(dolt_dir)

    # Build the row, excluding the auto-increment id
    row = {k: v for k, v in summary.items() if k != "id"}

    sql = _build_insert_ignore("scenario_runs", [row])
    if not sql:
        return None

    # --- Execute SQL ---------------------------------------------------------
    if not _execute_dolt_sql(dolt_path, sql, "Dolt summary insert failed"):
        return None

    scenario_id = summary.get("scenario_id", "unknown")
    seed = summary.get("seed", "?")
    commit_msg = f"Log run summary: {scenario_id} seed={seed}"

    _dolt_exec(["add", "."], cwd=str(dolt_path))
    commit_result = _dolt_exec(
        ["commit", "-m", commit_msg],
        cwd=str(dolt_path),
    )

    if commit_result.returncode != 0:
        stderr = commit_result.stderr.strip()
        if "nothing to commit" in stderr.lower() or "no changes added" in stderr.lower():
            print("Dolt: no new summary data to commit (duplicate)")
            return None
        logger.error("dolt commit failed: %s", stderr)
        print(f"Warning: Dolt summary commit failed: {stderr}")
        return None

    commit_hash = _parse_commit_hash(commit_result)

    print(f"Dolt: committed run summary for {scenario_id}")
    if commit_hash:
        print(f"Dolt commit: {commit_hash}")

    return commit_hash
