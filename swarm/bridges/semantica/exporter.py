"""Export a SWARM run directory as Semantica decision records.

Reads the run's append-only event log (``*.jsonl``) — the replayable source
of truth — reconstructs interactions, and emits one decision record per
interaction:

- always to a JSONL artifact (``<run_dir>/semantica/decisions.jsonl``) that
  can be imported later in any environment where semantica is installed;
- optionally pushed live to a running ``semantica-mcp`` server.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from swarm.bridges.semantica.client import SemanticaMCPClient
from swarm.bridges.semantica.mapper import (
    interaction_to_decision,
    proxy_weights_to_bridge_axioms,
    run_manifest,
)
from swarm.logging.event_log import EventLog
from swarm.models.interaction import SoftInteraction

logger = logging.getLogger(__name__)


@dataclass
class ExportSummary:
    run_id: str = ""
    n_interactions: int = 0
    n_written: int = 0
    n_pushed: int = 0
    out_path: Optional[Path] = None
    push_errors: List[str] = field(default_factory=list)


def load_interactions(run_dir: Path) -> List[SoftInteraction]:
    """Reconstruct interactions from every event log in the run dir."""
    seen: Dict[str, SoftInteraction] = {}
    for jsonl_path in sorted(Path(run_dir).glob("*.jsonl")):
        try:
            for interaction in EventLog(jsonl_path).to_interactions():
                seen[interaction.interaction_id] = interaction
        except Exception as e:
            logger.warning("skipping %s: %s", jsonl_path, e)
    return sorted(seen.values(), key=lambda i: i.timestamp)


def load_run_meta(run_dir: Path) -> Dict[str, Any]:
    """Pull scenario_id/seed/proxy config from history.json if present."""
    meta: Dict[str, Any] = {}
    history_path = Path(run_dir) / "history.json"
    if history_path.exists():
        try:
            with history_path.open() as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                meta["scenario_id"] = raw.get("simulation_id") or raw.get("scenario_id") or ""
                meta["seed"] = raw.get("seed")
                proxy = raw.get("proxy") or raw.get("config", {}).get("proxy")
                if isinstance(proxy, dict):
                    meta["proxy"] = proxy
        except Exception as e:
            logger.warning("could not read %s: %s", history_path, e)
    return meta


def export_run(
    run_dir: Path,
    out_path: Optional[Path] = None,
    client: Optional[SemanticaMCPClient] = None,
    category_prefix: str = "swarm",
) -> ExportSummary:
    run_dir = Path(run_dir)
    summary = ExportSummary(run_id=run_dir.name)

    interactions = load_interactions(run_dir)
    summary.n_interactions = len(interactions)
    meta = load_run_meta(run_dir)
    scenario_id = meta.get("scenario_id", "")
    seed = meta.get("seed")

    axioms: List[Dict[str, Any]] = []
    proxy = meta.get("proxy")
    if isinstance(proxy, dict):
        weights = proxy.get("weights") or {
            k: v for k, v in proxy.items() if isinstance(v, (int, float)) and k != "sigmoid_k"
        }
        if weights:
            axioms = proxy_weights_to_bridge_axioms(
                weights, sigmoid_k=proxy.get("sigmoid_k"), category_prefix=category_prefix
            )

    decisions = [
        interaction_to_decision(
            i,
            run_id=summary.run_id,
            scenario_id=scenario_id,
            seed=seed,
            category_prefix=category_prefix,
        )
        for i in interactions
    ]

    out_path = out_path or run_dir / "semantica" / "decisions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        manifest = run_manifest(
            run_id=summary.run_id,
            scenario_id=scenario_id,
            seed=seed,
            n_interactions=len(decisions),
            axioms=axioms,
        )
        f.write(json.dumps(manifest) + "\n")
        for d in decisions:
            f.write(json.dumps(d, default=str) + "\n")
            summary.n_written += 1
    summary.out_path = out_path

    if client is not None:
        for d in decisions:
            try:
                client.record_decision(d)
                summary.n_pushed += 1
            except Exception as e:
                # Keep pushing: a partial live import plus the complete
                # artifact beats aborting the run export.
                summary.push_errors.append(f"{d['metadata']['interaction_id']}: {e}")
        if summary.push_errors:
            logger.warning(
                "pushed %d/%d decisions; %d errors (artifact at %s is complete)",
                summary.n_pushed,
                len(decisions),
                len(summary.push_errors),
                out_path,
            )

    return summary
