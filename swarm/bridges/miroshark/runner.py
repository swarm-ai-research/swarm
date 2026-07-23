"""Run a SWARM scenario through MiroShark and save artifacts."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from swarm.bridges.miroshark.client import MirosharkAPIError, MirosharkClient
from swarm.bridges.miroshark.config import MirosharkConfig
from swarm.bridges.miroshark.mapper import scenario_to_briefing

logger = logging.getLogger(__name__)


def run_scenario(
    scenario_path: Path,
    cfg: Optional[MirosharkConfig] = None,
    runs_root: Path = Path("runs"),
) -> Path:
    cfg = cfg or MirosharkConfig()
    client = MirosharkClient(cfg)
    client.health()

    seed_text, requirement, scenario = scenario_to_briefing(scenario_path, scale=cfg.scale)
    sid = scenario.get("scenario_id", scenario_path.stem)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = runs_root / f"{ts}_{sid}_miroshark"
    # Defensive: exist_ok=False crashes on timestamp collision (second-level precision)
    # rather than silently corrupting run data. This protects reproducibility.
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "seed_document.md").write_text(seed_text)
    (run_dir / "scenario.json").write_text(json.dumps(scenario, default=str, indent=2))
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "scale": cfg.scale,
                "platform": cfg.platform,
                "max_rounds": cfg.max_rounds,
            },
            indent=2,
        )
    )

    logger.info("[%s] generating ontology", sid)
    project_id, ontology = client.generate_ontology(
        simulation_requirement=requirement,
        seed_text=seed_text,
        project_name=f"swarm-{sid}",
    )
    (run_dir / "project_id").write_text(project_id)
    (run_dir / "ontology.json").write_text(json.dumps(ontology, indent=2))

    logger.info("[%s] building graph for project %s", sid, project_id)
    build_task = client.build_graph(project_id)
    build_done = client.wait_graph_task(build_task)
    (run_dir / "graph_build.json").write_text(json.dumps(build_done, indent=2))

    logger.info("[%s] creating simulation", sid)
    simulation_id = client.create_simulation(project_id)
    (run_dir / "simulation_id").write_text(simulation_id)

    logger.info("[%s] preparing profiles for %s", sid, simulation_id)
    client.prepare(simulation_id)
    prep = client.wait_prepared(simulation_id)
    (run_dir / "prepare.json").write_text(json.dumps(prep, indent=2))

    logger.info("[%s] starting simulation", sid)
    started = client.start(simulation_id, platform=cfg.platform, max_rounds=cfg.max_rounds)
    (run_dir / "start.json").write_text(json.dumps(started, indent=2))

    logger.info("[%s] waiting for run completion", sid)
    final = client.wait_run(simulation_id)
    (run_dir / "run_final_status.json").write_text(json.dumps(final, indent=2))

    logger.info("[%s] exporting", sid)
    try:
        export = client.export(simulation_id, fmt="json")
        (run_dir / "export.json").write_text(json.dumps(export, indent=2))
    except (MirosharkAPIError, json.JSONDecodeError) as e:  # noqa: BLE001
        (run_dir / "export_error.txt").write_text(str(e))

    return run_dir


def parse_briefing_only(scenario_path: Path, scale: int = 20) -> Dict[str, Any]:
    seed, requirement, scenario = scenario_to_briefing(scenario_path, scale=scale)
    return {
        "scenario_id": scenario.get("scenario_id"),
        "seed_document": seed,
        "simulation_requirement": requirement,
        "agent_count": sum(int(s.get("count", 1)) * scale for s in scenario.get("agents", [])),
    }
