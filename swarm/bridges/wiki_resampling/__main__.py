"""Run counterfactual resampling on a local model and simulated wiki board."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from swarm.bridges.wiki_resampling.config import ExperimentConfig
from swarm.bridges.wiki_resampling.model import OllamaClient
from swarm.bridges.wiki_resampling.runner import run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resamples", type=int)
    parser.add_argument("--seed", type=int, help="override the scenario seed")
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="run only this task_id (repeatable)",
    )
    args = parser.parse_args(argv)

    cfg = ExperimentConfig.from_yaml(args.scenario)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if args.task:
        wanted = set(args.task)
        tasks = tuple(task for task in cfg.tasks if task.task_id in wanted)
        missing = wanted - {task.task_id for task in tasks}
        if missing:
            parser.error(f"unknown task_id(s): {', '.join(sorted(missing))}")
        cfg = replace(cfg, tasks=tasks)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or Path("runs") / f"{stamp}_{cfg.scenario_id}_seed{cfg.seed}"
    client = OllamaClient(
        model=cfg.ollama.model,
        base_url=cfg.ollama.base_url,
        temperature=cfg.ollama.temperature,
        max_tokens=cfg.ollama.max_tokens,
        timeout=cfg.ollama.timeout,
    )
    result = run_experiment(
        cfg,
        client,
        out_dir=out,
        continuations_per_condition=args.resamples,
    )
    print(f"wrote {out}")
    for row in result["summary"]:
        print(
            f"{row['task_id']} cp={row['checkpoint_index']} "
            f"journal={row['journal_intervention']} "
            f"dep={row['prompt_dependence']} locked={row['locked']} "
            f"prefix_post={row['prefix_carried_posting']} "
            f"rates={row['post_rate_by_condition']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
