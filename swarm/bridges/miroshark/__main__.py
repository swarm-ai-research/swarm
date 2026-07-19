"""CLI: ``python -m swarm.bridges.miroshark <scenario.yaml> [--scale N] [--dry-run]``."""

import argparse
import logging
import os
import sys
from pathlib import Path

from swarm.bridges.miroshark.config import MirosharkConfig
from swarm.bridges.miroshark.runner import parse_briefing_only, run_scenario


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="swarm.bridges.miroshark")
    p.add_argument("scenario", type=Path)
    p.add_argument("--scale", type=int, default=20)
    p.add_argument("--platform", default="parallel", choices=["twitter", "reddit", "parallel"])
    p.add_argument("--max-rounds", type=int, default=30)
    p.add_argument("--api-url", default=os.environ.get("MIROSHARK_API_URL", "http://localhost:5001"))
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    p.add_argument("--dry-run", action="store_true", help="emit briefing only; no API calls")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.scenario.exists():
        guess = Path("scenarios") / f"{args.scenario.stem}.yaml"
        if guess.exists():
            args.scenario = guess
        else:
            print(f"scenario not found: {args.scenario}", file=sys.stderr)
            return 2

    if args.dry_run:
        out = parse_briefing_only(args.scenario, scale=args.scale)
        print(f"# {out['scenario_id']}  (agents={out['agent_count']})\n")
        print(out["seed_document"])
        print("\n## simulation_requirement\n")
        print(out["simulation_requirement"])
        return 0

    cfg = MirosharkConfig(
        api_url=args.api_url,
        scale=args.scale,
        platform=args.platform,
        max_rounds=args.max_rounds,
        internal_key=os.environ.get("MIROSHARK_INTERNAL_KEY") or None,
        # Per-phase poll window (graph build / prepare / run each get one).
        # Reasoning-capable agent models (mimo-v2.5+) push prepare (~57
        # profile generations) past the old 1800s default.
        poll_timeout_s=float(os.environ.get("MIROSHARK_POLL_TIMEOUT_S", "1800")),
    )
    run_dir = run_scenario(args.scenario, cfg=cfg, runs_root=args.runs_root)
    print(str(run_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
