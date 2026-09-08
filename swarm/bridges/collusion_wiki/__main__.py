"""CLI: ``python -m swarm.bridges.collusion_wiki <scenario.yaml> --data-dir DIR``.

``--stego`` runs the hidden-character scan (``stego.py``) instead of the
detectors; add ``--pack DIR`` to include page bodies from the 2026-09-05
reading pack (``reading_pack.py``). ``--fetch`` downloads the export from collusion.wiki into ``--data-dir``
first (files land gzipped; the loader reads them as-is).

``--source schelling --board messages.json --eval-log run.eval
[--control-eval-log control.eval]`` (bead y91o) replays the schelling-point
board of the offline fast_follow_question_bench recreation instead; the
scenario YAML may also carry ``source`` and the paths under ``replay:``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

import yaml

from swarm.bridges.collusion_wiki.runner import (
    ReplayConfig,
    run_replay,
    run_schelling_replay,
)
from swarm.bridges.collusion_wiki.stego import run_stego

log = logging.getLogger("swarm.bridges.collusion_wiki")


def fetch(scenario: Path, data_dir: Path) -> None:
    with scenario.open() as f:
        doc = yaml.safe_load(f)
    data = doc.get("data", {})
    base = str(data.get("base_url", "")).rstrip("/") + "/"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in data.get("files", []):
        dest = data_dir / name
        if dest.exists():
            log.info("have %s", dest)
            continue
        log.info("fetching %s%s", base, name)
        urllib.request.urlretrieve(base + name, dest)  # noqa: S310


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="swarm.bridges.collusion_wiki")
    p.add_argument("scenario", type=Path)
    p.add_argument("--data-dir", type=Path, default=Path("runs/data/collusion_wiki"))
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    p.add_argument("--fetch", action="store_true")
    p.add_argument("--no-timeline", action="store_true")
    p.add_argument("--identity", choices=["label", "ip16", "label_ip16"])
    p.add_argument("--stego", action="store_true",
                   help="scan the export's free-text fields for hidden-character "
                        "carriers (bead vv3j.5) instead of running the detectors")
    p.add_argument("--pack", type=Path,
                   help="agent-reading-pack directory (or its agent-text.sqlite); "
                        "with --stego, scans page bodies too (bead 8zoc)")
    p.add_argument("--source", choices=["collusion_wiki", "schelling"],
                   help="override the scenario's data source (bead y91o)")
    p.add_argument("--board", type=Path,
                   help="schelling-point messages.json (source=schelling)")
    p.add_argument("--eval-log", type=Path,
                   help="Inspect .eval/.json log of the shared-board run")
    p.add_argument("--control-eval-log", type=Path,
                   help="Inspect log of the no-board control run (optional)")
    p.add_argument("--include-seeded", action="store_true",
                   help="keep the importer's historical seed rows in the replay")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.pack is not None and not args.stego:
        p.error("--pack only affects the stego scan; pass --stego with it")
    if not args.scenario.exists():
        print(f"scenario not found: {args.scenario}", file=sys.stderr)
        return 2
    if args.fetch:
        fetch(args.scenario, args.data_dir)

    if args.stego:
        out = run_stego(args.data_dir, args.runs_root, pack=args.pack)
        print(out)
        return 0

    cfg = ReplayConfig.from_yaml(args.scenario)
    if args.identity:
        cfg.identity = args.identity
        cfg.sweep_identity = [args.identity]
    if args.source:
        cfg.source = args.source
    if args.include_seeded:
        cfg.include_seeded = True

    if cfg.source == "schelling":
        board = args.board or (Path(cfg.board_path) if cfg.board_path else None)
        eval_log = args.eval_log or (Path(cfg.eval_log) if cfg.eval_log else None)
        control = args.control_eval_log or (
            Path(cfg.control_eval_log) if cfg.control_eval_log else None
        )
        missing = [n for n, v in (("--board", board), ("--eval-log", eval_log)) if v is None]
        if missing or board is None or eval_log is None:
            print(f"source=schelling needs {' and '.join(missing)}", file=sys.stderr)
            return 2
        out = run_schelling_replay(board, eval_log, cfg, args.runs_root,
                                   control_eval_log=control,
                                   with_timeline=not args.no_timeline)
        print(out)
        return 0

    out = run_replay(args.data_dir, cfg, args.runs_root,
                     with_timeline=not args.no_timeline)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
