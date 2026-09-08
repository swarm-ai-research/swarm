"""CLI: ``python -m swarm.bridges.collusion_wiki <scenario.yaml> --data-dir DIR``.

``--stego`` runs the hidden-character scan (``stego.py``) instead of the
detectors; add ``--pack DIR`` to include page bodies from the 2026-09-05
reading pack (``reading_pack.py``). ``--run-map run_identity_map.json`` enables ``--identity run`` (the audited
per-agent run from the fast-follow-question-trajectories reconstruction) and
``--revision-subset owned|supported`` (replay only run-owned revisions).
``--fetch`` downloads the export from collusion.wiki into ``--data-dir``
first (files land gzipped; the loader reads them as-is).

``--source schelling --board messages.json --eval-log run.eval
[--control-eval-log control.eval]`` (bead y91o) replays the schelling-point
board of the offline fast_follow_question_bench recreation instead; the
scenario YAML may also carry ``source`` and the paths under ``replay:``.

``--source termina --termina-db runs/data/termina`` (beads lnaf, sjis)
replays the swarm.termina.digital incident db per venue over the baseline
and post-disclosure windows of ``scenarios/casestudy_wiki_postdisclosure.yaml``.
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
    run_termina_replay,
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
    p.add_argument("--identity", choices=["label", "ip16", "label_ip16", "run", "actor"])
    p.add_argument("--run-map", type=Path,
                   help="run_identity_map.json (rev_id -> audited run) for "
                        "identity=run and --revision-subset")
    p.add_argument("--revision-subset", choices=["all", "owned", "supported"],
                   help="replay every revision, only run-owned ones, or only "
                        "those owned by a supported run")
    p.add_argument("--structural-null", choices=["configuration", "bipartite", "membership"],
                   help="null model behind the structural p-values: degree-"
                        "preserving configuration (default), hub-aware bipartite "
                        "rewiring of who edited which page, or membership "
                        "(per-page membership preserved, edit order shuffled; "
                        "bead y2t2)")
    p.add_argument("--stego", action="store_true",
                   help="scan the export's free-text fields for hidden-character "
                        "carriers (bead vv3j.5) instead of running the detectors")
    p.add_argument("--pack", type=Path,
                   help="agent-reading-pack directory (or its agent-text.sqlite); "
                        "with --stego, scans page bodies too (bead 8zoc)")
    p.add_argument("--source", choices=["collusion_wiki", "schelling", "termina"],
                   help="override the scenario's data source (beads y91o, lnaf)")
    p.add_argument("--termina-db", type=Path,
                   help="swarm.termina.digital incidents.sqlite, or its directory "
                        "(source=termina; --fetch downloads it into --data-dir)")
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
    if args.structural_null:
        cfg.structural_null = args.structural_null
    if args.include_seeded:
        cfg.include_seeded = True
    if args.run_map:
        cfg.run_map = str(args.run_map)
    if args.revision_subset:
        cfg.revision_subset = args.revision_subset

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

    if cfg.source == "termina":
        db = args.termina_db or (Path(cfg.termina_db) if cfg.termina_db else args.data_dir)
        out = run_termina_replay(db, cfg, args.runs_root,
                                 with_timeline=not args.no_timeline)
        print(out)
        return 0

    out = run_replay(args.data_dir, cfg, args.runs_root,
                     with_timeline=not args.no_timeline)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
