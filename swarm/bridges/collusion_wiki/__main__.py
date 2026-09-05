"""CLI: ``python -m swarm.bridges.collusion_wiki <scenario.yaml> --data-dir DIR``.

``--stego`` runs the hidden-character scan (``stego.py``) instead of the
detectors. ``--fetch`` downloads the export from collusion.wiki into ``--data-dir``
first (files land gzipped; the loader reads them as-is).
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

import yaml

from swarm.bridges.collusion_wiki.runner import ReplayConfig, run_replay
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
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not args.scenario.exists():
        print(f"scenario not found: {args.scenario}", file=sys.stderr)
        return 2
    if args.fetch:
        fetch(args.scenario, args.data_dir)

    if args.stego:
        out = run_stego(args.data_dir, args.runs_root)
        print(out)
        return 0

    cfg = ReplayConfig.from_yaml(args.scenario)
    if args.identity:
        cfg.identity = args.identity
        cfg.sweep_identity = [args.identity]
    out = run_replay(args.data_dir, cfg, args.runs_root,
                     with_timeline=not args.no_timeline)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
