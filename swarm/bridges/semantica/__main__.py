"""CLI: export a SWARM run to Semantica decision records.

Usage:
    python -m swarm.bridges.semantica runs/<run_dir>
    python -m swarm.bridges.semantica runs/<run_dir> --out decisions.jsonl
    python -m swarm.bridges.semantica runs/<run_dir> --push --mcp-cmd semantica-mcp
"""

import argparse
import logging
import sys
from pathlib import Path

from swarm.bridges.semantica.client import SemanticaMCPClient
from swarm.bridges.semantica.config import SemanticaConfig
from swarm.bridges.semantica.exporter import export_run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="swarm.bridges.semantica",
        description="Export a SWARM run's interactions as Semantica decision records.",
    )
    parser.add_argument("run_dir", type=Path, help="Run directory containing event log JSONL")
    parser.add_argument("--out", type=Path, default=None, help="Output JSONL path")
    parser.add_argument("--push", action="store_true", help="Also push to a semantica-mcp server")
    parser.add_argument("--mcp-cmd", default="semantica-mcp", help="Command for the MCP server")
    parser.add_argument("--category-prefix", default="swarm")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if not args.run_dir.is_dir():
        parser.error(f"{args.run_dir} is not a directory")

    client = None
    if args.push:
        cfg = SemanticaConfig(mcp_command=tuple(args.mcp_cmd.split()))
        client = SemanticaMCPClient(cfg)
        client.start()
    try:
        summary = export_run(
            args.run_dir,
            out_path=args.out,
            client=client,
            category_prefix=args.category_prefix,
        )
    finally:
        if client is not None:
            client.close()

    print(
        f"run={summary.run_id} interactions={summary.n_interactions} "
        f"written={summary.n_written} pushed={summary.n_pushed} -> {summary.out_path}"
    )
    if summary.push_errors:
        print(f"push errors: {len(summary.push_errors)} (see log)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
