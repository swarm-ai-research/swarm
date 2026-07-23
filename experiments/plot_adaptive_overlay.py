"""Plot the arm-2 adaptive-vs-static overlay.

Usage:
    python -m experiments.plot_adaptive_overlay \
        --adaptive runs/<adaptive-grid>/grid_summary.csv \
        --static runs/<static-grid>/static_summary.csv \
        --output docs/research/figures/adaptive-vs-static-overlay.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from swarm.analysis.adaptive_overlay_plot import plot_overlay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive", type=Path, required=True,
                        help="grid_summary.csv from adaptive_arm2_grid")
    parser.add_argument("--static", type=Path, required=True,
                        help="static_summary.csv from adaptive_arm2_static_grid")
    parser.add_argument("--output", type=Path, required=True,
                        help="output PNG path")
    parser.add_argument("--title", type=str,
                        default="Adaptive vs static — arm 2 (5 seeds × 6 ρ)")
    args = parser.parse_args(argv)

    if not args.adaptive.exists():
        print(f"ERROR: adaptive CSV not found: {args.adaptive}", file=sys.stderr)
        return 2
    if not args.static.exists():
        print(f"ERROR: static CSV not found: {args.static}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)

    plot_overlay(
        adaptive_path=str(args.adaptive),
        static_path=str(args.static),
        output_path=str(args.output),
        title=args.title,
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
