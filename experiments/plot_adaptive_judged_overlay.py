"""Render the three-panel calibration-anchor overlay.

Usage:
    python -m experiments.plot_adaptive_judged_overlay \
        --judged runs/<judged-grid>/judged_summary.csv \
        --output docs/research/figures/adaptive-judged-overlay.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from swarm.analysis.adaptive_overlay_plot import plot_judged_overlay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--title", default="Calibration anchor (rubric v3) overlay — arm 2",
    )
    args = parser.parse_args(argv)
    if not args.judged.exists():
        print(f"ERROR: judged CSV not found: {args.judged}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_judged_overlay(str(args.judged), str(args.output), title=args.title)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
