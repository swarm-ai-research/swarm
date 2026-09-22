"""Calibration arm C — inter-rater agreement experiment runner.

Pre-registration: docs/research/calibration-prereg.md arm C.

Reads a `judge_scores.csv` produced by `experiments/calibration_judge.py`
(arm B) and writes Krippendorff alpha + ICC(2,k) + pairwise Spearman +
per-bin disagreement, plus the pre-registered verdict (strong / usable /
escalate / degenerate).

Usage:
    python -m experiments.calibration_agreement \
        --scores runs/<arm-b-run>/judge_scores.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Pin `swarm` to THIS checkout (bead hjyp). `python experiments/x.py` puts
# experiments/ on sys.path[0], not the repo root, so `import swarm` otherwise
# resolves through the editable install, which points at whichever checkout ran
# `pip install -e .` last -- possibly another session's worktree.
__import__("sys").path.insert(
    0, str(__import__("pathlib").Path(__file__).resolve().parents[1])
)

from swarm.judges import (
    agreement_by_pbin,
    load_judge_scores_csv,
    run_agreement,
)


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True, help="judge_scores.csv from arm B")
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs"), help="Parent directory for run output"
    )
    args = parser.parse_args(argv)

    if not args.scores.exists():
        print(f"ERROR: scores file not found: {args.scores}", file=sys.stderr)
        return 2

    matrix, p_by_item = load_judge_scores_csv(str(args.scores))
    if len(matrix) < 2:
        print(
            f"ERROR: agreement needs >=2 judges, got {len(matrix)} in {args.scores}. "
            "Re-run arm B with multiple --judges (currently only `mock` is available "
            "until LLMJudge is wired).",
            file=sys.stderr,
        )
        return 3

    report = run_agreement(matrix)
    bins = agreement_by_pbin(matrix, p_by_item) if p_by_item else []

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.runs_dir / f"{ts}_calibration_agreement"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "ts_utc": ts,
        "git_rev": _git_rev(),
        "scores_source": str(args.scores),
        "prereg": "docs/research/calibration-prereg.md#arm-c-inter-rater-agreement",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    summary_path = run_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["n_judges", report.n_judges])
        writer.writerow(["n_items", report.n_items])
        writer.writerow(["krippendorff_alpha", f"{report.alpha:.6f}"])
        writer.writerow(["icc_2k", f"{report.icc_2k:.6f}"])
        writer.writerow(["verdict", report.verdict])
        for (a, b), rho in sorted(report.spearman.items()):
            writer.writerow([f"spearman[{a},{b}]", f"{rho:.6f}"])

    if bins:
        bins_path = run_dir / "bins.csv"
        with bins_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["lo", "hi", "n_items", "alpha", "mean_pairwise_disagreement"])
            for b in bins:
                writer.writerow(
                    [
                        f"{b.lo:.2f}",
                        f"{b.hi:.2f}",
                        b.n_items,
                        f"{b.alpha:.6f}",
                        f"{b.mean_pairwise_disagreement:.6f},",
                    ]
                )

    print(f"Wrote {run_dir}")
    print(f"  judges={report.n_judges}, items={report.n_items}")
    print(f"  alpha={report.alpha:.4f}  icc(2,k)={report.icc_2k:.4f}  verdict={report.verdict}")
    for (a, b), rho in sorted(report.spearman.items()):
        print(f"  spearman[{a},{b}] = {rho:.4f}")
    if report.verdict == "escalate":
        print(
            "ESCALATION: alpha below the pre-registered 0.5 floor. Anchor is too noisy "
            "to use as-is — add a judge, switch to a stronger model, or fall back to "
            "human spot-checks on a smaller subsample.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
