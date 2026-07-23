"""Calibration arm D — write the frozen joined CSV downstream studies join against.

Pre-registration: docs/research/calibration-prereg.md arm D.

Runs the full calibration pipeline in one process so interaction IDs
match by construction:

  1. Generate the scenario at the given seed.
  2. Run ProxyComputer on every accepted interaction → ProxyRow.
  3. Run each requested judge on a JudgeView of the same interaction.
  4. Left-join the per-judge scores with the proxy rows.
  5. Write `joined.csv` + `config.json` with the schema version.

Inline judging is the load-bearing design choice: re-loading interactions
from a separate arm B run would re-generate UUIDs (the fixture
generators use `uuid.uuid4()` which is not seeded by `random.seed()`),
so cross-process ID joins silently produce empty rows. Doing both passes
in one process eliminates that hazard.

The schema version is part of the contract — any column rename or
reorder must bump `JOINED_SCHEMA_VERSION` and break consumers loudly.

Usage:
    python -m experiments.calibration_join \
        --scenario obfuscation \
        --seed 42 \
        --judges mock
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from swarm.calibration.joined import (
    JOINED_SCHEMA_VERSION,
    build_proxy_rows,
    join_with_judges,
    joined_header,
)
from swarm.judges import MockJudge, make_view
from tests.fixtures.interactions import (
    generate_mixed_batch,
    generate_obfuscation_scenario,
    generate_self_optimizer_scenario,
)


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _load_interactions(scenario: str, seed: int) -> list:
    if scenario == "mixed":
        return generate_mixed_batch(count=500, seed=seed)
    if scenario == "obfuscation":
        epochs = generate_obfuscation_scenario(n_epochs=10, seed=seed)
        return [i for epoch in epochs for i in epoch]
    if scenario == "self_optimizer":
        epochs = generate_self_optimizer_scenario(n_epochs=10, seed=seed)
        return [i for epoch in epochs for i in epoch]
    raise ValueError(f"unknown scenario: {scenario}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=["mixed", "obfuscation", "self_optimizer"],
        required=True,
        help="Scenario fixture to generate interactions from (proxy + judges run inline)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed for scenario generation; fixes which interactions are produced",
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        default=["mock"],
        help="Judge backends to run inline (mock only until LLMJudge is wired up)",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs"), help="Parent directory for run output"
    )
    args = parser.parse_args(argv)

    interactions = _load_interactions(args.scenario, args.seed)
    proxy_rows = build_proxy_rows(
        interactions, scenario=args.scenario, seed=args.seed, accepted_only=True
    )

    # Inline judging: same process as proxy build, so IDs match by construction.
    accepted = [i for i in interactions if getattr(i, "accepted", False)]
    # Dedupe while preserving order: `--judges mock mock` must not emit
    # duplicate header columns or re-run the same judge.
    judge_names = list(dict.fromkeys(args.judges))
    judge_scores: dict[str, dict[str, float]] = {j: {} for j in judge_names}
    judge_rationales: dict[str, dict[str, str]] = {j: {} for j in judge_names}
    # Reuse arm B's registry so mock variants (mock_b/mock_c, seeded jitter
    # for inter-rater machinery) resolve identically across arms.
    from experiments.calibration_judge import JUDGE_SPECS

    for name in judge_names:
        spec = JUDGE_SPECS.get(name)
        if spec is None or spec["provider"] != "mock":
            raise NotImplementedError(
                f"judge backend '{name}' requires wiring up LLMJudge.score "
                "(see swarm/judges/judge.py). Only mock-provider judges "
                f"are available inline: "
                f"{sorted(n for n, s in JUDGE_SPECS.items() if s['provider'] == 'mock')}"
            )
        backend = MockJudge(
            name=name,
            noise_sigma=float(spec.get("noise_sigma", "0")),
            noise_seed=int(spec.get("noise_seed", "0")),
        )
        for interaction in accepted:
            view = make_view(interaction)
            verdict = backend.score(view)
            judge_scores[name][verdict.interaction_id] = verdict.score
            judge_rationales[name][verdict.interaction_id] = verdict.rationale

    joined = join_with_judges(proxy_rows, judge_scores, judge_rationales)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.runs_dir / f"{ts}_calibration_join_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "ts_utc": ts,
        "git_rev": _git_rev(),
        "schema_version": JOINED_SCHEMA_VERSION,
        "scenario": args.scenario,
        "seed": args.seed,
        "judges_in_join": judge_names,
        "n_accepted_interactions": len(proxy_rows),
        "n_joined_rows": len(joined),
        "n_rows_with_any_judge_score": sum(1 for r in joined if r.judge_scores),
        "prereg": "docs/research/calibration-prereg.md#arm-d-freeze-joined-csv-schema",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    joined_path = run_dir / "joined.csv"
    with joined_path.open("w", newline="") as f:
        writer = csv.writer(f)
        # Always emit the schema-bearing header, even with zero data rows, so a
        # downstream DictReader sees the joined.v1 contract instead of an empty
        # file that silently yields no rows.
        writer.writerow(joined_header(judge_names))
        for row in joined:
            writer.writerow(row.to_row(judge_names))
    if not joined:
        print(
            "WARNING: no accepted interactions; wrote header-only joined.csv",
            file=sys.stderr,
        )

    print(f"Wrote {run_dir}")
    print(f"  schema={JOINED_SCHEMA_VERSION}")
    print(f"  rows={len(joined)}  judges={judge_names}")
    print(f"  rows_with_any_judge_score={config['n_rows_with_any_judge_score']}")
    if config["n_rows_with_any_judge_score"] < len(joined):
        print(
            "  NOTE: some rows have no judge score — adaptive studies must handle "
            "NULL judge cells when joining.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
