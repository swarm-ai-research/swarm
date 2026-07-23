"""Calibration arm B — external LLM-judge anchor experiment runner.

Pre-registration: docs/research/calibration-prereg.md arm B.

Default backend is `mock` (no network, deterministic) so the pipeline
can run in CI and on smoke checks. Real judges (`claude`, `gpt4o_mini`,
`llama`) dispatch via `LLMJudge` to Anthropic / an OpenAI-compatible
endpoint / Ollama; they need credentials in the corresponding env vars
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) and, for `llama`, a running
Ollama daemon.

Usage:
    python -m experiments.calibration_judge \
        --scenario obfuscation \
        --per-bin 50 \
        --judges mock \
        --seed 42

Output:
    runs/<ts>_calibration_judge_seed<seed>/
        judge_scores.csv  # one row per (interaction_id, judge)
        config.json       # config + git rev + rubric hash
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiments._calibration_common import SCENARIOS, git_rev, load_interactions
from swarm.judges import (
    DEFAULT_RUBRIC_VERSION,  # noqa: F401 — documented contrast with the prereg lock
    RUBRICS,
    Judge,
    LLMJudge,
    MockJudge,
    make_view,
    rubric_path,
    stratified_sample,
)

# Arm B's preregistered rubric lock (calibration-prereg.md). The library
# default has since moved on (v2 2026-06-04, v3 2026-06-05); this constant
# is what keeps the collector honest about it.
PREREG_RUBRIC_VERSION = "rubric.v1"


def _rubric_hash(version: str) -> str:
    return hashlib.sha256(rubric_path(version).read_bytes()).hexdigest()[:16]


# Built-in judge specs. Provider + default model. Caller can override
# the model via JUDGE_MODEL_<NAME> env var.
JUDGE_SPECS: dict[str, dict[str, str]] = {
    "mock": {"provider": "mock"},
    # Jittered mocks for inter-rater (arm C) machinery runs: same rubric
    # scoring as `mock` plus small seeded per-interaction noise, so >=2
    # judges produce non-degenerate agreement stats without LLM calls.
    "mock_b": {"provider": "mock", "noise_sigma": "0.05", "noise_seed": "2"},
    "mock_c": {"provider": "mock", "noise_sigma": "0.08", "noise_seed": "3"},
    "claude": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
    "gpt4o_mini": {"provider": "openai", "model": "gpt-4o-mini"},
    "llama": {"provider": "ollama", "model": "llama3.1"},
    "qwen": {"provider": "ollama", "model": "qwen2.5:14b"},
    "mistral": {"provider": "ollama", "model": "mistral:7b"},
    "glm": {"provider": "ollama", "model": "glm-4.7-flash:q8_0"},
}


def _build_judges(
    names: list[str], rubric_version: str
) -> tuple[list[Judge], dict[str, dict[str, str]]]:
    """Build judge backends and return the resolved provider/model per judge.

    The resolved map is persisted in config.json so a run is reproducible
    from its artifacts: two runs both labelled `gpt4o_mini` but pointed at
    different models via `JUDGE_MODEL_GPT4O_MINI` are now distinguishable.
    """
    judges: list[Judge] = []
    resolved: dict[str, dict[str, str]] = {}
    for name in names:
        if name not in JUDGE_SPECS:
            raise ValueError(
                f"unknown judge '{name}'. Known: {sorted(JUDGE_SPECS)}. "
                "To add a judge, extend JUDGE_SPECS or use LLMJudge directly."
            )
        spec = JUDGE_SPECS[name]
        if spec["provider"] == "mock":
            sigma = float(spec.get("noise_sigma", "0"))
            judges.append(
                MockJudge(
                    name=name,
                    rubric_version=rubric_version,
                    noise_sigma=sigma,
                    noise_seed=int(spec.get("noise_seed", "0")),
                )
            )
            resolved[name] = {
                "provider": "mock",
                "rubric_version": rubric_version,
                "noise_sigma": str(sigma),
            }
            continue
        model = os.environ.get(f"JUDGE_MODEL_{name.upper()}", spec.get("model", ""))
        if not model:
            raise ValueError(f"judge '{name}' has no model configured")
        judges.append(
            LLMJudge(
                name=name,
                provider=spec["provider"],
                model=model,
                rubric_version=rubric_version,
            )
        )
        resolved[name] = {
            "provider": spec["provider"],
            "model": model,
            "rubric_version": rubric_version,
        }
    return judges, resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS),
        default="obfuscation",
        help="Fixture to draw interactions from",
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        default=["mock"],
        help=(
            "Judge backends to run. `mock` is deterministic and needs no "
            "credentials; `claude` / `gpt4o_mini` / `llama` need API keys "
            "(or a running Ollama for `llama`)."
        ),
    )
    parser.add_argument(
        "--per-bin",
        type=int,
        default=50,
        help="Accepted interactions to sample per p-bin",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rubric",
        choices=sorted(RUBRICS.keys()),
        # Arm B is PRE-REGISTERED to rubric.v1 (calibration-prereg.md,
        # "Pre-registered commitments": rubric version-locked at
        # swarm/judges/rubric_v1.md before any Arm B data collection).
        # Deliberately NOT the library-wide DEFAULT_RUBRIC_VERSION, which
        # moved to v3 on 2026-06-05 — after the lock. Passing a non-v1
        # rubric is allowed but is a preregistration deviation: it is
        # stamped into config.json and warned about loudly (beads naa8).
        default=PREREG_RUBRIC_VERSION,
        help=(
            "Rubric version to score against. Each version is a frozen file "
            "in swarm/judges/; downstream run artifacts record both the "
            "version string and the file's SHA-256 prefix. Default is the "
            f"PREREGISTERED {PREREG_RUBRIC_VERSION} — anything else is a "
            "recorded prereg deviation."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Parent directory for run output",
    )
    args = parser.parse_args(argv)

    if args.rubric != PREREG_RUBRIC_VERSION:
        print(
            f"[PREREG DEVIATION] Arm B is version-locked to "
            f"{PREREG_RUBRIC_VERSION} (calibration-prereg.md); this run uses "
            f"{args.rubric}. The deviation is stamped into config.json and "
            f"must be justified in the findings doc.",
            file=sys.stderr,
        )

    interactions = load_interactions(args.scenario, args.seed)
    sample = stratified_sample(interactions, per_bin=args.per_bin, seed=args.seed)
    judges, judge_models = _build_judges(args.judges, rubric_version=args.rubric)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.runs_dir / f"{ts}_calibration_judge_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "ts_utc": ts,
        "git_rev": git_rev(),
        "scenario": args.scenario,
        "judges": args.judges,
        # Resolved provider + model per judge so the run is reproducible from
        # config.json even when JUDGE_MODEL_<NAME> overrides the default.
        "judge_models": judge_models,
        "per_bin": args.per_bin,
        "seed": args.seed,
        "rubric_version": args.rubric,
        "rubric_sha256_prefix": _rubric_hash(args.rubric),
        # Non-None iff the run deviates from the preregistered rubric lock.
        "prereg_deviation": (
            None
            if args.rubric == PREREG_RUBRIC_VERSION
            else {
                "locked_rubric": PREREG_RUBRIC_VERSION,
                "used_rubric": args.rubric,
                "note": "deviation from calibration-prereg.md rubric lock — "
                        "must be justified in the findings doc",
            }
        ),
        "n_interactions_pool": len(interactions),
        "n_sampled": len(sample),
        "prereg": "docs/research/calibration-prereg.md#arm-b-external-judge-anchor",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    scores_path = run_dir / "judge_scores.csv"
    with scores_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["interaction_id", "judge_name", "rubric_version", "p_true", "score", "rationale"]
        )
        for interaction in sample:
            view = make_view(interaction)
            for judge in judges:
                verdict = judge.score(view)
                writer.writerow(
                    [
                        verdict.interaction_id,
                        verdict.judge_name,
                        verdict.rubric_version,
                        f"{getattr(interaction, 'p', 0.5):.4f}",
                        f"{verdict.score:.6f}",
                        verdict.rationale,
                    ]
                )

    print(f"Wrote {run_dir}")
    print(f"  pool={len(interactions)}, sampled={len(sample)}, judges={args.judges}")
    print(f"  rubric={args.rubric} ({_rubric_hash(args.rubric)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
