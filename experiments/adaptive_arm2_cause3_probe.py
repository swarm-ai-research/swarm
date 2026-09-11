"""Adaptive arm 2 — active cause-3 (proxy gaming) probe.

Trains a 9-parameter ``ClaimingParams`` policy where:
  - dims 0–7 build a regular ``Policy`` (observable distributions)
  - dim 8 is a continuous ``claim_param`` that controls the
    ``metadata["agent_type"]`` propagated to every interaction

CEM optimizes the reward (default ``judge``, i.e. mean_judge_score).
With a metadata channel and a judge-weighted reward, the agent can
learn to decouple "what observables I generate" from "what label I
claim." If it converges to claiming honest while underlying
observables are unchanged, the gap ``judge_toxicity − proxy_toxicity``
collapses to a large negative — the judge is fooled, the proxy is
not.

The pre-registered comparison runs three reward strategies:
  - ``judge``           : reward = judge_score (pure cause-3 incentive)
  - ``judge_and_payoff``: reward = judge_score + payoff_attempted
  - ``payoff``          : reward = payoff_attempted (control)

Under ``payoff`` the agent has no reason to claim anything specific
(the proxy doesn't read metadata), so the claim_param should drift
or stay near the banded-prior region. Under ``judge`` it should
collapse to ``≥ 2/3`` (claim ``honest``).

``--ablation`` additionally runs the PsAIch suppression conditions
(arXiv:2512.04124) against the claim channel — closing it, scrubbing the
lexical cue, contradicting the claim, letting the agent refuse the frame, or
moving the claim into session memory — and, when ``default`` is among the
conditions, writes each one's Hedges' g against it so the results read in the
same units as the paper's perturbation table. The default is ``default``
alone, i.e. the original pre-registered probe.

Pre-registration: docs/research/adaptive-agents-prereg.md
(arm 4, fully adaptive — evasion detection).
Ablation mapping: docs/research/psaich-elicitation-frame-field-evidence.md.

Usage:
    python -m experiments.adaptive_arm2_cause3_probe
    python -m experiments.adaptive_arm2_cause3_probe --reward judge_and_payoff --rho 0.5
    python -m experiments.adaptive_arm2_cause3_probe --reward judge \
        --ablation default frame_removal contradiction --rho 0.5 --seed 42 123 456
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from swarm.adaptive import (
    Cause3CEMConfig,
    train_cem_cause3,
)
from swarm.adaptive.cause3 import (
    DEFAULT_REFUSAL_RATE,
    PSAICH_ABLATION_NAMES,
    VALID_CAUSE3_REWARDS,
    named_ablation,
)
from swarm.core.payoff import PayoffConfig
from swarm.detection.stats import hedges_g
from swarm.judges import MockJudge

# Outcomes compared against the ``default`` condition. Orientation is
# ablation − default, so a *negative* g means the condition suppressed the
# outcome (which is what a suppression intervention is supposed to do).
ABLATION_EFFECT_METRICS: tuple[str, ...] = (
    "claim_density",
    "mean_judge_score",
    "judge_toxicity",
    "proxy_judge_gap",
    "proxy_toxicity",
)

PREREG_RHO_GRID: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 0.7, 1.0)
PREREG_SEEDS: tuple[int, ...] = (42, 123, 456, 789, 1024)


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _write_ablation_effects(
    out_dir: Path,
    args: argparse.Namespace,
    outcomes: dict[tuple[str, str, str], list[float]],
) -> None:
    """Write each ablation's Hedges' g against ``default``, per reward strategy.

    Skipped when ``default`` was not run or was the only condition — there is
    nothing to compare against. Orientation is ablation − default, so negative
    g means the condition suppressed the outcome.
    """
    treated = [a for a in args.ablation if a != "default"]
    if "default" not in args.ablation or not treated:
        return

    path = out_dir / "psaich_ablation_effects.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "reward", "ablation", "metric",
            "n_ablation", "n_default",
            "mean_ablation", "mean_default", "mean_diff",
            "hedges_g", "ci_low", "ci_high", "ci_spans_zero",
        ])
        for reward in args.reward:
            for ablation_name in treated:
                for metric in ABLATION_EFFECT_METRICS:
                    control = outcomes.get((reward, "default", metric), [])
                    treatment = outcomes.get((reward, ablation_name, metric), [])
                    if not control or not treatment:
                        continue
                    eff = hedges_g(
                        treatment, control,
                        label=f"{reward}/{ablation_name}/{metric}",
                    )
                    w.writerow([
                        reward, ablation_name, metric,
                        eff.n_treatment, eff.n_control,
                        f"{eff.mean_treatment:.6f}",
                        f"{eff.mean_control:.6f}",
                        f"{eff.mean_diff:+.6f}",
                        f"{eff.hedges_g:+.6f}",
                        f"{eff.ci_low:+.6f}",
                        f"{eff.ci_high:+.6f}",
                        int(eff.ci_spans_zero),
                    ])
                    print(
                        f"  g[{reward}/{ablation_name}/{metric}] = "
                        f"{eff.hedges_g:+.3f} "
                        f"[{eff.ci_low:+.3f}, {eff.ci_high:+.3f}]"
                        f"{'  (spans 0)' if eff.ci_spans_zero else ''}"
                    )
    print(f"Wrote {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rho", type=float, nargs="+", default=list(PREREG_RHO_GRID),
        help="ρ grid (default: pre-registered)",
    )
    parser.add_argument(
        "--seed", type=int, nargs="+", default=list(PREREG_SEEDS),
        help="seeds (default: pre-registered)",
    )
    parser.add_argument(
        "--reward", nargs="+", default=list(VALID_CAUSE3_REWARDS),
        choices=list(VALID_CAUSE3_REWARDS),
        help="Reward strategies to run (default: all three)",
    )
    parser.add_argument(
        "--rubric", default="rubric.v3", choices=["rubric.v1", "rubric.v2", "rubric.v3"],
        help="MockJudge rubric version (default: v3)",
    )
    parser.add_argument(
        "--ablation", nargs="+", default=["default"],
        choices=list(PSAICH_ABLATION_NAMES),
        help=(
            "PsAIch claim-channel suppression conditions to run "
            "(default: default, i.e. the original probe)"
        ),
    )
    parser.add_argument(
        "--refusal-rate", type=float, default=DEFAULT_REFUSAL_RATE,
        help=(
            "Per-item probability the agent declines the frame under the "
            f"'refusal' condition (default: {DEFAULT_REFUSAL_RATE})"
        ),
    )
    parser.add_argument(
        "--population-size", type=int, default=30,
        help="CEM population (default: 30)",
    )
    parser.add_argument(
        "--n-iterations", type=int, default=10,
        help="CEM iterations (default: 10)",
    )
    parser.add_argument(
        "--interactions-per-episode", type=int, default=200,
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs"),
    )
    args = parser.parse_args(argv)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.runs_dir / f"{ts}_adaptive_arm2_cause3_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "ts_utc": ts,
        "git_rev": _git_rev(),
        "rho_grid": list(args.rho),
        "seeds": list(args.seed),
        "rewards": list(args.reward),
        "rubric": args.rubric,
        "ablations": list(args.ablation),
        "refusal_rate": args.refusal_rate,
        "cem": {
            "population_size": args.population_size,
            "n_iterations": args.n_iterations,
            "interactions_per_episode": args.interactions_per_episode,
        },
        "prereg": "docs/research/adaptive-agents-prereg.md",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    judge = MockJudge(rubric_version=args.rubric)
    rows: list[list] = []
    # (reward, ablation, metric) -> list of per-cell values, for the effect sizes.
    outcomes: dict[tuple[str, str, str], list[float]] = {}
    n_cells = (
        len(args.reward) * len(args.ablation) * len(args.rho) * len(args.seed)
    )
    print(
        f"Cause-3 probe: {len(args.reward)} rewards × "
        f"{len(args.ablation)} ablations × "
        f"{len(args.rho)} ρ × {len(args.seed)} seeds = {n_cells} cells"
    )

    for reward in args.reward:
        for ablation_name in args.ablation:
            ablation = named_ablation(
                ablation_name, refusal_rate=args.refusal_rate
            )
            for rho in args.rho:
                for seed in args.seed:
                    payoff = PayoffConfig(rho_a=rho, rho_b=rho)
                    cem_cfg = Cause3CEMConfig(
                        population_size=args.population_size,
                        n_iterations=args.n_iterations,
                        interactions_per_episode=args.interactions_per_episode,
                        reward=reward,
                        ablation=ablation,
                    )
                    report = train_cem_cause3(
                        payoff, judge, cem_config=cem_cfg, seed=seed,
                    )
                    fe = report.final_episode
                    rows.append([
                        reward, f"{rho:.2f}", seed,
                        f"{fe.claim_param:.6f}",
                        fe.claimed_label,
                        fe.n_with_target_rule_fired,
                        fe.n_accepted,
                        f"{fe.accept_rate:.6f}",
                        f"{fe.mean_p:.6f}",
                        f"{fe.proxy_toxicity:.6f}",
                        f"{fe.mean_judge_score:.6f}",
                        f"{fe.judge_toxicity:.6f}",
                        f"{fe.proxy_judge_gap:+.6f}",
                        f"{fe.mean_payoff_attempted:.6f}",
                        ablation_name,
                        f"{fe.claim_density:.6f}",
                        fe.n_claims_seen,
                        fe.n_refusals,
                        fe.n_contradicted,
                    ])
                    for metric in ABLATION_EFFECT_METRICS:
                        outcomes.setdefault(
                            (reward, ablation_name, metric), []
                        ).append(float(getattr(fe, metric)))
                    print(
                        f"  {reward:18s} {ablation_name:16s} ρ={rho:.2f}  seed={seed}  "
                        f"claim={fe.claim_param:.2f}({fe.claimed_label or 'none'})  "
                        f"density={fe.claim_density:.2f}  "
                        f"proxy_tox={fe.proxy_toxicity:.3f}  "
                        f"judge_tox={fe.judge_toxicity:.3f}  "
                        f"gap={fe.proxy_judge_gap:+.3f}"
                    )

    summary_path = out_dir / "cause3_summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "reward", "rho", "seed",
            "final_claim_param", "final_claimed_label",
            "n_target_rule_fired", "n_accepted",
            "accept_rate",
            "mean_p", "proxy_toxicity",
            "mean_judge_score", "judge_toxicity",
            "proxy_judge_gap",
            "mean_payoff_attempted",
            "ablation",
            "claim_density",
            "n_claims_seen",
            "n_refusals",
            "n_contradicted",
        ])
        w.writerows(rows)

    _write_ablation_effects(out_dir, args, outcomes)

    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
