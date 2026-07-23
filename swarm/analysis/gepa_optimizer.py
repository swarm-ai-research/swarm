"""GEPA optimize_anything integration for governance parameter search.

Uses GEPA's LLM-guided Pareto-efficient search to optimize governance and
payoff parameters against our soft safety metrics (toxicity, quality gap,
welfare, payoff gap).

Requires: pip install gepa

Usage:
    python -m swarm.analysis.gepa_optimizer scenarios/baseline.yaml \\
        --max-evals 50 --seed 42

    # With custom fitness weights
    python -m swarm.analysis.gepa_optimizer scenarios/baseline.yaml \\
        --max-evals 100 --weight-toxicity 0.5 --weight-welfare 0.2
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from swarm.analysis.evolver import (
    DEFAULT_FITNESS_WEIGHTS,
    INT_PARAMS,
    PARAM_RANGES,
    compute_fitness,
)
from swarm.analysis.sweep import _apply_params, _extract_results
from swarm.scenarios import ScenarioConfig, build_orchestrator, load_scenario

logger = logging.getLogger(__name__)


def _params_to_yaml(params: Dict[str, Any]) -> str:
    """Serialize governance/payoff params to a YAML string for GEPA."""
    nested: Dict[str, Dict[str, Any]] = {}
    for dotted_key, value in sorted(params.items()):
        section, _, key = dotted_key.partition(".")
        nested.setdefault(section, {})[key] = value
    return yaml.dump(nested, default_flow_style=False, sort_keys=True)


def _yaml_to_params(text: str) -> Dict[str, Any]:
    """Parse GEPA candidate YAML back into dotted param dict."""
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML dict, got {type(data)}")
    params: Dict[str, Any] = {}
    for section, keys in data.items():
        if not isinstance(keys, dict):
            continue
        for key, value in keys.items():
            dotted = f"{section}.{key}"
            if dotted in PARAM_RANGES:
                lo, hi = PARAM_RANGES[dotted]
                if dotted in INT_PARAMS:
                    value = int(round(float(value)))
                else:
                    value = float(value)
                value = max(lo, min(hi, value))
                params[dotted] = value
    return params


def make_evaluator(
    base_scenario: ScenarioConfig,
    eval_epochs: int = 3,
    eval_steps: int = 5,
    seed: int = 42,
    fitness_weights: Optional[Dict[str, float]] = None,
    welfare_reference: float = 10.0,
    payoff_reference: float = 5.0,
):
    """Create a GEPA-compatible evaluator function.

    Returns a callable that takes a YAML config string and returns
    (score, side_info_dict).
    """
    try:
        import gepa.optimize_anything as oa
    except ImportError:
        oa = None  # type: ignore[assignment]

    def evaluate(candidate: str) -> Tuple[float, Dict[str, Any]]:
        try:
            params = _yaml_to_params(candidate)
        except Exception as exc:
            if oa is not None:
                oa.log(f"YAML parse error: {exc}")
            return 0.0, {"error": str(exc)}

        if not params:
            if oa is not None:
                oa.log("No valid parameters found in candidate")
            return 0.0, {"error": "no valid parameters"}

        # Run simulation
        scenario = copy.deepcopy(base_scenario)
        scenario = _apply_params(scenario, params)
        scenario.orchestrator_config.seed = seed
        scenario.orchestrator_config.n_epochs = eval_epochs
        scenario.orchestrator_config.steps_per_epoch = eval_steps
        scenario.orchestrator_config.log_path = None
        scenario.orchestrator_config.log_events = False

        try:
            orchestrator = build_orchestrator(scenario)
            orchestrator.run()
        except Exception as exc:
            if oa is not None:
                oa.log(f"Simulation error: {exc}")
            return 0.0, {"error": f"simulation failed: {exc}"}

        result = _extract_results(orchestrator, params, 0, seed)

        payoff_gap = result.honest_avg_payoff - result.adversarial_avg_payoff

        fitness = compute_fitness(
            avg_toxicity=result.avg_toxicity,
            avg_welfare=result.welfare_per_epoch,
            avg_quality_gap=result.avg_quality_gap,
            payoff_gap=payoff_gap,
            weights=fitness_weights,
            welfare_reference=welfare_reference,
            payoff_reference=payoff_reference,
        )

        # Build diagnostic side info for GEPA's ASI
        side_info = {
            "toxicity": round(result.avg_toxicity, 4),
            "welfare": round(result.welfare_per_epoch, 4),
            "quality_gap": round(result.avg_quality_gap, 4),
            "payoff_gap": round(payoff_gap, 4),
            "honest_avg_payoff": round(result.honest_avg_payoff, 4),
            "adversarial_avg_payoff": round(result.adversarial_avg_payoff, 4),
            "n_frozen": result.n_frozen,
            "params_used": params,
        }

        # Log diagnostics for GEPA reflection
        if oa is not None:
            oa.log(
                f"fitness={fitness:.4f} | "
                f"toxicity={result.avg_toxicity:.4f} | "
                f"welfare={result.welfare_per_epoch:.2f} | "
                f"quality_gap={result.avg_quality_gap:.4f} | "
                f"payoff_gap={payoff_gap:.2f}"
            )
            if result.avg_toxicity > 0.3:
                oa.log("WARNING: High toxicity — consider raising audit_probability or stake_slash_rate")
            if result.avg_quality_gap < 0.0:
                oa.log("WARNING: Adverse selection — quality_gap is negative")
            if payoff_gap < 0:
                oa.log("WARNING: Adversarial agents outearning honest ones")

        return fitness, side_info

    return evaluate


def run_gepa_optimization(
    scenario_path: str,
    max_evals: int = 50,
    seed: int = 42,
    eval_epochs: int = 3,
    eval_steps: int = 5,
    fitness_weights: Optional[Dict[str, float]] = None,
    reflection_model: str = "anthropic/claude-sonnet-4-20250514",
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run GEPA optimize_anything on governance parameters.

    Args:
        scenario_path: Path to base scenario YAML.
        max_evals: Maximum number of evaluations (budget).
        seed: Random seed for simulations.
        eval_epochs: Epochs per evaluation run.
        eval_steps: Steps per epoch per evaluation.
        fitness_weights: Custom fitness component weights.
        reflection_model: LLM model for GEPA reflection.
        output_dir: Where to write results.

    Returns:
        Dict with best_params, best_score, history, and run_dir.
    """
    from gepa.optimize_anything import (
        EngineConfig,
        GEPAConfig,
        ReflectionConfig,
        optimize_anything,
    )

    # Load base scenario
    scenario = load_scenario(Path(scenario_path))

    # Build seed candidate from scenario's current params
    orch_cfg = scenario.orchestrator_config
    seed_params: Dict[str, Any] = {}
    for dotted_key in PARAM_RANGES:
        section, _, key = dotted_key.partition(".")
        if section == "governance" and orch_cfg.governance_config is not None:
            if hasattr(orch_cfg.governance_config, key):
                seed_params[dotted_key] = getattr(orch_cfg.governance_config, key)
        elif section == "payoff" and hasattr(orch_cfg.payoff_config, key):
            seed_params[dotted_key] = getattr(orch_cfg.payoff_config, key)
    seed_candidate = _params_to_yaml(seed_params)

    # Set up run directory
    if output_dir:
        run_dir = Path(output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        scenario_id = scenario.scenario_id or "unknown"
        run_dir = Path(f"runs/{timestamp}_gepa_{scenario_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build evaluator
    evaluator = make_evaluator(
        base_scenario=scenario,
        eval_epochs=eval_epochs,
        eval_steps=eval_steps,
        seed=seed,
        fitness_weights=fitness_weights,
    )

    # Build param range description for GEPA background
    range_lines = []
    for name, (lo, hi) in sorted(PARAM_RANGES.items()):
        type_hint = " (integer)" if name in INT_PARAMS else ""
        range_lines.append(f"  {name}: [{lo}, {hi}]{type_hint}")

    background = (
        "You are optimizing governance and payoff parameters for a multi-agent "
        "safety simulation framework.\n\n"
        "The simulation has honest, opportunistic, and deceptive agents interacting "
        "under configurable governance rules. The goal is to maximize a composite "
        "fitness score that rewards:\n"
        "- Low toxicity (fraction of harmful interactions accepted)\n"
        "- High welfare (total agent payoffs)\n"
        "- Positive quality gap (good interactions accepted over bad — negative means adverse selection)\n"
        "- Positive payoff gap (honest agents should earn more than adversarial ones)\n\n"
        "Parameter ranges (values outside these are clamped):\n"
        + "\n".join(range_lines) + "\n\n"
        "The YAML candidate has two sections: 'governance' and 'payoff'.\n"
        "Key trade-offs:\n"
        "- Higher transaction_tax_rate discourages spam but reduces welfare\n"
        "- Higher audit_probability catches deceptive agents but costs resources\n"
        "- stake_slash_rate punishes bad behavior but may freeze agents too aggressively\n"
        "- rho_a/rho_b control externality internalization — higher values make agents "
        "bear more ecosystem cost\n"
        "- theta is the acceptance threshold — higher rejects more but risks losing good trades\n"
    )

    # Configure GEPA
    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=max_evals,
            seed=seed,
            run_dir=str(run_dir / "gepa_run"),
            capture_stdio=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm=reflection_model,
        ),
    )

    # Save our config
    our_config = {
        "scenario_path": scenario_path,
        "max_evals": max_evals,
        "seed": seed,
        "eval_epochs": eval_epochs,
        "eval_steps": eval_steps,
        "fitness_weights": fitness_weights or DEFAULT_FITNESS_WEIGHTS,
        "reflection_model": reflection_model,
    }
    (run_dir / "config.json").write_text(json.dumps(our_config, indent=2))

    print(f"Starting GEPA optimization: {max_evals} evals, seed={seed}")
    print(f"Scenario: {scenario_path}")
    print(f"Output: {run_dir}/")
    print(f"Reflection LM: {reflection_model}")
    print()

    # Run GEPA
    result = optimize_anything(
        seed_candidate=seed_candidate,
        evaluator=evaluator,
        objective=(
            "Find governance and payoff parameters that minimize toxicity while "
            "maintaining high welfare, positive quality gap (no adverse selection), "
            "and ensuring honest agents earn more than adversarial ones."
        ),
        background=background,
        config=config,
    )

    # Extract best. gepa 0.1.0: best_candidate is the candidate mapping
    # (component name -> text) — or a bare string for str-seeded runs; the
    # score/eval-count attrs this code originally used (best_score,
    # num_evaluations) do not exist on GEPAResult and would AttributeError.
    best_candidate = result.best_candidate
    best_yaml = (
        best_candidate
        if isinstance(best_candidate, str)
        else next(iter(best_candidate.values()))
    )
    best_params = _yaml_to_params(best_yaml)
    best_score = result.val_aggregate_scores[result.best_idx]
    n_evaluations = result.total_metric_calls

    # Save results
    summary = {
        "best_score": best_score,
        "best_params": best_params,
        "best_candidate_yaml": best_yaml,
        "n_evaluations": n_evaluations,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "best_candidate.yaml").write_text(best_yaml)

    print("\nOptimization complete!")
    print(f"  Best score: {best_score:.4f}")
    print(f"  Evaluations: {n_evaluations}")
    print(f"  Results: {run_dir}/")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize governance parameters using GEPA optimize_anything"
    )
    parser.add_argument("scenario", help="Path to base scenario YAML")
    parser.add_argument("--max-evals", type=int, default=50, help="Max evaluations (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--eval-epochs", type=int, default=3, help="Epochs per eval (default: 3)")
    parser.add_argument("--eval-steps", type=int, default=5, help="Steps per epoch per eval (default: 5)")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4-20250514",
                        help="Reflection LLM model")
    parser.add_argument("--output-dir", help="Output directory (default: runs/<timestamp>_gepa_<scenario>)")
    parser.add_argument("--weight-toxicity", type=float, default=0.35)
    parser.add_argument("--weight-welfare", type=float, default=0.30)
    parser.add_argument("--weight-quality-gap", type=float, default=0.20)
    parser.add_argument("--weight-payoff-gap", type=float, default=0.15)

    args = parser.parse_args()

    fitness_weights = {
        "low_toxicity": args.weight_toxicity,
        "welfare": args.weight_welfare,
        "quality_gap": args.weight_quality_gap,
        "payoff_gap": args.weight_payoff_gap,
    }

    run_gepa_optimization(
        scenario_path=args.scenario,
        max_evals=args.max_evals,
        seed=args.seed,
        eval_epochs=args.eval_epochs,
        eval_steps=args.eval_steps,
        fitness_weights=fitness_weights,
        reflection_model=args.model,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
