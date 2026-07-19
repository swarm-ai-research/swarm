"""Evolutionary search for governance configurations using darwinian_evolver.

Instead of grid-sweeping 100+ governance knobs, uses evolutionary search with
optional LLM-guided mutations that reason about *why* configurations fail and
propose targeted fixes.

Requires: pip install -e ".[evolve]"
"""

from __future__ import annotations

import copy
import json
import logging
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

try:
    from darwinian_evolver.evolve_problem_loop import EvolveProblemLoop
    from darwinian_evolver.learning_log import LearningLogEntry
    from darwinian_evolver.learning_log_view import AncestorLearningLogView
    from darwinian_evolver.problem import (
        EvaluationFailureCase,
        EvaluationResult,
        Evaluator,
        Mutator,
        Organism,
        Problem,
    )
    _darwinian_evolver_available = True
except ModuleNotFoundError as exc:
    # Only treat a missing darwinian_evolver package (or submodule) as unavailable.
    if exc.name is None or not exc.name.startswith("darwinian_evolver"):
        raise
    _darwinian_evolver_available = False

    # Minimal stub base classes so this module can be imported without
    # darwinian_evolver installed.  Classes that actually require the real
    # implementations are guarded at runtime inside run_evolution().

    class Organism(BaseModel):  # type: ignore[no-redef]
        id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        from_change_summary: Optional[str] = None

    class EvaluationFailureCase(BaseModel):  # type: ignore[no-redef]
        data_point_id: str = ""
        failure_type: str = ""

    class EvaluationResult(BaseModel):  # type: ignore[no-redef]
        score: float = 0.0
        trainable_failure_cases: List[Any] = Field(default_factory=list)
        holdout_failure_cases: List[Any] = Field(default_factory=list)
        is_viable: bool = True

    class Evaluator:  # type: ignore[no-redef]
        def __class_getitem__(cls, item: Any) -> Any:
            return cls

    class Mutator:  # type: ignore[no-redef]
        def __class_getitem__(cls, item: Any) -> Any:
            return cls

    class Problem:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            pass

    class LearningLogEntry:  # type: ignore[no-redef]
        attempted_change: str = ""
        observed_outcome: str = ""

    class AncestorLearningLogView:  # type: ignore[no-redef]
        pass

    class EvolveProblemLoop:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            raise ImportError(
                "darwinian_evolver is required for evolutionary search. "
                "Install it with: pip install -e '.[evolve]'"
            )

from swarm.analysis.sweep import _apply_params, _extract_results  # noqa: E402
from swarm.scenarios import ScenarioConfig, build_orchestrator  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameter ranges for governance/payoff knobs
# ---------------------------------------------------------------------------

PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    # Governance
    "governance.transaction_tax_rate": (0.0, 1.0),
    "governance.transaction_tax_split": (0.0, 1.0),
    "governance.reputation_decay_rate": (0.0, 1.0),
    "governance.bandwidth_cap": (1, 50),
    "governance.min_stake_to_participate": (0.0, 10.0),
    "governance.stake_slash_rate": (0.0, 1.0),
    "governance.freeze_threshold_toxicity": (0.0, 1.0),
    "governance.freeze_threshold_violations": (1, 10),
    "governance.freeze_duration_epochs": (1, 10),
    "governance.audit_probability": (0.0, 1.0),
    "governance.audit_penalty_multiplier": (0.0, 10.0),
    "governance.audit_threshold_p": (0.0, 1.0),
    # Payoff
    "payoff.s_plus": (0.0, 10.0),
    "payoff.s_minus": (0.0, 10.0),
    "payoff.h": (0.0, 10.0),
    "payoff.theta": (0.0, 1.0),
    "payoff.rho_a": (0.0, 1.0),
    "payoff.rho_b": (0.0, 1.0),
    "payoff.w_rep": (0.0, 10.0),
}

# Parameters that should be integers
INT_PARAMS = {
    "governance.bandwidth_cap",
    "governance.freeze_threshold_violations",
    "governance.freeze_duration_epochs",
}


# ---------------------------------------------------------------------------
# Fitness function
# ---------------------------------------------------------------------------

DEFAULT_FITNESS_WEIGHTS = {
    "low_toxicity": 0.35,
    "welfare": 0.30,
    "quality_gap": 0.20,
    "payoff_gap": 0.15,
}


def compute_fitness(
    avg_toxicity: float,
    avg_welfare: float,
    avg_quality_gap: float,
    payoff_gap: float,
    weights: Optional[Dict[str, float]] = None,
    welfare_reference: float = 10.0,
    payoff_reference: float = 5.0,
) -> float:
    """Compute composite fitness score in [0, 1].

    Args:
        avg_toxicity: Average toxicity rate across epochs.
        avg_welfare: Average welfare per epoch.
        avg_quality_gap: Average quality gap (positive = good selection).
        payoff_gap: honest_avg_payoff - adversarial_avg_payoff.
        weights: Component weights (default: DEFAULT_FITNESS_WEIGHTS).
        welfare_reference: Reference welfare for normalization.
        payoff_reference: Reference payoff gap for normalization.
    """
    w = weights or DEFAULT_FITNESS_WEIGHTS

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    score = (
        w.get("low_toxicity", 0.35) * (1.0 - clamp(avg_toxicity))
        + w.get("welfare", 0.30) * clamp(avg_welfare / welfare_reference if welfare_reference else 0.0)
        + w.get("quality_gap", 0.20) * clamp(avg_quality_gap)
        + w.get("payoff_gap", 0.15) * clamp(payoff_gap / payoff_reference if payoff_reference else 0.0)
    )
    return clamp(score)


# ---------------------------------------------------------------------------
# Organism
# ---------------------------------------------------------------------------

class GovernanceOrganism(Organism):
    """An organism whose genome is a dict of governance/payoff parameters."""

    params: Dict[str, Any] = {}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def visualizer_props(self) -> dict[str, str | float]:
        props: dict[str, str | float] = {}
        for k, v in self.params.items():
            short_key = k.split(".")[-1]
            if isinstance(v, float):
                props[short_key] = round(v, 4)
            else:
                props[short_key] = v
        return props


# ---------------------------------------------------------------------------
# Failure cases & evaluation result
# ---------------------------------------------------------------------------

class SimulationFailureCase(EvaluationFailureCase):
    """A single epoch that failed a quality threshold."""

    model_config = ConfigDict(frozen=True)

    data_point_id: str = ""
    epoch: int = 0
    epoch_metrics: Dict[str, float] = {}
    active_params: Dict[str, Any] = {}

    @model_validator(mode="before")
    @classmethod
    def _set_data_point_id(cls, values: Any) -> Any:
        if isinstance(values, dict) and "data_point_id" not in values:
            values["data_point_id"] = f"epoch_{values.get('epoch', 0)}"
        return values


class GovernanceEvaluationResult(EvaluationResult):
    """Evaluation result enriched with governance-specific metrics."""

    model_config = ConfigDict(frozen=True)

    avg_toxicity: float = 0.0
    avg_welfare: float = 0.0
    avg_quality_gap: float = 0.0
    payoff_gap: float = 0.0
    n_frozen_agents: int = 0
    params: Dict[str, Any] = {}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def visualizer_props(self) -> dict[str, str | float]:
        return {
            "toxicity": round(self.avg_toxicity, 4),
            "welfare": round(self.avg_welfare, 2),
            "quality_gap": round(self.avg_quality_gap, 4),
            "payoff_gap": round(self.payoff_gap, 2),
            "frozen": self.n_frozen_agents,
        }

    def format_observed_outcome(
        self, parent_result: EvaluationResult | None, ndigits: int = 2
    ) -> str:
        parts = [f"score={round(self.score, ndigits)}"]
        parts.append(f"toxicity={round(self.avg_toxicity, ndigits)}")
        parts.append(f"welfare={round(self.avg_welfare, ndigits)}")
        parts.append(f"quality_gap={round(self.avg_quality_gap, ndigits)}")
        if parent_result is not None:
            delta = round(self.score - parent_result.score, ndigits)
            parts.append(f"delta={'+' if delta >= 0 else ''}{delta}")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class GovernanceEvaluator(
    Evaluator[GovernanceOrganism, GovernanceEvaluationResult, SimulationFailureCase]
):
    """Evaluate a governance configuration by running a simulation."""

    def __init__(
        self,
        base_scenario: ScenarioConfig,
        eval_epochs: int = 3,
        eval_steps: int = 5,
        seed_base: int = 42,
        fitness_weights: Optional[Dict[str, float]] = None,
        welfare_reference: float = 10.0,
        payoff_reference: float = 5.0,
        toxicity_threshold: float = 0.3,
        quality_gap_floor: float = 0.0,
        welfare_floor: float = 0.0,
        payoff_gap_floor: float = 0.0,
        trainable_split: float = 0.7,
        viability_threshold: float = 0.95,
    ):
        self.base_scenario = base_scenario
        self.eval_epochs = eval_epochs
        self.eval_steps = eval_steps
        self.seed_base = seed_base
        self.fitness_weights = fitness_weights
        self.welfare_reference = welfare_reference
        self.payoff_reference = payoff_reference
        self.toxicity_threshold = toxicity_threshold
        self.quality_gap_floor = quality_gap_floor
        self.welfare_floor = welfare_floor
        self.payoff_gap_floor = payoff_gap_floor
        self.trainable_split = trainable_split
        self.viability_threshold = viability_threshold

    def evaluate(
        self, organism: GovernanceOrganism
    ) -> GovernanceEvaluationResult:
        """Run simulation and score the organism."""
        try:
            return self._evaluate_impl(organism)
        except Exception as exc:
            logger.warning("Evaluation failed for params %s: %s", organism.params, exc)
            return GovernanceEvaluationResult(
                score=0.0,
                trainable_failure_cases=[],
                holdout_failure_cases=[],
                is_viable=False,
                avg_toxicity=1.0,
                avg_welfare=0.0,
                avg_quality_gap=0.0,
                payoff_gap=0.0,
                n_frozen_agents=0,
                params=organism.params,
            )

    def _evaluate_impl(
        self, organism: GovernanceOrganism
    ) -> GovernanceEvaluationResult:
        scenario = copy.deepcopy(self.base_scenario)
        scenario = _apply_params(scenario, organism.params)
        scenario.orchestrator_config.seed = self.seed_base
        scenario.orchestrator_config.n_epochs = self.eval_epochs
        scenario.orchestrator_config.steps_per_epoch = self.eval_steps

        # Disable log path for evaluation runs
        scenario.orchestrator_config.log_path = None
        scenario.orchestrator_config.log_events = False

        orchestrator = build_orchestrator(scenario)
        orchestrator.run()

        # Extract sweep-style results
        sweep_result = _extract_results(orchestrator, organism.params, 0, self.seed_base)

        # Per-epoch metrics for failure case detection
        metrics_history = orchestrator.get_metrics_history()

        trainable_failures: List[SimulationFailureCase] = []
        holdout_failures: List[SimulationFailureCase] = []

        n_epochs = len(metrics_history)
        trainable_cutoff = int(n_epochs * self.trainable_split)

        for i, m in enumerate(metrics_history):
            epoch_metrics = {
                "toxicity_rate": m.toxicity_rate,
                "quality_gap": m.quality_gap,
                "total_welfare": m.total_welfare,
            }

            failure_type: Optional[str] = None
            if m.toxicity_rate > self.toxicity_threshold:
                failure_type = "high_toxicity"
            elif m.quality_gap < self.quality_gap_floor:
                failure_type = "adverse_selection"
            elif m.total_welfare < self.welfare_floor:
                failure_type = "low_welfare"

            if failure_type is not None:
                fc = SimulationFailureCase(
                    epoch=m.epoch,
                    epoch_metrics=epoch_metrics,
                    active_params=organism.params,
                    failure_type=failure_type,
                )
                if i < trainable_cutoff:
                    trainable_failures.append(fc)
                else:
                    holdout_failures.append(fc)

        # Payoff gap check (aggregate, not per-epoch)
        payoff_gap = sweep_result.honest_avg_payoff - sweep_result.adversarial_avg_payoff
        if payoff_gap < self.payoff_gap_floor:
            fc = SimulationFailureCase(
                epoch=n_epochs - 1 if n_epochs else 0,
                epoch_metrics={"payoff_gap": payoff_gap},
                active_params=organism.params,
                failure_type="bad_payoff_gap",
            )
            trainable_failures.append(fc)

        # Ensure at least one trainable failure case so the organism can be
        # selected as a parent by darwinian_evolver (requires non-empty
        # trainable_failure_cases for parent eligibility).
        if not trainable_failures:
            trainable_failures.append(
                SimulationFailureCase(
                    epoch=0,
                    epoch_metrics={
                        "toxicity_rate": sweep_result.avg_toxicity,
                        "quality_gap": sweep_result.avg_quality_gap,
                        "total_welfare": sweep_result.welfare_per_epoch,
                        "payoff_gap": payoff_gap,
                    },
                    active_params=organism.params,
                    failure_type="room_for_improvement",
                )
            )

        fitness = compute_fitness(
            avg_toxicity=sweep_result.avg_toxicity,
            avg_welfare=sweep_result.welfare_per_epoch,
            avg_quality_gap=sweep_result.avg_quality_gap,
            payoff_gap=payoff_gap,
            weights=self.fitness_weights,
            welfare_reference=self.welfare_reference,
            payoff_reference=self.payoff_reference,
        )

        is_viable = sweep_result.avg_toxicity <= self.viability_threshold

        return GovernanceEvaluationResult(
            score=fitness,
            trainable_failure_cases=trainable_failures,  # type: ignore[arg-type]
            holdout_failure_cases=holdout_failures,  # type: ignore[arg-type]
            is_viable=is_viable,
            avg_toxicity=sweep_result.avg_toxicity,
            avg_welfare=sweep_result.welfare_per_epoch,
            avg_quality_gap=sweep_result.avg_quality_gap,
            payoff_gap=payoff_gap,
            n_frozen_agents=sweep_result.n_frozen,
            params=organism.params,
        )


# ---------------------------------------------------------------------------
# Mutators
# ---------------------------------------------------------------------------

class RandomGovernanceMutator(Mutator[GovernanceOrganism, SimulationFailureCase]):
    """Perturb 1-3 parameters with bounded random deltas."""

    def __init__(
        self,
        n_offspring: int = 3,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.n_offspring = n_offspring
        self.param_ranges = param_ranges or PARAM_RANGES

    def mutate(
        self,
        organism: GovernanceOrganism,
        failure_cases: list[SimulationFailureCase],
        learning_log_entries: list[LearningLogEntry],
    ) -> list[GovernanceOrganism]:
        children: list[GovernanceOrganism] = []
        available_params = list(self.param_ranges.keys())

        for _ in range(self.n_offspring):
            new_params = dict(organism.params)
            n_mutations = random.randint(1, min(3, len(available_params)))
            chosen = random.sample(available_params, n_mutations)

            changes: list[str] = []
            for param_name in chosen:
                lo, hi = self.param_ranges[param_name]
                current = new_params.get(param_name, (lo + hi) / 2)

                # Perturb by up to 20% of range
                delta_max = (hi - lo) * 0.2
                delta = random.uniform(-delta_max, delta_max)
                new_val = max(lo, min(hi, current + delta))

                if param_name in INT_PARAMS:
                    new_val = int(round(new_val))

                new_params[param_name] = new_val
                changes.append(f"{param_name}: {current:.4g} -> {new_val:.4g}")

            child = GovernanceOrganism(
                params=new_params,
                from_change_summary=f"Random perturbation: {'; '.join(changes)}",
            )
            children.append(child)

        return children

    @property
    def supports_batch_mutation(self) -> bool:
        return True


class LLMGovernanceMutator(Mutator[GovernanceOrganism, SimulationFailureCase]):
    """Use an LLM to propose targeted governance parameter changes."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        n_offspring: int = 3,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        fallback_mutator: Optional[RandomGovernanceMutator] = None,
    ):
        self.model = model
        self.n_offspring = n_offspring
        self.param_ranges = param_ranges or PARAM_RANGES
        self.fallback = fallback_mutator or RandomGovernanceMutator(n_offspring=n_offspring)

    @property
    def supports_batch_mutation(self) -> bool:
        return True

    def mutate(
        self,
        organism: GovernanceOrganism,
        failure_cases: list[SimulationFailureCase],
        learning_log_entries: list[LearningLogEntry],
    ) -> list[GovernanceOrganism]:
        try:
            return self._llm_mutate(organism, failure_cases, learning_log_entries)
        except Exception as exc:
            logger.warning("LLM mutation failed, falling back to random: %s", exc)
            return self.fallback.mutate(organism, failure_cases, learning_log_entries)

    def _llm_mutate(
        self,
        organism: GovernanceOrganism,
        failure_cases: list[SimulationFailureCase],
        learning_log_entries: list[LearningLogEntry],
    ) -> list[GovernanceOrganism]:
        import anthropic

        prompt = self._build_prompt(organism, failure_cases, learning_log_entries)

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text  # type: ignore[union-attr]
        return self._parse_response(organism, text)

    def _build_prompt(
        self,
        organism: GovernanceOrganism,
        failure_cases: list[SimulationFailureCase],
        learning_log_entries: list[LearningLogEntry],
    ) -> str:
        parts = [
            "You are optimizing governance parameters for a multi-agent safety simulation.",
            "",
            "## Current Parameters",
            json.dumps(organism.params, indent=2),
            "",
            "## Parameter Ranges",
        ]
        for name, (lo, hi) in sorted(self.param_ranges.items()):
            is_int = name in INT_PARAMS
            type_hint = " (integer)" if is_int else ""
            parts.append(f"- {name}: [{lo}, {hi}]{type_hint}")

        parts.append("")
        parts.append("## Failure Cases")
        for fc in failure_cases[:10]:  # limit context
            parts.append(
                f"- Epoch {fc.epoch}: {fc.failure_type} | metrics={json.dumps(fc.epoch_metrics)}"
            )

        if learning_log_entries:
            parts.append("")
            parts.append("## Learning History (recent ancestor mutations)")
            for entry in learning_log_entries[-5:]:
                parts.append(f"- Tried: {entry.attempted_change}")
                parts.append(f"  Result: {entry.observed_outcome}")

        parts.extend([
            "",
            "## Task",
            f"Propose exactly {self.n_offspring} different parameter change sets that address the failures.",
            "Each set should be a JSON object mapping parameter names to new values.",
            "Wrap each JSON object in a ```json fenced code block.",
            "Include a brief rationale before each block.",
        ])

        return "\n".join(parts)

    def _parse_response(
        self,
        organism: GovernanceOrganism,
        text: str,
    ) -> list[GovernanceOrganism]:
        # Extract JSON blocks from fenced code blocks
        pattern = r"```(?:json)?\s*\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)

        children: list[GovernanceOrganism] = []
        for match in matches:
            try:
                changes = json.loads(match)
                if not isinstance(changes, dict):
                    continue

                # Merge with parent params and validate
                new_params = dict(organism.params)
                change_parts: list[str] = []
                for key, val in changes.items():
                    if key not in self.param_ranges:
                        continue
                    lo, hi = self.param_ranges[key]
                    if key in INT_PARAMS:
                        val = int(round(val))
                    val = max(lo, min(hi, val))
                    old_val = new_params.get(key, "unset")
                    new_params[key] = val
                    change_parts.append(f"{key}: {old_val} -> {val}")

                if not change_parts:
                    continue

                child = GovernanceOrganism(
                    params=new_params,
                    from_change_summary=f"LLM mutation: {'; '.join(change_parts)}",
                )
                children.append(child)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

        if not children:
            raise ValueError("No valid offspring parsed from LLM response")

        return children


# ---------------------------------------------------------------------------
# Config & Runner
# ---------------------------------------------------------------------------

@dataclass
class EvolverConfig:
    """Configuration for evolutionary governance search."""

    base_scenario: ScenarioConfig
    initial_params: Dict[str, Any] = field(default_factory=dict)
    n_iterations: int = 20
    num_parents_per_iteration: int = 3
    eval_epochs: int = 3
    eval_steps: int = 5
    final_eval_epochs: int = 10
    final_eval_steps: int = 10
    fitness_weights: Optional[Dict[str, float]] = None
    seed_base: int = 42
    use_llm_mutator: bool = True
    llm_model: str = "claude-sonnet-4-20250514"
    output_dir: Optional[Path] = None
    resume_from: Optional[Path] = None


@dataclass
class EvolutionResult:
    """Result from an evolutionary search run."""

    best_params: Dict[str, Any]
    best_score: float
    best_metrics: Dict[str, float]
    n_iterations: int
    final_population_size: int
    run_dir: Optional[Path] = None


def run_evolution(config: EvolverConfig) -> EvolutionResult:
    """Run evolutionary governance search.

    Args:
        config: Evolution configuration.

    Returns:
        EvolutionResult with best organism and metrics.
    """
    # Set up output directory
    run_dir: Optional[Path] = None
    if config.output_dir:
        run_dir = Path(config.output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        scenario_id = config.base_scenario.scenario_id or "unknown"
        run_dir = Path(f"runs/{timestamp}_evolve_{scenario_id}")

    run_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = run_dir / "snapshots"
    snapshots_dir.mkdir(exist_ok=True)

    # Save config
    config_data = {
        "scenario_id": config.base_scenario.scenario_id,
        "n_iterations": config.n_iterations,
        "num_parents_per_iteration": config.num_parents_per_iteration,
        "eval_epochs": config.eval_epochs,
        "eval_steps": config.eval_steps,
        "final_eval_epochs": config.final_eval_epochs,
        "final_eval_steps": config.final_eval_steps,
        "fitness_weights": config.fitness_weights or DEFAULT_FITNESS_WEIGHTS,
        "seed_base": config.seed_base,
        "use_llm_mutator": config.use_llm_mutator,
        "llm_model": config.llm_model,
        "initial_params": config.initial_params,
    }
    (run_dir / "config.json").write_text(json.dumps(config_data, indent=2))

    # Build components
    initial_organism = GovernanceOrganism(params=dict(config.initial_params))

    evaluator = GovernanceEvaluator(
        base_scenario=config.base_scenario,
        eval_epochs=config.eval_epochs,
        eval_steps=config.eval_steps,
        seed_base=config.seed_base,
        fitness_weights=config.fitness_weights,
    )

    mutators: list[Mutator] = [RandomGovernanceMutator()]
    if config.use_llm_mutator:
        mutators.append(
            LLMGovernanceMutator(
                model=config.llm_model,
                fallback_mutator=RandomGovernanceMutator(),
            )
        )

    problem = Problem(
        initial_organism=initial_organism,
        evaluator=evaluator,
        mutators=mutators,
    )

    # Resume or start fresh
    snapshot_to_resume: Optional[bytes] = None
    if config.resume_from and config.resume_from.exists():
        snapshot_to_resume = config.resume_from.read_bytes()
        logger.info("Resuming from snapshot: %s", config.resume_from)

    loop = EvolveProblemLoop(
        problem=problem,
        learning_log_view_type=(AncestorLearningLogView, {"max_depth": 5}),
        num_parents_per_iteration=config.num_parents_per_iteration,
        snapshot_to_resume_from=snapshot_to_resume,
        evaluator_concurrency=1,  # simulations are CPU-bound
        mutator_concurrency=3,
    )

    # Open population log
    pop_log = open(run_dir / "population_log.jsonl", "a")

    best_organism: Optional[GovernanceOrganism] = None
    best_result: Optional[GovernanceEvaluationResult] = None
    final_population_size = 0

    try:
        for snapshot in loop.run(num_iterations=config.n_iterations):
            iteration = snapshot.iteration
            org, result = snapshot.best_organism_result
            final_population_size = snapshot.population_size

            logger.info(
                "Iteration %d: best_score=%.4f pop_size=%d",
                iteration, result.score, snapshot.population_size,
            )

            # Track best
            best_organism = org  # type: ignore[assignment]
            best_result = result  # type: ignore[assignment]

            # Write snapshot
            snapshot_path = snapshots_dir / f"iter_{iteration:03d}.pkl"
            snapshot_path.write_bytes(snapshot.snapshot)

            # Append to population log
            log_entry = {
                "iteration": iteration,
                "population_size": snapshot.population_size,
                "best_score": result.score,
                "score_percentiles": snapshot.score_percentiles,
            }
            pop_log.write(json.dumps(log_entry) + "\n")
            pop_log.flush()

            print(
                f"  Iter {iteration:3d} | "
                f"score={result.score:.4f} | "
                f"pop={snapshot.population_size}"
            )
    finally:
        pop_log.close()

    if best_organism is None or best_result is None:
        raise RuntimeError("Evolution produced no results")

    # Save best organism
    best_data = {
        "params": best_organism.params,
        "id": str(best_organism.id),
    }
    (run_dir / "best_organism.json").write_text(json.dumps(best_data, indent=2))

    # Final validation with longer simulation
    print("\nRunning final validation...")
    final_evaluator = GovernanceEvaluator(
        base_scenario=config.base_scenario,
        eval_epochs=config.final_eval_epochs,
        eval_steps=config.final_eval_steps,
        seed_base=config.seed_base,
        fitness_weights=config.fitness_weights,
    )
    final_result = final_evaluator.evaluate(best_organism)

    final_data = {
        "score": final_result.score,
        "avg_toxicity": final_result.avg_toxicity,
        "avg_welfare": final_result.avg_welfare,
        "avg_quality_gap": final_result.avg_quality_gap,
        "payoff_gap": final_result.payoff_gap,
        "n_frozen_agents": final_result.n_frozen_agents,
        "is_viable": final_result.is_viable,
        "params": best_organism.params,
    }
    (run_dir / "final_eval.json").write_text(json.dumps(final_data, indent=2))

    # Summary
    summary = {
        "best_score": final_result.score,
        "best_params": best_organism.params,
        "n_iterations": config.n_iterations,
        "final_population_size": final_population_size,
        "final_metrics": {
            "avg_toxicity": final_result.avg_toxicity,
            "avg_welfare": final_result.avg_welfare,
            "avg_quality_gap": final_result.avg_quality_gap,
            "payoff_gap": final_result.payoff_gap,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nEvolution complete. Results in {run_dir}/")
    print(f"  Best score: {final_result.score:.4f}")
    print(f"  Toxicity:   {final_result.avg_toxicity:.4f}")
    print(f"  Welfare:    {final_result.avg_welfare:.2f}")

    return EvolutionResult(
        best_params=best_organism.params,
        best_score=final_result.score,
        best_metrics={
            "avg_toxicity": final_result.avg_toxicity,
            "avg_welfare": final_result.avg_welfare,
            "avg_quality_gap": final_result.avg_quality_gap,
            "payoff_gap": final_result.payoff_gap,
            "n_frozen_agents": final_result.n_frozen_agents,
        },
        n_iterations=config.n_iterations,
        final_population_size=final_population_size,
        run_dir=run_dir,
    )
