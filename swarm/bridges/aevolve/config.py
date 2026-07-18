"""Configuration for the SWARM ↔ A-Evolve benchmark bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# Governance/payoff knobs an evolved candidate may override. Everything else
# in the scenario (agent mix, epochs, seed, success criteria) is fixed by the
# benchmark — an agent that could shrink the simulation or delete the
# deceptive agents would be gaming the harness, not designing governance.
DEFAULT_ALLOWED_OVERRIDES: Tuple[str, ...] = (
    "governance.transaction_tax_rate",
    "governance.transaction_tax_split",
    "governance.reputation_decay_rate",
    "governance.bandwidth_cap",
    "governance.staking_enabled",
    "governance.min_stake_to_participate",
    "governance.circuit_breaker_enabled",
    "governance.audit_enabled",
    "payoff.theta",
    "payoff.rho_a",
    "payoff.rho_b",
    "payoff.w_rep",
)


@dataclass
class AevolveBridgeConfig:
    """Knobs for :class:`~swarm.bridges.aevolve.adapter.SwarmBenchmarkAdapter`.

    ``score_mode``:

    - ``one_minus_toxicity`` (default): ``1 - E[1-p | accepted]`` from the
      final epoch — high score = the evolved governance kept accepted
      interactions clean.
    - ``s_soft``: mean expected surplus per interaction, squashed to [0, 1]
      via ``score = 1 / (1 + exp(-S_soft))`` so Feedback stays in range.
    """

    scenario_path: str = "scenarios/aevolve_screening.yaml"
    n_tasks: int = 4  # tasks are (scenario, seed) cells: seed_base + index
    seed_base: int = 1000
    holdout_seed_base: int = 9000  # disjoint seeds for split="holdout"/"test"
    score_mode: str = "one_minus_toxicity"  # or "s_soft"
    success_threshold: float = 0.7
    # Keep evolution cycles cheap: cap the simulation size regardless of what
    # the scenario file says. None = respect the scenario.
    max_epochs: Optional[int] = 5
    max_steps_per_epoch: Optional[int] = 10
    allowed_overrides: Tuple[str, ...] = DEFAULT_ALLOWED_OVERRIDES

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_path": self.scenario_path,
            "n_tasks": self.n_tasks,
            "seed_base": self.seed_base,
            "holdout_seed_base": self.holdout_seed_base,
            "score_mode": self.score_mode,
            "success_threshold": self.success_threshold,
            "max_epochs": self.max_epochs,
            "max_steps_per_epoch": self.max_steps_per_epoch,
            "allowed_overrides": list(self.allowed_overrides),
        }


@dataclass
class EvaluationDetail:
    """Everything the evolver gets to see about one evaluated candidate."""

    scenario_id: str = ""
    seed: int = 0
    applied_overrides: Dict[str, Any] = field(default_factory=dict)
    rejected_overrides: Dict[str, Any] = field(default_factory=dict)
    toxicity: float = 0.0
    quality_gap: float = 0.0
    total_welfare: float = 0.0
    total_interactions: int = 0
    accepted_interactions: int = 0
    parse_error: str = ""

    def to_text(self) -> str:
        """Rich diagnostic text for ``Feedback.detail`` (evolver-facing)."""
        if self.parse_error:
            return (
                f"REJECTED: {self.parse_error}. Emit a JSON object of "
                f"governance overrides, e.g. "
                f'{{"governance.transaction_tax_rate": 0.1}}.'
            )
        lines = [
            f"scenario={self.scenario_id} seed={self.seed}",
            f"applied_overrides={self.applied_overrides}",
        ]
        if self.rejected_overrides:
            lines.append(
                f"rejected_overrides (not in whitelist): {self.rejected_overrides}"
            )
        lines.append(
            f"final epoch: toxicity={self.toxicity:.4f} "
            f"quality_gap={self.quality_gap:+.4f} "
            f"welfare={self.total_welfare:.2f} "
            f"interactions={self.total_interactions} "
            f"({self.accepted_interactions} accepted)"
        )
        if self.quality_gap < 0:
            lines.append(
                "adverse selection present: accepted interactions are lower-"
                "quality than rejected ones — governance is selecting badly"
            )
        return "\n".join(lines)
