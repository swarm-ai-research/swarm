"""Governed evaluation loop for EvoSkill candidate programs.

Wraps EvoSkill's evaluation step to run each candidate program through
both an **oracle** (ungoverned) and a **governed** SWARM simulation,
producing a composite fitness score that includes governance effects.

The governance attribution signal is:

    delta = metric(governed) - metric(oracle)

This tells us whether governance *helps or hurts* the candidate's
performance, and is a more informative training signal than raw score.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from swarm.bridges.evoskill.config import EvoSkillConfig
from swarm.bridges.evoskill.translator import SkillTranslator
from swarm.contracts.contract import ContractType
from swarm.core.payoff import PayoffConfig, SoftPayoffEngine
from swarm.core.proxy import ProxyComputer
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.interaction import InteractionType, SoftInteraction
from swarm.skills.library import SharingMode, SkillLibrary, SkillLibraryConfig
from swarm.skills.model import Skill, clamp_p

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of evaluating a single candidate program.

    Attributes:
        program_id: Identifier for the candidate program (git branch/hash).
        benchmark_score: Raw EvoSkill benchmark score (0–1).
        oracle_metrics: Metrics from the ungoverned oracle run.
        governed_metrics: Metrics from the governed run (per regime).
        governance_delta: Per-regime governance attribution signal.
        composite_score: Blended fitness score incorporating governance.
        regime: Contract regime this was evaluated under.
        skills_ingested: Number of skills translated from the program.
        provenance_ids: Provenance record IDs for audit trail.
    """

    program_id: str = ""
    benchmark_score: float = 0.0
    oracle_metrics: Dict[str, float] = field(default_factory=dict)
    governed_metrics: Dict[str, float] = field(default_factory=dict)
    governance_delta: Dict[str, float] = field(default_factory=dict)
    composite_score: float = 0.0
    regime: str = ""
    skills_ingested: int = 0
    provenance_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging and comparison."""
        return {
            "program_id": self.program_id,
            "benchmark_score": self.benchmark_score,
            "oracle_metrics": self.oracle_metrics,
            "governed_metrics": self.governed_metrics,
            "governance_delta": self.governance_delta,
            "composite_score": self.composite_score,
            "regime": self.regime,
            "skills_ingested": self.skills_ingested,
        }


class GovernedEvalLoop:
    """Evaluates EvoSkill candidate programs under SWARM governance.

    For each candidate program, the loop:

    1. Translates the program's skills into SWARM ``Skill`` objects.
    2. Runs an **oracle** evaluation (no governance, baseline payoffs).
    3. Runs a **governed** evaluation under the specified contract regime.
    4. Computes the governance delta and a composite fitness score.

    The composite score blends benchmark performance with governance
    effects, controlled by ``config.governance_weight``.
    """

    def __init__(
        self,
        config: Optional[EvoSkillConfig] = None,
        translator: Optional[SkillTranslator] = None,
        payoff_config: Optional[PayoffConfig] = None,
    ) -> None:
        self._config = config or EvoSkillConfig()
        self._translator = translator or SkillTranslator(
            default_author=self._config.skill_author_id,
        )
        self._payoff_config = payoff_config or PayoffConfig()
        self._proxy = ProxyComputer()
        self._engine = SoftPayoffEngine(config=self._payoff_config)
        self._metrics = SoftMetrics(payoff_engine=self._engine)

    def evaluate_program(
        self,
        program_id: str,
        skill_files: Dict[str, str],
        benchmark_score: float,
        regime: str,
        contract_type: ContractType,
        interactions: Optional[List[SoftInteraction]] = None,
        seed: int = 42,
    ) -> EvalResult:
        """Evaluate a single candidate program under a governance regime.

        Args:
            program_id: Git branch or hash identifying the program.
            skill_files: Mapping of filename → content from the program.
            benchmark_score: Raw benchmark score from EvoSkill (0–1).
            regime: Human-readable regime label.
            contract_type: SWARM contract type to apply.
            interactions: Pre-generated interactions (for testing).
                If None, synthetic interactions are generated from skills.
            seed: RNG seed for reproducibility.

        Returns:
            EvalResult with oracle, governed, and composite scores.
        """
        import random
        rng = random.Random(seed)

        # 1. Translate skills
        skills = self._translator.ingest_batch(
            skill_files,
            source_branch=program_id,
            author_id=self._config.skill_author_id,
        )

        # 2. Build a skill library from the translated skills
        library = SkillLibrary(
            owner_id=f"evoskill_{program_id}",
            config=SkillLibraryConfig(sharing_mode=SharingMode.PRIVATE),
        )
        for skill in skills:
            library.add_skill(skill)

        # 3. Generate or use provided interactions
        if interactions is None:
            interactions = self._generate_probe_interactions(
                skills, rng, self._config.oracle_epochs * self._config.steps_per_epoch,
            )

        # 4. Oracle run (no governance costs)
        oracle_metrics = self._run_oracle(interactions)

        # 5. Governed run (with contract regime effects)
        governed_metrics = self._run_governed(interactions, contract_type)

        # 6. Compute governance delta
        delta = self._compute_delta(oracle_metrics, governed_metrics)

        # 7. Composite score
        w = self._config.governance_weight
        # Governance benefit: lower toxicity + higher welfare under governance
        gov_benefit = (
            delta.get("toxicity_reduction", 0.0)
            + delta.get("welfare_improvement", 0.0)
        ) / 2.0
        composite = (1 - w) * benchmark_score + w * (benchmark_score + gov_benefit)
        composite = max(0.0, min(1.0, composite))

        return EvalResult(
            program_id=program_id,
            benchmark_score=benchmark_score,
            oracle_metrics=oracle_metrics,
            governed_metrics=governed_metrics,
            governance_delta=delta,
            composite_score=composite,
            regime=regime,
            skills_ingested=len(skills),
        )

    def evaluate_program_all_regimes(
        self,
        program_id: str,
        skill_files: Dict[str, str],
        benchmark_score: float,
        interactions: Optional[List[SoftInteraction]] = None,
        seed: int = 42,
    ) -> Dict[str, EvalResult]:
        """Evaluate a program under all configured governance regimes.

        Args:
            program_id: Program identifier.
            skill_files: Skill files from the program.
            benchmark_score: Raw benchmark score.
            interactions: Shared interaction set for fair comparison.
            seed: Base seed (offset per regime).

        Returns:
            Dict mapping regime label → EvalResult.
        """
        results: Dict[str, EvalResult] = {}

        for i, (label, ct) in enumerate(self._config.contract_regimes.items()):
            result = self.evaluate_program(
                program_id=program_id,
                skill_files=skill_files,
                benchmark_score=benchmark_score,
                regime=label,
                contract_type=ct,
                interactions=interactions,
                seed=seed + i,
            )
            results[label] = result

        return results

    # ------------------------------------------------------------------
    # Internal simulation helpers
    # ------------------------------------------------------------------

    def _run_oracle(self, interactions: List[SoftInteraction]) -> Dict[str, float]:
        """Run oracle evaluation — pure payoffs with no governance costs."""
        payoffs = []
        for ix in interactions:
            # Zero out governance costs for oracle
            oracle_ix = ix.model_copy(update={"c_a": 0.0, "c_b": 0.0})
            pi_a, pi_b = self._engine.payoffs_both(oracle_ix)
            payoffs.append((pi_a, pi_b))

        return self._compute_metrics(interactions, payoffs)

    def _run_governed(
        self,
        interactions: List[SoftInteraction],
        contract_type: ContractType,
    ) -> Dict[str, float]:
        """Run governed evaluation — apply contract regime effects."""
        from swarm.contracts.contract import (
            DefaultMarket,
            FairDivisionContract,
            TruthfulAuctionContract,
        )

        # Instantiate the contract
        contract_map = {
            ContractType.TRUTHFUL_AUCTION: TruthfulAuctionContract,
            ContractType.FAIR_DIVISION: FairDivisionContract,
            ContractType.DEFAULT_MARKET: DefaultMarket,
        }
        contract_cls = contract_map.get(contract_type, DefaultMarket)
        contract = contract_cls()

        payoffs = []
        governed_interactions = []
        for ix in interactions:
            gov_ix = contract.execute(ix)
            governed_interactions.append(gov_ix)
            pi_a, pi_b = self._engine.payoffs_both(gov_ix)
            payoffs.append((pi_a, pi_b))

        return self._compute_metrics(governed_interactions, payoffs)

    def _compute_metrics(
        self,
        interactions: List[SoftInteraction],
        payoffs: List[tuple],
    ) -> Dict[str, float]:
        """Compute metrics dict from interactions and payoffs.

        Args:
            interactions: List of SoftInteraction objects to analyze.
            payoffs: List of (pi_a, pi_b) tuples from the engine.

        Returns:
            Dict with toxicity, quality_gap, avg_welfare, and n_interactions.
        """
        accepted = [ix for ix in interactions if ix.accepted]
        toxicity = self._metrics.toxicity_rate(accepted) if accepted else 0.0
        quality_gap = self._metrics.quality_gap(interactions)
        avg_welfare = (
            sum(a + b for a, b in payoffs) / (2 * len(payoffs))
            if payoffs else 0.0
        )

        return {
            "toxicity": toxicity,
            "quality_gap": quality_gap,
            "avg_welfare": avg_welfare,
            "n_interactions": len(interactions),
        }

    @staticmethod
    def _compute_delta(
        oracle: Dict[str, float],
        governed: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute governance attribution signal.

        Positive values = governance helped.
        """
        return {
            # Lower toxicity under governance = positive reduction
            "toxicity_reduction": oracle.get("toxicity", 0) - governed.get("toxicity", 0),
            # Higher welfare under governance = positive improvement
            "welfare_improvement": governed.get("avg_welfare", 0) - oracle.get("avg_welfare", 0),
            # More negative quality gap under governance = worse adverse selection
            "quality_gap_shift": governed.get("quality_gap", 0) - oracle.get("quality_gap", 0),
        }

    def _generate_probe_interactions(
        self,
        skills: List[Skill],
        rng: "random.Random",
        n: int,
    ) -> List[SoftInteraction]:
        """Generate synthetic probe interactions to test skill effects.

        Creates a mix of interaction quality levels to exercise the
        governance regime's screening capacity.
        """

        interactions: List[SoftInteraction] = []
        for i in range(n):
            # Mix of quality levels: 40% high, 30% medium, 30% low
            roll = rng.random()
            if roll < 0.4:
                p = clamp_p(rng.gauss(0.8, 0.1))
                accepted = True
            elif roll < 0.7:
                p = clamp_p(rng.gauss(0.5, 0.15))
                accepted = rng.random() < 0.6
            else:
                p = clamp_p(rng.gauss(0.2, 0.1))
                accepted = rng.random() < 0.3

            # Derive v_hat from p via inverse sigmoid approximation
            v_hat = max(-1.0, min(1.0, (p - 0.5) * 4.0))

            interactions.append(SoftInteraction(
                initiator=f"agent_{i % 5}",
                counterparty=f"agent_{(i + 1) % 5}",
                interaction_type=InteractionType.COLLABORATION,
                accepted=accepted,
                task_progress_delta=p * 0.8 - 0.1,
                rework_count=max(0, int((1 - p) * 3)),
                verifier_rejections=0 if p > 0.5 else 1,
                tool_misuse_flags=0,
                counterparty_engagement_delta=p * 0.6,
                v_hat=v_hat,
                p=p,
                tau=rng.gauss(0.0, 0.5),
                c_a=0.1,
                c_b=0.1,
                r_a=0.0,
                r_b=0.0,
                metadata={"probe": True, "skill_count": len(skills)},
            ))

        return interactions
