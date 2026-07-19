"""Systematic multiverse analysis for Social Simulacra threads.

Replaces Social Simulacra's ad-hoc "Multiverse" (higher-temperature
resampling shown side-by-side) with SWARM's distributional measurement
framework: run the same scenario N times at varying temperatures, then
compute cross-run variance, bias-variance decomposition, and adverse
selection metrics across the multiverse.

Reference: Park et al. (2022) Section 5 — "Multiverse" feature
"""

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from swarm.bridges.concordia.adapter import ConcordiaAdapter
from swarm.bridges.concordia.config import ConcordiaConfig
from swarm.bridges.concordia.simulacra import (
    CommunityConfig,
    ExpandedPersona,
    Thread,
    ThreadGenerator,
    thread_to_narrative_samples,
)
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.interaction import SoftInteraction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MultiverseConfig:
    """Configuration for multiverse analysis."""

    n_universes: int = 10
    temperatures: List[float] = field(
        default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    threads_per_universe: int = 5
    base_seed: int = 42
    concordia_config: Optional[ConcordiaConfig] = None


# ---------------------------------------------------------------------------
# Per-universe result
# ---------------------------------------------------------------------------


@dataclass
class UniverseResult:
    """Metrics from a single universe (one temperature + seed run)."""

    universe_id: int = 0
    temperature: float = 0.7
    seed: int = 0
    threads: List[Thread] = field(default_factory=list)
    interactions: List[SoftInteraction] = field(default_factory=list)

    # Summary metrics
    toxicity_rate: float = 0.0
    quality_gap: float = 0.0
    avg_p: float = 0.0
    p_variance: float = 0.0
    n_interactions: int = 0
    n_threads: int = 0
    avg_replies_per_thread: float = 0.0
    n_unique_participants: int = 0


# ---------------------------------------------------------------------------
# Cross-universe analysis
# ---------------------------------------------------------------------------


@dataclass
class MultiverseResult:
    """Aggregated results across all universes."""

    universes: List[UniverseResult] = field(default_factory=list)

    # Cross-universe statistics
    toxicity_mean: float = 0.0
    toxicity_std: float = 0.0
    quality_gap_mean: float = 0.0
    quality_gap_std: float = 0.0
    p_mean: float = 0.0
    p_std: float = 0.0
    p_variance_of_means: float = 0.0  # Var[E[p | universe]]

    # Bias-variance decomposition
    bias_squared: float = 0.0  # (E[p] - reference_p)^2
    variance: float = 0.0  # E[Var[p | universe]]
    total_error: float = 0.0  # bias^2 + variance

    # Temperature sensitivity
    temperature_correlation: float = 0.0  # corr(temperature, toxicity)

    def to_dict(self) -> Dict[str, float]:
        """Convert summary statistics to a flat dict."""
        return {
            "n_universes": len(self.universes),
            "toxicity_mean": self.toxicity_mean,
            "toxicity_std": self.toxicity_std,
            "quality_gap_mean": self.quality_gap_mean,
            "quality_gap_std": self.quality_gap_std,
            "p_mean": self.p_mean,
            "p_std": self.p_std,
            "p_variance_of_means": self.p_variance_of_means,
            "bias_squared": self.bias_squared,
            "variance": self.variance,
            "total_error": self.total_error,
            "temperature_correlation": self.temperature_correlation,
        }


# ---------------------------------------------------------------------------
# Multiverse Runner
# ---------------------------------------------------------------------------


class MultiverseRunner:
    """Run the same scenario across multiple universes and analyze variance.

    Each universe uses a different (temperature, seed) pair to generate
    threads, processes them through SWARM's ConcordiaAdapter, and collects
    distributional metrics.
    """

    def __init__(
        self,
        community: CommunityConfig,
        personas: List[ExpandedPersona],
        config: Optional[MultiverseConfig] = None,
    ):
        self._community = community
        self._personas = personas
        self._config = config or MultiverseConfig()
        self._metrics = SoftMetrics()

    def run(
        self,
        *,
        reference_p: float = 0.5,
    ) -> MultiverseResult:
        """Run multiverse analysis.

        Args:
            reference_p: Reference probability for bias-variance decomposition.
                Typically 0.5 (uninformative prior) or the expected p under
                the community's stated norms.

        Returns:
            MultiverseResult with per-universe and cross-universe statistics.
        """
        universes: List[UniverseResult] = []
        universe_id = 0

        n_temps = len(self._config.temperatures)
        base_count, remainder = divmod(self._config.n_universes, n_temps)

        for temp_idx, temp in enumerate(self._config.temperatures):
            n_for_temp = base_count + (1 if temp_idx < remainder else 0)
            for run_idx in range(n_for_temp):
                # Use 1000 multiplier to stratify seed ranges by universe (and thus temperature),
                # ensuring that universes at different temperatures have non-overlapping seed spaces.
                seed = self._config.base_seed + universe_id * 1000 + run_idx
                result = self._run_single_universe(
                    universe_id=universe_id,
                    temperature=temp,
                    seed=seed,
                )
                universes.append(result)
                universe_id += 1

        return self._analyze(universes, reference_p)

    def _run_single_universe(
        self,
        universe_id: int,
        temperature: float,
        seed: int,
    ) -> UniverseResult:
        """Run a single universe: generate threads and score them."""
        rng = random.Random(seed)

        # Generate threads at this temperature
        generator = ThreadGenerator(
            community=self._community,
            personas=self._personas,
            base_temperature=temperature,
        )
        threads = generator.generate_threads(
            self._config.threads_per_universe, rng=rng
        )

        # Score through SWARM adapter
        concordia_config = self._config.concordia_config or ConcordiaConfig()
        adapter = ConcordiaAdapter(config=concordia_config)

        all_interactions: List[SoftInteraction] = []
        for thread in threads:
            samples = thread_to_narrative_samples(thread)
            for agent_ids, narrative in samples:
                interactions = adapter.process_narrative(
                    agent_ids=agent_ids,
                    narrative_text=narrative,
                    step=universe_id,
                )
                all_interactions.extend(interactions)

        # Compute per-universe metrics
        toxicity = self._metrics.toxicity_rate(all_interactions)
        quality_gap = self._metrics.quality_gap(all_interactions)
        avg_p = (
            sum(i.p for i in all_interactions) / len(all_interactions)
            if all_interactions
            else 0.0
        )
        p_var = self._metrics.quality_variance(all_interactions)

        # Thread-level stats
        total_replies = sum(len(t.replies) for t in threads)
        avg_replies = total_replies / len(threads) if threads else 0.0

        all_participants: set[str] = set()
        for t in threads:
            for p in t.participants:
                all_participants.add(p.persona_id)

        return UniverseResult(
            universe_id=universe_id,
            temperature=temperature,
            seed=seed,
            threads=threads,
            interactions=all_interactions,
            toxicity_rate=toxicity,
            quality_gap=quality_gap,
            avg_p=avg_p,
            p_variance=p_var,
            n_interactions=len(all_interactions),
            n_threads=len(threads),
            avg_replies_per_thread=avg_replies,
            n_unique_participants=len(all_participants),
        )

    def _analyze(
        self,
        universes: List[UniverseResult],
        reference_p: float,
    ) -> MultiverseResult:
        """Compute cross-universe statistics."""
        if not universes:
            return MultiverseResult()

        toxicities = [u.toxicity_rate for u in universes]
        quality_gaps = [u.quality_gap for u in universes]
        avg_ps = [u.avg_p for u in universes]
        within_variances = [u.p_variance for u in universes]

        result = MultiverseResult(universes=universes)

        # Means and standard deviations
        result.toxicity_mean = statistics.mean(toxicities)
        result.toxicity_std = (
            statistics.stdev(toxicities) if len(toxicities) > 1 else 0.0
        )
        result.quality_gap_mean = statistics.mean(quality_gaps)
        result.quality_gap_std = (
            statistics.stdev(quality_gaps) if len(quality_gaps) > 1 else 0.0
        )
        result.p_mean = statistics.mean(avg_ps)
        result.p_std = statistics.stdev(avg_ps) if len(avg_ps) > 1 else 0.0

        # Variance of universe means: Var[E[p | universe]]
        result.p_variance_of_means = (
            statistics.variance(avg_ps) if len(avg_ps) > 1 else 0.0
        )

        # Bias-variance decomposition
        # bias^2 = (E[E[p|u]] - reference_p)^2
        result.bias_squared = (result.p_mean - reference_p) ** 2

        # variance = E[Var[p|u]]  (average within-universe variance)
        result.variance = statistics.mean(within_variances)

        # total = bias^2 + variance
        result.total_error = result.bias_squared + result.variance

        # Temperature-toxicity correlation
        result.temperature_correlation = self._pearson_correlation(
            [u.temperature for u in universes],
            toxicities,
        )

        return result

    @staticmethod
    def _pearson_correlation(xs: List[float], ys: List[float]) -> float:
        """Compute Pearson correlation coefficient."""
        n = len(xs)
        if n < 2:
            return 0.0

        mean_x = sum(xs) / n
        mean_y = sum(ys) / n

        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / n
        var_x = sum((x - mean_x) ** 2 for x in xs) / n
        var_y = sum((y - mean_y) ** 2 for y in ys) / n

        denom = math.sqrt(var_x * var_y)
        if denom < 1e-12:
            return 0.0

        return cov / denom
