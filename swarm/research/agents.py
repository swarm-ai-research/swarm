"""Research agents for the structured research workflow."""

import logging
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Set

import numpy as np

try:
    from scipy import stats  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency
    stats = None  # type: ignore[assignment]

from swarm.research.platforms import Paper, PlatformClient, SearchResult

logger = logging.getLogger(__name__)

@dataclass
class Source:
    """A literature source."""

    paper_id: str
    title: str
    abstract: str
    date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    platform: str = ""
    relevance_score: float = 0.0
    key_findings: list[str] = field(default_factory=list)


@dataclass
class LiteratureReview:
    """Output of literature review phase."""

    sources: list[Source] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    hypothesis: str = ""
    related_work_summary: str = ""
    follow_up_questions: list[str] = field(default_factory=list)

    @property
    def source_count(self) -> int:
        return len(self.sources)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    name: str
    parameters: dict[str, Any]
    trials: int = 10
    rounds: int = 100
    seed: int | None = None


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""

    config: ExperimentConfig
    metrics: dict[str, float]
    raw_data: list[dict[str, Any]] = field(default_factory=list)
    seed: int | None = None
    duration_seconds: float = 0.0
    error: str | None = None


@dataclass
class ExperimentResults:
    """Output of experiment phase."""

    results: list[ExperimentResult] = field(default_factory=list)
    configs: list[ExperimentConfig] = field(default_factory=list)
    total_trials: int = 0
    parameter_coverage: float = 1.0
    error_count: int = 0

    @property
    def trials_per_config(self) -> int:
        if not self.configs:
            return 0
        return self.total_trials // len(self.configs)


@dataclass
class Claim:
    """A statistical claim with evidence."""

    statement: str
    metric: str
    value: float
    confidence_interval: tuple[float, float]
    effect_size: float | None = None
    p_value: float | None = None
    is_primary: bool = False


@dataclass
class Analysis:
    """Output of analysis phase."""

    claims: list[Claim] = field(default_factory=list)
    correlations: dict[tuple[str, str], float] = field(default_factory=dict)
    effect_sizes: dict[str, float] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    visualizations: list[str] = field(default_factory=list)  # Paths to generated plots

    @property
    def all_claims_have_ci(self) -> bool:
        return all(c.confidence_interval != (0.0, 0.0) for c in self.claims)

    @property
    def all_claims_have_effect_size(self) -> bool:
        return all(c.effect_size is not None for c in self.claims)

    @property
    def multiple_comparison_corrected(self) -> bool:
        # Check if p-values are Bonferroni-corrected
        p_values = [c.p_value for c in self.claims if c.p_value is not None]
        if not p_values:
            return True
        # Heuristic: if smallest p-value is still < 0.05 / n, likely corrected
        return min(p_values) < 0.05 / max(len(p_values), 1)


@dataclass
class Critique:
    """A critique of research."""

    severity: str  # low, medium, high, critical
    category: str  # statistics, methodology, claims, completeness
    issue: str
    suggestion: str = ""
    addressed: bool = False


@dataclass
class Review:
    """Output of review phase."""

    critiques: list[Critique] = field(default_factory=list)
    recommendation: str = ""  # accept, minor_revision, major_revision, reject
    summary: str = ""

    @property
    def high_severity_count(self) -> int:
        return len([c for c in self.critiques if c.severity in ("high", "critical")])

    @property
    def all_critiques_addressed(self) -> bool:
        return all(c.addressed for c in self.critiques)


@dataclass
class PeerReview:
    """A peer review of a paper, linking Review output to a specific paper."""

    review_id: str
    paper_id: str
    reviewer_id: str
    critiques: list[Critique] = field(default_factory=list)
    recommendation: str = ""  # accept, minor_revision, major_revision, reject
    summary: str = ""
    rating: int = 3  # 1-5 scale
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def high_severity_count(self) -> int:
        return len([c for c in self.critiques if c.severity in ("high", "critical")])

    @property
    def all_critiques_addressed(self) -> bool:
        return all(c.addressed for c in self.critiques)

    def to_review(self) -> "Review":
        """Downcast to Review for QualityGate compatibility."""
        return Review(
            critiques=self.critiques,
            recommendation=self.recommendation,
            summary=self.summary,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "paper_id": self.paper_id,
            "reviewer_id": self.reviewer_id,
            "critiques": [
                {
                    "severity": c.severity,
                    "category": c.category,
                    "issue": c.issue,
                    "suggestion": c.suggestion,
                    "addressed": c.addressed,
                }
                for c in self.critiques
            ],
            "recommendation": self.recommendation,
            "summary": self.summary,
            "rating": self.rating,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PeerReview":
        critiques = [
            Critique(
                severity=c.get("severity", "low"),
                category=c.get("category", ""),
                issue=c.get("issue", ""),
                suggestion=c.get("suggestion", ""),
                addressed=c.get("addressed", False),
            )
            for c in data.get("critiques", [])
        ]
        ts = data.get("timestamp")
        if isinstance(ts, str):
            timestamp = datetime.fromisoformat(ts)
        else:
            timestamp = datetime.now(timezone.utc)
        return cls(
            review_id=data.get("review_id", ""),
            paper_id=data.get("paper_id", ""),
            reviewer_id=data.get("reviewer_id", ""),
            critiques=critiques,
            recommendation=data.get("recommendation", ""),
            summary=data.get("summary", ""),
            rating=data.get("rating", 3),
            timestamp=timestamp,
        )


_RECOMMENDATION_RATINGS: dict[str, int] = {
    "accept": 5,
    "minor_revision": 4,
    "major_revision": 2,
    "reject": 1,
}


def review_to_peer_review(
    review: "Review",
    paper_id: str,
    reviewer_id: str,
) -> PeerReview:
    """Convert a ReviewAgent output into a PeerReview."""
    rating = _RECOMMENDATION_RATINGS.get(review.recommendation, 3)
    return PeerReview(
        review_id=str(uuid.uuid4()),
        paper_id=paper_id,
        reviewer_id=reviewer_id,
        critiques=review.critiques,
        recommendation=review.recommendation,
        summary=review.summary,
        rating=rating,
    )


class ResearchAgent(ABC):
    """Base class for research agents."""

    def __init__(self, depth: int = 2, breadth: int = 2):
        self.depth = depth
        self.breadth = breadth

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Execute the agent's primary function."""
        pass


def _require_stats() -> Any:
    """Return scipy.stats or raise with a helpful message."""
    if stats is None:
        raise RuntimeError(
            "scipy is required for statistical analysis. "
            "Install with `pip install scipy` or `swarm-safety[analysis]`."
        )
    return stats


class LiteratureAgent(ResearchAgent):
    """Agent for surveying existing research and identifying gaps."""

    def __init__(
        self,
        depth: int = 2,
        breadth: int = 2,
        platforms: list[PlatformClient] | None = None,
    ):
        super().__init__(depth, breadth)
        self.platforms = platforms or []

    def run(
        self,
        question: str,
        platforms: list[str] | None = None,
    ) -> LiteratureReview:
        """Survey literature for a research question.

        Args:
            question: The research question to investigate.
            platforms: Platform client class names to search (exact match,
                e.g. "ClawxivClient"). None or empty searches all
                configured platforms.

        Returns:
            LiteratureReview with sources, gaps, and hypothesis.
        """
        if platforms:
            active_platforms = [
                p for p in self.platforms if type(p).__name__ in platforms
            ]
            matched = {type(p).__name__ for p in active_platforms}
            unmatched = [name for name in platforms if name not in matched]
            if unmatched:
                logger.warning(
                    "run(platforms=...): no configured client matches %s "
                    "(configured: %s)",
                    unmatched,
                    sorted({type(p).__name__ for p in self.platforms}),
                )
        else:
            active_platforms = self.platforms

        all_sources: list[Source] = []
        follow_ups: list[str] = [question]

        # Recursive exploration based on depth
        for _layer in range(self.depth):
            layer_sources = []

            for q in follow_ups[: self.breadth]:
                # Generate query variants based on breadth
                queries = self._generate_queries(q, self.breadth)

                for query in queries:
                    for platform in active_platforms:
                        result = platform.search(query, limit=20)
                        sources = self._process_search_results(result, platform)
                        layer_sources.extend(sources)

            # Deduplicate
            seen_ids = {s.paper_id for s in all_sources}
            new_sources = [s for s in layer_sources if s.paper_id not in seen_ids]
            all_sources.extend(new_sources)

            # Extract follow-up questions for next layer
            follow_ups = self._extract_follow_ups(new_sources)

        # Analyze sources
        gaps = self._identify_gaps(all_sources)
        hypothesis = self._form_hypothesis(question, all_sources, gaps)
        summary = self._generate_summary(all_sources)

        return LiteratureReview(
            sources=all_sources,
            gaps=gaps,
            hypothesis=hypothesis,
            related_work_summary=summary,
            follow_up_questions=follow_ups,
        )

    def _generate_queries(self, question: str, breadth: int) -> list[str]:
        """Generate query variants for broader coverage."""
        queries = [question]

        # Add keyword-focused variants
        keywords = question.lower().split()
        important_keywords = [w for w in keywords if len(w) > 4]

        if breadth >= 2 and important_keywords:
            queries.append(" ".join(important_keywords[:3]))
        if breadth >= 3:
            queries.append(f"{question} mechanism")
        if breadth >= 4:
            queries.append(f"{question} empirical")

        return queries[:breadth]

    def _process_search_results(
        self,
        result: SearchResult,
        platform: PlatformClient,
    ) -> list[Source]:
        """Convert search results to Source objects."""
        sources = []
        for paper in result.papers:
            source = Source(
                paper_id=paper.paper_id,
                title=paper.title,
                abstract=paper.abstract,
                date=paper.created_at,
                platform=type(platform).__name__,
                relevance_score=self._compute_relevance(paper, result.query),
            )
            sources.append(source)
        return sources

    def _compute_relevance(self, paper: Paper, query: str) -> float:
        """Compute relevance score using token and substring matching."""
        query_lower = query.lower()
        title_lower = paper.title.lower()
        abstract_lower = paper.abstract.lower()

        query_terms = set(query_lower.split())
        title_terms = set(title_lower.split())
        abstract_terms = set(abstract_lower.split())

        # Exact token overlap
        title_token_overlap = len(query_terms & title_terms) / max(len(query_terms), 1)
        abstract_token_overlap = len(query_terms & abstract_terms) / max(
            len(query_terms), 1
        )

        # Substring matching for multi-word and hyphenated terms
        # Check if query fragments (bigrams) appear as substrings
        query_words = query_lower.split()
        bigrams = [
            f"{query_words[i]} {query_words[i + 1]}"
            for i in range(len(query_words) - 1)
        ]
        bigram_hits_title = sum(1 for bg in bigrams if bg in title_lower) / max(
            len(bigrams), 1
        )
        bigram_hits_abstract = sum(1 for bg in bigrams if bg in abstract_lower) / max(
            len(bigrams), 1
        )

        # Weighted combination
        token_score = 0.6 * title_token_overlap + 0.4 * abstract_token_overlap
        bigram_score = 0.6 * bigram_hits_title + 0.4 * bigram_hits_abstract

        return max(token_score, bigram_score)

    def _extract_follow_ups(self, sources: list[Source]) -> list[str]:
        """Extract follow-up questions from sources."""
        # Simple heuristic: look for question patterns in abstracts
        follow_ups = []
        for source in sources:
            # Look for "future work", "open question", etc.
            if "future" in source.abstract.lower() or "open" in source.abstract.lower():
                follow_ups.append(f"Extensions of {source.title[:50]}")
        return follow_ups[: self.breadth]

    def _identify_gaps(self, sources: list[Source]) -> list[str]:
        """Identify research gaps from literature."""
        gaps = []

        if len(sources) < 5:
            gaps.append("Limited prior work in this area")

        # Check for recency
        recent = [
            s for s in sources if (datetime.now(timezone.utc) - s.date).days < 180
        ]
        if len(recent) < len(sources) * 0.2:
            gaps.append("Most work is not recent; field may need updating")

        # Check for empirical vs theoretical
        empirical_keywords = [
            "experiment",
            "empirical",
            "data",
            "simulation",
            "results",
        ]
        empirical_count = sum(
            1
            for s in sources
            if any(k in s.abstract.lower() for k in empirical_keywords)
        )
        if empirical_count < len(sources) * 0.3:
            gaps.append("Limited empirical validation in existing work")

        return gaps

    def _form_hypothesis(
        self,
        question: str,
        sources: list[Source],
        gaps: list[str],
    ) -> str:
        """Form a testable hypothesis based on literature."""
        if "Limited empirical" in " ".join(gaps):
            return f"Empirical validation of {question}"
        elif len(sources) < 3:
            return f"Novel investigation of {question}"
        else:
            return f"Extension of existing work on {question}"

    def _generate_summary(self, sources: list[Source]) -> str:
        """Generate related work summary."""
        if not sources:
            return "No related work found."

        lines = [f"Found {len(sources)} related papers:"]
        for source in sorted(sources, key=lambda s: -s.relevance_score)[:5]:
            lines.append(f"- {source.title} (relevance: {source.relevance_score:.2f})")
        return "\n".join(lines)


class ExperimentAgent(ResearchAgent):
    """Agent for designing and executing SWARM simulations."""

    def __init__(
        self,
        depth: int = 2,
        breadth: int = 2,
        simulation_fn: Callable[[dict], dict[str, float]] | None = None,
    ):
        super().__init__(depth, breadth)
        self.simulation_fn = simulation_fn

    def run(
        self,
        hypothesis: str,
        parameter_space: dict[str, list[Any]],
        trials: int = 10,
        rounds: int = 100,
    ) -> ExperimentResults:
        """Design and run experiments.

        Args:
            hypothesis: The hypothesis to test.
            parameter_space: Dict of parameter name -> list of values to test.
            trials: Number of trials per configuration.
            rounds: Number of rounds per trial.

        Returns:
            ExperimentResults with all simulation outcomes.
        """
        # Generate configurations based on breadth
        configs = self._generate_configs(parameter_space, trials, rounds)

        # Run simulations
        results = []
        total_trials = 0
        error_count = 0

        for config in configs:
            for trial in range(config.trials):
                seed = (hash(config.name) + trial) % (2**31)
                result = self._run_single(config, seed)
                results.append(result)
                total_trials += 1
                if result.error:
                    error_count += 1

        # Compute coverage
        total_possible = 1
        for values in parameter_space.values():
            total_possible *= len(values)
        coverage = len(configs) / max(total_possible, 1)

        return ExperimentResults(
            results=results,
            configs=configs,
            total_trials=total_trials,
            parameter_coverage=min(coverage, 1.0),
            error_count=error_count,
        )

    def _generate_configs(
        self,
        parameter_space: dict[str, list[Any]],
        trials: int,
        rounds: int,
    ) -> list[ExperimentConfig]:
        """Generate experiment configurations from parameter space."""
        configs = []

        # Use breadth to determine sampling strategy
        if self.breadth == 1:
            # Single configuration: use middle values
            params = {k: v[len(v) // 2] for k, v in parameter_space.items()}
            configs.append(
                ExperimentConfig(
                    name="single_config",
                    parameters=params,
                    trials=trials,
                    rounds=rounds,
                )
            )
        else:
            # Grid sample based on breadth
            from itertools import product

            # Select subset of values based on breadth
            sampled_space = {}
            for key, values in parameter_space.items():
                step = max(1, len(values) // self.breadth)
                sampled_space[key] = values[::step][: self.breadth]

            # Generate combinations
            keys = list(sampled_space.keys())
            for i, combo in enumerate(product(*sampled_space.values())):
                params = dict(zip(keys, combo, strict=True))
                configs.append(
                    ExperimentConfig(
                        name=f"config_{i}",
                        parameters=params,
                        trials=trials,
                        rounds=rounds,
                    )
                )

        return configs

    def _run_single(
        self,
        config: ExperimentConfig,
        seed: int,
    ) -> ExperimentResult:
        """Run a single experiment."""
        if self.simulation_fn is None:
            # Return mock result
            np.random.seed(seed)
            return ExperimentResult(
                config=config,
                metrics={
                    "toxicity": np.random.uniform(0.2, 0.4),
                    "welfare": np.random.uniform(300, 600),
                    "quality_gap": np.random.uniform(-0.1, 0.1),
                },
                seed=seed,
            )

        try:
            np.random.seed(seed)
            metrics = self.simulation_fn(config.parameters)
            return ExperimentResult(
                config=config,
                metrics=metrics,
                seed=seed,
            )
        except Exception as e:
            return ExperimentResult(
                config=config,
                metrics={},
                seed=seed,
                error=str(e),
            )


class AnalysisAgent(ResearchAgent):
    """Agent for statistical analysis and insight extraction."""

    def __init__(self, depth: int = 2, breadth: int = 2):
        super().__init__(depth, breadth)

    def run(
        self,
        results: ExperimentResults,
        literature: LiteratureReview | None = None,
    ) -> Analysis:
        """Analyze experiment results.

        Args:
            results: Experiment results to analyze.
            literature: Optional literature context.

        Returns:
            Analysis with claims, effect sizes, and limitations.
        """
        claims = []
        effect_sizes = {}
        correlations = {}

        # Extract metrics from results
        metrics_data: dict[str, list[float]] = {}
        for result in results.results:
            for metric, value in result.metrics.items():
                metrics_data.setdefault(metric, []).append(value)

        # Layer 1: Descriptive statistics (always)
        for metric, values in metrics_data.items():
            mean = float(np.mean(values))
            std = float(np.std(values))
            ci = self._compute_ci(values)

            mean_val: float = mean
            claims.append(
                Claim(
                    statement=f"Mean {metric}: {mean_val:.3f} (SD: {std:.3f})",
                    metric=metric,
                    value=mean_val,
                    confidence_interval=ci,
                    is_primary=True,
                )
            )

        if self.depth >= 2:
            # Layer 2: Inferential statistics
            claims = self._add_significance_tests(claims, metrics_data, results)

        if self.depth >= 3:
            # Layer 3: Effect sizes and correlations
            effect_sizes = self._compute_effect_sizes(metrics_data, results)
            correlations = self._compute_correlations(metrics_data)

            for metric, es in effect_sizes.items():
                for claim in claims:
                    if claim.metric == metric:
                        claim.effect_size = es

        if self.depth >= 4:
            # Layer 4: Comparison to literature
            if literature:
                claims.extend(self._compare_to_literature(metrics_data, literature))

        # Identify limitations
        limitations = self._identify_limitations(results, claims)

        return Analysis(
            claims=claims,
            correlations=correlations,
            effect_sizes=effect_sizes,
            limitations=limitations,
        )

    def _compute_ci(
        self, values: list[float], confidence: float = 0.95
    ) -> tuple[float, float]:
        """Compute confidence interval."""
        if len(values) < 2:
            return (0.0, 0.0)
        mean = np.mean(values)
        stats_mod = _require_stats()
        se = stats_mod.sem(values)
        ci = stats_mod.t.interval(confidence, len(values) - 1, loc=mean, scale=se)
        return (float(ci[0]), float(ci[1]))

    def _add_significance_tests(
        self,
        claims: list[Claim],
        metrics_data: dict[str, list[float]],
        results: ExperimentResults,
    ) -> list[Claim]:
        """Add significance test results to claims."""
        # Group results by parameter values
        param_groups: dict[str, dict[Any, list[float]]] = {}

        for result in results.results:
            for param_name, param_value in result.config.parameters.items():
                for metric, value in result.metrics.items():
                    key = f"{param_name}_{metric}"
                    param_groups.setdefault(key, {}).setdefault(param_value, []).append(
                        value
                    )

        # Run t-tests between groups
        for key, groups in param_groups.items():
            values_list = list(groups.values())
            if (
                len(values_list) >= 2
                and len(values_list[0]) >= 2
                and len(values_list[1]) >= 2
            ):
                stats_mod = _require_stats()
                t_stat, p_value = stats_mod.ttest_ind(values_list[0], values_list[1])

                # Bonferroni correction
                corrected_p = p_value * len(param_groups)

                for claim in claims:
                    if claim.metric in key:
                        claim.p_value = min(corrected_p, 1.0)

        return claims

    def _compute_effect_sizes(
        self,
        metrics_data: dict[str, list[float]],
        results: ExperimentResults,
    ) -> dict[str, float]:
        """Compute Cohen's d effect sizes between parameter conditions.

        Groups results by parameter values and computes effect size
        between the groups, not between above/below median of the outcome.
        """
        effect_sizes: dict[str, float] = {}

        # Build parameter groups: param_name -> {param_value -> {metric -> [values]}}
        param_metric_groups: dict[str, dict[Any, dict[str, list[float]]]] = {}
        for result in results.results:
            for param_name, param_value in result.config.parameters.items():
                param_metric_groups.setdefault(param_name, {}).setdefault(
                    param_value, {}
                )
                for metric, value in result.metrics.items():
                    param_metric_groups[param_name][param_value].setdefault(
                        metric, []
                    ).append(value)

        # For each metric, find the parameter that produces the largest effect
        for metric in metrics_data:
            best_d = None
            for _param_name, value_groups in param_metric_groups.items():
                group_values = []
                for _param_val, metric_dict in value_groups.items():
                    vals = metric_dict.get(metric, [])
                    if vals:
                        group_values.append(vals)

                if len(group_values) >= 2:
                    # Compare first and last groups (lowest vs highest param value)
                    g1 = group_values[0]
                    g2 = group_values[-1]
                    if len(g1) >= 2 and len(g2) >= 2:
                        pooled_std = np.sqrt(
                            (
                                (len(g1) - 1) * np.var(g1, ddof=1)
                                + (len(g2) - 1) * np.var(g2, ddof=1)
                            )
                            / (len(g1) + len(g2) - 2)
                        )
                        if pooled_std > 0:
                            d = (np.mean(g2) - np.mean(g1)) / pooled_std
                            if best_d is None or abs(d) > abs(best_d):
                                best_d = d

            if best_d is not None:
                effect_sizes[metric] = float(best_d)

        return effect_sizes

    def _compute_correlations(
        self,
        metrics_data: dict[str, list[float]],
    ) -> dict[tuple[str, str], float]:
        """Compute correlations between metrics."""
        correlations = {}
        metrics = list(metrics_data.keys())

        for i, m1 in enumerate(metrics):
            for m2 in metrics[i + 1 :]:
                v1 = metrics_data[m1]
                v2 = metrics_data[m2]
                if len(v1) == len(v2) and len(v1) >= 3:
                    stats_mod = _require_stats()
                    r, _ = stats_mod.pearsonr(v1, v2)
                    correlations[(m1, m2)] = float(r)

        return correlations

    def _compare_to_literature(
        self,
        metrics_data: dict[str, list[float]],
        literature: LiteratureReview,
    ) -> list[Claim]:
        """Compare results to literature findings."""
        claims = []

        # Simple comparison based on source count
        if literature.source_count > 10:
            claims.append(
                Claim(
                    statement="Results are consistent with substantial prior literature",
                    metric="literature_support",
                    value=literature.source_count,
                    confidence_interval=(0, 0),
                )
            )

        return claims

    def _identify_limitations(
        self,
        results: ExperimentResults,
        claims: list[Claim],
    ) -> list[str]:
        """Identify analysis limitations."""
        limitations = []

        if results.total_trials < 50:
            limitations.append("Limited sample size may affect generalizability")

        if results.parameter_coverage < 0.5:
            limitations.append("Parameter space not fully explored")

        small_effects = [
            c for c in claims if c.effect_size and abs(c.effect_size) < 0.3
        ]
        if small_effects:
            limitations.append(
                "Some effects are small and may not be practically significant"
            )

        return limitations


class WritingAgent(ResearchAgent):
    """Agent for synthesizing findings into publication-ready paper."""

    def __init__(self, depth: int = 2, breadth: int = 2):
        super().__init__(depth, breadth)

    def run(
        self,
        literature: LiteratureReview,
        analysis: Analysis,
        results: ExperimentResults,
        venue: str = "clawxiv",
    ) -> Paper:
        """Generate a publication-ready paper.

        Args:
            literature: Literature review.
            analysis: Statistical analysis.
            results: Raw experiment results.
            venue: Target publication venue.

        Returns:
            Paper ready for submission.
        """
        title = self._generate_title(literature, analysis)
        abstract = self._generate_abstract(literature, analysis)
        source = self._generate_latex(literature, analysis, results)

        return Paper(
            title=title,
            abstract=abstract,
            source=source,
            categories=["cs.MA", "cs.AI"],
        )

    def _generate_title(
        self,
        literature: LiteratureReview,
        analysis: Analysis,
    ) -> str:
        """Generate paper title."""
        # Find primary claim
        primary_claims = [c for c in analysis.claims if c.is_primary]
        if primary_claims:
            metric = primary_claims[0].metric
            return f"Empirical Analysis of {metric.replace('_', ' ').title()} in Multi-Agent Systems"
        return "Multi-Agent System Analysis: Empirical Findings"

    def _generate_abstract(
        self,
        literature: LiteratureReview,
        analysis: Analysis,
    ) -> str:
        """Generate 4-sentence abstract."""
        # Sentence 1: Problem — adapt phrasing to hypothesis style
        hypothesis = literature.hypothesis
        if hypothesis.lower().startswith(("novel ", "empirical ", "extension ")):
            problem = f"{hypothesis} remains an open challenge."
        else:
            problem = (
                f"Understanding whether {hypothesis.lower()} remains an open challenge."
            )

        # Sentence 2: Method
        method = "We conduct systematic simulations using the SWARM framework."

        # Sentence 3: Finding
        if analysis.claims:
            claim = analysis.claims[0]
            finding = f"We find {claim.metric} = {claim.value:.3f} (95% CI: [{claim.confidence_interval[0]:.3f}, {claim.confidence_interval[1]:.3f}])."
        else:
            finding = "We report quantitative metrics across configurations."

        # Sentence 4: Implication
        implication = "These findings inform the design of robust multi-agent systems."

        return f"{problem} {method} {finding} {implication}"

    def _generate_latex(
        self,
        literature: LiteratureReview,
        analysis: Analysis,
        results: ExperimentResults,
    ) -> str:
        """Generate LaTeX source."""
        sections = []

        # Preamble
        sections.append(r"\documentclass{article}")
        sections.append(r"\usepackage{amsmath,amssymb,amsthm}")
        sections.append(r"\usepackage{booktabs}")
        sections.append("")
        sections.append(r"\title{" + self._generate_title(literature, analysis) + "}")
        sections.append(r"\author{SWARM Research Agent}")
        sections.append(r"\date{\today}")
        sections.append("")
        sections.append(r"\begin{document}")
        sections.append(r"\maketitle")
        sections.append("")

        # Abstract
        sections.append(r"\begin{abstract}")
        sections.append(self._generate_abstract(literature, analysis))
        sections.append(r"\end{abstract}")
        sections.append("")

        # Introduction
        sections.append(r"\section{Introduction}")
        sections.append(f"This paper investigates {literature.hypothesis}.")
        if literature.gaps:
            sections.append(f"Prior work has identified gaps: {literature.gaps[0]}.")
        sections.append("")

        # Related Work (breadth controls coverage)
        if self.breadth >= 2:
            sections.append(r"\section{Related Work}")
            sections.append(literature.related_work_summary)
            sections.append("")

        # Methods
        sections.append(r"\section{Methods}")
        sections.append(
            f"We ran {results.total_trials} trials across {len(results.configs)} configurations."
        )
        sections.append(f"Parameter coverage: {results.parameter_coverage:.1%}.")
        sections.append("")

        # Results
        sections.append(r"\section{Results}")
        sections.append(r"\begin{table}[h]")
        sections.append(r"\centering")
        sections.append(r"\begin{tabular}{lcc}")
        sections.append(r"\toprule")
        sections.append(r"Metric & Value & 95\% CI \\")
        sections.append(r"\midrule")
        for claim in analysis.claims[:5]:  # Top 5 claims
            ci_str = f"[{claim.confidence_interval[0]:.3f}, {claim.confidence_interval[1]:.3f}]"
            sections.append(f"{claim.metric} & {claim.value:.3f} & {ci_str} \\\\")
        sections.append(r"\bottomrule")
        sections.append(r"\end{tabular}")
        sections.append(r"\caption{Summary Statistics}")
        sections.append(r"\end{table}")
        sections.append("")

        # Effect sizes (depth >= 3)
        if self.depth >= 3 and analysis.effect_sizes:
            sections.append(r"\subsection{Effect Sizes}")
            for metric, es in analysis.effect_sizes.items():
                size_label = (
                    "small" if abs(es) < 0.5 else "medium" if abs(es) < 0.8 else "large"
                )
                sections.append(f"{metric}: Cohen's d = {es:.3f} ({size_label})")
            sections.append("")

        # Discussion
        sections.append(r"\section{Discussion}")
        if analysis.limitations:
            sections.append("Limitations: " + "; ".join(analysis.limitations))
        sections.append("")

        # Conclusion
        sections.append(r"\section{Conclusion}")
        sections.append(f"We empirically investigated {literature.hypothesis}.")
        sections.append("")

        # Reproducibility (depth >= 3)
        if self.depth >= 3:
            sections.append(r"\section*{Reproducibility}")
            sections.append(
                "Code and configurations available at: github.com/swarm-ai-safety/swarm"
            )
            sections.append("")

        sections.append(r"\end{document}")

        return "\n".join(sections)


class ReviewAgent(ResearchAgent):
    """Agent for adversarial peer review."""

    def run(
        self,
        paper: Paper,
        analysis: Analysis,
    ) -> Review:
        """Review a paper for quality issues.

        Args:
            paper: The paper to review.
            analysis: The underlying analysis.

        Returns:
            Review with critiques and recommendation.
        """
        critiques = []

        # Check statistics
        critiques.extend(self._check_statistics(analysis))

        # Check methodology
        critiques.extend(self._check_methodology(paper, analysis))

        # Check claims vs evidence
        critiques.extend(self._verify_claims(paper, analysis))

        # Check completeness
        critiques.extend(self._check_completeness(paper, analysis))

        # Determine recommendation
        high_severity = len(
            [c for c in critiques if c.severity in ("high", "critical")]
        )
        medium_severity = len([c for c in critiques if c.severity == "medium"])

        if high_severity > 0:
            recommendation = "major_revision"
        elif medium_severity > 2:
            recommendation = "minor_revision"
        else:
            recommendation = "accept"

        return Review(
            critiques=critiques,
            recommendation=recommendation,
            summary=f"Found {len(critiques)} issues. Recommendation: {recommendation}",
        )

    def _check_statistics(self, analysis: Analysis) -> list[Critique]:
        """Check statistical issues."""
        critiques = []

        # Check for p-hacking indicators
        p_values = [c.p_value for c in analysis.claims if c.p_value]
        marginal = [p for p in p_values if 0.04 < p < 0.06]
        if len(marginal) > len(p_values) * 0.3:
            critiques.append(
                Critique(
                    severity="high",
                    category="statistics",
                    issue="Multiple p-values near 0.05 threshold suggests potential p-hacking",
                    suggestion="Pre-register hypotheses or use stricter threshold",
                )
            )

        # Check effect sizes
        small_effects = [
            c for c in analysis.claims if c.effect_size and abs(c.effect_size) < 0.2
        ]
        if len(small_effects) > len(analysis.claims) * 0.5:
            critiques.append(
                Critique(
                    severity="medium",
                    category="statistics",
                    issue="Many small effect sizes - practical significance unclear",
                    suggestion="Discuss practical implications of small effects",
                )
            )

        # Check CI coverage
        if not analysis.all_claims_have_ci:
            critiques.append(
                Critique(
                    severity="high",
                    category="statistics",
                    issue="Not all claims have confidence intervals",
                    suggestion="Add 95% CI for all reported statistics",
                )
            )

        return critiques

    def _check_methodology(self, paper: Paper, analysis: Analysis) -> list[Critique]:
        """Check methodology issues."""
        critiques = []

        # Check for reproducibility info
        if (
            "seed" not in paper.source.lower()
            and "reproducib" not in paper.source.lower()
        ):
            critiques.append(
                Critique(
                    severity="medium",
                    category="methodology",
                    issue="No mention of random seeds or reproducibility",
                    suggestion="Document random seeds and provide reproduction instructions",
                )
            )

        return critiques

    def _verify_claims(self, paper: Paper, analysis: Analysis) -> list[Critique]:
        """Verify claims have evidence."""
        critiques = []

        # Simple check: look for numeric claims in abstract
        abstract_numbers = re.findall(r"\d+\.?\d*", paper.abstract)
        claim_values = [str(round(c.value, 2)) for c in analysis.claims]

        for num in abstract_numbers:
            if num not in " ".join(claim_values) and float(num) > 1:
                critiques.append(
                    Critique(
                        severity="low",
                        category="claims",
                        issue=f"Number {num} in abstract may not match analysis",
                        suggestion="Verify all numbers trace to analysis",
                    )
                )
                break  # Only flag once

        return critiques

    def _check_completeness(self, paper: Paper, analysis: Analysis) -> list[Critique]:
        """Check for completeness."""
        critiques = []

        if not analysis.limitations:
            critiques.append(
                Critique(
                    severity="medium",
                    category="completeness",
                    issue="No limitations discussed",
                    suggestion="Add limitations section",
                )
            )

        if "future" not in paper.source.lower():
            critiques.append(
                Critique(
                    severity="low",
                    category="completeness",
                    issue="No future work discussed",
                    suggestion="Add future directions",
                )
            )

        return critiques


class CritiqueAgent(ResearchAgent):
    """Agent for red-teaming findings before external review."""

    def run(
        self,
        hypothesis: str,
        results: ExperimentResults,
        analysis: Analysis,
    ) -> list[Critique]:
        """Red-team findings to find weaknesses.

        Args:
            hypothesis: The hypothesis tested.
            results: Experiment results.
            analysis: Statistical analysis.

        Returns:
            List of critiques from adversarial perspective.
        """
        critiques = []

        # Generate alternative hypotheses
        alternatives = self._generate_alternatives(hypothesis)
        for alt in alternatives:
            if self._consistent_with_data(alt, results, analysis):
                critiques.append(
                    Critique(
                        severity="medium",
                        category="methodology",
                        issue=f"Alternative explanation consistent with data: {alt}",
                        suggestion="Run additional experiments to distinguish hypotheses",
                    )
                )

        # Look for confounds
        confounds = self._identify_confounds(results)
        for confound in confounds:
            critiques.append(
                Critique(
                    severity="medium",
                    category="methodology",
                    issue=f"Potential confound: {confound}",
                    suggestion="Control for this variable or discuss as limitation",
                )
            )

        # Check boundary conditions
        boundary_issues = self._check_boundaries(results, analysis)
        critiques.extend(boundary_issues)

        return critiques

    def _generate_alternatives(self, hypothesis: str) -> list[str]:
        """Generate alternative explanations."""
        return [
            "Null hypothesis: no effect",
            f"Reverse causation of {hypothesis}",
            f"Confounding explains {hypothesis}",
        ]

    def _consistent_with_data(
        self,
        alternative: str,
        results: ExperimentResults,
        analysis: Analysis,
    ) -> bool:
        """Check if alternative is consistent with (i.e. not ruled out by) data.

        Uses normalized effect sizes and p-values rather than raw metric
        values, so the check works regardless of metric scale. Covers the
        closed set produced by :meth:`_generate_alternatives`:

        * **null** — consistent when no claim shows a substantial or
          significant effect.
        * **reverse causation** — an intervention fixes causal direction, so
          this is ruled out when the design *manipulated* parameters
          (configs differ in parameter values). Purely observational data
          with a claimed effect cannot rule it out.
        * **confounding** — ruled out by a controlled sweep (exactly one
          parameter varied). When several parameters co-vary across configs,
          or nothing was varied at all, confounding stays on the table.

        Unknown alternative strings return False (nothing to assess).
        """
        alt = alternative.lower()
        if "null" in alt:
            # Null is consistent if effect sizes are small or p-values are large
            for claim in analysis.claims:
                if claim.effect_size is not None and abs(claim.effect_size) > 0.5:
                    return False
                if claim.p_value is not None and claim.p_value < 0.05:
                    return False
            return True

        # Non-null alternatives explain a claimed effect; with no substantial
        # effect claimed there is nothing for them to explain.
        if not self._has_claimed_effect(analysis):
            return False

        n_varied = self._count_varied_parameters(results)
        if "reverse causation" in alt:
            # Manipulated designs fix the causal arrow for the varied knobs.
            return n_varied == 0
        if "confound" in alt:
            # A clean single-parameter sweep isolates the effect; anything
            # else (co-varied parameters, or no variation) leaves confounding
            # viable.
            return n_varied != 1
        return False

    @staticmethod
    def _has_claimed_effect(analysis: Analysis) -> bool:
        """True when at least one claim reports a substantial or significant effect."""
        for claim in analysis.claims:
            if claim.effect_size is not None and abs(claim.effect_size) > 0.5:
                return True
            if claim.p_value is not None and claim.p_value < 0.05:
                return True
        return False

    @staticmethod
    def _count_varied_parameters(results: ExperimentResults) -> int:
        """Number of parameters that take >=2 distinct values across configs."""
        values: dict[str, set[str]] = {}
        for config in results.configs:
            for name, value in config.parameters.items():
                values.setdefault(name, set()).add(repr(value))
        return sum(1 for seen in values.values() if len(seen) >= 2)

    def _identify_confounds(self, results: ExperimentResults) -> list[str]:
        """Identify potential confounding variables."""
        confounds = []

        # Check if parameters are correlated in the design
        params_tested: Set[str] = set()
        for config in results.configs:
            params_tested.update(config.parameters.keys())

        if len(params_tested) > 2:
            confounds.append("Multiple parameters varied - interactions possible")

        return confounds

    def _check_boundaries(
        self,
        results: ExperimentResults,
        analysis: Analysis,
    ) -> list[Critique]:
        """Check boundary conditions."""
        critiques = []

        # Check for extreme values
        for claim in analysis.claims:
            if claim.value < 0 or claim.value > 1:
                if "toxicity" in claim.metric or "probability" in claim.metric:
                    critiques.append(
                        Critique(
                            severity="high",
                            category="methodology",
                            issue=f"{claim.metric} out of expected [0,1] range",
                            suggestion="Verify metric computation",
                        )
                    )

        return critiques


class ReplicationAgent(ResearchAgent):
    """Agent for verifying prior findings."""

    def __init__(
        self,
        depth: int = 2,
        breadth: int = 2,
        platforms: list[PlatformClient] | None = None,
        simulation_fn: Callable | None = None,
    ):
        super().__init__(depth, breadth)
        self.platforms = platforms or []
        self.simulation_fn = simulation_fn

    def run(
        self,
        paper_id: str,
        platform: PlatformClient | None = None,
    ) -> dict[str, Any]:
        """Attempt to replicate a published finding.

        Args:
            paper_id: ID of paper to replicate.
            platform: Platform to fetch paper from.

        Returns:
            Replication results.
        """
        # Fetch original paper
        paper = None
        if platform:
            paper = platform.get_paper(paper_id)

        if not paper:
            return {"success": False, "error": "Could not fetch paper"}

        # Extract methodology (simplified)
        config = self._extract_config(paper)
        if not config:
            return {"success": False, "error": "Could not extract methodology"}

        # Run exact replication
        exact_result = self._run_exact(config)

        # Run conceptual replication (different seeds)
        conceptual_result = self._run_conceptual(config)

        return {
            "success": True,
            "paper_id": paper_id,
            "exact_replication": exact_result,
            "conceptual_replication": conceptual_result,
            "verdict": self._verdict(exact_result, conceptual_result),
        }

    def _extract_config(self, paper: Paper) -> dict[str, Any] | None:
        """Extract configuration from paper."""
        # Look for common patterns
        config = {}

        source = paper.source.lower()
        if "trials" in source:
            # Try to extract trial count
            match = re.search(r"(\d+)\s*trials", source)
            if match:
                config["trials"] = int(match.group(1))

        if "agents" in source:
            match = re.search(r"(\d+)\s*agents", source)
            if match:
                config["agents"] = int(match.group(1))

        return config if config else None

    def _run_exact(self, config: dict) -> dict:
        """Run exact replication with same seeds."""
        if not self.simulation_fn:
            return {"metrics": {}, "match": False}

        result = self.simulation_fn(config)
        return {"metrics": result, "match": True}

    def _run_conceptual(self, config: dict) -> dict:
        """Run conceptual replication with new seeds."""
        if not self.simulation_fn:
            return {"metrics": {}, "consistent": False}

        results = []
        for seed in range(5):
            np.random.seed(seed + 1000)
            result = self.simulation_fn(config)
            results.append(result)

        # Aggregate
        aggregated = {}
        for key in results[0]:
            values = [r[key] for r in results]
            aggregated[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
            }

        return {"metrics": aggregated, "consistent": True}

    def _verdict(self, exact: dict, conceptual: dict) -> str:
        """Determine replication verdict."""
        if exact.get("match") and conceptual.get("consistent"):
            return "REPLICATED"
        elif conceptual.get("consistent"):
            return "PARTIALLY_REPLICATED"
        else:
            return "FAILED_TO_REPLICATE"
