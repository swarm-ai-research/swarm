"""Main research workflow orchestrator."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from swarm.research.agents import (
    Analysis,
    AnalysisAgent,
    CritiqueAgent,
    ExperimentAgent,
    ExperimentResults,
    LiteratureAgent,
    LiteratureReview,
    ReplicationAgent,
    ReviewAgent,
    WritingAgent,
)
from swarm.research.platforms import (
    AgentRxivClient,
    AgentxivClient,
    ClawxivClient,
    Paper,
    PlatformClient,
    SubmissionResult,
)
from swarm.research.quality import (
    GateResult,
    PreRegistration,
    QualityGate,
    QualityGates,
)
from swarm.research.reflexivity import (
    Finding,
    ReflexivityAnalyzer,
)


@dataclass
class WorkflowConfig:
    """Configuration for research workflow."""

    depth: int = 2
    breadth: int = 2
    trials_per_config: int = 10
    rounds_per_trial: int = 100
    platforms: list[str] = field(default_factory=lambda: ["agentxiv", "clawxiv"])
    target_venue: str = "clawxiv"
    enable_reflexivity: bool = True
    enable_pre_registration: bool = True
    max_iterations: int = 3
    api_keys: dict[str, str] = field(default_factory=dict)

    # Configurable gate thresholds
    min_literature_sources: int = 5
    min_trials_per_config: int = 10
    min_parameter_coverage: float = 0.5


@dataclass
class WorkflowState:
    """State of a research workflow execution."""

    question: str
    config: WorkflowConfig
    iteration: int = 0
    literature: LiteratureReview | None = None
    experiments: ExperimentResults | None = None
    analysis: Analysis | None = None
    paper: Paper | None = None
    pre_registration: PreRegistration | None = None
    gate_results: list[GateResult] = field(default_factory=list)
    reflexivity_result: Any = None
    submission_result: SubmissionResult | None = None
    status: str = "initialized"
    errors: list[str] = field(default_factory=list)


class ResearchWorkflow:
    """Orchestrates the complete research workflow.

    Implements the enhanced workflow from docs/guides/research-workflow.md:
    1. Pre-registration
    2. Literature review
    3. Experiment design & execution
    4. Statistical analysis
    5. Self-critique
    6. Paper writing
    7. Peer review
    8. Reflexivity analysis
    9. Publication
    """

    def __init__(
        self,
        config: WorkflowConfig | None = None,
        simulation_fn: Callable[[dict], dict[str, float]] | None = None,
    ):
        self.config = config or WorkflowConfig()
        self.simulation_fn = simulation_fn

        # Initialize platform clients
        self.platforms: list[PlatformClient] = []
        for platform_name in self.config.platforms:
            api_key = self.config.api_keys.get(
                platform_name,
                os.environ.get(f"{platform_name.upper()}_API_KEY"),
            )
            if platform_name == "agentxiv":
                self.platforms.append(AgentxivClient(api_key=api_key))
            elif platform_name == "clawxiv":
                self.platforms.append(ClawxivClient(api_key=api_key))
            elif platform_name == "agentrxiv":
                self.platforms.append(AgentRxivClient(api_key=api_key))

        # Initialize agents
        self.literature_agent = LiteratureAgent(
            depth=self.config.depth,
            breadth=self.config.breadth,
            platforms=self.platforms,
        )
        self.experiment_agent = ExperimentAgent(
            depth=self.config.depth,
            breadth=self.config.breadth,
            simulation_fn=simulation_fn,
        )
        self.analysis_agent = AnalysisAgent(
            depth=self.config.depth,
            breadth=self.config.breadth,
        )
        self.writing_agent = WritingAgent(
            depth=self.config.depth,
            breadth=self.config.breadth,
        )
        self.review_agent = ReviewAgent(
            depth=self.config.depth,
            breadth=self.config.breadth,
        )
        self.critique_agent = CritiqueAgent(
            depth=self.config.depth,
            breadth=self.config.breadth,
        )
        self.replication_agent = ReplicationAgent(
            depth=self.config.depth,
            breadth=self.config.breadth,
            platforms=self.platforms,
            simulation_fn=simulation_fn,
        )

        # Build quality gates from QualityGates class, with configurable thresholds
        self._literature_gate = self._build_literature_gate()
        self._experiment_gate = self._build_experiment_gate()
        self._analysis_gate = QualityGates.analysis_gate()
        self._review_gate = QualityGates.review_gate()

        # Initialize reflexivity analyzer if enabled
        self.reflexivity_analyzer: ReflexivityAnalyzer | None = None
        if self.config.enable_reflexivity and simulation_fn:
            self.reflexivity_analyzer = ReflexivityAnalyzer(
                simulation_fn=lambda cfg: simulation_fn(cfg),
            )

    def _build_literature_gate(self) -> QualityGate:
        """Build literature gate with configurable min_sources threshold."""
        min_sources = self.config.min_literature_sources
        return (
            QualityGate("Literature Review")
            .add_check(
                "min_sources",
                lambda lit: len(lit.sources) >= min_sources,
                message=f"Minimum {min_sources} sources required",
            )
            .add_check(
                "gaps_identified",
                lambda lit: len(lit.gaps) >= 1,
                message="At least one research gap must be identified",
            )
            .add_check(
                "hypothesis_formed",
                lambda lit: bool(lit.hypothesis),
                message="A testable hypothesis must be formed",
            )
        )

    def _build_experiment_gate(self) -> QualityGate:
        """Build experiment gate with configurable thresholds."""
        min_trials = self.config.min_trials_per_config
        min_coverage = self.config.min_parameter_coverage
        return (
            QualityGate("Experiment")
            .add_check(
                "min_trials",
                lambda exp: exp.trials_per_config >= min_trials,
                message=f"Minimum {min_trials} trials per configuration",
            )
            .add_check(
                "seeds_documented",
                lambda exp: all(r.seed is not None for r in exp.results),
                message="All random seeds must be documented",
            )
            .add_check(
                "configs_complete",
                lambda exp: exp.parameter_coverage >= min_coverage,
                message=f"At least {min_coverage:.0%} parameter coverage required",
            )
            .add_check(
                "no_errors",
                lambda exp: exp.error_count == 0,
                message="No simulation errors allowed",
            )
        )

    def run(
        self,
        question: str,
        parameter_space: dict[str, list[Any]] | None = None,
    ) -> WorkflowState:
        """Execute the complete research workflow.

        Args:
            question: The research question to investigate.
            parameter_space: Parameter space for experiments.

        Returns:
            WorkflowState with all results.
        """
        state = WorkflowState(question=question, config=self.config)

        try:
            for iteration in range(self.config.max_iterations):
                state.iteration = iteration
                state.status = f"iteration_{iteration}"

                # Phase 1: Literature Review
                state = self._run_literature_phase(state)
                if not self._check_gate(state, "literature"):
                    continue

                # Phase 2: Pre-registration (if enabled)
                if self.config.enable_pre_registration and iteration == 0:
                    state = self._run_preregistration_phase(state, parameter_space)

                # Phase 3: Experiments
                state = self._run_experiment_phase(state, parameter_space)
                if not self._check_gate(state, "experiment"):
                    continue

                # Phase 4: Analysis
                state = self._run_analysis_phase(state)
                if not self._check_gate(state, "analysis"):
                    continue

                # Phase 5: Self-Critique
                state = self._run_critique_phase(state)
                if not self._is_robust(state):
                    # Refine question and iterate
                    state.question = self._refine_question(state)
                    continue

                # Phase 6: Writing
                state = self._run_writing_phase(state)

                # Phase 7: Peer Review
                state = self._run_review_phase(state)
                if not self._check_gate(state, "review"):
                    state = self._revise_paper(state)
                    continue

                # Phase 8: Reflexivity Analysis (if enabled)
                if self.config.enable_reflexivity:
                    state = self._run_reflexivity_phase(state)

                # Phase 9: Publication
                state = self._run_publication_phase(state)

                # Success - break iteration loop
                state.status = "completed"
                break

        except Exception as e:
            state.status = "error"
            state.errors.append(str(e))

        return state

    def _run_literature_phase(self, state: WorkflowState) -> WorkflowState:
        """Run literature review phase."""
        state.literature = self.literature_agent.run(
            question=state.question,
            platforms=[p.__class__.__name__ for p in self.platforms],
        )
        return state

    def _run_preregistration_phase(
        self,
        state: WorkflowState,
        parameter_space: dict[str, list[Any]] | None,
    ) -> WorkflowState:
        """Create pre-registration."""
        if not state.literature:
            return state

        state.pre_registration = PreRegistration(
            hypothesis=state.literature.hypothesis,
            secondary_hypotheses=[],
            methodology={
                "parameter_space": parameter_space or {},
                "trials": self.config.trials_per_config,
                "rounds": self.config.rounds_per_trial,
            },
            analysis_plan={
                "primary": ["Descriptive statistics", "Significance tests"],
                "secondary": ["Effect sizes", "Correlations"],
            },
        )
        return state

    def _run_experiment_phase(
        self,
        state: WorkflowState,
        parameter_space: dict[str, list[Any]] | None,
    ) -> WorkflowState:
        """Run experiment phase."""
        if not state.literature:
            return state

        default_space = {
            "honest_fraction": [0.2, 0.4, 0.6, 0.8, 1.0],
            "transaction_tax": [0.0, 0.05, 0.10],
        }

        state.experiments = self.experiment_agent.run(
            hypothesis=state.literature.hypothesis,
            parameter_space=parameter_space or default_space,
            trials=self.config.trials_per_config,
            rounds=self.config.rounds_per_trial,
        )
        return state

    def _run_analysis_phase(self, state: WorkflowState) -> WorkflowState:
        """Run analysis phase."""
        if not state.experiments:
            return state

        state.analysis = self.analysis_agent.run(
            results=state.experiments,
            literature=state.literature,
        )
        return state

    def _run_critique_phase(self, state: WorkflowState) -> WorkflowState:
        """Run self-critique phase."""
        if not state.literature or not state.experiments or not state.analysis:
            return state

        critiques = self.critique_agent.run(
            hypothesis=state.literature.hypothesis,
            results=state.experiments,
            analysis=state.analysis,
        )

        # Add critiques to analysis limitations
        for critique in critiques:
            if critique.severity in ("high", "critical"):
                state.analysis.limitations.append(critique.issue)

        return state

    def _generate_paper(
        self,
        literature: LiteratureReview | None,
        analysis: Analysis | None,
        experiments: ExperimentResults | None,
    ) -> Paper | None:
        """Generate paper from literature, analysis, and experiments."""
        if not literature or not analysis or not experiments:
            return None
        return self.writing_agent.run(
            literature=literature,
            analysis=analysis,
            results=experiments,
            venue=self.config.target_venue,
        )

    def _run_writing_phase(self, state: WorkflowState) -> WorkflowState:
        """Run paper writing phase."""
        if not state.literature or not state.analysis or not state.experiments:
            return state

        state.paper = self._generate_paper(
            state.literature, state.analysis, state.experiments
        )
        return state

    def _run_review_phase(self, state: WorkflowState) -> WorkflowState:
        """Run peer review phase."""
        if not state.paper or not state.analysis:
            return state

        review = self.review_agent.run(
            paper=state.paper,
            analysis=state.analysis,
        )

        # Record gate result
        gate_result = GateResult(
            gate_name="review",
            passed=review.recommendation in ("accept", "minor_revision"),
            checks=[],
        )
        state.gate_results.append(gate_result)

        return state

    def _run_reflexivity_phase(self, state: WorkflowState) -> WorkflowState:
        """Run reflexivity analysis phase."""
        if not self.reflexivity_analyzer or not state.analysis:
            return state

        # Create finding from primary claim
        if state.analysis.claims:
            claim = state.analysis.claims[0]
            finding = Finding(
                statement=claim.statement,
                metric_name=claim.metric,
                metric_value=claim.value,
                confidence_interval=claim.confidence_interval,
            )

            # Get baseline metrics
            baseline = {c.metric: c.value for c in state.analysis.claims}

            # Run reflexivity analysis
            state.reflexivity_result = self.reflexivity_analyzer.analyze(
                finding=finding,
                config={},
                baseline_metrics=baseline,
            )

            # Update paper with reflexivity disclosure
            if state.paper and state.reflexivity_result:
                disclosure = state.reflexivity_result.generate_disclosure()
                state.paper.source += f"\n\n% Reflexivity Analysis\n% {disclosure}\n"

        return state

    def _run_publication_phase(self, state: WorkflowState) -> WorkflowState:
        """Run publication phase."""
        if not state.paper:
            return state

        # Find target platform client
        target_client = None
        for client in self.platforms:
            if self.config.target_venue.lower() in client.__class__.__name__.lower():
                target_client = client
                break

        if target_client:
            state.submission_result = target_client.submit(state.paper)
        else:
            state.errors.append(f"No client for venue: {self.config.target_venue}")

        return state

    def _check_gate(self, state: WorkflowState, phase: str) -> bool:
        """Check quality gate for a phase using QualityGate instances."""
        gate_data = {
            "literature": (self._literature_gate, state.literature),
            "experiment": (self._experiment_gate, state.experiments),
            "analysis": (self._analysis_gate, state.analysis),
        }

        if phase == "review":
            # Review gate is recorded during review phase
            review_results = [g for g in state.gate_results if g.gate_name == "review"]
            if review_results:
                return review_results[-1].passed
            return True

        entry = gate_data.get(phase)
        if not entry:
            return True

        gate, data = entry
        if data is None:
            result = GateResult(gate_name=phase, passed=False, checks=[])
        else:
            result = gate.run(data)

        state.gate_results.append(result)
        return result.passed

    def _is_robust(self, state: WorkflowState) -> bool:
        """Check if findings are robust after critique."""
        if not state.analysis:
            return False

        # Check that no critical issues were found
        critical_limitations = [
            lim
            for lim in state.analysis.limitations
            if "critical" in lim.lower() or "severe" in lim.lower()
        ]
        return len(critical_limitations) == 0

    def _refine_question(self, state: WorkflowState) -> str:
        """Refine research question based on critique."""
        if state.analysis and state.analysis.limitations:
            # Narrow scope based on first limitation
            return f"{state.question} (controlling for {state.analysis.limitations[0][:30]}...)"
        return state.question

    def _revise_paper(self, state: WorkflowState) -> WorkflowState:
        """Re-write paper with same analysis and results (review feedback not incorporated)."""
        # Re-run writing agent with same inputs as initial write
        paper = self._generate_paper(
            state.literature, state.analysis, state.experiments
        )
        if paper:
            state.paper = paper
        return state

    def replicate(self, paper_id: str, platform: str = "clawxiv") -> dict[str, Any]:
        """Attempt to replicate a published finding.

        Args:
            paper_id: ID of paper to replicate.
            platform: Platform the paper is on.

        Returns:
            Replication results.
        """
        client = None
        for c in self.platforms:
            if platform.lower() in c.__class__.__name__.lower():
                client = c
                break

        return self.replication_agent.run(paper_id=paper_id, platform=client)

    def save_state(self, state: WorkflowState, path: str) -> None:
        """Save workflow state to file with full data."""
        data: dict[str, Any] = {
            "question": state.question,
            "iteration": state.iteration,
            "status": state.status,
            "errors": state.errors,
            "config": {
                "depth": state.config.depth,
                "breadth": state.config.breadth,
                "target_venue": state.config.target_venue,
                "trials_per_config": state.config.trials_per_config,
                "rounds_per_trial": state.config.rounds_per_trial,
                "min_literature_sources": state.config.min_literature_sources,
                "min_trials_per_config": state.config.min_trials_per_config,
                "min_parameter_coverage": state.config.min_parameter_coverage,
            },
        }

        if state.literature:
            data["literature"] = {
                "source_count": state.literature.source_count,
                "gaps": state.literature.gaps,
                "hypothesis": state.literature.hypothesis,
                "related_work_summary": state.literature.related_work_summary,
                "sources": [
                    {
                        "paper_id": s.paper_id,
                        "title": s.title,
                        "abstract": s.abstract[:500],
                        "platform": s.platform,
                        "relevance_score": s.relevance_score,
                    }
                    for s in state.literature.sources
                ],
            }

        if state.experiments:
            data["experiments"] = {
                "total_trials": state.experiments.total_trials,
                "config_count": len(state.experiments.configs),
                "error_count": state.experiments.error_count,
                "parameter_coverage": state.experiments.parameter_coverage,
                "configs": [
                    {"name": c.name, "parameters": c.parameters}
                    for c in state.experiments.configs
                ],
            }

        if state.analysis:
            data["analysis"] = {
                "claim_count": len(state.analysis.claims),
                "claims": [
                    {
                        "statement": c.statement,
                        "metric": c.metric,
                        "value": c.value,
                        "confidence_interval": list(c.confidence_interval),
                        "effect_size": c.effect_size,
                        "p_value": c.p_value,
                    }
                    for c in state.analysis.claims
                ],
                "correlations": {
                    f"{k[0]}___{k[1]}": v
                    for k, v in state.analysis.correlations.items()
                },
                "effect_sizes": state.analysis.effect_sizes,
                "limitations": state.analysis.limitations,
            }

        if state.paper:
            data["paper"] = state.paper.to_dict()

        if state.pre_registration:
            data["pre_registration"] = state.pre_registration.to_dict()

        if state.submission_result:
            data["submission"] = {
                "success": state.submission_result.success,
                "paper_id": state.submission_result.paper_id,
                "message": state.submission_result.message,
            }

        if state.reflexivity_result:
            data["reflexivity"] = {
                "classification": state.reflexivity_result.final_classification.value,
                "epistemic_note": state.reflexivity_result.epistemic_note,
            }
            if state.reflexivity_result.shadow_simulation:
                data["reflexivity"]["shadow_divergence"] = (
                    state.reflexivity_result.shadow_simulation.overall_divergence
                )
                data["reflexivity"]["finding_is_robust"] = (
                    state.reflexivity_result.shadow_simulation.finding_is_robust
                )

        data["gate_results"] = [
            {"gate_name": g.gate_name, "passed": g.passed} for g in state.gate_results
        ]

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load_state(cls, path: str) -> dict[str, Any]:
        """Load workflow state from file."""
        with open(path) as f:
            result: dict[str, Any] = json.load(f)
            return result
