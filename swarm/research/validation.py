"""Validation workflow for paper claims via SWARM simulation.

Runs generated scenarios against SWARM and compares results
to testable claims extracted from papers.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from swarm.research.annotator import PaperAnnotation, VerifiableClaim
from swarm.research.scenario_gen import ScenarioGenerator


@dataclass
class ClaimResult:
    """Result of testing a single claim."""

    claim: VerifiableClaim = field(default_factory=VerifiableClaim)
    metric_values: list[float] = field(default_factory=list)
    mean_value: float = 0.0
    expected: str = "positive"
    matched: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim.to_dict(),
            "metric_values": list(self.metric_values),
            "mean_value": self.mean_value,
            "expected": self.expected,
            "matched": self.matched,
        }


@dataclass
class ValidationResult:
    """Full validation result for a paper."""

    paper_id: str = ""
    annotation: PaperAnnotation = field(default_factory=PaperAnnotation)
    scenario: dict[str, Any] = field(default_factory=dict)
    run_results: list[dict[str, Any]] = field(default_factory=list)
    claim_results: list[ClaimResult] = field(default_factory=list)
    overall_verdict: str = "inconclusive"
    report: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "annotation": self.annotation.to_dict(),
            "scenario": self.scenario,
            "run_results": list(self.run_results),
            "claim_results": [cr.to_dict() for cr in self.claim_results],
            "overall_verdict": self.overall_verdict,
            "report": self.report,
        }


# Type alias for simulation function
# Takes (scenario_dict, seed) -> dict with metric keys
SimulationFn = Callable[[dict[str, Any], int], dict[str, Any]]


class ValidationWorkflow:
    """End-to-end validation of paper claims via SWARM simulation."""

    def __init__(
        self,
        annotator: Any = None,
        generator: ScenarioGenerator | None = None,
        simulation_fn: SimulationFn | None = None,
        num_runs: int = 10,
    ):
        self._annotator = annotator
        self._generator = generator or ScenarioGenerator()
        self._simulation_fn = simulation_fn
        self._num_runs = num_runs

    def validate(self, paper_id: str) -> ValidationResult:
        """Full validation pipeline: annotate, generate, simulate, compare."""
        if self._annotator is None:
            return ValidationResult(
                paper_id=paper_id,
                overall_verdict="error",
                report="No annotator configured",
            )

        annotation = self._annotator.annotate(paper_id)
        return self.validate_from_annotation(annotation)

    def validate_from_annotation(
        self, annotation: PaperAnnotation
    ) -> ValidationResult:
        """Validate from an existing annotation."""
        scenario = self._generator.from_paper(annotation)
        run_results = self._run_scenarios(scenario, self._num_runs)
        claim_results = self._compare_claims(annotation.claims, run_results)
        report = self._generate_report(annotation, claim_results)

        # Determine overall verdict
        testable_claims = [cr for cr in claim_results if cr.claim.testable]
        if not testable_claims:
            verdict = "no_testable_claims"
        else:
            matched = sum(1 for cr in testable_claims if cr.matched)
            ratio = matched / len(testable_claims)
            if ratio >= 0.7:
                verdict = "supported"
            elif ratio >= 0.3:
                verdict = "partially_supported"
            else:
                verdict = "not_supported"

        return ValidationResult(
            paper_id=annotation.paper_id,
            annotation=annotation,
            scenario=scenario,
            run_results=run_results,
            claim_results=claim_results,
            overall_verdict=verdict,
            report=report,
        )

    def _run_scenarios(
        self, scenario_dict: dict[str, Any], num_runs: int
    ) -> list[dict[str, Any]]:
        """Run simulation with different seeds."""
        if self._simulation_fn is None:
            return []

        results: list[dict[str, Any]] = []
        for i in range(num_runs):
            seed = 42 + i
            result = self._simulation_fn(scenario_dict, seed)
            results.append(result)
        return results

    def _compare_claims(
        self,
        claims: list[VerifiableClaim],
        run_results: list[dict[str, Any]],
    ) -> list[ClaimResult]:
        """Compare expected claim directions against simulation results."""
        results: list[ClaimResult] = []

        for claim in claims:
            if not claim.testable or not claim.metric:
                results.append(
                    ClaimResult(claim=claim, expected=claim.expected, matched=False)
                )
                continue

            # Gather metric values across runs
            values = []
            for run in run_results:
                if claim.metric in run:
                    val = run[claim.metric]
                    if isinstance(val, (int, float)):
                        values.append(float(val))

            if not values:
                results.append(
                    ClaimResult(claim=claim, expected=claim.expected, matched=False)
                )
                continue

            mean_val = sum(values) / len(values)
            matched = self._check_direction(mean_val, claim.expected)

            results.append(
                ClaimResult(
                    claim=claim,
                    metric_values=values,
                    mean_value=mean_val,
                    expected=claim.expected,
                    matched=matched,
                )
            )

        return results

    def _check_direction(self, mean_value: float, expected: str) -> bool:
        """Check if the mean value matches the expected direction."""
        threshold = 0.01  # Small threshold for "zero"
        if expected == "positive":
            return mean_value > threshold
        elif expected == "negative":
            return mean_value < -threshold
        else:  # zero
            return abs(mean_value) <= threshold

    def _generate_report(
        self,
        annotation: PaperAnnotation,
        claim_results: list[ClaimResult],
    ) -> str:
        """Generate a text summary of validation results."""
        lines = [
            f"Validation Report for: {annotation.title}",
            f"Paper ID: {annotation.paper_id}",
            f"Risk Profile: density={annotation.risk_profile.interaction_density}, "
            f"failure_modes={annotation.risk_profile.failure_modes}",
            "",
            "Claim Results:",
        ]

        for i, cr in enumerate(claim_results, 1):
            status = "MATCHED" if cr.matched else "NOT MATCHED"
            lines.append(
                f"  {i}. [{status}] {cr.claim.claim[:80]}"
                f" (metric={cr.claim.metric}, expected={cr.expected}, "
                f"mean={cr.mean_value:.4f})"
            )

        testable = [cr for cr in claim_results if cr.claim.testable]
        if testable:
            matched = sum(1 for cr in testable if cr.matched)
            lines.append(f"\nSummary: {matched}/{len(testable)} testable claims matched")
        else:
            lines.append("\nNo testable claims found")

        return "\n".join(lines)
