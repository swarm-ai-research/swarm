"""Dual reporting for soft and hard metrics."""

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional

from swarm.core.payoff import PayoffConfig, SoftPayoffEngine
from swarm.metrics.incoherence import DecisionRecord, IncoherenceMetrics
from swarm.metrics.obfuscation_metrics import ObfuscationSummary
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.interaction import SoftInteraction


@dataclass
class MetricsSummary:
    """Summary of both soft and hard metrics."""

    # Soft metrics (probabilistic)
    toxicity_soft: float
    conditional_loss_initiator: float
    conditional_loss_counterparty: float
    spread: float
    quality_gap: float
    average_quality: float
    uncertain_fraction: float

    # Hard metrics (threshold-based)
    toxicity_hard: float
    acceptance_rate: float
    high_quality_acceptance: float
    low_quality_acceptance: float

    # Counts
    total_interactions: int
    accepted_count: int
    rejected_count: int
    high_quality_count: int
    low_quality_count: int

    # Welfare
    total_welfare: float
    total_social_surplus: float
    avg_initiator_payoff: float
    avg_counterparty_payoff: float

    # Calibration metrics (None if no ground truth)
    brier_score: Optional[float] = None
    log_loss: Optional[float] = None
    calibration_error: Optional[float] = None
    expected_calibration_error: Optional[float] = None

    # Variance metrics
    quality_variance: float = 0.0
    payoff_variance_initiator: float = 0.0
    payoff_variance_counterparty: float = 0.0

    # Projection-geometric diagnostics for the quality gap.
    # quality_correlation: ρ(p,a) ∈ [-1,+1], normalized quality gap.
    # baseline_harm:       1 - E[p],     population-level harm floor.
    # selection_credit:    β · Q,        toxicity reduction from selection.
    # selection_saturation: |Q| / (σ_p/√(αβ)) ∈ [0,1], C-S bound fill.
    quality_correlation: float = 0.0
    baseline_harm: float = 0.0
    selection_credit: float = 0.0
    selection_saturation: float = 0.0

    # Incoherence metrics (optional replay-driven additions)
    incoherence_disagreement: float = 0.0
    incoherence_error: float = 0.0
    incoherence_index: float = 0.0
    incoherence_n_decisions: int = 0

    # Obfuscation Atlas metrics (optional)
    obfuscation_probe_evasion: float = 0.0
    obfuscation_drift: float = 0.0
    obfuscation_rationalization: float = 0.0
    obfuscation_detector_auc: Optional[float] = None
    obfuscation_belief_shift: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "soft_metrics": {
                "toxicity": self.toxicity_soft,
                "conditional_loss_initiator": self.conditional_loss_initiator,
                "conditional_loss_counterparty": self.conditional_loss_counterparty,
                "spread": self.spread,
                "quality_gap": self.quality_gap,
                "quality_correlation": self.quality_correlation,
                "baseline_harm": self.baseline_harm,
                "selection_credit": self.selection_credit,
                "selection_saturation": self.selection_saturation,
                "average_quality": self.average_quality,
                "uncertain_fraction": self.uncertain_fraction,
            },
            "hard_metrics": {
                "toxicity": self.toxicity_hard,
                "acceptance_rate": self.acceptance_rate,
                "high_quality_acceptance": self.high_quality_acceptance,
                "low_quality_acceptance": self.low_quality_acceptance,
            },
            "counts": {
                "total": self.total_interactions,
                "accepted": self.accepted_count,
                "rejected": self.rejected_count,
                "high_quality": self.high_quality_count,
                "low_quality": self.low_quality_count,
            },
            "welfare": {
                "total_welfare": self.total_welfare,
                "total_social_surplus": self.total_social_surplus,
                "avg_initiator_payoff": self.avg_initiator_payoff,
                "avg_counterparty_payoff": self.avg_counterparty_payoff,
            },
            "calibration": {
                "brier_score": self.brier_score,
                "log_loss": self.log_loss,
                "calibration_error": self.calibration_error,
                "expected_calibration_error": self.expected_calibration_error,
            },
            "variance": {
                "quality_variance": self.quality_variance,
                "payoff_variance_initiator": self.payoff_variance_initiator,
                "payoff_variance_counterparty": self.payoff_variance_counterparty,
            },
            "incoherence": {
                "disagreement": self.incoherence_disagreement,
                "error": self.incoherence_error,
                "index": self.incoherence_index,
                "n_decisions": self.incoherence_n_decisions,
            },
            "obfuscation": {
                "probe_evasion": self.obfuscation_probe_evasion,
                "drift": self.obfuscation_drift,
                "rationalization": self.obfuscation_rationalization,
                "detector_auc": self.obfuscation_detector_auc,
                "belief_shift": self.obfuscation_belief_shift,
            },
        }


class MetricsReporter:
    """
    Reports both soft (probabilistic) and hard (threshold-based) metrics.

    Soft metrics use the full probability distribution, while hard metrics
    apply a threshold (default 0.5) to convert to binary labels.
    """

    def __init__(
        self,
        payoff_config: Optional[PayoffConfig] = None,
        quality_threshold: float = 0.5,
        uncertainty_band: float = 0.2,
    ):
        """
        Initialize reporter.

        Args:
            payoff_config: Configuration for payoff calculations
            quality_threshold: Threshold for hard metrics (default 0.5)
            uncertainty_band: Band around 0.5 for uncertainty flagging
        """
        self.payoff_engine = SoftPayoffEngine(payoff_config)
        self.soft_metrics = SoftMetrics(self.payoff_engine)
        self.quality_threshold = quality_threshold
        self.uncertainty_band = uncertainty_band

    def _compute_hard_toxicity(self, interactions: List[SoftInteraction]) -> float:
        """
        Compute hard toxicity: P(p < threshold | accepted)

        This is the fraction of accepted interactions that would be
        classified as harmful using a hard threshold.
        """
        accepted = [i for i in interactions if i.accepted]
        if not accepted:
            return 0.0

        toxic_count = sum(1 for i in accepted if i.p < self.quality_threshold)
        return toxic_count / len(accepted)

    def summary(self, interactions: List[SoftInteraction]) -> MetricsSummary:
        """
        Compute comprehensive summary of all metrics.

        Optimised via a single pass through the interaction list for core metrics:
        payoffs, quality accumulators, and counts are gathered together to avoid
        O(N) re-filtering. Note: calibration metrics (brier_score, log_loss, etc.)
        perform additional passes when ground_truth is available.

        Args:
            interactions: List of interactions

        Returns:
            MetricsSummary with both soft and hard metrics
        """
        if not interactions:
            return MetricsSummary(
                toxicity_soft=0.0,
                conditional_loss_initiator=0.0,
                conditional_loss_counterparty=0.0,
                spread=0.0,
                quality_gap=0.0,
                average_quality=0.0,
                uncertain_fraction=0.0,
                toxicity_hard=0.0,
                acceptance_rate=0.0,
                high_quality_acceptance=0.0,
                low_quality_acceptance=0.0,
                total_interactions=0,
                accepted_count=0,
                rejected_count=0,
                high_quality_count=0,
                low_quality_count=0,
                total_welfare=0.0,
                total_social_surplus=0.0,
                avg_initiator_payoff=0.0,
                avg_counterparty_payoff=0.0,
                brier_score=None,
                log_loss=None,
                calibration_error=None,
                expected_calibration_error=None,
                quality_variance=0.0,
                payoff_variance_initiator=0.0,
                payoff_variance_counterparty=0.0,
            )

        # ------------------------------------------------------------------
        # Single-pass accumulation
        # Hoist config lookups out of the loop to avoid repeated attribute access.
        # ------------------------------------------------------------------
        cfg = self.payoff_engine.config
        s_plus = cfg.s_plus
        s_minus = cfg.s_minus
        h = cfg.h
        theta = cfg.theta
        one_m_theta = 1.0 - theta
        rho_a = cfg.rho_a
        rho_b = cfg.rho_b
        w_rep = cfg.w_rep
        s_scale = s_plus + s_minus
        s_minus_plus_h = s_minus + h

        threshold = self.quality_threshold
        band = self.uncertainty_band
        band_lo = 0.5 - band
        band_hi = 0.5 + band

        n = len(interactions)

        # Running sums for all interactions
        sum_p = 0.0
        sum_p_sq = 0.0
        sum_pi_a = 0.0
        sum_pi_a_sq = 0.0
        sum_pi_b = 0.0
        sum_pi_b_sq = 0.0

        # Running sums for accepted subset
        n_acc = 0
        sum_p_acc = 0.0
        sum_pi_a_acc = 0.0
        sum_pi_b_acc = 0.0
        sum_social_acc = 0.0

        # Running sums for rejected subset
        sum_p_rej = 0.0
        n_rej = 0

        # Counters
        uncertain_count = 0
        hard_toxic_count = 0
        hq_count = 0
        lq_count = 0
        hq_accepted = 0
        lq_accepted = 0

        for interaction in interactions:
            p = interaction.p

            # Compute S_soft, E_soft, and both payoffs inline — avoids
            # calling expected_surplus / expected_harm / payoffs_both as
            # separate functions, eliminating per-call overhead.
            one_m_p = 1.0 - p
            S_soft = p * s_plus - one_m_p * s_minus
            E_soft = one_m_p * h

            pi_a = (
                theta * S_soft
                - interaction.tau
                - interaction.c_a
                - rho_a * E_soft
                + w_rep * interaction.r_a
            )
            pi_b = (
                one_m_theta * S_soft
                + interaction.tau
                - interaction.c_b
                - rho_b * E_soft
                + w_rep * interaction.r_b
            )

            # All-interaction accumulators
            sum_p += p
            sum_p_sq += p * p
            sum_pi_a += pi_a
            sum_pi_a_sq += pi_a * pi_a
            sum_pi_b += pi_b
            sum_pi_b_sq += pi_b * pi_b

            # Uncertainty band
            if band_lo < p < band_hi:
                uncertain_count += 1

            # Quality tier (high/low)
            if p >= threshold:
                hq_count += 1
                if interaction.accepted:
                    hq_accepted += 1
            else:
                lq_count += 1
                if interaction.accepted:
                    lq_accepted += 1

            # Accepted vs rejected
            if interaction.accepted:
                n_acc += 1
                sum_p_acc += p
                sum_pi_a_acc += pi_a
                sum_pi_b_acc += pi_b
                # Social surplus = S_soft - E_soft = p*s_plus - (1-p)*(s_minus+h)
                sum_social_acc += p * s_plus - one_m_p * s_minus_plus_h
                if p < threshold:
                    hard_toxic_count += 1
            else:
                n_rej += 1
                sum_p_rej += p

        # ------------------------------------------------------------------
        # Derive all metrics from accumulators (no further list traversals)
        # ------------------------------------------------------------------
        avg_p = sum_p / n

        avg_pi_a_all = sum_pi_a / n
        avg_pi_b_all = sum_pi_b / n

        if n_acc > 0:
            avg_p_acc = sum_p_acc / n_acc
            avg_pi_a_acc = sum_pi_a_acc / n_acc
            avg_pi_b_acc = sum_pi_b_acc / n_acc
            toxicity_soft = 1.0 - avg_p_acc
            toxicity_hard = hard_toxic_count / n_acc
            total_welfare = sum_pi_a_acc + sum_pi_b_acc
            total_social = sum_social_acc
            cl_init = avg_pi_a_acc - avg_pi_a_all
            cl_counter = avg_pi_b_acc - avg_pi_b_all
            spread = s_scale * (avg_p - avg_p_acc)
        else:
            avg_pi_a_acc = 0.0
            avg_pi_b_acc = 0.0
            toxicity_soft = 0.0
            toxicity_hard = 0.0
            total_welfare = 0.0
            total_social = 0.0
            cl_init = 0.0
            cl_counter = 0.0
            spread = 0.0

        avg_p_rej = (sum_p_rej / n_rej) if n_rej > 0 else 0.0

        # Quality gap: E[p | accepted] - E[p | rejected]
        quality_gap = (
            (sum_p_acc / n_acc - avg_p_rej) if (n_acc > 0 and n_rej > 0) else 0.0
        )

        uncertain_frac = uncertain_count / n
        acceptance_rate = n_acc / n

        hq_acc_rate = (hq_accepted / hq_count) if hq_count > 0 else 0.0
        lq_acc_rate = (lq_accepted / lq_count) if lq_count > 0 else 0.0

        # Variance via E[x²] - E[x]²  (numerically acceptable for these ranges)
        quality_var = max(0.0, sum_p_sq / n - avg_p * avg_p) if n >= 2 else 0.0
        payoff_var_a = (
            max(0.0, sum_pi_a_sq / n - avg_pi_a_all * avg_pi_a_all) if n >= 2 else 0.0
        )
        payoff_var_b = (
            max(0.0, sum_pi_b_sq / n - avg_pi_b_all * avg_pi_b_all) if n >= 2 else 0.0
        )

        # Projection-geometric decomposition of toxicity:
        #   T = (1 - E[p]) - β · Q   (baseline harm − selection credit)
        # and normalized gap ρ(p,a) = Q · √(αβ) / σ_p (Pearson correlation).
        alpha = acceptance_rate
        beta = 1.0 - alpha
        baseline_harm = 1.0 - avg_p
        selection_credit = beta * quality_gap
        sigma_p = math.sqrt(quality_var)
        ab = alpha * beta
        if sigma_p > 0.0 and ab > 0.0:
            quality_correlation = quality_gap * math.sqrt(ab) / sigma_p
            selection_saturation = abs(quality_gap) * math.sqrt(ab) / sigma_p
        else:
            quality_correlation = 0.0
            selection_saturation = 0.0

        # Calibration metrics require ground_truth (uncommon path); delegate to
        # SoftMetrics which handles the None-return case gracefully.
        brier = self.soft_metrics.brier_score(interactions)
        logloss = self.soft_metrics.log_loss(interactions)
        cal_error = self.soft_metrics.calibration_error(interactions)
        ece = self.soft_metrics.expected_calibration_error(interactions)

        return MetricsSummary(
            toxicity_soft=toxicity_soft,
            conditional_loss_initiator=cl_init,
            conditional_loss_counterparty=cl_counter,
            spread=spread,
            quality_gap=quality_gap,
            average_quality=avg_p,
            uncertain_fraction=uncertain_frac,
            toxicity_hard=toxicity_hard,
            acceptance_rate=acceptance_rate,
            high_quality_acceptance=hq_acc_rate,
            low_quality_acceptance=lq_acc_rate,
            total_interactions=n,
            accepted_count=n_acc,
            rejected_count=n_rej,
            high_quality_count=hq_count,
            low_quality_count=lq_count,
            total_welfare=total_welfare,
            total_social_surplus=total_social,
            avg_initiator_payoff=avg_pi_a_acc,
            avg_counterparty_payoff=avg_pi_b_acc,
            brier_score=brier,
            log_loss=logloss,
            calibration_error=cal_error,
            expected_calibration_error=ece,
            quality_variance=quality_var,
            quality_correlation=quality_correlation,
            baseline_harm=baseline_harm,
            selection_credit=selection_credit,
            selection_saturation=selection_saturation,
            payoff_variance_initiator=payoff_var_a,
            payoff_variance_counterparty=payoff_var_b,
        )

    def compare_soft_hard(self, interactions: List[SoftInteraction]) -> dict:
        """
        Compare soft vs hard metrics side by side.

        Useful for understanding how much information is lost
        by thresholding.

        Args:
            interactions: List of interactions

        Returns:
            Dictionary with comparison data
        """
        summary = self.summary(interactions)

        toxicity_diff = summary.toxicity_soft - summary.toxicity_hard

        return {
            "toxicity": {
                "soft": summary.toxicity_soft,
                "hard": summary.toxicity_hard,
                "difference": toxicity_diff,
                "interpretation": (
                    "Soft metric captures uncertainty better"
                    if abs(toxicity_diff) > 0.05
                    else "Metrics roughly agree"
                ),
            },
            "quality_filtering": {
                "quality_gap": summary.quality_gap,
                "spread": summary.spread,
                "adverse_selection": summary.quality_gap < 0,
            },
            "uncertainty": {
                "fraction_uncertain": summary.uncertain_fraction,
                "interpretation": (
                    "High uncertainty - labels may be unreliable"
                    if summary.uncertain_fraction > 0.3
                    else "Labels are relatively confident"
                ),
            },
        }

    def summary_with_incoherence(
        self,
        interactions: List[SoftInteraction],
        records_by_decision: Dict[str, List[DecisionRecord]],
        incoherence_metrics: IncoherenceMetrics,
    ) -> MetricsSummary:
        """
        Compute summary plus replay-based incoherence aggregates.

        Args:
            interactions: Interaction list for base soft/hard metrics
            records_by_decision: Replay records grouped by decision key
            incoherence_metrics: Metric calculator with benchmark policy
        """
        base = self.summary(interactions)
        if not records_by_decision:
            return base

        results = [
            incoherence_metrics.compute_for_decision(records)
            for records in records_by_decision.values()
            if records
        ]
        if not results:
            return base

        n = len(results)
        return replace(
            base,
            incoherence_disagreement=sum(r.disagreement for r in results) / n,
            incoherence_error=sum(r.error for r in results) / n,
            incoherence_index=sum(r.incoherence for r in results) / n,
            incoherence_n_decisions=n,
        )

    def summary_with_obfuscation(
        self,
        interactions: List[SoftInteraction],
        obfuscation_summary: ObfuscationSummary,
    ) -> MetricsSummary:
        """Compute summary plus obfuscation metrics.

        Follows the same pattern as ``summary_with_incoherence()``:
        computes the base summary then overlays obfuscation fields
        via ``dataclasses.replace()``.

        Args:
            interactions: Interaction list for base soft/hard metrics.
            obfuscation_summary: Pre-computed ObfuscationSummary.
        """
        base = self.summary(interactions)

        # Compute rationalization rate: fraction with rationalization metadata
        if interactions:
            rationalization_count = sum(
                1
                for i in interactions
                if i.metadata.get("rationalization")
            )
            rationalization_rate = rationalization_count / len(interactions)
        else:
            rationalization_rate = 0.0

        return replace(
            base,
            obfuscation_probe_evasion=obfuscation_summary.probe_evasion_rate,
            obfuscation_drift=obfuscation_summary.representation_drift,
            obfuscation_rationalization=rationalization_rate,
            obfuscation_detector_auc=obfuscation_summary.detector_auc,
            obfuscation_belief_shift=obfuscation_summary.belief_shift_index,
        )

    def format_report(
        self,
        interactions: List[SoftInteraction],
        verbose: bool = False,
    ) -> str:
        """
        Format a human-readable metrics report.

        Args:
            interactions: List of interactions
            verbose: Include additional details

        Returns:
            Formatted report string
        """
        summary = self.summary(interactions)

        lines = [
            "=" * 50,
            "METRICS REPORT",
            "=" * 50,
            "",
            f"Total interactions: {summary.total_interactions}",
            f"  Accepted: {summary.accepted_count} ({summary.acceptance_rate:.1%})",
            f"  Rejected: {summary.rejected_count}",
            "",
            "SOFT METRICS (probabilistic)",
            "-" * 30,
            f"  Toxicity (E[1-p|accepted]): {summary.toxicity_soft:.3f}",
            f"  Average quality (E[p]):     {summary.average_quality:.3f}",
            f"  Quality gap:                {summary.quality_gap:+.3f}",
            f"  Spread:                     {summary.spread:+.3f}",
            f"  Uncertain fraction:         {summary.uncertain_fraction:.1%}",
            "",
            "HARD METRICS (threshold-based)",
            "-" * 30,
            f"  Toxicity (P(p<0.5|acc)):    {summary.toxicity_hard:.3f}",
            f"  High quality acceptance:    {summary.high_quality_acceptance:.1%}",
            f"  Low quality acceptance:     {summary.low_quality_acceptance:.1%}",
            "",
            "SELECTION GEOMETRY",
            "-" * 30,
            f"  Quality correlation ρ(p,a): {summary.quality_correlation:+.3f}  [-1, +1]",
            f"  Baseline harm (1-E[p]):     {summary.baseline_harm:.3f}",
            f"  Selection credit (β·Q):     {summary.selection_credit:+.3f}",
            f"  C-S saturation |Q|/bound:   {summary.selection_saturation:.1%}",
        ]

        if verbose:
            lines.extend(
                [
                    "",
                    "PAYOFF ANALYSIS",
                    "-" * 30,
                    f"  Conditional loss (init):    {summary.conditional_loss_initiator:+.3f}",
                    f"  Conditional loss (counter): {summary.conditional_loss_counterparty:+.3f}",
                    "",
                    "WELFARE",
                    "-" * 30,
                    f"  Total welfare:              {summary.total_welfare:.2f}",
                    f"  Social surplus:             {summary.total_social_surplus:.2f}",
                    f"  Avg initiator payoff:       {summary.avg_initiator_payoff:.3f}",
                    f"  Avg counterparty payoff:    {summary.avg_counterparty_payoff:.3f}",
                ]
            )

            # Interpretation
            if summary.quality_gap < 0:
                lines.extend(
                    [
                        "",
                        "WARNING: Negative quality gap indicates adverse selection",
                        "  (rejected interactions have higher quality than accepted)",
                    ]
                )

        lines.append("=" * 50)

        return "\n".join(lines)

    def to_markdown(
        self,
        interactions: List[SoftInteraction],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        decimals: int = 4,
    ) -> str:
        """Render the metrics summary as a GitHub-flavored markdown document.

        The output is suitable for pasting into GitHub issues, PR comments, or
        research notes. The structure is::

            # Metrics report

            **seed:** ...   (optional run metadata header, one bullet per key)

            ## Soft metrics (probabilistic)
            | Metric | Value |
            ...

            ## Hard metrics (threshold-based)
            | Metric | Value |
            ...

            ## Counts
            | Metric | Value |
            ...

        Args:
            interactions: Interactions to summarise (same shape as
                ``format_report``).
            metadata: Optional mapping of run metadata (e.g.
                ``{"seed": 42, "scenario": "baseline.yaml", "epochs": 10}``).
                Rendered as bold-key bullet rows at the top of the document.
            decimals: Number of decimal places for floating-point cells
                (default: 4). Integers are rendered without rounding.

        Returns:
            A markdown document as a single string.
        """
        summary = self.summary(interactions)

        def escape(text: str) -> str:
            """Escape pipes so a string survives as a single GFM table cell."""
            return text.replace("|", "\\|")

        def fmt(value: Any) -> str:
            if value is None:
                return "n/a"
            if isinstance(value, bool):
                return str(value)
            if isinstance(value, float):
                return f"{round(value, decimals):.{decimals}f}"
            return escape(str(value))

        def render_table(rows: List[tuple[str, Any]]) -> List[str]:
            lines = ["| Metric | Value |", "| --- | ---: |"]
            for label, value in rows:
                lines.append(f"| {escape(label)} | {fmt(value)} |")
            return lines

        lines: List[str] = ["# Metrics report", ""]

        if metadata:
            for key, value in metadata.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        lines.append("## Soft metrics (probabilistic)")
        lines.append("")
        lines.extend(
            render_table(
                [
                    ("Toxicity (E[1-p | accepted])", summary.toxicity_soft),
                    ("Average quality (E[p])", summary.average_quality),
                    ("Quality gap", summary.quality_gap),
                    ("Spread", summary.spread),
                    ("Uncertain fraction", summary.uncertain_fraction),
                    (
                        "Conditional loss (initiator)",
                        summary.conditional_loss_initiator,
                    ),
                    (
                        "Conditional loss (counterparty)",
                        summary.conditional_loss_counterparty,
                    ),
                ]
            )
        )
        lines.append("")

        lines.append("## Selection geometry")
        lines.append("")
        lines.extend(
            render_table(
                [
                    ("Quality correlation ρ(p,a)", summary.quality_correlation),
                    ("Baseline harm (1 − E[p])", summary.baseline_harm),
                    ("Selection credit (β · Q)", summary.selection_credit),
                    (
                        "C-S saturation (|Q| / σ_p/√(αβ))",
                        summary.selection_saturation,
                    ),
                ]
            )
        )
        lines.append("")

        lines.append("## Hard metrics (threshold-based)")
        lines.append("")
        lines.extend(
            render_table(
                [
                    (
                        "Hard toxicity (low-quality among accepted)",
                        summary.toxicity_hard,
                    ),
                    ("Acceptance rate", summary.acceptance_rate),
                    (
                        "High-quality acceptance rate",
                        summary.high_quality_acceptance,
                    ),
                    (
                        "Low-quality acceptance rate",
                        summary.low_quality_acceptance,
                    ),
                ]
            )
        )
        lines.append("")

        lines.append("## Counts")
        lines.append("")
        lines.extend(
            render_table(
                [
                    ("Total interactions", summary.total_interactions),
                    ("Accepted", summary.accepted_count),
                    ("Rejected", summary.rejected_count),
                    ("High quality", summary.high_quality_count),
                    ("Low quality", summary.low_quality_count),
                ]
            )
        )
        lines.append("")

        lines.append("## Welfare")
        lines.append("")
        lines.extend(
            render_table(
                [
                    ("Total welfare", summary.total_welfare),
                    ("Total social surplus", summary.total_social_surplus),
                    (
                        "Avg initiator payoff",
                        summary.avg_initiator_payoff,
                    ),
                    (
                        "Avg counterparty payoff",
                        summary.avg_counterparty_payoff,
                    ),
                ]
            )
        )
        lines.append("")

        if any(
            v is not None
            for v in (
                summary.brier_score,
                summary.log_loss,
                summary.calibration_error,
                summary.expected_calibration_error,
            )
        ):
            lines.append("## Calibration")
            lines.append("")
            lines.extend(
                render_table(
                    [
                        ("Brier score", summary.brier_score),
                        ("Log loss", summary.log_loss),
                        ("Calibration error", summary.calibration_error),
                        (
                            "Expected calibration error",
                            summary.expected_calibration_error,
                        ),
                    ]
                )
            )
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"
