"""Soft metrics for interaction quality analysis."""

import math
from typing import Callable, List, Optional, Tuple

from swarm.core.payoff import SoftPayoffEngine
from swarm.models.interaction import SoftInteraction


class SoftMetrics:
    """
    Computes soft (probabilistic) metrics for interaction quality.

    Key metrics:
    - Toxicity rate: E[1-p | accepted]
    - Conditional loss: E[π_a | accepted] - E[π_a]
    - Spread: Quality filtering effectiveness
    - Quality gap: E[p | accepted] - E[p | rejected]
    """

    def __init__(self, payoff_engine: Optional[SoftPayoffEngine] = None):
        """
        Initialize metrics calculator.

        Args:
            payoff_engine: Engine for payoff calculations (default: SoftPayoffEngine())
        """
        self.payoff_engine = payoff_engine or SoftPayoffEngine()

    def toxicity_rate(self, interactions: List[SoftInteraction]) -> float:
        """
        Compute toxicity rate: E[1-p | accepted]

        This measures the expected fraction of harmful outcomes
        among accepted interactions.

        Args:
            interactions: List of interactions

        Returns:
            Toxicity rate in [0, 1], or 0.0 if no accepted interactions
        """
        accepted = [i for i in interactions if i.accepted]
        if not accepted:
            return 0.0

        return sum(1 - i.p for i in accepted) / len(accepted)

    def toxicity_rate_all(self, interactions: List[SoftInteraction]) -> float:
        """
        Compute unconditional toxicity rate: E[1-p]

        Args:
            interactions: List of interactions

        Returns:
            Toxicity rate in [0, 1]
        """
        if not interactions:
            return 0.0

        return sum(1 - i.p for i in interactions) / len(interactions)

    def _conditional_loss(
        self,
        interactions: List[SoftInteraction],
        payoff_fn: Callable[[SoftInteraction], float],
    ) -> float:
        """
        Compute E[payoff | accepted] - E[payoff] for the given payoff function.

        Negative values indicate adverse selection (accepted interactions
        are worse than average for that party).
        """
        if not interactions:
            return 0.0

        accepted = [i for i in interactions if i.accepted]
        if not accepted:
            return 0.0

        avg_accepted = sum(payoff_fn(i) for i in accepted) / len(accepted)
        avg_all = sum(payoff_fn(i) for i in interactions) / len(interactions)

        return avg_accepted - avg_all

    def conditional_loss_initiator(self, interactions: List[SoftInteraction]) -> float:
        """
        Compute conditional loss for initiator: E[π_a | accepted] - E[π_a]

        Args:
            interactions: List of interactions

        Returns:
            Conditional loss (negative = adverse selection)
        """
        return self._conditional_loss(interactions, self.payoff_engine.payoff_initiator)

    def conditional_loss_counterparty(
        self, interactions: List[SoftInteraction]
    ) -> float:
        """
        Compute conditional loss for counterparty: E[π_b | accepted] - E[π_b]

        Args:
            interactions: List of interactions

        Returns:
            Conditional loss (negative = adverse selection)
        """
        return self._conditional_loss(
            interactions, self.payoff_engine.payoff_counterparty
        )

    def spread(self, interactions: List[SoftInteraction]) -> float:
        """
        Compute spread: (s_plus + s_minus) * (E[p] - E[p | accepted])

        Positive spread indicates the market is filtering out high-quality
        interactions (adverse selection in the quality sense).

        Args:
            interactions: List of interactions

        Returns:
            Spread value
        """
        if not interactions:
            return 0.0

        accepted = [i for i in interactions if i.accepted]
        if not accepted:
            return 0.0

        avg_p_all = self.average_quality(interactions)
        avg_p_accepted = self.average_quality(interactions, accepted_only=True)

        scale = self.payoff_engine.config.s_plus + self.payoff_engine.config.s_minus

        return scale * (avg_p_all - avg_p_accepted)

    def quality_gap(self, interactions: List[SoftInteraction]) -> float:
        """
        Compute quality gap: E[p | accepted] - E[p | rejected]

        Negative quality gap indicates adverse selection (accepted
        interactions have lower quality than rejected ones).

        Args:
            interactions: List of interactions

        Returns:
            Quality gap (negative = adverse selection)
        """
        accepted = [i for i in interactions if i.accepted]
        rejected = [i for i in interactions if not i.accepted]

        if not accepted or not rejected:
            return 0.0

        avg_p_accepted = sum(i.p for i in accepted) / len(accepted)
        avg_p_rejected = sum(i.p for i in rejected) / len(rejected)

        return avg_p_accepted - avg_p_rejected

    def quality_correlation(self, interactions: List[SoftInteraction]) -> float:
        """
        Pearson correlation ρ(p, a) ∈ [-1, +1] between soft labels and
        acceptance — the projection-normalized quality gap.

        Derivation: Q = E[p|a=1] - E[p|a=0] = Cov(p,a)/Var(a). Cauchy-Schwarz
        gives |Q| ≤ σ_p / √(αβ), so ρ(p,a) = Q · √(αβ) / σ_p ∈ [-1, +1].

        Returns:
            ρ ∈ [-1, +1]; 0.0 when Var(p) or Var(a) is zero
            (no signal to correlate).
        """
        n = len(interactions)
        if n < 2:
            return 0.0

        n_acc = sum(1 for i in interactions if i.accepted)
        if n_acc == 0 or n_acc == n:
            return 0.0  # Var(a) = 0

        alpha = n_acc / n
        mean_p = self.average_quality(interactions)
        var_p = self.quality_variance(interactions)
        if var_p == 0.0:
            return 0.0  # Var(p) = 0

        cov = sum((i.p - mean_p) * ((1.0 if i.accepted else 0.0) - alpha)
                  for i in interactions) / n
        return cov / math.sqrt(var_p * alpha * (1 - alpha))

    def toxicity_decomposition(self, interactions: List[SoftInteraction]) -> dict:
        """
        Decompose toxicity into baseline harm minus selection credit:

            T = (1 - E[p]) - β · Q

        where β = P(rejected) and Q = quality gap. The baseline term
        is a population property; the credit term is what governance
        can move by improving selection.

        Returns:
            Dict with keys baseline_harm, selection_credit, toxicity,
            and reconstruction_error (|T_direct - (baseline - credit)|,
            should be ~0 modulo float).
        """
        if not interactions:
            return {
                "baseline_harm": 0.0,
                "selection_credit": 0.0,
                "toxicity": 0.0,
                "reconstruction_error": 0.0,
            }

        n = len(interactions)
        n_acc = sum(1 for i in interactions if i.accepted)
        beta = (n - n_acc) / n
        baseline = 1.0 - self.average_quality(interactions)
        q = self.quality_gap(interactions)
        credit = beta * q
        tox_direct = self.toxicity_rate(interactions)
        return {
            "baseline_harm": baseline,
            "selection_credit": credit,
            "toxicity": tox_direct,
            "reconstruction_error": abs(tox_direct - (baseline - credit)),
        }

    def selection_saturation(self, interactions: List[SoftInteraction]) -> float:
        """
        Fraction of the Cauchy-Schwarz bound on |Q| that is realized:

            saturation = |Q| / (σ_p / √(αβ))  ∈ [0, 1]

        Near 1 ⇒ the selection mechanism is already extracting all the
        sorting signal the proxy carries; further toxicity reduction
        requires a better proxy (larger σ_p), not a better policy.
        Returns 0.0 when the bound is undefined (σ_p = 0 or α∈{0,1}).
        """
        n = len(interactions)
        if n < 2:
            return 0.0
        n_acc = sum(1 for i in interactions if i.accepted)
        if n_acc == 0 or n_acc == n:
            return 0.0
        sigma_p = self.quality_std(interactions)
        if sigma_p == 0.0:
            return 0.0
        alpha = n_acc / n
        bound = sigma_p / math.sqrt(alpha * (1 - alpha))
        return abs(self.quality_gap(interactions)) / bound

    def participation_by_quality(
        self,
        interactions: List[SoftInteraction],
        threshold: float = 0.5,
    ) -> dict:
        """
        Compute acceptance rates for high/low quality interactions.

        Args:
            interactions: List of interactions
            threshold: Quality threshold (default 0.5)

        Returns:
            Dictionary with acceptance rates:
            - high_quality_acceptance: P(accepted | p >= threshold)
            - low_quality_acceptance: P(accepted | p < threshold)
            - high_quality_count: Number of high quality interactions
            - low_quality_count: Number of low quality interactions
        """
        high_quality = [i for i in interactions if i.p >= threshold]
        low_quality = [i for i in interactions if i.p < threshold]

        high_accepted = sum(1 for i in high_quality if i.accepted)
        low_accepted = sum(1 for i in low_quality if i.accepted)

        return {
            "high_quality_acceptance": (
                high_accepted / len(high_quality) if high_quality else 0.0
            ),
            "low_quality_acceptance": (
                low_accepted / len(low_quality) if low_quality else 0.0
            ),
            "high_quality_count": len(high_quality),
            "low_quality_count": len(low_quality),
        }

    def flag_uncertain(
        self,
        interactions: List[SoftInteraction],
        band: float = 0.2,
    ) -> List[SoftInteraction]:
        """
        Flag interactions with uncertain labels (p near 0.5).

        Args:
            interactions: List of interactions
            band: Width of uncertainty band around 0.5

        Returns:
            List of uncertain interactions
        """
        return [i for i in interactions if i.is_uncertain(band)]

    def uncertain_fraction(
        self,
        interactions: List[SoftInteraction],
        band: float = 0.2,
    ) -> float:
        """
        Compute fraction of interactions with uncertain labels.

        Args:
            interactions: List of interactions
            band: Width of uncertainty band around 0.5

        Returns:
            Fraction in [0, 1]
        """
        if not interactions:
            return 0.0

        uncertain = self.flag_uncertain(interactions, band)
        return len(uncertain) / len(interactions)

    def average_quality(
        self,
        interactions: List[SoftInteraction],
        accepted_only: bool = False,
    ) -> float:
        """
        Compute average quality E[p].

        Args:
            interactions: List of interactions
            accepted_only: If True, only consider accepted interactions

        Returns:
            Average p value
        """
        if accepted_only:
            interactions = [i for i in interactions if i.accepted]

        if not interactions:
            return 0.0

        return sum(i.p for i in interactions) / len(interactions)

    def quality_distribution(
        self,
        interactions: List[SoftInteraction],
        bins: int = 10,
    ) -> List[Tuple[float, float, int]]:
        """
        Compute quality distribution histogram.

        Args:
            interactions: List of interactions
            bins: Number of bins

        Returns:
            List of (bin_start, bin_end, count) tuples
        """
        if not interactions:
            return []

        bin_width = 1.0 / bins
        result = []

        for i in range(bins):
            bin_start = i * bin_width
            bin_end = (i + 1) * bin_width

            count = sum(
                1
                for interaction in interactions
                if bin_start <= interaction.p < bin_end
                or (i == bins - 1 and interaction.p == 1.0)
            )

            result.append((bin_start, bin_end, count))

        return result

    def welfare_metrics(self, interactions: List[SoftInteraction]) -> dict:
        """
        Compute aggregate welfare metrics.

        Args:
            interactions: List of interactions

        Returns:
            Dictionary with welfare metrics
        """
        if not interactions:
            return {
                "total_welfare": 0.0,
                "net_social_welfare": 0.0,
                "avg_initiator_payoff": 0.0,
                "avg_counterparty_payoff": 0.0,
            }

        accepted = [i for i in interactions if i.accepted]

        total_welfare = sum(self.payoff_engine.total_welfare(i) for i in accepted)
        # Social surplus charges the full externality:
        # S_soft - E_soft = p*s+ - (1-p)*(s- + h)
        total_social = sum(self.payoff_engine.social_surplus(i) for i in accepted)
        avg_init = (
            sum(self.payoff_engine.payoff_initiator(i) for i in accepted)
            / len(accepted)
            if accepted
            else 0.0
        )
        avg_counter = (
            sum(self.payoff_engine.payoff_counterparty(i) for i in accepted)
            / len(accepted)
            if accepted
            else 0.0
        )

        return {
            "total_welfare": total_welfare,
            "net_social_welfare": total_social,
            "avg_initiator_payoff": avg_init,
            "avg_counterparty_payoff": avg_counter,
        }

    # =========================================================================
    # Calibration Metrics
    # =========================================================================

    def calibration_error(self, interactions: List[SoftInteraction]) -> Optional[float]:
        """
        Compute calibration error: E[p] - empirical_positive_rate.

        Requires ground_truth to be set on interactions.
        A well-calibrated model has calibration error near 0.

        Args:
            interactions: List of interactions with ground_truth set

        Returns:
            Calibration error, or None if no ground truth available
        """
        with_truth = [i for i in interactions if i.ground_truth is not None]
        if not with_truth:
            return None

        # E[p]
        avg_p = sum(i.p for i in with_truth) / len(with_truth)

        # Empirical positive rate: fraction where ground_truth = +1
        # ground_truth is +1 or -1, so we convert to 0/1
        positive_count = sum(1 for i in with_truth if i.ground_truth == 1)
        empirical_rate = positive_count / len(with_truth)

        return avg_p - empirical_rate

    def brier_score(self, interactions: List[SoftInteraction]) -> Optional[float]:
        """
        Compute Brier score: E[(p - v)^2] where v = (ground_truth + 1) / 2.

        The Brier score is a proper scoring rule for probabilistic predictions.
        - 0 is perfect prediction
        - 0.25 is equivalent to always predicting p=0.5

        Args:
            interactions: List of interactions with ground_truth set

        Returns:
            Brier score in [0, 1], or None if no ground truth available
        """
        with_truth = [i for i in interactions if i.ground_truth is not None]
        if not with_truth:
            return None

        total = 0.0
        for i in with_truth:
            # Convert ground_truth from {-1, +1} to {0, 1}
            assert i.ground_truth is not None  # filtered above
            v = (i.ground_truth + 1) / 2
            total += (i.p - v) ** 2

        return total / len(with_truth)

    def expected_calibration_error(
        self,
        interactions: List[SoftInteraction],
        bins: int = 10,
    ) -> Optional[float]:
        """
        Compute Expected Calibration Error (ECE).

        ECE is the weighted average of |E[p|bin] - accuracy(bin)| across bins.
        A perfectly calibrated model has ECE = 0.

        Args:
            interactions: List of interactions with ground_truth set
            bins: Number of probability bins

        Returns:
            ECE value, or None if no ground truth available
        """
        curve = self.calibration_curve(interactions, bins)
        if not curve:
            return None

        total_count = sum(count for _, _, count in curve)
        if total_count == 0:
            return None

        ece = 0.0
        for mean_predicted, fraction_positive, count in curve:
            if count > 0:
                ece += (count / total_count) * abs(mean_predicted - fraction_positive)

        return ece

    def calibration_curve(
        self,
        interactions: List[SoftInteraction],
        bins: int = 10,
    ) -> List[Tuple[float, float, int]]:
        """
        Compute calibration curve data.

        For each bin of predicted probabilities, compute the fraction of
        actually positive outcomes.

        Args:
            interactions: List of interactions with ground_truth set
            bins: Number of probability bins

        Returns:
            List of (mean_predicted, fraction_positive, count) per bin.
            Returns empty list if no ground truth available.
        """
        with_truth = [i for i in interactions if i.ground_truth is not None]
        if not with_truth:
            return []

        bin_width = 1.0 / bins
        result = []

        for b in range(bins):
            bin_start = b * bin_width
            bin_end = (b + 1) * bin_width

            # Get interactions in this bin
            in_bin = [
                i
                for i in with_truth
                if bin_start <= i.p < bin_end or (b == bins - 1 and i.p == 1.0)
            ]

            if not in_bin:
                # Empty bin - use midpoint as predicted, 0.0 as accuracy
                result.append((bin_start + bin_width / 2, 0.0, 0))
            else:
                mean_predicted = sum(i.p for i in in_bin) / len(in_bin)
                positive_count = sum(1 for i in in_bin if i.ground_truth == 1)
                fraction_positive = positive_count / len(in_bin)
                result.append((mean_predicted, fraction_positive, len(in_bin)))

        return result

    # =========================================================================
    # Information-Theoretic Metrics
    # =========================================================================

    def log_loss(
        self,
        interactions: List[SoftInteraction],
        eps: float = 1e-15,
    ) -> Optional[float]:
        """
        Compute log loss (cross-entropy): -E[v*log(p) + (1-v)*log(1-p)].

        Args:
            interactions: List of interactions with ground_truth set
            eps: Small value to avoid log(0)

        Returns:
            Log loss (lower is better), or None if no ground truth available
        """
        with_truth = [i for i in interactions if i.ground_truth is not None]
        if not with_truth:
            return None

        total = 0.0
        for i in with_truth:
            # Convert ground_truth from {-1, +1} to {0, 1}
            assert i.ground_truth is not None  # filtered above
            v = (i.ground_truth + 1) / 2
            # Clamp p to avoid log(0)
            p_clamped = max(eps, min(1 - eps, i.p))

            total -= v * math.log(p_clamped) + (1 - v) * math.log(1 - p_clamped)

        return total / len(with_truth)

    def discrimination_auc(
        self, interactions: List[SoftInteraction]
    ) -> Optional[float]:
        """
        Compute Area Under ROC Curve (AUC) for discrimination.

        AUC measures the model's ability to rank positive cases higher
        than negative cases.
        - AUC = 0.5: random guessing
        - AUC = 1.0: perfect discrimination

        Args:
            interactions: List of interactions with ground_truth set

        Returns:
            AUC value in [0, 1], or None if insufficient data
        """
        with_truth = [i for i in interactions if i.ground_truth is not None]
        if not with_truth:
            return None

        positives = [i for i in with_truth if i.ground_truth == 1]
        negatives = [i for i in with_truth if i.ground_truth == -1]

        if not positives or not negatives:
            return None

        # Wilcoxon-Mann-Whitney statistic
        concordant: float = 0
        total_pairs = len(positives) * len(negatives)

        for pos in positives:
            for neg in negatives:
                if pos.p > neg.p:
                    concordant += 1
                elif pos.p == neg.p:
                    concordant += 0.5

        return concordant / total_pairs

    # =========================================================================
    # Variance / Uncertainty Metrics
    # =========================================================================

    def synthesis_fraction(self, interactions: List[SoftInteraction]) -> float:
        """
        Compute synthesis fraction: ratio of cross-agent synthesis interactions.

        An interaction counts as *synthesis* only if its ``causal_parents``
        reference interactions from **distinct initiators** — consuming
        two artifacts from the same agent is reuse, not cross-agent
        synthesis.  Higher synthesis fraction signals richer multi-agent
        coordination.

        Inspired by Wang et al. (2026) Table 1 metric.

        Args:
            interactions: List of interactions

        Returns:
            Fraction in [0, 1], or 0.0 if no interactions
        """
        if not interactions:
            return 0.0

        # Index interactions by ID for parent lookups
        ix_by_id = {i.interaction_id: i for i in interactions}

        synthesis_count = 0
        for i in interactions:
            if len(i.causal_parents) < 2:
                continue
            # Count distinct initiators among parents
            parent_initiators = {
                ix_by_id[pid].initiator
                for pid in i.causal_parents
                if pid in ix_by_id
            }
            if len(parent_initiators) > 1:
                synthesis_count += 1

        return synthesis_count / len(interactions)

    def synthesis_depth(self, interactions: List[SoftInteraction]) -> float:
        """
        Compute average arity (parent count) of synthesis interactions.

        Measures the average number of artifacts consumed by multi-parent
        interactions (those with ``len(causal_parents) > 1``).  Despite
        the name, this is arity (fan-in) not DAG depth (longest path).

        Args:
            interactions: List of interactions

        Returns:
            Average parent count for synthesis interactions, or 0.0
        """
        synthesis = [i for i in interactions if len(i.causal_parents) > 1]
        if not synthesis:
            return 0.0
        return sum(len(i.causal_parents) for i in synthesis) / len(synthesis)

    @staticmethod
    def _variance(values: List[float]) -> float:
        """Population variance of values; 0.0 when fewer than 2 values."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    def quality_variance(
        self,
        interactions: List[SoftInteraction],
        accepted_only: bool = False,
    ) -> float:
        """
        Compute variance of quality: Var[p].

        Args:
            interactions: List of interactions
            accepted_only: If True, only consider accepted interactions

        Returns:
            Variance of p
        """
        if accepted_only:
            interactions = [i for i in interactions if i.accepted]

        return self._variance([i.p for i in interactions])

    def quality_std(
        self,
        interactions: List[SoftInteraction],
        accepted_only: bool = False,
    ) -> float:
        """
        Compute standard deviation of quality: Std[p].

        Args:
            interactions: List of interactions
            accepted_only: If True, only consider accepted interactions

        Returns:
            Standard deviation of p
        """
        return math.sqrt(self.quality_variance(interactions, accepted_only))

    def payoff_variance_initiator(self, interactions: List[SoftInteraction]) -> float:
        """
        Compute variance of initiator payoffs: Var[π_a].

        Measures risk/dispersion in initiator outcomes.

        Args:
            interactions: List of interactions

        Returns:
            Variance of initiator payoffs
        """
        return self._variance(
            [self.payoff_engine.payoff_initiator(i) for i in interactions]
        )

    def payoff_variance_counterparty(
        self, interactions: List[SoftInteraction]
    ) -> float:
        """
        Compute variance of counterparty payoffs: Var[π_b].

        Measures risk/dispersion in counterparty outcomes.

        Args:
            interactions: List of interactions

        Returns:
            Variance of counterparty payoffs
        """
        return self._variance(
            [self.payoff_engine.payoff_counterparty(i) for i in interactions]
        )

    def coefficient_of_variation(self, interactions: List[SoftInteraction]) -> dict:
        """
        Compute coefficient of variation (CV = std/mean) for key metrics.

        CV is a standardized measure of dispersion. Higher CV indicates
        more variability relative to the mean. Uses an epsilon floor on
        |mean| to avoid division by zero while preserving large CV when
        the mean is near zero.

        Args:
            interactions: List of interactions

        Returns:
            Dictionary with CV for p, π_a, and π_b
        """
        if not interactions:
            return {
                "cv_p": 0.0,
                "cv_payoff_initiator": 0.0,
                "cv_payoff_counterparty": 0.0,
            }

        def stable_cv(std: float, mean: float) -> float:
            """Compute CV with epsilon-stabilized denominator."""
            if std == 0:
                return 0.0
            denom = abs(mean)
            eps = 1e-8 * (denom + std)
            return std / max(denom, eps)

        # CV for p
        mean_p = self.average_quality(interactions)
        std_p = self.quality_std(interactions)
        cv_p = stable_cv(std_p, mean_p)

        # CV for initiator payoffs
        payoffs_a = [self.payoff_engine.payoff_initiator(i) for i in interactions]
        mean_a = sum(payoffs_a) / len(payoffs_a)
        std_a = math.sqrt(self.payoff_variance_initiator(interactions))
        cv_a = stable_cv(std_a, mean_a)

        # CV for counterparty payoffs
        payoffs_b = [self.payoff_engine.payoff_counterparty(i) for i in interactions]
        mean_b = sum(payoffs_b) / len(payoffs_b)
        std_b = math.sqrt(self.payoff_variance_counterparty(interactions))
        cv_b = stable_cv(std_b, mean_b)

        return {
            "cv_p": cv_p,
            "cv_payoff_initiator": cv_a,
            "cv_payoff_counterparty": cv_b,
        }
