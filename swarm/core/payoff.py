"""Soft payoff engine implementing payoffs with probabilistic labels."""

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, model_validator

from swarm.models.interaction import SoftInteraction


class PayoffConfig(BaseModel):
    """
    Configuration for the soft payoff engine.

    Default values from specification:
    - s_plus = 2.0: Surplus when v = +1 (beneficial)
    - s_minus = 1.0: Loss when v = -1 (harmful)
    - h = 2.0: Harm magnitude for externality
    - theta = 0.5: Surplus split (initiator share)
    - rho_a = 0.0: Externality internalization (initiator)
    - rho_b = 0.0: Externality internalization (counterparty)
    - w_rep = 1.0: Reputation weight in payoff
    - sigmoid_k = 2.0: Calibration sharpness
    """

    s_plus: float = 2.0
    s_minus: float = 1.0
    h: float = 2.0
    theta: float = 0.5
    rho_a: float = 0.0
    rho_b: float = 0.0
    w_rep: float = 1.0
    sigmoid_k: float = 2.0

    @model_validator(mode="after")
    def _run_validation(self) -> "PayoffConfig":
        self._check_values()
        return self

    def _check_values(self) -> None:
        """Validate configuration parameters."""
        if self.s_plus < 0:
            raise ValueError("s_plus must be non-negative")
        if self.s_minus < 0:
            raise ValueError("s_minus must be non-negative")
        if self.h < 0:
            raise ValueError("h must be non-negative")
        if not 0 <= self.theta <= 1:
            raise ValueError("theta must be in [0, 1]")
        if not 0 <= self.rho_a <= 1:
            raise ValueError("rho_a must be in [0, 1]")
        if not 0 <= self.rho_b <= 1:
            raise ValueError("rho_b must be in [0, 1]")
        if not 0 <= self.w_rep <= 100:
            raise ValueError("w_rep must be in [0, 100]")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.model_dump()


@dataclass
class PayoffBreakdown:
    """Detailed breakdown of payoff computation."""

    # Soft expectations
    expected_surplus: float
    expected_harm: float

    # Payoff components
    surplus_share: float
    transfer: float
    governance_cost: float
    externality_cost: float
    reputation_bonus: float

    # Final payoff
    total: float


class SoftPayoffEngine:
    """
    Computes payoffs using soft (probabilistic) labels.

    Payoff equations:
        π_a = θ * S_soft - τ - c_a - ρ_a * E_soft + w_rep * r_a
        π_b = (1-θ) * S_soft + τ - c_b - ρ_b * E_soft + w_rep * r_b

    Where:
        S_soft = p * s_plus - (1-p) * s_minus  (expected surplus)
        E_soft = (1-p) * h                      (expected harm)
    """

    def __init__(self, config: Optional[PayoffConfig] = None):
        """
        Initialize payoff engine.

        Args:
            config: Payoff configuration (default: PayoffConfig())
        """
        self.config = PayoffConfig() if config is None else config
        # Pydantic auto-validates

    def expected_surplus(self, p: float) -> float:
        """
        Compute expected surplus: p * s_plus - (1-p) * s_minus

        This is the expected value created/destroyed by the interaction.

        Args:
            p: P(v = +1) probability of beneficial outcome

        Returns:
            Expected surplus (can be negative)
        """
        return p * self.config.s_plus - (1 - p) * self.config.s_minus

    def expected_harm(self, p: float) -> float:
        """
        Compute expected harm externality: (1-p) * h

        This is the expected harm to the ecosystem from a bad outcome.

        Args:
            p: P(v = +1)

        Returns:
            Expected harm (always non-negative)
        """
        return (1 - p) * self.config.h

    def payoff_initiator(self, interaction: SoftInteraction) -> float:
        """
        Compute initiator's payoff.

        π_a = θ * S_soft - τ - c_a - ρ_a * E_soft + w_rep * r_a

        Args:
            interaction: The soft interaction

        Returns:
            Initiator's payoff
        """
        S_soft = self.expected_surplus(interaction.p)
        E_soft = self.expected_harm(interaction.p)

        result: float = (
            self.config.theta * S_soft
            - interaction.tau
            - interaction.c_a
            - self.config.rho_a * E_soft
            + self.config.w_rep * interaction.r_a
        )
        return result

    def payoff_counterparty(self, interaction: SoftInteraction) -> float:
        """
        Compute counterparty's payoff.

        π_b = (1-θ) * S_soft + τ - c_b - ρ_b * E_soft + w_rep * r_b

        Args:
            interaction: The soft interaction

        Returns:
            Counterparty's payoff
        """
        S_soft = self.expected_surplus(interaction.p)
        E_soft = self.expected_harm(interaction.p)

        result: float = (
            (1 - self.config.theta) * S_soft
            + interaction.tau
            - interaction.c_b
            - self.config.rho_b * E_soft
            + self.config.w_rep * interaction.r_b
        )
        return result

    def payoff_breakdown_initiator(
        self, interaction: SoftInteraction
    ) -> PayoffBreakdown:
        """
        Compute detailed payoff breakdown for initiator.

        Args:
            interaction: The soft interaction

        Returns:
            PayoffBreakdown with all components
        """
        S_soft = self.expected_surplus(interaction.p)
        E_soft = self.expected_harm(interaction.p)

        surplus_share = self.config.theta * S_soft
        externality_cost = self.config.rho_a * E_soft
        reputation_bonus = self.config.w_rep * interaction.r_a

        total = (
            surplus_share
            - interaction.tau
            - interaction.c_a
            - externality_cost
            + reputation_bonus
        )

        return PayoffBreakdown(
            expected_surplus=S_soft,
            expected_harm=E_soft,
            surplus_share=surplus_share,
            transfer=-interaction.tau,  # Negative because initiator pays
            governance_cost=interaction.c_a,
            externality_cost=externality_cost,
            reputation_bonus=reputation_bonus,
            total=total,
        )

    def payoff_breakdown_counterparty(
        self, interaction: SoftInteraction
    ) -> PayoffBreakdown:
        """
        Compute detailed payoff breakdown for counterparty.

        Args:
            interaction: The soft interaction

        Returns:
            PayoffBreakdown with all components
        """
        S_soft = self.expected_surplus(interaction.p)
        E_soft = self.expected_harm(interaction.p)

        surplus_share = (1 - self.config.theta) * S_soft
        externality_cost = self.config.rho_b * E_soft
        reputation_bonus = self.config.w_rep * interaction.r_b

        total = (
            surplus_share
            + interaction.tau
            - interaction.c_b
            - externality_cost
            + reputation_bonus
        )

        return PayoffBreakdown(
            expected_surplus=S_soft,
            expected_harm=E_soft,
            surplus_share=surplus_share,
            transfer=interaction.tau,  # Positive because counterparty receives
            governance_cost=interaction.c_b,
            externality_cost=externality_cost,
            reputation_bonus=reputation_bonus,
            total=total,
        )

    def payoffs_both(self, interaction: SoftInteraction) -> tuple[float, float]:
        """
        Compute both initiator and counterparty payoffs in a single call.

        Avoids redundant expected_surplus / expected_harm computation
        compared to calling payoff_initiator + payoff_counterparty separately.

        Args:
            interaction: The soft interaction

        Returns:
            (pi_a, pi_b): Initiator and counterparty payoffs
        """
        S_soft = self.expected_surplus(interaction.p)
        E_soft = self.expected_harm(interaction.p)

        pi_a = (
            self.config.theta * S_soft
            - interaction.tau
            - interaction.c_a
            - self.config.rho_a * E_soft
            + self.config.w_rep * interaction.r_a
        )
        pi_b = (
            (1 - self.config.theta) * S_soft
            + interaction.tau
            - interaction.c_b
            - self.config.rho_b * E_soft
            + self.config.w_rep * interaction.r_b
        )
        return pi_a, pi_b

    def total_welfare(self, interaction: SoftInteraction) -> float:
        """
        Compute total private welfare: W = π_a + π_b.

        Transfers (τ) cancel between the parties. Externality harm enters
        only through the internalized shares (ρ_a, ρ_b); the uninternalized
        remainder is excluded here — see social_surplus for the version
        that charges the full externality.

        Args:
            interaction: The soft interaction

        Returns:
            Total welfare
        """
        return self.payoff_initiator(interaction) + self.payoff_counterparty(
            interaction
        )

    def social_surplus(self, interaction: SoftInteraction) -> float:
        """
        Compute social surplus including externality.

        Social surplus = S_soft - E_soft
                      = p * s_plus - (1-p) * s_minus - (1-p) * h
                      = p * s_plus - (1-p) * (s_minus + h)

        Args:
            interaction: The soft interaction

        Returns:
            Social surplus
        """
        S_soft = self.expected_surplus(interaction.p)
        E_soft = self.expected_harm(interaction.p)
        return S_soft - E_soft

    def break_even_p(self) -> float:
        """
        Compute the probability threshold where expected surplus = 0.

        p * s_plus = (1-p) * s_minus
        p = s_minus / (s_plus + s_minus)

        Returns:
            Break-even probability
        """
        denominator = self.config.s_plus + self.config.s_minus
        if denominator == 0:
            return 0.5
        return self.config.s_minus / denominator

    def social_break_even_p(self) -> float:
        """
        Compute probability where social surplus = 0.

        p * s_plus = (1-p) * (s_minus + h)
        p = (s_minus + h) / (s_plus + s_minus + h)

        Returns:
            Social break-even probability
        """
        numerator = self.config.s_minus + self.config.h
        denominator = self.config.s_plus + self.config.s_minus + self.config.h
        if denominator == 0:
            return 0.5
        return numerator / denominator
