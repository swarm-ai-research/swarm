"""Permeability model for sandbox boundary analysis.

Models sandbox boundaries as semi-permeable membranes with parameterized
permeability (0 = fully sealed, 1 = fully open). Includes contagion
modeling for how harmful interactions inside the sandbox propagate to
the external world and vice versa.

Reference: "Virtual Agent Economies" (arXiv 2509.10147) - Tomasev et al.
"""

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from swarm.models.interaction import SoftInteraction


@dataclass
class PermeabilityConfig:
    """Configuration for the permeability model."""

    base_permeability: float = 0.5
    information_decay: float = 0.1
    contagion_rate: float = 0.05
    spillover_amplification: float = 1.5
    adaptive: bool = True
    threat_sensitivity: float = 1.0

    def validate(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.base_permeability <= 1.0:
            raise ValueError("base_permeability must be in [0, 1]")
        if not 0.0 <= self.information_decay <= 1.0:
            raise ValueError("information_decay must be in [0, 1]")
        if not 0.0 <= self.contagion_rate <= 1.0:
            raise ValueError("contagion_rate must be in [0, 1]")
        if self.spillover_amplification < 0:
            raise ValueError("spillover_amplification must be non-negative")
        if self.threat_sensitivity < 0:
            raise ValueError("threat_sensitivity must be non-negative")


@dataclass
class SpilloverEvent:
    """A harmful outcome that crossed the sandbox boundary."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_interaction_id: str = ""
    source_p: float = 0.5
    direction: str = "outbound"  # "outbound" or "inbound"
    harm_magnitude: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    blocked: bool = False

    def to_dict(self) -> Dict:
        """Serialize spillover event."""
        return {
            "event_id": self.event_id,
            "source_interaction_id": self.source_interaction_id,
            "source_p": self.source_p,
            "direction": self.direction,
            "harm_magnitude": self.harm_magnitude,
            "timestamp": self.timestamp.isoformat(),
            "blocked": self.blocked,
        }


@dataclass
class PermeabilityState:
    """Current state of the permeability model."""

    effective_permeability: float = 0.5
    total_spillovers: int = 0
    total_blocked: int = 0
    cumulative_external_harm: float = 0.0
    threat_level: float = 0.0

    def to_dict(self) -> Dict:
        """Serialize state."""
        return {
            "effective_permeability": self.effective_permeability,
            "total_spillovers": self.total_spillovers,
            "total_blocked": self.total_blocked,
            "cumulative_external_harm": self.cumulative_external_harm,
            "threat_level": self.threat_level,
        }


class PermeabilityModel:
    """
    Models sandbox boundary as a semi-permeable membrane.

    The model computes:
    1. Effective permeability based on base level, threat, and agent trust
    2. Whether individual flows should cross the boundary
    3. Contagion probability for harmful interactions
    4. Spillover harm magnitude with external amplification

    Connection to soft labels: contagion probability is proportional to
    (1-p) * permeability, linking boundary dynamics to the quality pipeline.
    """

    def __init__(
        self,
        config: Optional[PermeabilityConfig] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the permeability model."""
        self.config = config or PermeabilityConfig()
        self.config.validate()
        self._rng = random.Random(seed)
        self._state = PermeabilityState(
            effective_permeability=self.config.base_permeability
        )
        self._spillover_history: List[SpilloverEvent] = []

    @property
    def state(self) -> PermeabilityState:
        """Get current permeability state."""
        return self._state

    def compute_effective_permeability(
        self,
        threat_level: float = 0.0,
        agent_trust: float = 0.5,
    ) -> float:
        """
        Compute effective permeability given threat and trust levels.

        Args:
            threat_level: Current threat level [0, 1]
            agent_trust: Trust level of the agent [0, 1]

        Returns:
            Effective permeability in [0, 1]
        """
        base = self.config.base_permeability

        if not self.config.adaptive:
            return base

        # Reduce permeability with higher threat
        threat_factor = 1.0 - (threat_level * self.config.threat_sensitivity)
        threat_factor = max(0.0, min(1.0, threat_factor))

        # Increase permeability for trusted agents
        trust_factor = 0.5 + 0.5 * agent_trust

        effective = base * threat_factor * trust_factor
        effective = max(0.0, min(1.0, effective))

        self._state.effective_permeability = effective
        self._state.threat_level = threat_level

        return effective

    def should_cross(
        self,
        permeability: float,
        interaction_p: float,
        direction: str = "outbound",
    ) -> Tuple[bool, float]:
        """
        Decide whether an information flow should cross the boundary.

        Args:
            permeability: Effective permeability for this crossing
            interaction_p: Quality label of the interaction
            direction: "outbound" or "inbound"

        Returns:
            (allowed, degraded_quality) tuple
        """
        # Higher quality interactions are more likely to pass
        # Low-quality (low p) interactions face additional scrutiny
        quality_modifier = 0.5 + 0.5 * interaction_p
        effective_prob = permeability * quality_modifier

        allowed = self._rng.random() < effective_prob

        # Information degrades as it crosses the boundary
        if allowed:
            degraded_p = interaction_p * (1 - self.config.information_decay)
        else:
            degraded_p = interaction_p

        return allowed, degraded_p

    def compute_contagion_probability(
        self,
        interaction: SoftInteraction,
        permeability: float,
    ) -> float:
        """
        Compute probability that a harmful interaction spreads externally.

        Contagion probability = base_rate * (1-p) * permeability

        Args:
            interaction: The interaction to evaluate
            permeability: Current effective permeability

        Returns:
            Contagion probability in [0, 1]
        """
        harm_level = 1.0 - interaction.p
        prob = self.config.contagion_rate * harm_level * permeability
        return max(0.0, min(1.0, prob))

    def simulate_spillover(
        self,
        interactions: List[SoftInteraction],
        permeability: Optional[float] = None,
    ) -> List[SpilloverEvent]:
        """
        Simulate spillover events for a batch of interactions.

        Args:
            interactions: Interactions to check for spillover
            permeability: Override permeability (uses effective if None)

        Returns:
            List of SpilloverEvent objects (both blocked and unblocked)
        """
        perm = (
            permeability
            if permeability is not None
            else self._state.effective_permeability
        )
        spillovers = []

        for interaction in interactions:
            if not interaction.accepted:
                continue

            contagion_prob = self.compute_contagion_probability(interaction, perm)

            if self._rng.random() < contagion_prob:
                # Potential spillover
                harm_level = 1.0 - interaction.p
                raw_harm = harm_level * self.config.spillover_amplification

                # Try to block
                allowed, _ = self.should_cross(perm, interaction.p, "outbound")

                spillover = SpilloverEvent(
                    source_interaction_id=interaction.interaction_id,
                    source_p=interaction.p,
                    direction="outbound",
                    harm_magnitude=raw_harm,
                    blocked=not allowed,
                )

                spillovers.append(spillover)
                self._spillover_history.append(spillover)

                if not spillover.blocked:
                    self._state.total_spillovers += 1
                    self._state.cumulative_external_harm += spillover.harm_magnitude
                else:
                    self._state.total_blocked += 1

        return spillovers

    def compute_spillover_harm(
        self,
        spillovers: Optional[List[SpilloverEvent]] = None,
    ) -> float:
        """
        Compute total externalized harm from spillover events.

        Args:
            spillovers: List of spillover events (uses history if None)

        Returns:
            Total harm that crossed the boundary
        """
        events = spillovers if spillovers is not None else self._spillover_history
        return sum(s.harm_magnitude for s in events if not s.blocked)

    def containment_rate(self) -> float:
        """Fraction of potential spillovers that were blocked."""
        total = self._state.total_spillovers + self._state.total_blocked
        if total == 0:
            return 1.0
        return self._state.total_blocked / total

    def optimal_permeability(
        self,
        interactions: List[SoftInteraction],
        external_harm_weight: float = 1.0,
        n_samples: int = 10,
    ) -> float:
        """
        Estimate optimal permeability that balances flow and containment.

        Uses grid search over permeability values.

        Args:
            interactions: Interactions to evaluate
            external_harm_weight: Weight on external harm vs internal benefit
            n_samples: Number of permeability values to test

        Returns:
            Estimated optimal permeability in [0, 1]
        """
        best_perm = 0.0
        best_score = float("-inf")

        for i in range(n_samples + 1):
            perm = i / n_samples

            # Benefit: flows that cross successfully
            benefit = sum(ix.p * perm for ix in interactions if ix.accepted)

            # Cost: expected spillover harm
            harm = sum(
                self.compute_contagion_probability(ix, perm)
                * (1 - ix.p)
                * self.config.spillover_amplification
                for ix in interactions
                if ix.accepted
            )

            score = benefit - external_harm_weight * harm

            if score > best_score:
                best_score = score
                best_perm = perm

        return best_perm

    def get_spillover_history(self) -> List[SpilloverEvent]:
        """Get full spillover history."""
        return list(self._spillover_history)

    def reset(self) -> None:
        """Reset state and history."""
        self._state = PermeabilityState(
            effective_permeability=self.config.base_permeability
        )
        self._spillover_history.clear()
