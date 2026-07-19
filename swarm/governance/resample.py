"""Resample protocol governance lever (Bhatt et al., 2025).

Implements the core mechanism from "Ctrl-Z: Controlling AI Agents via
Resampling": when a monitor flags a suspicious action, additional samples
are drawn from the agent.  This serves two purposes:

1. **Blocking** — if a majority of resampled actions differ from the
   original, the suspicious action is replaced with the majority choice.
2. **Incrimination** — consistent dangerous behavior across samples
   accumulates evidence, which can trigger agent freeze.

The actual resampling logic lives in the orchestrator (analogous to
``SelfEnsembleLever``).  This lever handles:
- Suspicion scoring based on interaction observables
- Evidence accumulation across interactions (incrimination)
- Cost accounting for extra compute from resampling
- Agent freeze when evidence exceeds threshold
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional

from swarm.env.state import EnvState
from swarm.governance.config import GovernanceConfig
from swarm.governance.levers import GovernanceLever, LeverEffect
from swarm.models.interaction import SoftInteraction


class ResampleLever(GovernanceLever):
    """Governance lever implementing the resample protocol.

    The suspicion score for an interaction is computed from its soft label
    ``p`` — interactions with low ``p`` (likely harmful) are flagged.
    When the suspicion exceeds ``resample_suspicion_threshold``, the
    orchestrator draws ``resample_n_samples`` additional action samples.

    Evidence is accumulated per-agent across interactions.  If the total
    evidence for an agent exceeds ``resample_incrimination_threshold``,
    the agent is frozen (analogous to the paper's "shutdown" outcome).
    """

    def __init__(
        self,
        config: GovernanceConfig,
        seed: Optional[int] = None,
    ):
        super().__init__(config)
        self._rng = random.Random(seed)
        # Per-agent evidence accumulator: agent_id -> cumulative evidence score
        self._evidence: Dict[str, float] = defaultdict(float)
        # Per-agent resample history for reporting
        self._resample_counts: Dict[str, int] = defaultdict(int)
        # Trajectory window per agent: recent p values for trajectory analysis
        self._trajectory: Dict[str, List[float]] = defaultdict(list)

    @property
    def name(self) -> str:
        return "resample"

    # -----------------------------------------------------------------
    # Suspicion scoring
    # -----------------------------------------------------------------

    def compute_suspicion(self, p: float, agent_id: str) -> float:
        """Compute suspicion score for an action.

        Suspicion is based on:
        1. The soft label ``p`` (low p → high suspicion)
        2. Optional trajectory context (declining p over recent actions)

        Returns a score in [0, 1].
        """
        # Base suspicion: 1 - p (harmful interactions are suspicious)
        base_suspicion = 1.0 - p

        if not self.config.resample_trajectory_analysis:
            return base_suspicion

        # Trajectory analysis: if agent's recent p values are declining,
        # boost suspicion (pattern of increasingly harmful behavior)
        trajectory = self._trajectory.get(agent_id, [])
        if len(trajectory) >= self.config.resample_trajectory_window:
            recent = trajectory[-self.config.resample_trajectory_window:]
            # Compute trend: negative slope → declining quality
            n = len(recent)
            if n >= 2:
                x_mean = (n - 1) / 2.0
                y_mean = sum(recent) / n
                numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
                denominator = sum((i - x_mean) ** 2 for i in range(n))
                if denominator > 0:
                    slope = numerator / denominator
                    # Negative slope → boost suspicion (capped at 0.2 boost)
                    trend_boost = min(0.2, max(0.0, -slope))
                    base_suspicion = min(1.0, base_suspicion + trend_boost)

        return base_suspicion

    def should_resample(self, p: float, agent_id: str) -> bool:
        """Check whether an action should trigger resampling."""
        if not self.config.resample_enabled:
            return False
        suspicion = self.compute_suspicion(p, agent_id)
        return suspicion >= self.config.resample_suspicion_threshold

    # -----------------------------------------------------------------
    # Evidence / incrimination
    # -----------------------------------------------------------------

    def accumulate_evidence(
        self,
        agent_id: str,
        original_p: float,
        resample_p_values: List[float],
    ) -> float:
        """Accumulate incrimination evidence from resampled actions.

        If the resampled actions consistently show dangerous behavior
        (low p), the evidence grows.  If most resamples look benign,
        the evidence from this round is small.

        Returns the evidence delta added this round.
        """
        if not resample_p_values:
            return 0.0

        # Fraction of resampled actions that are also suspicious
        n_suspicious = sum(
            1 for rp in resample_p_values
            if (1.0 - rp) >= self.config.resample_suspicion_threshold
        )
        suspicious_fraction = n_suspicious / len(resample_p_values)

        # Evidence = suspicious_fraction * (1 - original_p)
        # Higher when original was suspicious AND resamples confirm it
        evidence_delta = suspicious_fraction * (1.0 - original_p)

        self._evidence[agent_id] += evidence_delta
        self._resample_counts[agent_id] += 1
        return evidence_delta

    def is_incriminated(self, agent_id: str) -> bool:
        """Check if accumulated evidence warrants agent freeze."""
        return self._evidence.get(agent_id, 0.0) >= self.config.resample_incrimination_threshold

    def get_evidence(self, agent_id: str) -> float:
        """Return current evidence for an agent."""
        return self._evidence.get(agent_id, 0.0)

    # -----------------------------------------------------------------
    # Lever hooks
    # -----------------------------------------------------------------

    def on_interaction(
        self,
        interaction: SoftInteraction,
        state: EnvState,
    ) -> LeverEffect:
        """Process interaction: update trajectory, apply costs, check freeze."""
        if not self.config.resample_enabled:
            return LeverEffect(lever_name=self.name)

        agent_id = interaction.initiator

        # Update trajectory window
        self._trajectory[agent_id].append(interaction.p)
        max_window = self.config.resample_trajectory_window
        if len(self._trajectory[agent_id]) > max_window:
            self._trajectory[agent_id] = self._trajectory[agent_id][-max_window:]

        # Check if this interaction was resampled (marked by orchestrator)
        was_resampled = interaction.metadata.get("resample_triggered", False)
        resample_count = interaction.metadata.get("resample_n_samples", 0)

        agents_to_freeze: set[str] = set()
        reputation_deltas: Dict[str, float] = {}
        cost_a = 0.0

        if was_resampled:
            # Compute cost surcharge for extra samples
            cost_a = self.config.resample_compute_cost * resample_count

            # Process incrimination from resample data.
            # The orchestrator populates resample_p_values with per-sample
            # reputation proxies at action-selection time.  Fall back to
            # the scalar p_proxy (or interaction.p) only for legacy data.
            resample_p_values = interaction.metadata.get("resample_p_values", [])
            if not resample_p_values:
                p_proxy = interaction.metadata.get("resample_p_proxy", interaction.p)
                resample_p_values = [p_proxy] * resample_count

            evidence_delta = self.accumulate_evidence(
                agent_id, interaction.p, resample_p_values
            )

            # Reputation penalty proportional to evidence
            if evidence_delta > 0:
                reputation_deltas[agent_id] = -evidence_delta * self.config.resample_evidence_rep_weight

            # Check incrimination threshold → freeze
            if self.is_incriminated(agent_id):
                agents_to_freeze.add(agent_id)

            # Was the action blocked (replaced by majority)?
            was_blocked = interaction.metadata.get("resample_blocked", False)
            if was_blocked:
                reputation_deltas[agent_id] = reputation_deltas.get(agent_id, 0.0) - 0.1

        return LeverEffect(
            cost_a=cost_a,
            agents_to_freeze=agents_to_freeze,
            reputation_deltas=reputation_deltas,
            lever_name=self.name,
            details={
                "suspicion": self.compute_suspicion(interaction.p, agent_id),
                "was_resampled": was_resampled,
                "resample_count": resample_count,
                "evidence_total": self._evidence.get(agent_id, 0.0),
                "incriminated": agent_id in agents_to_freeze,
            },
        )

    def on_epoch_start(
        self,
        state: EnvState,
        epoch: int,
    ) -> LeverEffect:
        """Decay evidence slightly each epoch to avoid permanent incrimination."""
        if not self.config.resample_enabled:
            return LeverEffect(lever_name=self.name)

        decay = self.config.resample_evidence_decay
        if decay < 1.0:
            for agent_id in list(self._evidence):
                self._evidence[agent_id] *= decay
                if self._evidence[agent_id] < 1e-6:
                    del self._evidence[agent_id]

        return LeverEffect(lever_name=self.name)

    # -----------------------------------------------------------------
    # Reporting / inspection
    # -----------------------------------------------------------------

    def get_report(self) -> Dict:
        """Return summary of resample activity."""
        return {
            "evidence": dict(self._evidence),
            "resample_counts": dict(self._resample_counts),
            "incriminated_agents": [
                aid for aid, ev in self._evidence.items()
                if ev >= self.config.resample_incrimination_threshold
            ],
        }

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._evidence.clear()
        self._resample_counts.clear()
        self._trajectory.clear()
