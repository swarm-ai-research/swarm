"""Cascade risk governance lever.

Penalizes agents whose artifacts consistently produce low-quality
downstream chains.  Uses ``CausalCreditEngine.cascade_risk()`` to
compute a depth-weighted measure of bad descendants: each descendant's
contribution is weighted by ``decay^depth`` (default 0.5) so that
immediate failures matter more than distant ones.

This bridges the artifact layer (emergent tool chaining) with the
governance engine — it governs *chains*, not just individual nodes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from swarm.governance.levers import GovernanceLever, LeverEffect
from swarm.metrics.causal_credit import CausalCreditEngine

if TYPE_CHECKING:
    from swarm.env.state import EnvState
    from swarm.governance.config import GovernanceConfig
    from swarm.models.interaction import SoftInteraction


class CascadeRiskLever(GovernanceLever):
    """Govern agents by the downstream quality of their artifact chains.

    When an agent's recent interactions have high cascade risk (many
    low-p descendants), the lever applies a cost penalty and negative
    reputation delta proportional to the risk.

    Configuration fields on ``GovernanceConfig``:

    - ``cascade_risk_enabled``: Enable/disable this lever.
    - ``cascade_risk_threshold``: Minimum cascade risk to trigger penalty.
    - ``cascade_risk_penalty_scale``: Multiplier for cost penalty.
    - ``cascade_risk_reputation_scale``: Multiplier for reputation penalty.
    - ``cascade_risk_p_threshold``: What counts as a "bad" descendant.
    - ``cascade_risk_window``: Number of recent interactions to analyze.
    """

    def __init__(self, config: GovernanceConfig) -> None:
        super().__init__(config)
        self._credit_engine = CausalCreditEngine(
            decay=0.5,
            max_depth=10,
        )
        # Rolling history of interactions for DAG analysis
        self._interaction_history: List[SoftInteraction] = []
        self._dag_dirty: bool = False

    @property
    def name(self) -> str:
        return "cascade_risk"

    def on_interaction(
        self,
        interaction: SoftInteraction,
        state: EnvState,
    ) -> LeverEffect:
        if not self.config.cascade_risk_enabled:
            return LeverEffect(lever_name=self.name)

        # Add to rolling history
        self._interaction_history.append(interaction)
        self._dag_dirty = True
        if len(self._interaction_history) > self.config.cascade_risk_window:
            self._interaction_history = self._interaction_history[
                -self.config.cascade_risk_window:
            ]

        # Only evaluate if this interaction has causal parents
        # (i.e. it consumed artifacts — it's part of a chain)
        if not interaction.causal_parents:
            return LeverEffect(lever_name=self.name)

        # Rebuild DAG only when history has changed since last build
        if self._dag_dirty:
            self._credit_engine.build_dag(self._interaction_history)
            self._dag_dirty = False

        # Check cascade risk of the parent interactions this one consumed
        max_risk = 0.0
        blame_agent = ""
        for parent_id in interaction.causal_parents:
            risk = self._credit_engine.cascade_risk(
                parent_id, p_threshold=self.config.cascade_risk_p_threshold
            )
            if risk > max_risk:
                max_risk = risk
                # Find who initiated the parent interaction
                parent_interaction = self._credit_engine.get_interaction(parent_id)
                if parent_interaction:
                    blame_agent = parent_interaction.initiator

        if max_risk < self.config.cascade_risk_threshold or not blame_agent:
            return LeverEffect(lever_name=self.name)

        cost = max_risk * self.config.cascade_risk_penalty_scale
        rep_penalty = max_risk * self.config.cascade_risk_reputation_scale

        return LeverEffect(
            lever_name=self.name,
            cost_a=cost,
            reputation_deltas={blame_agent: -rep_penalty},
            details={
                "cascade_risk": max_risk,
                "blamed_agent": blame_agent,
                "parent_count": len(interaction.causal_parents),
            },
        )

    def on_epoch_start(
        self,
        state: EnvState,
        epoch: int,
    ) -> LeverEffect:
        """No-op — artifact GC is handled by the orchestrator."""
        return LeverEffect(lever_name=self.name)
