"""Variance-reduction ensemble governance lever."""

from swarm.env.state import EnvState
from swarm.governance.levers import GovernanceLever, LeverEffect
from swarm.models.interaction import SoftInteraction


class SelfEnsembleLever(GovernanceLever):
    """
    Governance cost model for self-ensemble execution.

    The actual action-level ensembling is performed in the orchestrator; this
    lever accounts for additional compute/latency friction in payoff space.
    """

    reduces_variance = True

    @property
    def name(self) -> str:
        return "self_ensemble"

    def on_interaction(
        self,
        interaction: SoftInteraction,
        state: EnvState,
    ) -> LeverEffect:
        if not self.config.self_ensemble_enabled:
            return LeverEffect(lever_name=self.name)
        if not interaction.accepted:
            return LeverEffect(lever_name=self.name)

        extra_samples = max(0, self.config.self_ensemble_samples - 1)
        if extra_samples == 0:
            return LeverEffect(lever_name=self.name)

        # Small linear compute surcharge per extra sample.
        total_cost = 0.01 * extra_samples
        return LeverEffect(
            cost_a=total_cost * 0.5,
            cost_b=total_cost * 0.5,
            lever_name=self.name,
            details={
                "ensemble_samples": self.config.self_ensemble_samples,
                "compute_surcharge": total_cost,
            },
        )
