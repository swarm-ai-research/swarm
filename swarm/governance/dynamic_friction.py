"""Incoherence-proportional friction governance lever."""

from swarm.env.state import EnvState
from swarm.governance.levers import GovernanceLever, LeverEffect
from swarm.models.interaction import SoftInteraction


class IncoherenceFrictionLever(GovernanceLever):
    """Apply transaction friction proportional to interaction uncertainty."""

    reduces_variance = True

    @property
    def name(self) -> str:
        return "incoherence_friction"

    def on_interaction(
        self,
        interaction: SoftInteraction,
        state: EnvState,
    ) -> LeverEffect:
        if not self.config.incoherence_friction_enabled:
            return LeverEffect(lever_name=self.name)
        if not interaction.accepted:
            return LeverEffect(lever_name=self.name)

        uncertainty = 1.0 - abs(2 * interaction.p - 1.0)
        base = abs(interaction.tau) + 1.0
        total_cost = self.config.incoherence_friction_rate * uncertainty * base
        split = self.config.transaction_tax_split
        return LeverEffect(
            lever_name=self.name,
            cost_a=total_cost * split,
            cost_b=total_cost * (1.0 - split),
            details={
                "uncertainty": uncertainty,
                "base": base,
                "friction_rate": self.config.incoherence_friction_rate,
                "total_cost": total_cost,
            },
        )
