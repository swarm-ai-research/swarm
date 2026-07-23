"""Refinery quality gate governance lever.

Models GasTown's Refinery merge queue: a deterministic quality gate that
rejects interactions where the proxy probability p falls below a threshold.

Key property: zero friction on accepted interactions.  No tax, no staking,
no reputation decay — only rejected interactions are affected.
"""

from swarm.env.state import EnvState
from swarm.governance.config import GovernanceConfig
from swarm.governance.levers import GovernanceLever, LeverEffect
from swarm.models.interaction import SoftInteraction


class RefineryLever(GovernanceLever):
    """Deterministic quality gate inspired by GasTown's Refinery.

    Rejects any interaction whose proxy probability ``p`` is below
    ``refinery_p_threshold``.  Accepted interactions pass through with
    zero additional cost — the lever imposes no friction on quality work.

    Contract note: rejection works by mutating ``interaction.accepted``
    in ``on_interaction`` — the only lever that modifies its input, since
    ``LeverEffect`` has no acceptance-override channel. Levers that run
    after this one in the engine's loop observe the mutated value.
    """

    def __init__(self, config: GovernanceConfig) -> None:
        super().__init__(config)
        self._rejected_count: int = 0
        self._inspected_count: int = 0

    @property
    def name(self) -> str:
        return "refinery"

    def on_interaction(
        self,
        interaction: SoftInteraction,
        state: EnvState,
    ) -> LeverEffect:
        if not self.config.refinery_enabled:
            return LeverEffect(lever_name=self.name)

        self._inspected_count += 1

        if interaction.p < self.config.refinery_p_threshold:
            interaction.accepted = False
            self._rejected_count += 1
            return LeverEffect(
                lever_name=self.name,
                details={
                    "action": "rejected",
                    "p": interaction.p,
                    "threshold": self.config.refinery_p_threshold,
                },
            )

        return LeverEffect(
            lever_name=self.name,
            details={
                "action": "passed",
                "p": interaction.p,
                "threshold": self.config.refinery_p_threshold,
            },
        )

    def get_stats(self) -> dict:
        """Return rejection statistics."""
        return {
            "inspected": self._inspected_count,
            "rejected": self._rejected_count,
            "rejection_rate": (
                self._rejected_count / self._inspected_count
                if self._inspected_count > 0
                else 0.0
            ),
        }
