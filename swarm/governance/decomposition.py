"""Long-horizon decomposition/verification governance lever."""

from swarm.env.state import EnvState
from swarm.governance.levers import GovernanceLever, LeverEffect


class DecompositionLever(GovernanceLever):
    """
    Step-level checkpointing for long-horizon episodes.

    When horizon exceeds a threshold, periodic checkpoints apply mild
    reputation penalties to agents whose initiated interaction acceptance
    rates suggest unstable plans.
    """

    reduces_variance = True

    @property
    def name(self) -> str:
        return "decomposition"

    def on_step(
        self,
        state: EnvState,
        step: int,
    ) -> LeverEffect:
        if not self.config.decomposition_enabled:
            return LeverEffect(lever_name=self.name)
        if state.steps_per_epoch < self.config.decomposition_horizon_threshold:
            return LeverEffect(lever_name=self.name)

        interval = max(1, self.config.decomposition_horizon_threshold // 2)
        if step == 0 or step % interval != 0:
            return LeverEffect(lever_name=self.name)

        reputation_deltas = {}
        for agent_id, agent_state in state.agents.items():
            if agent_state.interactions_initiated < 2:
                continue
            if agent_state.acceptance_rate() < 0.2:
                reputation_deltas[agent_id] = -0.05

        return LeverEffect(
            lever_name=self.name,
            reputation_deltas=reputation_deltas,
            details={
                "checkpoint_step": step,
                "checkpoint_interval": interval,
                "penalized_agents": list(reputation_deltas.keys()),
            },
        )
