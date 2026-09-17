"""Admission control governance lever (staking)."""

from swarm.env.state import EnvState
from swarm.governance.levers import GovernanceLever, LeverEffect


class StakingLever(GovernanceLever):
    """
    Staking requirement for agent participation.

    Blocks agents from acting if their resources fall below
    min_stake_to_participate. Can also slash stakes on violations.
    """

    @property
    def name(self) -> str:
        return "staking"

    def can_agent_act(
        self,
        agent_id: str,
        state: EnvState,
    ) -> bool:
        """
        Check if agent has sufficient stake to participate.

        Args:
            agent_id: Agent attempting to act
            state: Current environment state

        Returns:
            True if agent has sufficient resources
        """
        if not self.config.staking_enabled:
            return True

        agent_state = state.get_agent(agent_id)
        if agent_state is None:
            return False

        balance = (
            agent_state.initial_resources + agent_state.total_payoff
            if self.config.stake_basis == "cumulative_payoff"
            else agent_state.resources
        )
        return balance >= self.config.min_stake_to_participate

    def slash_stake(
        self,
        agent_id: str,
        state: EnvState,
        reason: str = "violation",
    ) -> LeverEffect:
        """
        Slash an agent's stake for a violation.

        Args:
            agent_id: Agent to slash
            state: Current environment state
            reason: Reason for slashing

        Returns:
            Effect with resource delta
        """
        if not self.config.staking_enabled:
            return LeverEffect(lever_name=self.name)

        agent_state = state.get_agent(agent_id)
        if agent_state is None:
            return LeverEffect(lever_name=self.name)

        slash_amount = agent_state.resources * self.config.stake_slash_rate

        return LeverEffect(
            resource_deltas={agent_id: -slash_amount},
            lever_name=self.name,
            details={
                "agent_id": agent_id,
                "slash_amount": slash_amount,
                "slash_rate": self.config.stake_slash_rate,
                "reason": reason,
            },
        )
