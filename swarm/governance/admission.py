"""Admission control governance lever (staking)."""

from swarm.env.state import EnvState
from swarm.governance.levers import GovernanceLever, LeverEffect
from swarm.models.agent import AgentState


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

        return self.stake_balance(agent_state) >= self.config.min_stake_to_participate

    def stake_balance(self, agent_state: AgentState) -> float:
        """The balance the stake is measured against, per ``stake_basis``.

        The cumulative-payoff basis nets out slashes: earnings only ever
        accumulate, so without that term nothing could push an agent below
        the bar (beads-ms0f).
        """
        if self.config.stake_basis == "cumulative_payoff":
            return (
                agent_state.initial_resources
                + agent_state.total_payoff
                - agent_state.stake_slashed
            )
        return agent_state.resources

    def slash_stake(
        self,
        agent_id: str,
        state: EnvState,
        reason: str = "violation",
    ) -> LeverEffect:
        """
        Slash an agent's stake for a violation.

        Records the amount on ``AgentState.stake_slashed`` so the gate sees it
        under either basis; the returned effect still carries the resource
        delta and must be applied by the caller.

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

        # An agent already below zero has nothing left to take; slashing a
        # negative balance would hand it resources back.
        balance = max(0.0, self.stake_balance(agent_state))
        slash_amount = balance * self.config.stake_slash_rate
        agent_state.stake_slashed += slash_amount

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
