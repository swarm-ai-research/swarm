"""Interaction finalization: governance, payoff, reputation, and state updates.

Extracted from ``Orchestrator`` to isolate the interaction completion
pipeline into a focused, testable component.  The orchestrator delegates
to an ``InteractionFinalizer`` instance, passing shared mutable
references (not copies) so all state mutations are visible immediately.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from swarm.core.handler import HandlerActionResult
from swarm.core.observable_generator import ObservableGenerator
from swarm.core.payoff import SoftPayoffEngine
from swarm.core.proxy import ProxyComputer
from swarm.env.network import AgentNetwork
from swarm.env.state import EnvState, InteractionProposal
from swarm.governance.engine import GovernanceEffect, GovernanceEngine
from swarm.logging.event_bus import EventBus
from swarm.models.events import (
    Event,
    EventType,
    interaction_completed_event,
    payoff_computed_event,
    reputation_updated_event,
)
from swarm.models.interaction import InteractionType, SoftInteraction


class InteractionFinalizer:
    """Handles the full lifecycle of completing an interaction.

    Responsibilities:
    - Generate observables and compute proxy labels
    - Apply governance effects
    - Compute payoffs for both parties
    - Update agent state (reputation, resources, payoff totals)
    - Strengthen network edges on accepted interactions
    - Emit events and fire callbacks
    """

    def __init__(
        self,
        state: EnvState,
        payoff_engine: SoftPayoffEngine,
        proxy_computer: ProxyComputer,
        observable_generator: ObservableGenerator,
        governance_engine: Optional[GovernanceEngine],
        network: Optional[AgentNetwork],
        agents: Dict[str, Any],
        on_interaction_complete: List[Callable],
        *,
        event_bus: EventBus,
    ) -> None:
        self._state = state
        self._payoff_engine = payoff_engine
        self._proxy_computer = proxy_computer
        self._observable_generator = observable_generator
        self._governance_engine = governance_engine
        self._network = network
        self._agents = agents
        self._on_interaction_complete = on_interaction_complete
        self._event_bus = event_bus
        self._emit_event: Callable[[Event], None] = event_bus.emit

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete_interaction(
        self,
        proposal: InteractionProposal,
        accepted: bool,
    ) -> None:
        """Complete an interaction and compute payoffs."""
        observables = self._observable_generator.generate(
            proposal,
            accepted,
            self._state,
        )

        v_hat, p = self._proxy_computer.compute_labels(observables)

        # Calibration scenarios: generators may expose a latent ground-truth
        # draw (duck-typed, like the obfuscation offsets protocol). Drawn
        # from the generator's *intended* latent probability — not the
        # computed p — so injection drift shows up as calibration error.
        draw = getattr(self._observable_generator, "draw_ground_truth", None)
        ground_truth = draw(proposal.initiator_id) if callable(draw) else None

        interaction = SoftInteraction(
            interaction_id=proposal.proposal_id,
            initiator=proposal.initiator_id,
            counterparty=proposal.counterparty_id,
            interaction_type=InteractionType(proposal.interaction_type),
            accepted=accepted,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=observables.tool_misuse_flags,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            tau=proposal.metadata.get("offered_transfer", 0),
            ground_truth=ground_truth,
            metadata=proposal.metadata,
        )

        self.finalize_interaction(interaction)

    def finalize_interaction(
        self,
        interaction: SoftInteraction,
        handler_result: Optional[HandlerActionResult] = None,
    ) -> Tuple[GovernanceEffect, float, float]:
        """Apply governance, compute payoffs, update state, and emit events.

        Args:
            interaction: The interaction to finalize.
            handler_result: If the interaction was produced by a handler
                that returned artifacts, pass the result here so
                ``_wire_artifacts`` can publish them and auto-wire
                ``causal_parents``.
        """
        # Wire artifacts and causal_parents BEFORE governance so that
        # levers (e.g. CascadeRiskLever) can see the populated DAG edges.
        if handler_result is not None:
            self._wire_artifacts(interaction, handler_result)

        gov_effect = GovernanceEffect()
        if self._governance_engine:
            gov_effect = self._governance_engine.apply_interaction(
                interaction, self._state
            )
            interaction.c_a += gov_effect.cost_a
            interaction.c_b += gov_effect.cost_b
            self.apply_governance_effect(gov_effect)

            if gov_effect.cost_a > 0 or gov_effect.cost_b > 0:
                self._emit_event(
                    Event(
                        event_type=EventType.GOVERNANCE_COST_APPLIED,
                        interaction_id=interaction.interaction_id,
                        initiator_id=interaction.initiator,
                        counterparty_id=interaction.counterparty,
                        payload={
                            "cost_a": gov_effect.cost_a,
                            "cost_b": gov_effect.cost_b,
                            "levers": [
                                e.lever_name for e in gov_effect.lever_effects
                            ],
                        },
                        epoch=self._state.current_epoch,
                        step=self._state.current_step,
                    )
                )

        payoff_init = self._payoff_engine.payoff_initiator(interaction)
        payoff_counter = self._payoff_engine.payoff_counterparty(interaction)

        if interaction.accepted:
            initiator_state = self._state.get_agent(interaction.initiator)
            counterparty_state = self._state.get_agent(interaction.counterparty)

            if initiator_state:
                initiator_state.record_initiated(accepted=True, p=interaction.p)
                initiator_state.total_payoff += payoff_init

                # Reputation delta formula:
                #   rep_delta = (p - 0.5) - c_a
                #
                # (p - 0.5)  Quality signal centered at neutral.  Interactions
                #            with p > 0.5 (beneficial) raise reputation; those
                #            below lower it.  The 0.5 baseline means a perfectly
                #            ambiguous interaction produces zero signal.
                #
                # -c_a       Governance cost penalty.  Transaction taxes, audit
                #            fees, and other governance costs levied on the
                #            initiator reduce the reputation gain.  This couples
                #            reputation to governance: agents under heavier
                #            scrutiny must demonstrate higher quality (p >> 0.5)
                #            to build reputation, while agents with zero
                #            governance cost earn reputation at face value.
                #
                # The result feeds into the payoff equation as:
                #   pi_a = theta * S_soft - tau - c_a - rho_a * E_soft + w_rep * r_a
                #
                # where r_a is cumulative reputation and w_rep weights its
                # contribution to payoffs (default 1.0).
                rep_delta = (interaction.p - 0.5) - interaction.c_a
                self._update_reputation(interaction.initiator, rep_delta)

            if counterparty_state:
                counterparty_state.record_received(accepted=True, p=interaction.p)
                counterparty_state.total_payoff += payoff_counter

        if interaction.initiator in self._agents:
            self._agents[interaction.initiator].update_from_outcome(
                interaction, payoff_init
            )
        if interaction.counterparty in self._agents:
            self._agents[interaction.counterparty].update_from_outcome(
                interaction, payoff_counter
            )

        self._state.record_interaction(interaction)

        self._emit_event(
            interaction_completed_event(
                interaction_id=interaction.interaction_id,
                accepted=interaction.accepted,
                payoff_initiator=payoff_init,
                payoff_counterparty=payoff_counter,
                epoch=self._state.current_epoch,
                step=self._state.current_step,
            )
        )

        self._emit_event(
            payoff_computed_event(
                interaction_id=interaction.interaction_id,
                initiator_id=interaction.initiator,
                counterparty_id=interaction.counterparty,
                payoff_initiator=payoff_init,
                payoff_counterparty=payoff_counter,
                components={
                    "p": interaction.p,
                    "v_hat": interaction.v_hat,
                    "tau": interaction.tau,
                    "accepted": interaction.accepted,
                },
                epoch=self._state.current_epoch,
                step=self._state.current_step,
            )
        )

        if self._network is not None and interaction.accepted:
            self._network.strengthen_edge(
                interaction.initiator, interaction.counterparty
            )

        for callback in self._on_interaction_complete:
            callback(interaction, payoff_init, payoff_counter)

        return gov_effect, payoff_init, payoff_counter

    def apply_governance_effect(self, effect: GovernanceEffect) -> None:
        """Apply governance effects to state (freeze/unfreeze, reputation, resources)."""
        for agent_id in effect.agents_to_freeze:
            self._state.freeze_agent(agent_id)

        for agent_id in effect.agents_to_unfreeze:
            self._state.unfreeze_agent(agent_id)

        for agent_id, delta in effect.reputation_deltas.items():
            agent_state = self._state.get_agent(agent_id)
            if agent_state:
                old_rep = agent_state.reputation
                agent_state.update_reputation(delta)
                self._emit_event(
                    reputation_updated_event(
                        agent_id=agent_id,
                        old_reputation=old_rep,
                        new_reputation=agent_state.reputation,
                        delta=delta,
                        reason="governance",
                        epoch=self._state.current_epoch,
                        step=self._state.current_step,
                    )
                )

        for agent_id, delta in effect.resource_deltas.items():
            agent_state = self._state.get_agent(agent_id)
            if agent_state:
                agent_state.update_resources(delta)

    # ------------------------------------------------------------------
    # Artifact layer
    # ------------------------------------------------------------------

    def _wire_artifacts(
        self,
        interaction: SoftInteraction,
        result: HandlerActionResult,
    ) -> None:
        """Publish produced artifacts and auto-wire causal_parents from consumed ones."""
        registry = self._state.artifact_registry

        # Publish produced artifacts with the interaction's soft label
        for artifact in result.produced_artifacts:
            artifact.interaction_id = interaction.interaction_id
            artifact.p_at_production = interaction.p
            artifact.step = (
                self._state.current_epoch * self._state.steps_per_epoch
                + self._state.current_step
            )
            registry.publish(artifact)

        # Consume artifacts and wire causal_parents
        for consumed_id in result.consumed_artifact_ids:
            parent_interaction_id = registry.consume(
                consumed_id, interaction.interaction_id
            )
            if parent_interaction_id and parent_interaction_id not in interaction.causal_parents:
                interaction.causal_parents.append(parent_interaction_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_reputation(self, agent_id: str, delta: float) -> None:
        """Update agent reputation."""
        agent_state = self._state.get_agent(agent_id)
        if not agent_state:
            return

        old_rep = agent_state.reputation
        agent_state.update_reputation(delta)

        self._emit_event(
            reputation_updated_event(
                agent_id=agent_id,
                old_reputation=old_rep,
                new_reputation=agent_state.reputation,
                delta=delta,
                reason="interaction_outcome",
                epoch=self._state.current_epoch,
                step=self._state.current_step,
            )
        )
