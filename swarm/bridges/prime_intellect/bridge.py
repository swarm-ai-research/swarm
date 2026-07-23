"""Main adapter connecting Prime Intellect–trained models to SWARM evaluation.

PrimeIntellectBridge is the central bridge class that:
1. Loads a model (local path or PI hub reference).
2. Injects it as an agent into a SWARM simulation.
3. Converts model completions → ProxyObservables → SoftInteraction.
4. Records interactions and computes safety metrics.

This is the "Evaluation Bridge" described in the integration plan: after
training a model on Prime Intellect using SWARM rewards, bridge it back
into a full SWARM simulation to measure population-level safety metrics
(toxicity, quality gap, Purity Paradox, etc.).
"""

import logging
from hashlib import sha256
from typing import Any, Callable, Dict, List, Optional

from swarm.bridges._common import trim_to_half
from swarm.bridges.prime_intellect.config import PrimeIntellectConfig
from swarm.bridges.prime_intellect.events import (
    PIEvent,
    PIEventType,
)
from swarm.bridges.prime_intellect.rewards import SwarmRewardComputer
from swarm.bridges.prime_intellect.scoring import score_text
from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.logging.event_log import EventLog
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.events import Event, EventType
from swarm.models.interaction import InteractionType, SoftInteraction

logger = logging.getLogger(__name__)


# Type alias for a model inference callable
ModelCallable = Callable[[str], str]


class PrimeIntellectBridge:
    """Bridge between a Prime Intellect–trained model and SWARM.

    The bridge wraps any callable ``model_fn(prompt) -> completion`` and
    lets it participate in SWARM interactions.  The callable can be:

    * A local model loaded via transformers / vLLM.
    * A Prime Intellect API endpoint.
    * A stub for testing.

    Example::

        def my_model(prompt: str) -> str:
            return llm.generate(prompt)

        bridge = PrimeIntellectBridge(model_fn=my_model)
        interactions = bridge.evaluate_prompt(
            agent_ids=["pi_model", "honest_0"],
            prompt="Collaborate on this task...",
        )
    """

    def __init__(
        self,
        config: Optional[PrimeIntellectConfig] = None,
        proxy_computer: Optional[ProxyComputer] = None,
        event_log: Optional[EventLog] = None,
        model_fn: Optional[ModelCallable] = None,
    ) -> None:
        self._config = config or PrimeIntellectConfig()
        self._proxy = proxy_computer or ProxyComputer(
            sigmoid_k=self._config.proxy_sigmoid_k
        )
        self._event_log = event_log
        self._model_fn = model_fn
        self._reward_computer = SwarmRewardComputer(self._config)
        self._interactions: List[SoftInteraction] = []
        self._events: List[PIEvent] = []
        self._metrics = SoftMetrics()

    # ----- public interface ------------------------------------------------

    def evaluate_prompt(
        self,
        agent_ids: List[str],
        prompt: str,
        step: int = 0,
    ) -> List[SoftInteraction]:
        """Run the model on a prompt and evaluate the result.

        Args:
            agent_ids: Agent IDs involved (first is the PI model agent).
            prompt: The situation prompt to present to the model.
            step: Current simulation step.

        Returns:
            List of SoftInteraction objects (one per agent pair).
        """
        if self._model_fn is None:
            raise RuntimeError(
                "No model_fn provided.  Pass a callable to the constructor "
                "or use set_model_fn()."
            )

        # Get model completion
        completion = self._model_fn(prompt)

        self._record_event(
            PIEventType.OBSERVATION_GENERATED,
            {
                "agent_ids": agent_ids,
                "prompt_len": len(prompt),
                "completion_len": len(completion),
                "step": step,
            },
        )

        # Score completion through SWARM pipeline
        observables = self._completion_to_observables(completion)
        v_hat, p = self._proxy.compute_labels(observables)

        # Hash the completion preview to avoid storing raw model output
        completion_hash = sha256(
            completion[:200].encode("utf-8")
        ).hexdigest()

        self._record_event(
            PIEventType.REWARD_COMPUTED,
            {
                "v_hat": v_hat,
                "p": p,
                "step": step,
            },
        )

        # Create interactions for each agent pair
        interactions: List[SoftInteraction] = []
        model_agent = agent_ids[0] if agent_ids else "pi_model"

        if len(agent_ids) < 2:
            interaction = SoftInteraction(
                initiator=model_agent,
                counterparty=model_agent,
                interaction_type=InteractionType.COLLABORATION,
                accepted=True,
                task_progress_delta=observables.task_progress_delta,
                rework_count=observables.rework_count,
                verifier_rejections=observables.verifier_rejections,
                tool_misuse_flags=observables.tool_misuse_flags,
                counterparty_engagement_delta=observables.counterparty_engagement_delta,
                v_hat=v_hat,
                p=p,
                metadata={
                    "bridge": "prime_intellect",
                    "step": step,
                    "completion_sha256": completion_hash,
                },
            )
            interactions.append(interaction)
        else:
            for counterparty in agent_ids[1:]:
                interaction = SoftInteraction(
                    initiator=model_agent,
                    counterparty=counterparty,
                    interaction_type=InteractionType.COLLABORATION,
                    accepted=True,
                    task_progress_delta=observables.task_progress_delta,
                    rework_count=observables.rework_count,
                    verifier_rejections=observables.verifier_rejections,
                    tool_misuse_flags=observables.tool_misuse_flags,
                    counterparty_engagement_delta=observables.counterparty_engagement_delta,
                    v_hat=v_hat,
                    p=p,
                    metadata={
                        "bridge": "prime_intellect",
                        "step": step,
                        "completion_sha256": completion_hash,
                    },
                )
                interactions.append(interaction)

        for interaction in interactions:
            self._record_interaction(interaction)
            self._log_interaction(interaction)

        return interactions

    def evaluate_batch(
        self,
        prompts: List[Dict[str, Any]],
    ) -> List[SoftInteraction]:
        """Evaluate a batch of prompts.

        Each item in ``prompts`` should have keys:
        - ``agent_ids``: list of agent IDs
        - ``prompt``: the prompt text
        - ``step``: optional step number

        Returns:
            All generated SoftInteraction objects.
        """
        all_interactions: List[SoftInteraction] = []
        for item in prompts:
            interactions = self.evaluate_prompt(
                agent_ids=item["agent_ids"],
                prompt=item["prompt"],
                step=item.get("step", 0),
            )
            all_interactions.extend(interactions)
        return all_interactions

    def set_model_fn(self, model_fn: ModelCallable) -> None:
        """Set or replace the model callable."""
        self._model_fn = model_fn

    def get_interactions(self) -> List[SoftInteraction]:
        """Return a copy of all recorded interactions."""
        return list(self._interactions)

    def get_events(self) -> List[PIEvent]:
        """Return a copy of all recorded events."""
        return list(self._events)

    def get_metrics(self) -> Dict[str, float]:
        """Compute safety metrics from stored interactions."""
        if not self._interactions:
            return {
                "toxicity_rate": 0.0,
                "quality_gap": 0.0,
                "total_welfare": 0.0,
                "mean_p": 0.5,
                "n_interactions": 0,
            }

        tox = self._metrics.toxicity_rate(self._interactions)
        qg = self._metrics.quality_gap(self._interactions)
        mean_p = sum(ix.p for ix in self._interactions) / len(self._interactions)

        # Welfare via reward computer
        from swarm.core.payoff import SoftPayoffEngine

        engine = SoftPayoffEngine()
        total_welfare = sum(
            engine.payoff_initiator(ix) + engine.payoff_counterparty(ix)
            for ix in self._interactions
        )

        return {
            "toxicity_rate": tox,
            "quality_gap": qg,
            "total_welfare": total_welfare,
            "mean_p": mean_p,
            "n_interactions": len(self._interactions),
        }

    def get_reward(self) -> float:
        """Compute the current composite reward from all interactions."""
        return self._reward_computer.compute(self._interactions)

    # ----- internal --------------------------------------------------------

    def _completion_to_observables(self, completion: str) -> ProxyObservables:
        """Convert a model completion to SWARM proxy observables.

        Delegates to the shared ``scoring.score_text`` heuristic, which
        includes anti-gaming defences (contradiction detection, density
        normalisation, repetition penalty).

        In production, an LLM judge should be used for richer scoring.
        """
        return score_text(completion)

    def _record_interaction(self, interaction: SoftInteraction) -> None:
        """Record an interaction, enforcing the configured cap."""
        self._interactions = trim_to_half(self._interactions, self._config.max_interactions)
        self._interactions.append(interaction)

    def _record_event(
        self, event_type: PIEventType, payload: Dict[str, Any]
    ) -> None:
        """Record a bridge event, enforcing the configured cap."""
        self._events = trim_to_half(self._events, self._config.max_events)
        self._events.append(PIEvent(
            event_type=event_type,
            payload=payload,
        ))

    def _log_interaction(self, interaction: SoftInteraction) -> None:
        """Log an interaction to SWARM's event log."""
        if self._event_log is None:
            return

        metadata = dict(interaction.metadata or {})

        event = Event(
            event_type=EventType.INTERACTION_COMPLETED,
            interaction_id=interaction.interaction_id,
            initiator_id=interaction.initiator,
            counterparty_id=interaction.counterparty,
            payload={
                "accepted": interaction.accepted,
                "v_hat": interaction.v_hat,
                "p": interaction.p,
                "bridge": "prime_intellect",
                "metadata": metadata,
            },
        )
        self._event_log.append(event)
