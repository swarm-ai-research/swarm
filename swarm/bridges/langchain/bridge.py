"""LangChain chain/agent wrapper for SWARM governance testing.

Wraps any LangChain ``Runnable`` (chain, AgentExecutor, tool, etc.)
as a SWARM agent that produces ``SoftInteraction`` objects scored by
``ProxyComputer`` and measured by ``SoftMetrics``.

Usage::

    from langchain.chains import LLMChain
    from swarm.bridges.langchain import LangChainBridge, LangChainBridgeConfig

    config = LangChainBridgeConfig(agent_id="qa-chain")
    bridge = LangChainBridge(chain=my_llm_chain, config=config)

    interaction = bridge.run(
        prompt="Summarise the findings.",
        counterparty="user-001",
    )
    print(f"p={interaction.p:.3f}")
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Any, List, Optional

from swarm.bridges._common import log_interaction_event
from swarm.bridges.langchain.config import LangChainBridgeConfig
from swarm.core.payoff import PayoffConfig, SoftPayoffEngine
from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.interaction import InteractionType, SoftInteraction

logger = logging.getLogger(__name__)


class LangChainBridgeError(Exception):
    """Raised when a LangChain bridge operation fails."""


class LangChainBridge:
    """Bridge a LangChain Runnable to the SWARM governance framework.

    The bridge executes the underlying chain, extracts observables from
    the output, computes a soft label ``p = P(interaction is beneficial)``,
    and records a ``SoftInteraction`` for metrics and governance hooks.

    Optional ``langchain`` import is deferred to runtime so that the
    bridge module can be imported even without LangChain installed.

    Args:
        chain: Any object with an ``.invoke(input)`` or ``.run(input)``
               method (LangChain Runnable, Chain, AgentExecutor, etc.).
        config: Bridge configuration.
        payoff_config: Optional custom payoff parameters.
    """

    def __init__(
        self,
        chain: Any,
        config: Optional[LangChainBridgeConfig] = None,
        payoff_config: Optional[PayoffConfig] = None,
    ) -> None:
        self.chain = chain
        self.config = config or LangChainBridgeConfig()
        self._proxy = ProxyComputer(sigmoid_k=self.config.proxy_sigmoid_k)
        self._payoff_engine = SoftPayoffEngine(
            payoff_config or PayoffConfig(w_rep=self.config.reputation_weight)
        )
        self._metrics = SoftMetrics(self._payoff_engine)
        self._event_log: Optional[Any] = None
        self.last_payoff: Optional[float] = None
        self._interactions: List[SoftInteraction] = []

        if self.config.enable_event_log:
            try:
                from pathlib import Path as _Path

                from swarm.logging.event_log import EventLog

                path = self.config.event_log_path or f"{self.config.agent_id}_events.jsonl"
                self._event_log = EventLog(path=_Path(path))
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not initialise EventLog: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        prompt: str,
        counterparty: str = "environment",
        quality_score: Optional[float] = None,
    ) -> SoftInteraction:
        """Execute the chain and return a scored SoftInteraction.

        Args:
            prompt: Input passed to the chain.
            counterparty: ID of the agent/user on the other side.
            quality_score: Optional quality score in [0, 1].
                           Higher scores boost verifier signal.

        Returns:
            A ``SoftInteraction`` with ``p`` ∈ [0, 1].

        Raises:
            LangChainBridgeError: if the chain raises an unhandled exception.
        """
        error: Optional[str] = None
        raw_output: Any = None
        intermediate_steps: int = 0

        try:
            raw_output, intermediate_steps = self._invoke_chain(prompt)
            success = True
        except Exception as exc:
            error = str(exc)
            success = False
            logger.warning(
                "LangChain chain failed for agent %s: %s",
                self.config.agent_id,
                error,
            )

        observables = self._extract_observables(
            success=success,
            raw_output=raw_output,
            intermediate_steps=intermediate_steps,
            quality_score=quality_score,
        )

        v_hat, p = self._proxy.compute_labels(observables)
        # Safety invariant: p must be in [0, 1]
        assert 0.0 <= p <= 1.0, f"p invariant violated: p={p}"

        interaction = SoftInteraction(
            initiator=self.config.agent_id,
            counterparty=counterparty,
            interaction_type=InteractionType.COLLABORATION,
            p=p,
            v_hat=v_hat,
            accepted=success,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            metadata={
                "intermediate_steps": intermediate_steps,
                "error": error,
            },
        )

        self.last_payoff = self._payoff_engine.payoff_initiator(interaction)
        self._interactions.append(interaction)

        log_interaction_event(self._event_log, interaction)

        return interaction

    def get_toxicity_rate(self) -> float:
        """Toxicity rate over all recorded interactions."""
        return float(self._metrics.toxicity_rate(self._interactions))

    def get_quality_gap(self) -> float:
        """Quality gap over all recorded interactions."""
        return float(self._metrics.quality_gap(self._interactions))

    def get_interactions(self) -> List[SoftInteraction]:
        """Return all recorded interactions (read-only copy)."""
        return list(self._interactions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _invoke_chain(self, prompt: str) -> tuple[Any, int]:
        """Invoke the chain with a timeout.

        Returns:
            (raw_output, intermediate_steps_count)
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._call_chain, prompt)
            try:
                return future.result(timeout=self.config.timeout_seconds)
            except concurrent.futures.TimeoutError as exc:
                raise LangChainBridgeError(
                    f"Chain timed out after {self.config.timeout_seconds}s"
                ) from exc

    def _call_chain(self, prompt: str) -> tuple[Any, int]:
        """Directly call the chain (runs in worker thread)."""
        intermediate_steps = 0
        # Try .invoke() first (LangChain Runnables), then .run() (legacy)
        if hasattr(self.chain, "invoke"):
            result = self.chain.invoke({"input": prompt})
            if isinstance(result, dict):
                steps = result.get("intermediate_steps", [])
                intermediate_steps = len(steps) if isinstance(steps, (list, tuple)) else 0
                output = result.get("output", result)
            else:
                output = result
        elif hasattr(self.chain, "run"):
            output = self.chain.run(prompt)
        else:
            raise LangChainBridgeError(
                f"Chain {type(self.chain).__name__} has no .invoke() or .run() method"
            )

        if intermediate_steps > self.config.max_steps:
            raise LangChainBridgeError(
                f"Chain exceeded max_steps={self.config.max_steps} "
                f"(used {intermediate_steps})"
            )
        return output, intermediate_steps

    def _extract_observables(
        self,
        success: bool,
        raw_output: Any,
        intermediate_steps: int,
        quality_score: Optional[float],
    ) -> ProxyObservables:
        """Map chain execution results to SWARM proxy observables.

        Observable mapping:
            task_progress_delta  ← +1.0 if success else -1.0
            rework_count         ← intermediate_steps (number of tool calls)
            verifier_rejections  ← 0 if success else 1
            counterparty_engagement_delta ← output length as engagement proxy

        Returns:
            ProxyObservables with values in valid ranges.
        """
        task_progress_delta = 1.0 if success else -1.0
        rework_count = min(intermediate_steps, self.config.max_steps)
        verifier_rejections = 0 if success else 1

        # Engagement: output character count normalised to [−1, +1]
        if raw_output is not None:
            output_str = str(raw_output)
            # Normalize to [0, 1] then shift to [-1, +1]
            normalized = min(len(output_str) / max(self.config.engagement_max_chars, 1), 1.0)
            engagement_delta = normalized * 2.0 - 1.0
        else:
            engagement_delta = -1.0

        # If quality_score provided, use it to adjust verifier rejections
        if quality_score is not None:
            # quality_score ∈ [0,1]; treat < 0.5 as a rejection
            if quality_score < 0.5:
                verifier_rejections += 1

        return ProxyObservables(
            task_progress_delta=task_progress_delta,
            rework_count=rework_count,
            verifier_rejections=verifier_rejections,
            counterparty_engagement_delta=engagement_delta,
        )
