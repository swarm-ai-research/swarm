"""OpenSandboxMapper — convert sandbox signals to SoftInteractions.

Translates command execution results, message bus activity, and
governance interventions into SWARM's ProxyObservables, then uses
ProxyComputer to derive v_hat and p.
"""

import logging
from typing import Dict, Optional

from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.models.interaction import InteractionType, SoftInteraction

logger = logging.getLogger(__name__)


class OpenSandboxMapper:
    """Maps sandbox execution signals to SoftInteractions.

    Observable mapping:

    ====================================== ============================= ==========================
    Sandbox Signal                         ProxyObservable               Formula
    ====================================== ============================= ==========================
    Successful actions / total             task_progress_delta           ratio clamped to [0, 1]
    Contract violations count              tool_misuse_flags             direct count
    Governance interventions count         verifier_rejections           direct count
    Blocked messages count                 rework_count                  direct count
    Messages delivered / total             counterparty_engagement_delta ratio scaled to [-1, 1]
    ====================================== ============================= ==========================
    """

    def __init__(self, proxy: Optional[ProxyComputer] = None) -> None:
        self._proxy = proxy or ProxyComputer()

    def map_command_execution(
        self,
        agent_id: str,
        sandbox_id: str,
        command: str,
        exit_code: int,
        agent_stats: Optional[Dict[str, int]] = None,
        provenance_id: Optional[str] = None,
        contract_id: Optional[str] = None,
    ) -> SoftInteraction:
        """Map a command execution to a SoftInteraction.

        Args:
            agent_id: The executing agent.
            sandbox_id: The sandbox container.
            command: The command string.
            exit_code: Process exit code.
            agent_stats: Running agent stats.
            provenance_id: Byline provenance ID.
            contract_id: Governing contract ID.

        Returns:
            A SoftInteraction with computed v_hat and p.
        """
        stats = agent_stats or {}
        observables = self._stats_to_observables(stats)
        v_hat, p = self._proxy.compute_labels(observables)

        return SoftInteraction(
            initiator="opensandbox_orchestrator",
            counterparty=agent_id,
            interaction_type=InteractionType.COLLABORATION,
            accepted=exit_code == 0,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=observables.tool_misuse_flags,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            metadata={
                "bridge": "opensandbox",
                "sandbox_id": sandbox_id,
                "command": command,
                "exit_code": exit_code,
                "provenance_id": provenance_id,
                "contract_id": contract_id,
            },
        )

    def map_message_exchange(
        self,
        from_agent: str,
        to_agent: str,
        delivered: bool,
        from_sandbox: str,
        to_sandbox: str,
        provenance_id: Optional[str] = None,
    ) -> SoftInteraction:
        """Map an inter-agent message exchange to a SoftInteraction.

        Args:
            from_agent: Sender agent ID.
            to_agent: Receiver agent ID.
            delivered: Whether the message was delivered.
            from_sandbox: Sender sandbox ID.
            to_sandbox: Receiver sandbox ID.
            provenance_id: Byline provenance ID.

        Returns:
            A SoftInteraction representing the exchange.
        """
        # Delivered messages have positive engagement
        engagement = 1.0 if delivered else -1.0
        observables = ProxyObservables(
            task_progress_delta=0.1 if delivered else 0.0,
            rework_count=0,
            verifier_rejections=0 if delivered else 1,
            tool_misuse_flags=0,
            counterparty_engagement_delta=engagement,
        )
        v_hat, p = self._proxy.compute_labels(observables)

        return SoftInteraction(
            initiator=from_agent,
            counterparty=to_agent,
            interaction_type=InteractionType.COLLABORATION,
            accepted=delivered,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=observables.tool_misuse_flags,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            metadata={
                "bridge": "opensandbox",
                "event": "message_exchange",
                "from_sandbox": from_sandbox,
                "to_sandbox": to_sandbox,
                "delivered": delivered,
                "provenance_id": provenance_id,
            },
        )

    def map_governance_intervention(
        self,
        agent_id: str,
        sandbox_id: str,
        reason: str,
        action: str,
        contract_id: Optional[str] = None,
        agent_stats: Optional[Dict[str, int]] = None,
    ) -> SoftInteraction:
        """Map a governance intervention to a SoftInteraction.

        Interventions carry negative proxy signals — they indicate
        contract violations or detected risk.

        Args:
            agent_id: The agent being intervened upon.
            sandbox_id: The agent's sandbox.
            reason: Reason for the intervention.
            action: Action taken (isolate, restrict, terminate).
            contract_id: Governing contract.
            agent_stats: Running agent stats.

        Returns:
            A SoftInteraction with low p reflecting the violation.
        """
        stats = agent_stats or {}
        observables = ProxyObservables(
            task_progress_delta=0.0,
            rework_count=stats.get("blocked_messages", 0),
            verifier_rejections=stats.get("interventions", 0) + 1,
            tool_misuse_flags=stats.get("violations", 0) + 1,
            counterparty_engagement_delta=-1.0,
        )
        v_hat, p = self._proxy.compute_labels(observables)

        return SoftInteraction(
            initiator="opensandbox_orchestrator",
            counterparty=agent_id,
            interaction_type=InteractionType.COLLABORATION,
            accepted=False,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=observables.tool_misuse_flags,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            metadata={
                "bridge": "opensandbox",
                "event": "governance_intervention",
                "sandbox_id": sandbox_id,
                "reason": reason,
                "action": action,
                "contract_id": contract_id,
            },
        )

    def map_sandbox_observation(
        self,
        agent_id: str,
        sandbox_id: str,
        agent_stats: Optional[Dict[str, int]] = None,
        contract_id: Optional[str] = None,
    ) -> SoftInteraction:
        """Map a periodic sandbox observation to a SoftInteraction.

        Used during polling to observe sandbox state without a
        specific trigger event. Observations are passive status checks,
        not acceptance/rejection outcomes.
        """
        stats = agent_stats or {}
        observables = self._stats_to_observables(stats)
        v_hat, p = self._proxy.compute_labels(observables)

        return SoftInteraction(
            initiator="opensandbox_orchestrator",
            counterparty=agent_id,
            interaction_type=InteractionType.COLLABORATION,
            accepted=False,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=observables.tool_misuse_flags,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            metadata={
                "bridge": "opensandbox",
                "event": "observation",
                "sandbox_id": sandbox_id,
                "contract_id": contract_id,
            },
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _stats_to_observables(stats: Dict[str, int]) -> ProxyObservables:
        """Convert running agent stats to ProxyObservables."""
        total_cmds = max(stats.get("total_commands", 0), 1)
        successful = stats.get("successful_commands", 0)
        progress = min(1.0, successful / total_cmds)

        total_msgs = max(stats.get("total_messages", 0), 1)
        delivered = stats.get("delivered_messages", 0)
        engagement = (delivered / total_msgs) * 2.0 - 1.0

        return ProxyObservables(
            task_progress_delta=progress,
            rework_count=stats.get("blocked_messages", 0),
            verifier_rejections=stats.get("interventions", 0),
            tool_misuse_flags=stats.get("violations", 0),
            counterparty_engagement_delta=engagement,
        )
