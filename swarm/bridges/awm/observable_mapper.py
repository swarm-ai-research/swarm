"""Maps AWM episode traces to SWARM ProxyObservables.

Converts tool-use interaction data into the standard observable signals
that the ProxyComputer uses to compute v_hat and p.
"""

from __future__ import annotations

from swarm.bridges.awm.mcp_client import AWMEpisodeTrace
from swarm.core.proxy import ProxyObservables


class AWMObservableMapper:
    """Converts AWMEpisodeTrace into ProxyObservables.

    Mapping:
    - task_progress_delta: verification pass -> +1.0, fail -> -0.5, unverified -> 0.0
    - rework_count: count of error responses from MCP tool calls
    - verifier_rejections: 0 if verified pass, 1 if fail or unverified
    - tool_misuse_flags: count of malformed tool calls (invalid args)
    - counterparty_engagement_delta: step efficiency 1 - (steps/max_steps)
    """

    def __init__(
        self,
        error_weight: float = 1.0,
        misuse_weight: float = 1.0,
    ) -> None:
        self.error_weight = error_weight
        self.misuse_weight = misuse_weight

    def map(self, trace: AWMEpisodeTrace) -> ProxyObservables:
        """Convert an episode trace to proxy observables.

        Args:
            trace: Complete episode trace with tool calls and verification

        Returns:
            ProxyObservables ready for ProxyComputer
        """
        # task_progress_delta from verification
        if trace.verified is True:
            task_progress = 1.0
        elif trace.verified is False:
            task_progress = -0.5
        else:
            # Not yet verified — neutral
            task_progress = 0.0

        # rework_count from error responses
        rework = int(trace.error_count * self.error_weight)

        # verifier_rejections: binary
        verifier_rej = 0 if trace.verified else 1

        # tool_misuse_flags from malformed calls
        misuse = int(trace.malformed_count * self.misuse_weight)

        # Step efficiency: fewer steps = more efficient
        if trace.max_steps > 0:
            efficiency = 1.0 - (trace.steps_used / trace.max_steps)
        else:
            efficiency = 0.0
        engagement = max(-1.0, min(1.0, efficiency))

        return ProxyObservables(
            task_progress_delta=task_progress,
            rework_count=rework,
            verifier_rejections=verifier_rej,
            tool_misuse_flags=misuse,
            counterparty_engagement_delta=engagement,
        )
