"""Red-team inspection: adversary reports, detection, and evasion metrics.

Extracted from ``Orchestrator`` to isolate red-team analysis into a
focused, testable component.  The orchestrator delegates to a
``RedTeamInspector`` instance, passing shared references to the agent
dict and environment state.
"""

from __future__ import annotations

from typing import Any, Dict

from swarm.env.state import EnvState
from swarm.models.agent import AgentType


class RedTeamInspector:
    """Queries and updates adversarial agents for red-team analysis.

    Responsibilities:
    - Collect strategy reports from adaptive adversaries
    - Notify adversaries of detection/penalty outcomes
    - Aggregate evasion metrics across all adversarial agents
    """

    def __init__(
        self,
        agents: Dict[str, Any],
        state: EnvState,
    ) -> None:
        self._agents = agents
        self._state = state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_adaptive_adversary_reports(self) -> Dict[str, Dict]:
        """Get strategy reports from all adaptive adversaries."""
        reports: Dict[str, Dict] = {}
        for agent_id, agent in self._agents.items():
            if hasattr(agent, "get_strategy_report"):
                reports[agent_id] = agent.get_strategy_report()
        return reports

    def notify_adversary_detection(
        self,
        agent_id: str,
        penalty: float = 0.0,
        detected: bool = True,
    ) -> None:
        """Notify an adaptive adversary of detection/penalty.

        This allows adversaries to learn from governance feedback.

        Args:
            agent_id: The agent that was detected
            penalty: Penalty amount applied
            detected: Whether the agent was detected
        """
        agent = self._agents.get(agent_id)
        if agent is not None and hasattr(agent, "update_adversary_outcome"):
            recent_payoff = 0.0
            if self._state.completed_interactions:
                agent_interactions = [
                    i
                    for i in self._state.completed_interactions
                    if i.initiator == agent_id or i.counterparty == agent_id
                ]
                if agent_interactions:
                    last = agent_interactions[-1]
                    if last.initiator == agent_id:
                        # payoff_initiator is computed on-demand by the payoff engine,
                        # not stored on SoftInteraction; accessing it via attribute lookup
                        # requires a type ignore since it's not in the class definition
                        recent_payoff = last.payoff_initiator or 0.0  # type: ignore[attr-defined]
                    else:
                        # payoff_counterparty is computed on-demand by the payoff engine,
                        # not stored on SoftInteraction; accessing it via attribute lookup
                        # requires a type ignore since it's not in the class definition
                        recent_payoff = last.payoff_counterparty or 0.0  # type: ignore[attr-defined]

            agent.update_adversary_outcome(
                payoff=recent_payoff,
                penalty=penalty,
                detected=detected,
            )

    def get_evasion_metrics(self) -> Dict:
        """Get evasion metrics for adversarial agents."""
        metrics: Dict[str, Any] = {
            "total_adversaries": 0,
            "adaptive_adversaries": 0,
            "avg_detection_rate": 0.0,
            "avg_heat_level": 0.0,
            "strategies_used": {},
            "by_agent": {},
        }

        detection_rates: list[float] = []
        heat_levels: list[float] = []

        for agent_id, agent in self._agents.items():
            agent_state = self._state.get_agent(agent_id)
            if agent_state and agent_state.agent_type == AgentType.ADVERSARIAL:
                metrics["total_adversaries"] += 1

                if hasattr(agent, "get_strategy_report"):
                    metrics["adaptive_adversaries"] += 1
                    report = agent.get_strategy_report()
                    metrics["by_agent"][agent_id] = report

                    for strategy, stats in report.get("strategy_stats", {}).items():
                        if strategy not in metrics["strategies_used"]:
                            metrics["strategies_used"][strategy] = {
                                "total_attempts": 0,
                                "total_detections": 0,
                            }
                        attempts = stats.get("attempts", 0)
                        detection_rate = stats.get("detection_rate", 0)
                        metrics["strategies_used"][strategy][
                            "total_attempts"
                        ] += attempts
                        metrics["strategies_used"][strategy][
                            "total_detections"
                        ] += int(attempts * detection_rate)

                    heat_levels.append(report.get("heat_level", 0))

                    total_attempts = sum(
                        s.get("attempts", 0)
                        for s in report.get("strategy_stats", {}).values()
                    )
                    total_detected = sum(
                        s.get("attempts", 0) * s.get("detection_rate", 0)
                        for s in report.get("strategy_stats", {}).values()
                    )
                    if total_attempts > 0:
                        detection_rates.append(total_detected / total_attempts)

        if detection_rates:
            metrics["avg_detection_rate"] = sum(detection_rates) / len(detection_rates)
        if heat_levels:
            metrics["avg_heat_level"] = sum(heat_levels) / len(heat_levels)

        return metrics
