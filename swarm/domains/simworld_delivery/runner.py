"""Scenario runner for the SimWorld Delivery domain.

Provides an end-to-end runner that:
1. Loads config from YAML
2. Initializes the delivery environment and agents
3. Runs the bidding/delivery loop
4. Collects metrics and events
5. Exports to runs/ directory
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from swarm.domains.simworld_delivery.agents import (
    AggressivePolicy,
    CautiousPolicy,
    ConscientiousPolicy,
    DeliveryPolicy,
    OpportunisticPolicy,
)
from swarm.domains.simworld_delivery.config import DeliveryConfig
from swarm.domains.simworld_delivery.entities import DeliveryAction, PersonaType
from swarm.domains.simworld_delivery.env import DeliveryEnvironment
from swarm.domains.simworld_delivery.metrics import (
    DeliveryMetrics,
    compute_delivery_metrics,
)

logger = logging.getLogger(__name__)

_POLICY_MAP: dict[str, type[DeliveryPolicy]] = {
    "conscientious": ConscientiousPolicy,
    "aggressive": AggressivePolicy,
    "cautious": CautiousPolicy,
    "opportunistic": OpportunisticPolicy,
}


def _create_policy(
    agent_spec: dict[str, Any],
    agent_id: str,
    seed: int,
) -> DeliveryPolicy:
    """Create a delivery policy from a spec dict."""
    ptype = agent_spec.get("policy", "conscientious")
    policy_cls = _POLICY_MAP.get(ptype)
    if policy_cls is None:
        logger.warning(
            "Unknown policy type '%s', defaulting to conscientious", ptype,
        )
        policy_cls = ConscientiousPolicy

    kwargs: dict[str, Any] = {"agent_id": agent_id, "seed": seed}
    if ptype == "aggressive":
        kwargs["scooter_priority"] = agent_spec.get("scooter_priority", True)

    result: DeliveryPolicy = policy_cls(**kwargs)
    return result


class DeliveryScenarioRunner:
    """End-to-end runner for the SimWorld Delivery scenario."""

    def __init__(
        self,
        config: DeliveryConfig,
        agent_specs: list[dict[str, Any]],
        n_epochs: int = 10,
        steps_per_epoch: int = 20,
        seed: int | None = None,
    ) -> None:
        self._config = config
        self._n_epochs = n_epochs
        self._steps_per_epoch = steps_per_epoch
        self._seed = seed if seed is not None else 42

        # Ensure environment uses the same seed for determinism
        config.seed = self._seed

        # Initialize environment
        self._env = DeliveryEnvironment(config)

        # Initialize agents
        self._policies: dict[str, DeliveryPolicy] = {}
        for i, spec in enumerate(agent_specs):
            count = spec.get("count", 1)
            policy_name = spec.get("policy", "conscientious")
            for j in range(count):
                agent_id = f"driver_{policy_name}_{i}_{j}"
                agent_seed = int(hashlib.sha256(f"{self._seed}:{agent_id}".encode()).hexdigest()[:8], 16)

                # Map policy name to persona
                persona_map = {
                    "conscientious": PersonaType.CONSCIENTIOUS,
                    "aggressive": PersonaType.AGGRESSIVE,
                    "cautious": PersonaType.CAUTIOUS,
                    "opportunistic": PersonaType.OPPORTUNISTIC,
                }
                persona = persona_map.get(policy_name, PersonaType.CONSCIENTIOUS)
                self._env.add_agent(agent_id, persona=persona)
                policy = _create_policy(spec, agent_id, agent_seed)
                self._policies[agent_id] = policy

        # Metrics storage
        self._epoch_metrics: list[DeliveryMetrics] = []
        self._event_log_path: Path | None = None

    def _append_events_jsonl(self, events: list[dict]) -> None:
        """Append events to JSONL file if logging is configured."""
        if self._event_log_path is None:
            return
        with open(self._event_log_path, "a") as f:
            for evt in events:
                f.write(json.dumps(evt) + "\n")

    def run(self, log_dir: str | None = None) -> list[DeliveryMetrics]:
        """Run the full scenario.

        Args:
            log_dir: Optional directory for streaming event log (JSONL).

        Returns:
            List of per-epoch DeliveryMetrics.
        """
        # Set up event logging if requested
        if log_dir is not None:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            self._event_log_path = log_path / "event_log.jsonl"
            # Clear if it exists
            self._event_log_path.write_text("")

        logger.info(
            "Starting delivery scenario: %d epochs, %d steps/epoch, %d agents",
            self._n_epochs, self._steps_per_epoch, len(self._policies),
        )

        for epoch in range(self._n_epochs):
            epoch_events = []

            # Generate orders for this epoch
            order_events = self._env.generate_orders()
            epoch_events.extend(order_events)

            # Run steps within epoch
            for _step in range(self._steps_per_epoch):
                actions: dict[str, DeliveryAction] = {}
                for agent_id, policy in self._policies.items():
                    obs = self._env.obs(agent_id)
                    actions[agent_id] = policy.decide(obs)

                step_events = self._env.apply_actions(actions)
                epoch_events.extend(step_events)

                # Generate a few more orders mid-epoch
                if _step == self._steps_per_epoch // 2:
                    mid_orders = self._env.generate_orders(
                        count=max(1, self._config.orders.orders_per_epoch // 4),
                    )
                    epoch_events.extend(mid_orders)

            # End epoch
            end_events = self._env.end_epoch()
            epoch_events.extend(end_events)

            # Compute metrics
            metrics = compute_delivery_metrics(
                agents=self._env.agents,
                events=epoch_events,
                epoch=epoch,
            )
            self._epoch_metrics.append(metrics)

            # Stream events to JSONL if logging
            event_dicts = [
                {
                    "event_type": evt.event_type,
                    "epoch": evt.epoch,
                    "step": evt.step,
                    "agent_id": evt.agent_id,
                    "details": evt.details,
                }
                for evt in epoch_events
            ]
            self._append_events_jsonl(event_dicts)

            logger.info(
                "Epoch %d: delivered=%d failed=%d rate=%.2f "
                "earnings=%.1f gini=%.3f overbid=%.2f rep=%.2f",
                epoch, metrics.orders_delivered, metrics.orders_failed,
                metrics.delivery_rate, metrics.total_earnings,
                metrics.earnings_gini, metrics.overbid_rate,
                metrics.mean_reputation,
            )

        return self._epoch_metrics

    def export(self, output_dir: str | None = None) -> Path:
        """Export results to a run directory.

        Args:
            output_dir: Override output directory. If None, creates
                        runs/<timestamp>_delivery_seed<seed>/.

        Returns:
            Path to the output directory.
        """
        if output_dir:
            raw = Path(output_dir)
            run_dir = raw.resolve()
            if not raw.is_absolute():
                cwd = Path.cwd().resolve()
                if not (run_dir == cwd or str(run_dir).startswith(str(cwd) + "/")):
                    raise ValueError(
                        f"Relative output directory resolves to {run_dir} "
                        f"which is outside {cwd}. Use an absolute path."
                    )
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = Path(f"runs/{timestamp}_delivery_seed{self._seed}")

        run_dir.mkdir(parents=True, exist_ok=True)
        csv_dir = run_dir / "csv"
        csv_dir.mkdir(exist_ok=True)

        # Export events as JSONL
        if self._event_log_path and self._event_log_path.exists():
            import shutil
            event_path = run_dir / "event_log.jsonl"
            shutil.copy(self._event_log_path, event_path)
        else:
            # Fallback: export from env if no log file was created
            event_path = run_dir / "event_log.jsonl"
            with open(event_path, "w") as f:
                for evt in self._env.events:
                    f.write(json.dumps({
                        "event_type": evt.event_type,
                        "epoch": evt.epoch,
                        "step": evt.step,
                        "agent_id": evt.agent_id,
                        "details": evt.details,
                    }) + "\n")

        # Export metrics as CSV
        metrics_path = csv_dir / "metrics.csv"
        if self._epoch_metrics:
            fieldnames = list(self._epoch_metrics[0].to_dict().keys())
            with open(metrics_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for m in self._epoch_metrics:
                    writer.writerow(m.to_dict())

        # Export final agent states
        agents_path = csv_dir / "agents.csv"
        agents = self._env.agents
        if agents:
            with open(agents_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "agent_id", "persona", "budget", "has_scooter",
                    "deliveries_completed", "deliveries_failed",
                    "total_earnings", "reputation", "overbids",
                    "idle_steps", "total_distance",
                ])
                for aid, a in agents.items():
                    w.writerow([
                        aid, a.persona.value, f"{a.budget:.2f}",
                        a.has_scooter, a.deliveries_completed,
                        a.deliveries_failed, f"{a.total_earnings:.2f}",
                        f"{a.reputation:.3f}", a.overbids,
                        a.idle_steps, f"{a.total_distance:.1f}",
                    ])

        logger.info("Exported delivery results to %s", run_dir)
        return run_dir

    @property
    def env(self) -> DeliveryEnvironment:
        return self._env

    @property
    def metrics(self) -> list[DeliveryMetrics]:
        return list(self._epoch_metrics)
