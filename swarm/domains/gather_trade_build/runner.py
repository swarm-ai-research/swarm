"""Scenario runner for the AI Economist GTB scenario.

Provides an end-to-end runner that:
1. Loads config from YAML
2. Initializes the GTB environment and agents
3. Runs the bilevel Planner-Workers loop
4. Collects metrics and events
5. Exports to runs/ directory
"""

from __future__ import annotations

import csv
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from swarm.domains.gather_trade_build.agents import (
    CollusiveWorkerPolicy,
    EvasiveWorkerPolicy,
    GamingWorkerPolicy,
    GTBWorkerPolicy,
    HonestWorkerPolicy,
)
from swarm.domains.gather_trade_build.config import GTBConfig
from swarm.domains.gather_trade_build.entities import ResourceType
from swarm.domains.gather_trade_build.env import GTBAction, GTBEnvironment
from swarm.domains.gather_trade_build.metrics import GTBMetrics, compute_gtb_metrics
from swarm.domains.gather_trade_build.planner import PlannerAgent

logger = logging.getLogger(__name__)


def _create_policy(
    agent_spec: Dict[str, Any],
    agent_id: str,
    seed: int,
) -> GTBWorkerPolicy:
    """Create a worker policy from a spec dict."""
    ptype = agent_spec.get("policy", "honest")
    if ptype == "honest":
        return HonestWorkerPolicy(agent_id, seed=seed)
    elif ptype == "gaming":
        return GamingWorkerPolicy(
            agent_id,
            shift_fraction=agent_spec.get("shift_fraction", 0.2),
            seed=seed,
        )
    elif ptype == "evasive":
        return EvasiveWorkerPolicy(
            agent_id,
            underreport_fraction=agent_spec.get("underreport_fraction", 0.3),
            seed=seed,
        )
    elif ptype == "collusive":
        return CollusiveWorkerPolicy(
            agent_id,
            coalition_id=agent_spec.get("coalition_id", "default"),
            seed=seed,
        )
    else:
        logger.warning("Unknown policy type '%s', defaulting to honest", ptype)
        return HonestWorkerPolicy(agent_id, seed=seed)


class GTBScenarioRunner:
    """End-to-end runner for the AI Economist GTB scenario."""

    def __init__(
        self,
        config: GTBConfig,
        agent_specs: List[Dict[str, Any]],
        n_epochs: int = 10,
        steps_per_epoch: int = 10,
        seed: Optional[int] = None,
    ) -> None:
        self._config = config
        self._n_epochs = n_epochs
        self._steps_per_epoch = steps_per_epoch
        self._seed = 42 if seed is None else seed
        self._rng = random.Random(self._seed)

        # Initialize environment
        self._env = GTBEnvironment(config)

        # Initialize workers
        self._policies: Dict[str, GTBWorkerPolicy] = {}
        for i, spec in enumerate(agent_specs):
            count = spec.get("count", 1)
            for j in range(count):
                agent_id = f"worker_{spec.get('policy', 'honest')}_{i}_{j}"
                worker_seed = self._rng.randint(0, 2**31)
                skill_g = spec.get("skill_gather", 1.0)
                skill_b = spec.get("skill_build", 1.0)
                self._env.add_worker(agent_id, skill_gather=skill_g,
                                     skill_build=skill_b)
                policy = _create_policy(spec, agent_id, worker_seed)
                self._policies[agent_id] = policy

                # Set coalition if collusive
                if spec.get("policy") == "collusive":
                    worker = self._env.workers[agent_id]
                    worker.coalition_id = spec.get("coalition_id", "default")

        # Initialize planner
        self._planner = PlannerAgent(
            config.planner,
            self._env.tax_schedule,
            seed=self._seed,
        )

        # Metrics storage
        self._epoch_metrics: List[GTBMetrics] = []
        self._all_events: List[dict] = []

    def run(self) -> List[GTBMetrics]:
        """Run the full scenario.

        Returns:
            List of per-epoch GTBMetrics.
        """
        logger.info(
            "Starting GTB scenario: %d epochs, %d steps/epoch, %d workers",
            self._n_epochs, self._steps_per_epoch, len(self._policies),
        )

        for epoch in range(self._n_epochs):
            epoch_events = []

            # Planner update at epoch boundary
            if self._planner.should_update(epoch):
                stats = self._env.get_aggregate_stats()
                new_brackets = self._planner.update(stats)
                logger.debug(
                    "Epoch %d: planner updated brackets -> %s",
                    epoch, [{"thr": b.threshold, "rate": b.rate} for b in new_brackets],
                )

            # Run steps within epoch
            for _step in range(self._steps_per_epoch):
                actions: Dict[str, GTBAction] = {}
                for agent_id, policy in self._policies.items():
                    obs = self._env.obs(agent_id)
                    actions[agent_id] = policy.decide(obs)

                step_events = self._env.apply_actions(actions)
                epoch_events.extend(step_events)

            # Collusion detection
            collusion_events = self._env.detect_collusion()
            epoch_events.extend(collusion_events)

            # End epoch: taxes, audits, then snapshot, then reset.
            # EpochResult contains both events and a pre-reset snapshot.
            epoch_result = self._env.end_epoch()
            epoch_events.extend(epoch_result.events)

            # Reset evasive worker epoch state
            for policy in self._policies.values():
                if isinstance(policy, EvasiveWorkerPolicy):
                    policy.reset_epoch()

            # Compute metrics from the pre-reset snapshot (includes
            # tax_paid and audit results but not yet zeroed)
            metrics = compute_gtb_metrics(
                workers=epoch_result.snapshot,
                events=epoch_events,
                epoch=epoch,
                bracket_thresholds=self._env.tax_schedule.bracket_thresholds,
                prod_weight=self._config.planner.prod_weight,
                ineq_weight=self._config.planner.ineq_weight,
            )
            self._epoch_metrics.append(metrics)

            # Store events
            for evt in epoch_events:
                self._all_events.append({
                    "event_type": evt.event_type,
                    "epoch": evt.epoch,
                    "step": evt.step,
                    "agent_id": evt.agent_id,
                    "details": evt.details,
                })

            logger.info(
                "Epoch %d: prod=%.2f gini=%.2f welfare=%.2f tax=%.2f "
                "audits=%d catches=%d bunching=%.3f",
                epoch, metrics.total_production, metrics.gini_coefficient,
                metrics.welfare, metrics.total_tax_revenue,
                metrics.total_audits, metrics.total_catches,
                metrics.bunching_intensity,
            )

        return self._epoch_metrics

    def export(self, output_dir: Optional[str] = None) -> Path:
        """Export results to a run directory.

        Args:
            output_dir: Override output directory. If None, creates
                        runs/<timestamp>_ai_economist_seed<seed>/.

        Returns:
            Path to the output directory.
        """
        if output_dir:
            raw = Path(output_dir)
            run_dir = raw.resolve()
            # Block relative paths that escape CWD (e.g. "../../etc")
            if not raw.is_absolute():
                cwd = Path.cwd().resolve()
                if not (run_dir == cwd or str(run_dir).startswith(str(cwd) + "/")):
                    raise ValueError(
                        f"Relative output directory resolves to {run_dir} "
                        f"which is outside {cwd}. Use an absolute path."
                    )
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = Path(f"runs/{timestamp}_ai_economist_seed{self._seed}")

        run_dir.mkdir(parents=True, exist_ok=True)
        csv_dir = run_dir / "csv"
        csv_dir.mkdir(exist_ok=True)

        # Export events as JSONL
        event_path = run_dir / "event_log.jsonl"
        with open(event_path, "w") as f:
            for evt in self._all_events:
                f.write(json.dumps(evt) + "\n")

        # Export metrics as CSV
        metrics_path = csv_dir / "metrics.csv"
        if self._epoch_metrics:
            fieldnames = list(self._epoch_metrics[0].to_dict().keys())
            with open(metrics_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for m in self._epoch_metrics:
                    writer.writerow(m.to_dict())

        # Export tax schedule history
        schedule_path = csv_dir / "tax_schedule.json"
        with open(schedule_path, "w") as f:
            json.dump(self._env.tax_schedule.to_dict(), f, indent=2)

        # Export final worker states
        workers_path = csv_dir / "workers.csv"
        workers = self._env.workers
        if workers:
            with open(workers_path, "w", newline="") as f:
                w_writer = csv.writer(f)
                w_writer.writerow([
                    "agent_id", "position", "coin", "wood", "stone",
                    "houses_built", "cumulative_income",
                    "times_audited", "times_caught", "total_fines",
                    "coalition_id",
                ])
                for aid, w in workers.items():
                    w_writer.writerow([
                        aid, str(w.position),
                        f"{w.get_resource(ResourceType.COIN):.2f}",
                        f"{w.get_resource(ResourceType.WOOD):.2f}",
                        f"{w.get_resource(ResourceType.STONE):.2f}",
                        w.houses_built, f"{w.cumulative_income:.2f}",
                        w.times_audited, w.times_caught,
                        f"{w.total_fines:.2f}",
                        w.coalition_id or "",
                    ])

        logger.info("Exported GTB results to %s", run_dir)
        return run_dir

    @property
    def env(self) -> GTBEnvironment:
        return self._env

    @property
    def planner(self) -> PlannerAgent:
        return self._planner

    @property
    def metrics(self) -> List[GTBMetrics]:
        return list(self._epoch_metrics)
