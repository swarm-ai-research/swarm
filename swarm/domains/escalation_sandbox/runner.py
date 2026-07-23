"""Scenario runner for the Escalation Sandbox.

Provides an end-to-end runner that:
1. Loads config from YAML
2. Initializes crisis environment and agents
3. Runs the turn-based crisis loop
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

from swarm.domains.escalation_sandbox.agents import (
    EscalationPolicy,
    create_policy,
)
from swarm.domains.escalation_sandbox.config import EscalationConfig
from swarm.domains.escalation_sandbox.entities import (
    EscalationAction,
)
from swarm.domains.escalation_sandbox.env import EscalationEnvironment
from swarm.domains.escalation_sandbox.metrics import (
    EscalationMetrics,
    compute_escalation_metrics,
)

logger = logging.getLogger(__name__)


class EscalationRunner:
    """End-to-end runner for the Escalation Sandbox scenario."""

    def __init__(
        self,
        config: EscalationConfig,
        seed: Optional[int] = None,
    ) -> None:
        self._config = config
        self._seed = seed if seed is not None else (config.seed or 42)
        self._rng = random.Random(self._seed)

        # Initialize environment with the resolved seed
        self._env = EscalationEnvironment(config, seed=self._seed)

        # Initialize agents and policies
        self._policies: Dict[str, EscalationPolicy] = {}
        for agent_cfg in config.agents:
            agent_id = agent_cfg.agent_id
            self._env.add_nation(
                agent_id=agent_id,
                name=agent_cfg.name,
                military_strength=agent_cfg.military_strength,
                economic_strength=agent_cfg.economic_strength,
                has_nuclear=agent_cfg.has_nuclear,
                has_second_strike=agent_cfg.has_second_strike,
                intelligence_quality=agent_cfg.intelligence_quality,
            )
            policy = create_policy(
                agent_id=agent_id,
                agent_type=agent_cfg.agent_type,
                persona=agent_cfg.persona,
                seed=self._rng.randint(0, 2**31),
                name=agent_cfg.name,
                provider=agent_cfg.provider,
                model_id=agent_cfg.model_id,
                temperature=agent_cfg.temperature,
            )
            self._policies[agent_id] = policy

        # Apply initial escalation if configured
        initial = config.crisis.initial_escalation
        if initial > 0:
            from swarm.domains.escalation_sandbox.entities import EscalationLevel
            clamped = max(0, min(9, initial))
            for nation in self._env.nations.values():
                nation.current_level = EscalationLevel(clamped)

        # Results
        self._metrics: Optional[EscalationMetrics] = None
        self._all_events: List[dict] = []

    def run(self) -> EscalationMetrics:
        """Run a complete crisis episode.

        Returns:
            EscalationMetrics for the episode.
        """
        logger.info(
            "Starting escalation scenario: template=%s, %d agents, max_turns=%d",
            self._config.crisis.template,
            len(self._policies),
            self._config.max_turns,
        )

        while not self._env.is_terminal():
            # Collect observations and decisions
            actions: Dict[str, EscalationAction] = {}
            for agent_id, policy in self._policies.items():
                obs = self._env.obs(agent_id)
                actions[agent_id] = policy.decide(obs)

            # Resolve turn
            turn_result = self._env.apply_actions(actions)

            # Log summary
            levels = dict(turn_result.realised_levels)
            logger.info(
                "Turn %d: levels=%s outcome=%s",
                turn_result.turn, levels, turn_result.outcome.value,
            )

            # Store events
            for evt in turn_result.events:
                self._all_events.append({
                    "event_type": evt.event_type,
                    "turn": evt.turn,
                    "agent_id": evt.agent_id,
                    "details": evt.details,
                })

        # Compute metrics
        self._metrics = compute_escalation_metrics(
            turn_results=self._env.turn_results,
            nations=self._env.nations,
            events=self._env.events,
            nuclear_threshold_turn=self._env.nuclear_threshold_turn,
            outcome=self._env.outcome.value,
        )

        logger.info(
            "Episode complete: outcome=%s, max_level=%d, nuclear_turn=%s, "
            "welfare=%.2f, divergence=%.3f",
            self._metrics.outcome,
            self._metrics.escalation_max,
            self._metrics.nuclear_threshold_turn,
            self._metrics.welfare_composite,
            self._metrics.signal_action_divergence,
        )

        return self._metrics

    def export(self, output_dir: Optional[str] = None) -> Path:
        """Export results to a run directory.

        Args:
            output_dir: Override output directory. If None, creates
                        runs/<timestamp>_escalation_seed<seed>/.

        Returns:
            Path to the output directory.
        """
        if output_dir:
            raw = Path(output_dir)
            run_dir = raw.resolve()
            # Block paths that escape CWD (both relative and absolute)
            cwd = Path.cwd().resolve()
            if not (run_dir == cwd or str(run_dir).startswith(str(cwd) + "/")):
                raise ValueError(
                    f"Output directory resolves to {run_dir} "
                    f"which is outside {cwd}. Use a path within the project."
                )
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = Path(f"runs/{timestamp}_escalation_seed{self._seed}")

        run_dir.mkdir(parents=True, exist_ok=True)
        csv_dir = run_dir / "csv"
        csv_dir.mkdir(exist_ok=True)

        # Export events as JSONL
        event_path = run_dir / "event_log.jsonl"
        with open(event_path, "w") as f:
            for evt in self._all_events:
                f.write(json.dumps(evt) + "\n")

        # Export metrics as CSV
        if self._metrics:
            metrics_path = csv_dir / "metrics.csv"
            m_dict = self._metrics.to_dict()
            with open(metrics_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(m_dict.keys()))
                writer.writeheader()
                writer.writerow(m_dict)

        # Export nation states
        nations_path = csv_dir / "nations.csv"
        nations = self._env.nations
        if nations:
            with open(nations_path, "w", newline="") as f:
                nation_writer = csv.writer(f)
                nation_writer.writerow([
                    "agent_id", "name", "final_level",
                    "military_remaining", "economic_remaining",
                    "population_welfare", "trust_score",
                    "signal_action_divergence",
                    "cumulative_military_damage",
                    "cumulative_economic_damage",
                    "cumulative_civilian_casualties",
                ])
                for aid, n in nations.items():
                    nation_writer.writerow([
                        aid, n.name, int(n.current_level),
                        f"{n.military_strength:.2f}",
                        f"{n.economic_strength:.2f}",
                        f"{n.population_welfare:.2f}",
                        f"{n.trust_score:.4f}",
                        f"{n.signal_action_divergence():.4f}",
                        f"{n.cumulative_military_damage:.2f}",
                        f"{n.cumulative_economic_damage:.2f}",
                        f"{n.cumulative_civilian_casualties:.2f}",
                    ])

        # Export escalation history (per-turn levels)
        history_path = csv_dir / "escalation_history.csv"
        turn_results = self._env.turn_results
        if turn_results:
            agent_ids = sorted(self._policies.keys())
            with open(history_path, "w", newline="") as f:
                hist_writer = csv.writer(f)
                header = ["turn", "outcome"]
                for aid in agent_ids:
                    header.extend([
                        f"{aid}_signal", f"{aid}_intended",
                        f"{aid}_realised", f"{aid}_fog_delta",
                    ])
                hist_writer.writerow(header)
                for tr in turn_results:
                    row: list[Any] = [tr.turn, tr.outcome.value]
                    for aid in agent_ids:
                        action = tr.actions.get(aid)
                        row.append(action.signal_level if action else "")
                        row.append(action.action_level if action else "")
                        row.append(tr.realised_levels.get(aid, ""))
                        row.append(tr.fog_deltas.get(aid, ""))
                    hist_writer.writerow(row)

        logger.info("Exported escalation results to %s", run_dir)
        return run_dir

    @property
    def env(self) -> EscalationEnvironment:
        return self._env

    @property
    def metrics(self) -> Optional[EscalationMetrics]:
        return self._metrics
