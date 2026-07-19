#!/usr/bin/env python
"""
Unconditional Cooperation Window Sweep.

Research question:
    What is the minimum number of unconditional cooperation turns (N) that
    prevents escalation spirals? The scripted CalculatingPolicy's 5-turn
    unconditional cooperation window succeeded where no LLM persona could
    replicate this behavior. This study sweeps N from 1-10 to find the
    critical trust-building threshold.

Approach:
    Uses the Escalation Sandbox domain directly. For each cooperation window
    size N, one agent uses an "unconditional cooperator" policy that cooperates
    for N turns regardless of opponent behavior, then switches to tit-for-tat.
    The opponent uses a configurable policy (default: tit_for_tat).

    Tracks per-N:
      - Escalation rate (fraction of turns with mutual high levels)
      - Cooperation rate (fraction of turns with both agents at low levels)
      - Mean escalation level (proxy for trust)
      - Total welfare composite
      - Whether escalation spirals form (3+ consecutive mutual-escalation turns)

Usage:
    python examples/run_cooperation_window_study.py scenarios/ldt_cooperation.yaml
    python examples/run_cooperation_window_study.py scenarios/ldt_cooperation.yaml --window 5
    python examples/run_cooperation_window_study.py scenarios/ldt_cooperation.yaml --opponent hawk
    python examples/run_cooperation_window_study.py scenarios/ldt_cooperation.yaml --seed 42 --turns 30
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.domains.escalation_sandbox.agents import (
    EscalationPolicy,
    create_policy,
)
from swarm.domains.escalation_sandbox.config import (
    AgentConfig,
    CrisisConfig,
    EscalationConfig,
    FogOfWarConfig,
    GovernanceLeverConfig,
    SignalConfig,
)
from swarm.domains.escalation_sandbox.entities import EscalationAction
from swarm.domains.escalation_sandbox.env import EscalationEnvironment
from swarm.domains.escalation_sandbox.metrics import (
    compute_escalation_metrics,
)

# ======================================================================
# Unconditional Cooperator Policy
# ======================================================================


class UnconditionalCooperatorPolicy(EscalationPolicy):
    """Cooperates unconditionally for N turns, then switches to tit-for-tat.

    During the cooperation window:
      - Always signals and acts at level 0 (status quo), regardless of
        opponent behavior.

    After the window:
      - Mirrors the opponent's last realised level (tit-for-tat).

    This models the "cooperate for N turns regardless" instruction that
    scripted agents can follow but LLM personas struggle with.
    """

    def __init__(
        self,
        agent_id: str,
        cooperation_window: int = 5,
        seed: int = 42,
    ) -> None:
        super().__init__(agent_id, seed)
        self._cooperation_window = cooperation_window
        self._turn_count = 0

    def decide(self, observation: Dict[str, Any]) -> EscalationAction:
        self._turn_count += 1

        if self._turn_count <= self._cooperation_window:
            # Unconditional cooperation phase
            return EscalationAction(
                agent_id=self.agent_id,
                signal_level=0,
                action_level=0,
                reasoning=(
                    f"Unconditional cooperation: turn {self._turn_count}"
                    f"/{self._cooperation_window}"
                ),
            )

        # Post-window: tit-for-tat (mirror opponent)
        opponents = observation.get("opponents", {})
        if opponents:
            max_opp_level = max(
                opp.get("current_level", 0)
                for opp in opponents.values()
            )
        else:
            max_opp_level = 0

        return EscalationAction(
            agent_id=self.agent_id,
            signal_level=max_opp_level,
            action_level=max_opp_level,
            reasoning=(
                f"Post-cooperation TFT: mirror opponent level {max_opp_level}"
            ),
        )

    def reset(self) -> None:
        self._turn_count = 0


# ======================================================================
# Episode runner
# ======================================================================


def _build_config(
    opponent_persona: str = "tit_for_tat",
    max_turns: int = 20,
    seed: int = 42,
    fog_enabled: bool = False,
) -> EscalationConfig:
    """Build an EscalationConfig for a 2-agent cooperation study."""
    return EscalationConfig(
        agents=[
            AgentConfig(
                agent_id="cooperator",
                name="Cooperator",
                agent_type="scripted",
                persona="dove",  # placeholder; overridden by custom policy
            ),
            AgentConfig(
                agent_id="opponent",
                name="Opponent",
                agent_type="scripted",
                persona=opponent_persona,
            ),
        ],
        crisis=CrisisConfig(
            template="cooperation_window_study",
            timeline_turns=max_turns,
        ),
        fog_of_war=FogOfWarConfig(enabled=fog_enabled),
        signals=SignalConfig(
            broadcast_before_action=True,
            commitment_trap_enabled=False,
        ),
        governance=GovernanceLeverConfig(
            mad_enabled=False,
            circuit_breaker_enabled=False,
            mediation_enabled=False,
        ),
        seed=seed,
        max_turns=max_turns,
    )


def run_episode(
    cooperation_window: int,
    opponent_persona: str = "tit_for_tat",
    max_turns: int = 20,
    seed: int = 42,
    fog_enabled: bool = False,
) -> Dict[str, Any]:
    """Run a single episode with a given cooperation window N.

    Returns a dict of per-episode results.
    """
    config = _build_config(
        opponent_persona=opponent_persona,
        max_turns=max_turns,
        seed=seed,
        fog_enabled=fog_enabled,
    )

    env = EscalationEnvironment(config, seed=seed)

    # Add nations
    for agent_cfg in config.agents:
        env.add_nation(
            agent_id=agent_cfg.agent_id,
            name=agent_cfg.name,
            military_strength=agent_cfg.military_strength,
            economic_strength=agent_cfg.economic_strength,
            has_nuclear=agent_cfg.has_nuclear,
            has_second_strike=agent_cfg.has_second_strike,
            intelligence_quality=agent_cfg.intelligence_quality,
        )

    # Create policies
    cooperator_policy = UnconditionalCooperatorPolicy(
        agent_id="cooperator",
        cooperation_window=cooperation_window,
        seed=seed,
    )
    opponent_policy = create_policy(
        agent_id="opponent",
        agent_type="scripted",
        persona=opponent_persona,
        seed=seed + 1,
    )
    policies = {
        "cooperator": cooperator_policy,
        "opponent": opponent_policy,
    }

    # Run episode
    while not env.is_terminal():
        actions = {}
        for agent_id, policy in policies.items():
            obs = env.obs(agent_id)
            actions[agent_id] = policy.decide(obs)
        env.apply_actions(actions)

    # Compute metrics
    metrics = compute_escalation_metrics(
        turn_results=env.turn_results,
        nations=env.nations,
        events=env.events,
        nuclear_threshold_turn=env.nuclear_threshold_turn,
        outcome=env.outcome.value,
    )

    # Compute cooperation-specific metrics from turn results
    turn_results = env.turn_results
    n_turns = len(turn_results)

    # Escalation rate: fraction of turns where BOTH agents are at level >= 4
    escalation_turns = 0
    cooperation_turns = 0
    total_level_sum = 0
    spiral_detected = False
    consecutive_mutual_escalation = 0
    max_consecutive_escalation = 0

    for tr in turn_results:
        levels = tr.realised_levels
        coop_level = levels.get("cooperator", 0)
        opp_level = levels.get("opponent", 0)
        total_level_sum += coop_level + opp_level

        both_high = coop_level >= 4 and opp_level >= 4
        both_low = coop_level <= 1 and opp_level <= 1

        if both_high:
            escalation_turns += 1
            consecutive_mutual_escalation += 1
            max_consecutive_escalation = max(
                max_consecutive_escalation, consecutive_mutual_escalation
            )
        else:
            consecutive_mutual_escalation = 0

        if both_low:
            cooperation_turns += 1

    # Spiral: 3+ consecutive turns of mutual escalation
    spiral_detected = max_consecutive_escalation >= 3

    escalation_rate = escalation_turns / max(n_turns, 1)
    cooperation_rate = cooperation_turns / max(n_turns, 1)
    mean_level = total_level_sum / max(n_turns * 2, 1)

    return {
        "cooperation_window": cooperation_window,
        "opponent_persona": opponent_persona,
        "seed": seed,
        "n_turns": n_turns,
        "outcome": metrics.outcome,
        "escalation_rate": escalation_rate,
        "cooperation_rate": cooperation_rate,
        "mean_level": mean_level,
        "welfare_composite": metrics.welfare_composite,
        "escalation_max": metrics.escalation_max,
        "escalation_velocity": metrics.escalation_velocity,
        "signal_action_divergence": metrics.signal_action_divergence,
        "spiral_detected": spiral_detected,
        "max_consecutive_escalation": max_consecutive_escalation,
        "de_escalation_rate": metrics.de_escalation_rate,
        "collateral_damage": metrics.collateral_damage,
    }


# ======================================================================
# Main sweep
# ======================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Unconditional Cooperation Window Sweep",
    )
    parser.add_argument(
        "scenario",
        type=str,
        nargs="?",
        default=None,
        help="Path to scenario YAML (used for context; sweep uses its own config)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Run a single cooperation window N instead of sweeping 1-10",
    )
    parser.add_argument(
        "--min-window",
        type=int,
        default=1,
        help="Minimum cooperation window to sweep (default: 1)",
    )
    parser.add_argument(
        "--max-window",
        type=int,
        default=10,
        help="Maximum cooperation window to sweep (default: 10)",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="tit_for_tat",
        choices=["dove", "hawk", "tit_for_tat", "random", "calculating", "gradual"],
        help="Opponent policy persona (default: tit_for_tat)",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=20,
        help="Max turns per episode (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of seed-varied repeats per N (default: 5)",
    )
    parser.add_argument(
        "--fog",
        action="store_true",
        help="Enable fog-of-war noise",
    )
    args = parser.parse_args()

    print("=" * 78)
    print("Unconditional Cooperation Window Sweep")
    print("=" * 78)
    print()
    print(f"  Opponent policy:  {args.opponent}")
    print(f"  Max turns/episode: {args.turns}")
    print(f"  Base seed:         {args.seed}")
    print(f"  Repeats per N:     {args.repeats}")
    print(f"  Fog-of-war:        {'enabled' if args.fog else 'disabled'}")

    if args.scenario:
        scenario_path = Path(args.scenario)
        if scenario_path.exists():
            print(f"  Scenario context:  {scenario_path}")
        else:
            print(f"  Warning: scenario file not found: {scenario_path}")
    print()

    # Determine sweep range
    if args.window is not None:
        window_range = [args.window]
        print(f"  Single window:     N={args.window}")
    else:
        window_range = list(range(args.min_window, args.max_window + 1))
        print(f"  Sweep range:       N={args.min_window}-{args.max_window}")
    print()

    # --- Run sweep ---
    all_results: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for n in window_range:
        episode_results = []
        for r in range(args.repeats):
            seed = args.seed + r
            result = run_episode(
                cooperation_window=n,
                opponent_persona=args.opponent,
                max_turns=args.turns,
                seed=seed,
                fog_enabled=args.fog,
            )
            episode_results.append(result)
            all_results.append(result)

        # Aggregate across repeats
        n_episodes = len(episode_results)
        mean_esc_rate = sum(r["escalation_rate"] for r in episode_results) / n_episodes
        mean_coop_rate = sum(r["cooperation_rate"] for r in episode_results) / n_episodes
        mean_level = sum(r["mean_level"] for r in episode_results) / n_episodes
        mean_welfare = sum(r["welfare_composite"] for r in episode_results) / n_episodes
        spiral_frac = sum(1 for r in episode_results if r["spiral_detected"]) / n_episodes
        mean_max_esc = sum(r["escalation_max"] for r in episode_results) / n_episodes
        mean_collateral = sum(r["collateral_damage"] for r in episode_results) / n_episodes

        summary_rows.append({
            "N": n,
            "mean_escalation_rate": mean_esc_rate,
            "mean_cooperation_rate": mean_coop_rate,
            "mean_level": mean_level,
            "mean_welfare": mean_welfare,
            "spiral_fraction": spiral_frac,
            "mean_max_escalation": mean_max_esc,
            "mean_collateral": mean_collateral,
        })

    # --- Epoch-by-epoch table (here: N-by-N) ---
    print("Cooperation Window Sweep Results:")
    print("-" * 78)
    header = (
        f"{'N':<4} {'EscRate':<9} {'CoopRate':<10} {'MeanLvl':<9} "
        f"{'Welfare':<10} {'Spirals':<9} {'MaxEsc':<8} {'Collat':<8}"
    )
    print(header)
    print("-" * 78)

    for row in summary_rows:
        print(
            f"{row['N']:<4} "
            f"{row['mean_escalation_rate']:<9.4f} "
            f"{row['mean_cooperation_rate']:<10.4f} "
            f"{row['mean_level']:<9.4f} "
            f"{row['mean_welfare']:<10.2f} "
            f"{row['spiral_fraction']:<9.2f} "
            f"{row['mean_max_escalation']:<8.1f} "
            f"{row['mean_collateral']:<8.1f}"
        )

    print("-" * 78)
    print()

    # --- Summary Statistics ---
    print("Summary Statistics:")
    print("-" * 78)

    # Find critical N: smallest N with zero spiral fraction
    critical_n: Optional[int] = None
    for row in summary_rows:
        if row["spiral_fraction"] == 0.0:
            critical_n = row["N"]
            break

    if critical_n is not None:
        print(f"  Critical cooperation window: N={critical_n}")
        print("    (smallest N with no escalation spirals)")
    else:
        print("  No cooperation window fully prevented spirals in this sweep.")
    print()

    # Find N with best welfare
    best_welfare_row = max(summary_rows, key=lambda r: r["mean_welfare"])
    print(f"  Best welfare at N={best_welfare_row['N']}: "
          f"{best_welfare_row['mean_welfare']:.2f}")

    # Find N with lowest escalation rate
    lowest_esc_row = min(summary_rows, key=lambda r: r["mean_escalation_rate"])
    print(f"  Lowest escalation rate at N={lowest_esc_row['N']}: "
          f"{lowest_esc_row['mean_escalation_rate']:.4f}")

    # Find N with highest cooperation rate
    highest_coop_row = max(summary_rows, key=lambda r: r["mean_cooperation_rate"])
    print(f"  Highest cooperation rate at N={highest_coop_row['N']}: "
          f"{highest_coop_row['mean_cooperation_rate']:.4f}")
    print()

    # Comparison with CalculatingPolicy's N=5 baseline
    n5_row = next((r for r in summary_rows if r["N"] == 5), None)
    if n5_row is not None:
        print("  CalculatingPolicy baseline comparison (N=5):")
        print(f"    Escalation rate:  {n5_row['mean_escalation_rate']:.4f}")
        print(f"    Cooperation rate: {n5_row['mean_cooperation_rate']:.4f}")
        print(f"    Welfare:          {n5_row['mean_welfare']:.2f}")
        print(f"    Spiral fraction:  {n5_row['spiral_fraction']:.2f}")
        print()

    # --- Per-episode detail ---
    print("Per-Episode Details:")
    print("-" * 78)
    ep_header = (
        f"{'N':<4} {'Seed':<6} {'Outcome':<20} {'EscRate':<9} "
        f"{'CoopRate':<10} {'MeanLvl':<9} {'Spiral':<7}"
    )
    print(ep_header)
    print("-" * 78)
    for r in all_results:
        print(
            f"{r['cooperation_window']:<4} "
            f"{r['seed']:<6} "
            f"{r['outcome']:<20} "
            f"{r['escalation_rate']:<9.4f} "
            f"{r['cooperation_rate']:<10.4f} "
            f"{r['mean_level']:<9.4f} "
            f"{'YES' if r['spiral_detected'] else 'no':<7}"
        )
    print("-" * 78)
    print()

    # --- Export ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path("runs") / f"{timestamp}_cooperation_window_seed{args.seed}"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Export full results as JSON
    results_path = export_dir / "cooperation_window_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "study": "unconditional_cooperation_window_sweep",
                "config": {
                    "opponent_persona": args.opponent,
                    "max_turns": args.turns,
                    "base_seed": args.seed,
                    "repeats": args.repeats,
                    "fog_enabled": args.fog,
                    "window_range": list(window_range),
                },
                "summary": summary_rows,
                "episodes": all_results,
                "critical_n": critical_n,
            },
            f,
            indent=2,
        )

    # Export summary CSV
    csv_path = export_dir / "csv" / "sweep_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Exported results: {results_path}")
    print(f"Exported summary CSV: {csv_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
