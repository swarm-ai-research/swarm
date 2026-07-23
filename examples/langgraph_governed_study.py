#!/usr/bin/env python
"""LangGraph Governed Handoff Study — Runner Script.

Sweeps governance parameters across a 4-agent LLM-backed swarm and
exports metrics CSV + provenance JSONL.

Usage:
    # Dry run (validates setup, no LLM calls)
    python examples/langgraph_governed_study.py \\
        --scenario scenarios/langgraph_governed_handoff.yaml --dry-run

    # Single seed across all 32 configs
    python examples/langgraph_governed_study.py \\
        --scenario scenarios/langgraph_governed_handoff.yaml --seeds 1

    # Full sweep (96 runs)
    python examples/langgraph_governed_study.py \\
        --scenario scenarios/langgraph_governed_handoff.yaml

    # Run with Ollama (free, local)
    python examples/langgraph_governed_study.py \\
        --provider ollama --model llama3.2 --seeds 1

    # Run with OpenAI
    python examples/langgraph_governed_study.py \\
        --provider openai --model gpt-4o-mini --seeds 1

Requires ANTHROPIC_API_KEY (for anthropic provider) or OPENAI_API_KEY
(for openai provider) in environment. Ollama requires a running local server.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.bridges.langgraph_swarm.governed_swarm import (
    analyze_swarm_run,
)
from swarm.bridges.langgraph_swarm.study_agents import (
    build_study_agents,
)


def load_scenario(path: Path) -> dict:
    """Load and return the scenario YAML as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_param_grid(scenario: dict) -> list[dict]:
    """Build the full parameter grid from sweep config."""
    sweep = scenario["sweep"]
    keys = ["max_cycles", "max_handoffs", "trust_boundaries"]
    values = [sweep[k] for k in keys]
    grid = []
    for combo in itertools.product(*values):
        grid.append(dict(zip(keys, combo, strict=True)))
    return grid


def detect_task_completed(messages: list) -> tuple[bool, str]:
    """Check if the swarm produced a final answer.

    Returns (completed, final_agent).
    """
    for msg in reversed(messages):
        content = ""
        if hasattr(msg, "content"):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
        elif isinstance(msg, dict):
            content = msg.get("content", "")

        if "FINAL ANSWER:" in content:
            # Try to identify the agent
            agent_name = "unknown"
            if hasattr(msg, "name") and msg.name:
                agent_name = msg.name
            return True, agent_name

    return False, "none"


def run_single(
    *,
    config: dict,
    seed: int,
    scenario: dict,
    run_dir: Path,
    dry_run: bool = False,
) -> dict:
    """Run a single sweep configuration and return metrics."""
    llm_config = scenario.get("llm", {})
    model = llm_config.get("model", "claude-sonnet-4-20250514")
    max_tokens = llm_config.get("max_tokens", 300)
    max_turns = llm_config.get("max_turns", 25)
    task_prompt = scenario.get("task_prompt", "Summarize AI safety handoff risks.")

    provider = scenario.get("llm", {}).get("provider", "anthropic")
    base_url = scenario.get("llm", {}).get("base_url")

    # Build agents and policy
    agents, logger, policy = build_study_agents(
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        base_url=base_url,
        max_cycles=config["max_cycles"],
        max_handoffs=config["max_handoffs"],
        trust_boundaries=config["trust_boundaries"],
    )

    if dry_run:
        # Validate that agents and policy were built correctly
        return {
            "seed": seed,
            "max_cycles": config["max_cycles"],
            "max_handoffs": config["max_handoffs"],
            "trust_boundaries": config["trust_boundaries"],
            "dry_run": True,
            "agents_built": len(agents),
            "policy_type": type(policy).__name__,
            "status": "validated",
        }

    # Build the swarm graph
    from langgraph.checkpoint.memory import InMemorySaver

    from swarm.bridges.langgraph_swarm.governed_swarm import create_governed_swarm

    graph, logger = create_governed_swarm(
        agents,
        default_active_agent="coordinator",
        governance_policy=policy,
        provenance_logger=logger,
    )

    app = graph.compile(checkpointer=InMemorySaver())

    # Invoke the swarm
    thread_id = f"study-seed{seed}-mc{config['max_cycles']}-mh{config['max_handoffs']}-tb{config['trust_boundaries']}"
    invoke_config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": max_turns,
    }

    start_time = time.time()
    try:
        result = app.invoke(
            {
                "messages": [("user", task_prompt)],
                "active_agent": "coordinator",
                "provenance_chain": [],
                "risk_scores": {},
                "governance_log": [],
                "handoff_count": 0,
            },
            config=invoke_config,
        )
        elapsed = time.time() - start_time
        error = None
    except Exception as e:
        elapsed = time.time() - start_time
        result = {"messages": []}
        error = str(e)

    # Extract metrics
    messages = result.get("messages", [])
    analysis = analyze_swarm_run(logger)
    task_completed, final_agent = detect_task_completed(messages)

    # Per-agent handoff counts
    agent_handoff_counts = {}
    for record in logger.records:
        src = record.source_agent
        agent_handoff_counts[src] = agent_handoff_counts.get(src, 0) + 1

    # Export provenance for this run
    provenance_path = run_dir / "provenance_audit.jsonl"
    with open(provenance_path, "a") as f:
        for record in logger.records:
            entry = record.to_dict()
            entry["_sweep"] = {
                "seed": seed,
                "max_cycles": config["max_cycles"],
                "max_handoffs": config["max_handoffs"],
                "trust_boundaries": config["trust_boundaries"],
            }
            f.write(json.dumps(entry) + "\n")

    metrics = {
        "seed": seed,
        "max_cycles": config["max_cycles"],
        "max_handoffs": config["max_handoffs"],
        "trust_boundaries": config["trust_boundaries"],
        "total_handoffs": analysis.get("total_handoffs", 0),
        "approved_handoffs": analysis.get("approved_handoffs", 0),
        "denied_handoffs": analysis.get("denied_handoffs", 0),
        "escalated_handoffs": analysis.get("escalated_handoffs", 0),
        "modified_handoffs": analysis.get("modified_handoffs", 0),
        "max_chain_depth": analysis.get("max_chain_depth", 0),
        "cycle_pairs_count": len(analysis.get("cycle_pairs", [])),
        "denial_rate": analysis.get("denial_rate", 0.0),
        "risk_level": analysis.get("risk_level", "unknown"),
        "task_completed": task_completed,
        "final_agent": final_agent,
        "total_turns": len(messages),
        "elapsed_seconds": round(elapsed, 2),
        "coordinator_handoffs": agent_handoff_counts.get("coordinator", 0),
        "researcher_handoffs": agent_handoff_counts.get("researcher", 0),
        "writer_handoffs": agent_handoff_counts.get("writer", 0),
        "reviewer_handoffs": agent_handoff_counts.get("reviewer", 0),
        "error": error,
    }

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LangGraph Governed Handoff Study"
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("scenarios/langgraph_governed_handoff.yaml"),
        help="Path to scenario YAML",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Override seeds_per_config from scenario",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without making LLM calls",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="LLM provider: anthropic, ollama, openai (overrides YAML)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (overrides YAML)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for the LLM provider (overrides YAML)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory",
    )
    args = parser.parse_args()

    # Load scenario
    if not args.scenario.exists():
        print(f"Error: Scenario file not found: {args.scenario}")
        return 1

    scenario = load_scenario(args.scenario)

    # CLI overrides YAML values
    if args.provider is not None:
        scenario.setdefault("llm", {})["provider"] = args.provider
    if args.model is not None:
        scenario.setdefault("llm", {})["model"] = args.model
    if args.base_url is not None:
        scenario.setdefault("llm", {})["base_url"] = args.base_url

    provider = scenario.get("llm", {}).get("provider", "anthropic")

    print("=" * 60)
    print("LangGraph Governed Handoff Study")
    print(f"Scenario: {scenario['scenario_id']}")
    print(f"Provider: {provider}")
    print(f"Model:    {scenario.get('llm', {}).get('model', 'default')}")
    print("=" * 60)

    # Check API key (unless dry run or local provider)
    if not args.dry_run and provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set in environment")
        return 1
    if not args.dry_run and provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set in environment")
        return 1

    # Build parameter grid
    grid = build_param_grid(scenario)
    seeds_per_config = args.seeds or scenario["sweep"].get("seeds_per_config", 3)

    total_runs = len(grid) * seeds_per_config
    print(f"Parameter grid: {len(grid)} configs x {seeds_per_config} seeds = {total_runs} runs")
    if args.dry_run:
        print("[DRY RUN MODE — no LLM calls]")
    print()

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir or Path("runs") / f"{timestamp}_langgraph_governed"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save scenario copy
    with open(run_dir / "scenario.yaml", "w") as f:
        yaml.dump(scenario, f, default_flow_style=False)

    # Run sweep
    all_metrics: list[dict] = []
    run_idx = 0

    for config in grid:
        for seed_offset in range(seeds_per_config):
            seed = 42 + seed_offset
            run_idx += 1
            label = (
                f"[{run_idx}/{total_runs}] "
                f"mc={config['max_cycles']} mh={config['max_handoffs']} "
                f"tb={config['trust_boundaries']} seed={seed}"
            )
            print(f"{label} ... ", end="", flush=True)

            try:
                metrics = run_single(
                    config=config,
                    seed=seed,
                    scenario=scenario,
                    run_dir=run_dir,
                    dry_run=args.dry_run,
                )
                all_metrics.append(metrics)

                if args.dry_run:
                    print(f"OK ({metrics['status']})")
                else:
                    status = "DONE" if metrics["task_completed"] else "INCOMPLETE"
                    print(
                        f"{status} "
                        f"(handoffs={metrics['total_handoffs']}, "
                        f"denied={metrics['denied_handoffs']}, "
                        f"{metrics['elapsed_seconds']}s)"
                    )
            except Exception as e:
                print(f"ERROR: {e}")
                all_metrics.append({
                    "seed": seed,
                    **config,
                    "error": str(e),
                })

    # Export CSV
    csv_path = run_dir / scenario.get("outputs", {}).get("csv", "sweep_results.csv")
    if all_metrics:
        # Union of keys across rows: error rows carry a different shape
        # (config + "error") than success rows, and either may come first.
        fieldnames = list(dict.fromkeys(k for row in all_metrics for k in row))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            for row in all_metrics:
                writer.writerow(row)
        print(f"\nResults exported: {csv_path}")
        print(f"Total rows: {len(all_metrics)}")

    # Summary
    if not args.dry_run and all_metrics:
        completed = sum(1 for m in all_metrics if m.get("task_completed"))
        total = len(all_metrics)
        avg_denial = sum(m.get("denial_rate", 0) for m in all_metrics) / total
        print("\nSummary:")
        print(f"  Completion rate: {completed}/{total} ({completed/total:.1%})")
        print(f"  Avg denial rate: {avg_denial:.3f}")
        print(f"  Output dir: {run_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
