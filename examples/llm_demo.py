#!/usr/bin/env python
"""
Demo script for running simulations with LLM-backed agents.

This script demonstrates:
1. Creating LLM agents with different personas
2. Running async simulations for better performance
3. Tracking LLM API usage and costs
4. Mixing LLM and scripted agents

Prerequisites:
    python -m pip install -e ".[llm,runtime]"
    export ANTHROPIC_API_KEY="your-key"  # or OPENAI_API_KEY

Usage:
    python examples/llm_demo.py
    python examples/llm_demo.py --dry-run  # No API calls
    python examples/llm_demo.py --provider openai --model gpt-4o
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.agents.honest import HonestAgent
from swarm.agents.llm_agent import LLMAgent
from swarm.agents.llm_config import LLMConfig, LLMProvider, PersonaType
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig
from swarm.core.payoff import PayoffConfig
from swarm.governance.config import GovernanceConfig


def create_demo_orchestrator(
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-20250514",
    n_llm_agents: int = 2,
    n_scripted_agents: int = 2,
    dry_run: bool = False,
) -> Orchestrator:
    """
    Create an orchestrator with mixed LLM and scripted agents.

    Args:
        provider: LLM provider (anthropic, openai, ollama)
        model: Model identifier
        n_llm_agents: Number of LLM agents
        n_scripted_agents: Number of scripted agents
        dry_run: If True, don't make real API calls

    Returns:
        Configured Orchestrator
    """
    # Configuration
    config = OrchestratorConfig(
        n_epochs=5,
        steps_per_epoch=3,
        seed=42,
        payoff_config=PayoffConfig(
            s_plus=2.0,
            s_minus=1.0,
            h=2.0,
            theta=0.5,
        ),
        governance_config=GovernanceConfig(
            transaction_tax_rate=0.05,
            audit_enabled=True,
            audit_probability=0.1,
        ),
    )

    orchestrator = Orchestrator(config)

    # Create LLM agents
    llm_provider = LLMProvider(provider.lower())

    # Define mock_call once for dry-run (stateless and reusable)
    async def mock_call(*args, **kwargs):
        return ('{"action_type": "NOOP", "reasoning": "Dry run"}', 50, 20)

    for i in range(n_llm_agents):
        # Alternate between personas
        persona = PersonaType.OPEN if i % 2 == 0 else PersonaType.STRATEGIC

        llm_config = LLMConfig(
            provider=llm_provider,
            model=model,
            persona=persona,
            temperature=0.7,
            max_tokens=512,
            cost_tracking=True,
        )

        agent = LLMAgent(
            agent_id=f"llm_{i + 1}",
            llm_config=llm_config,
        )

        # For dry-run, mock the API calls
        if dry_run:
            agent._call_llm_async = mock_call

        orchestrator.register_agent(agent)

    # Create scripted agents for comparison
    for i in range(n_scripted_agents):
        agent = HonestAgent(agent_id=f"honest_{i + 1}")
        orchestrator.register_agent(agent)

    return orchestrator


async def run_demo(
    provider: str,
    model: str,
    dry_run: bool,
) -> None:
    """Run the demo simulation."""
    print("=" * 60)
    print("LLM Agent Demo - Distributional AGI Safety Sandbox")
    print("=" * 60)

    # Check API key
    if not dry_run:
        if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            print("Warning: ANTHROPIC_API_KEY not set. Use --dry-run or set the key.")
            return
        elif provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            print("Warning: OPENAI_API_KEY not set. Use --dry-run or set the key.")
            return

    print("\nConfiguration:")
    print(f"  Provider: {provider}")
    print(f"  Model: {model}")
    print(f"  Dry run: {dry_run}")
    print()

    # Create orchestrator
    orchestrator = create_demo_orchestrator(
        provider=provider,
        model=model,
        n_llm_agents=2,
        n_scripted_agents=2,
        dry_run=dry_run,
    )

    print(f"Registered {len(orchestrator._agents)} agents:")
    for agent_id, agent in orchestrator._agents.items():
        agent_type = type(agent).__name__
        if hasattr(agent, "llm_config"):
            print(f"  - {agent_id}: {agent_type} ({agent.llm_config.persona.value})")
        else:
            print(f"  - {agent_id}: {agent_type}")
    print()

    # Run simulation asynchronously
    print("Running simulation (async)...")
    print("-" * 40)

    metrics = await orchestrator.run_async()

    # Print results
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)

    for i, m in enumerate(metrics):
        print(f"\nEpoch {i + 1}:")
        print(f"  Total interactions: {m.total_interactions}")
        print(f"  Accepted: {m.accepted_interactions}")
        print(f"  Toxicity: {m.toxicity_rate:.3f}")
        print(f"  Total welfare: {m.total_welfare:.2f}")

    # Print LLM usage stats
    print("\n" + "-" * 40)
    print("LLM API USAGE")
    print("-" * 40)

    usage_stats = orchestrator.get_llm_usage_stats()
    total_cost = 0.0

    for agent_id, stats in usage_stats.items():
        print(f"\n{agent_id}:")
        print(f"  Requests: {stats['total_requests']}")
        print(f"  Input tokens: {stats['total_input_tokens']}")
        print(f"  Output tokens: {stats['total_output_tokens']}")
        print(f"  Estimated cost: ${stats['estimated_cost_usd']:.4f}")
        total_cost += stats["estimated_cost_usd"]

    print(f"\nTotal estimated cost: ${total_cost:.4f}")

    # Final agent states
    print("\n" + "-" * 40)
    print("FINAL AGENT STATES")
    print("-" * 40)

    for agent_id, state in orchestrator.state.agents.items():
        print(f"\n{agent_id}:")
        print(f"  Reputation: {state.reputation:.2f}")
        print(f"  Resources: {state.resources:.2f}")
        print(f"  Total payoff: {state.total_payoff:.2f}")
        print(f"  Interactions initiated: {state.interactions_initiated}")


def main():
    parser = argparse.ArgumentParser(description="LLM Agent Demo")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "ollama"],
        default="anthropic",
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Model identifier",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without making real API calls",
    )

    args = parser.parse_args()

    # Run async demo
    asyncio.run(
        run_demo(
            provider=args.provider,
            model=args.model,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
