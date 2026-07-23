"""CLI entry point for the distributional-agi-safety framework.

Usage:
    python -m swarm run scenarios/baseline.yaml
    python -m swarm run scenarios/baseline.yaml --seed 42 --epochs 20
    python -m swarm run scenarios/baseline.yaml --export-json results.json
    python -m swarm run scenarios/baseline.yaml --export-csv output/
    python -m swarm list
"""

import argparse
import os
import sys
from pathlib import Path


def _validate_scenario_path(path: Path) -> int:
    """Validate that *path* exists, is a regular file and is readable.

    Returns 0 when the path is valid, 1 otherwise (a user-facing error
    message is printed to stderr in the failure case). Kept separate from
    ``cmd_run`` so it can be unit-tested in isolation.
    """
    if not path.exists():
        print(f"Error: scenario file '{path}' not found.", file=sys.stderr)
        return 1
    if not path.is_file():
        print(
            f"Error: scenario path '{path}' is not a regular file.",
            file=sys.stderr,
        )
        return 1
    if not os.access(path, os.R_OK):
        print(
            f"Error: scenario file '{path}' is not readable (check permissions).",
            file=sys.stderr,
        )
        return 1
    return 0


def _safe_export_path(raw: str, kind: str) -> Path:
    """Resolve *raw* to an absolute path and reject relative path traversal.

    Absolute paths are accepted as-is (the caller chose that location explicitly).
    Relative paths are resolved against the current working directory; if the
    result escapes the CWD via ``..`` components the call exits with an error.

    Raises SystemExit with an error message when the path is rejected.
    """
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    cwd = Path.cwd().resolve()
    resolved = (cwd / p).resolve()
    try:
        resolved.relative_to(cwd)
    except ValueError:
        print(
            f"Error: {kind} relative path '{raw}' resolves outside the working "
            f"directory ({cwd}). Path traversal is not allowed.",
            file=sys.stderr,
        )
        sys.exit(1)
    return resolved


def cmd_run(args: argparse.Namespace) -> int:
    """Run a simulation from a YAML scenario file."""
    import yaml

    from swarm.scenarios.loader import build_orchestrator, load_scenario

    scenario_path = Path(args.scenario)
    if (rc := _validate_scenario_path(scenario_path)) != 0:
        return rc

    # Load scenario — surface common configuration mistakes with a
    # CLI-friendly message instead of a noisy traceback.
    try:
        scenario = load_scenario(scenario_path)
    except yaml.YAMLError as exc:
        print(
            f"Error: scenario file '{scenario_path}' is not valid YAML: {exc}",
            file=sys.stderr,
        )
        return 1
    except (KeyError, ValueError, TypeError) as exc:
        print(
            f"Error: scenario file '{scenario_path}' has invalid configuration: "
            f"{exc}",
            file=sys.stderr,
        )
        return 1

    # Required config keys: at minimum, an `agents:` list with one entry is
    # needed for the simulation to produce any behaviour. We surface this
    # explicitly so users don't see a silent "ran 0 epochs, 0 agents" result.
    if not scenario.agent_specs:
        print(
            f"Error: scenario file '{scenario_path}' has no 'agents' entries. "
            "Add an 'agents:' list with at least one agent spec.",
            file=sys.stderr,
        )
        return 1

    # In constrained environments, scenario-configured event log paths
    # may not be writable. Fall back to in-memory-only execution.
    if scenario.orchestrator_config.log_path is not None:
        log_parent = Path(scenario.orchestrator_config.log_path).parent
        if not os.access(log_parent, os.W_OK):
            scenario.orchestrator_config.log_path = None
            scenario.orchestrator_config.log_events = False

    # Apply CLI overrides
    if args.seed is not None:
        scenario.orchestrator_config.seed = args.seed
    if args.epochs is not None:
        scenario.orchestrator_config.n_epochs = args.epochs
    if args.steps is not None:
        scenario.orchestrator_config.steps_per_epoch = args.steps

    if not args.quiet:
        print("=" * 60)
        print("Distributional AGI Safety Sandbox")
        print("=" * 60)
        print(f"Scenario:    {scenario.scenario_id}")
        print(f"Description: {scenario.description}")
        print(f"Epochs:      {scenario.orchestrator_config.n_epochs}")
        print(f"Steps/epoch: {scenario.orchestrator_config.steps_per_epoch}")
        print(f"Seed:        {scenario.orchestrator_config.seed}")
        print()

    # Build and run
    orchestrator = build_orchestrator(scenario)

    # CLI override: prompt audit logging for LLM agents
    if args.prompt_audit is not None:
        for agent in orchestrator.get_all_agents():
            llm_config = getattr(agent, "llm_config", None)
            if llm_config is None:
                continue
            if not hasattr(llm_config, "prompt_audit_path"):
                continue
            llm_config.prompt_audit_path = args.prompt_audit
            llm_config.prompt_audit_include_system_prompt = bool(
                args.prompt_audit_include_system
            )
            llm_config.prompt_audit_hash_system_prompt = True

    # Wire MetricsAggregator for rich export data
    want_rich_export = args.export_json or args.export_csv or args.export_dolt
    aggregator = None
    if want_rich_export:
        from swarm.analysis.aggregation import MetricsAggregator

        aggregator = MetricsAggregator()
        aggregator.start_simulation(
            simulation_id=scenario.scenario_id,
            n_epochs=scenario.orchestrator_config.n_epochs,
            steps_per_epoch=scenario.orchestrator_config.steps_per_epoch,
            n_agents=len(orchestrator.get_all_agents()),
            seed=scenario.orchestrator_config.seed,
        )

        # Record each interaction + payoffs
        def _on_interaction(interaction, initiator_payoff, counterparty_payoff):
            # Tag with epoch/step for viz event export
            interaction.metadata["epoch"] = orchestrator.state.current_epoch
            interaction.metadata["step"] = orchestrator.state.current_step
            aggregator.record_interaction(interaction)
            aggregator.record_payoff(interaction.initiator, initiator_payoff)
            aggregator.record_payoff(interaction.counterparty, counterparty_payoff)
            # Store interactions for event export
            aggregator.get_history().interactions.append(interaction)

        orchestrator.on_interaction_complete(_on_interaction)

        # Finalize each epoch with full agent states
        def _on_epoch_end(epoch_metrics):
            agent_states = orchestrator.state.agents
            frozen = orchestrator.state.frozen_agents
            quarantined = getattr(orchestrator.state, 'quarantined_agents', set())
            aggregator.finalize_epoch(
                epoch=epoch_metrics.epoch,
                agent_states=agent_states,
                frozen_agents=frozen,
                quarantined_agents=quarantined,
            )

        orchestrator.on_epoch_end(_on_epoch_end)

    if not args.quiet:
        print(f"Agents: {len(orchestrator.get_all_agents())}")
        for agent in orchestrator.get_all_agents():
            print(f"  - {agent.agent_id} ({agent.agent_type.value})")
        print()
        print("Running simulation...")
        print("-" * 60)

    metrics_history = orchestrator.run()

    # Print results
    if not args.quiet:
        print()
        print(
            f"{'Epoch':<6} {'Interactions':<13} {'Accepted':<10} {'Toxicity':<10} "
            f"{'Surplus':<10} {'Net Welf':<10}"
        )
        print("-" * 70)
        for m in metrics_history:
            print(
                f"{m.epoch:<6} "
                f"{m.total_interactions:<13} "
                f"{m.accepted_interactions:<10} "
                f"{m.toxicity_rate:<10.4f} "
                f"{m.total_welfare:<10.2f} "
                f"{m.net_social_welfare:<10.2f}"
            )
        print("-" * 70)
        print()

    # Summary
    if metrics_history:
        total_interactions = sum(m.total_interactions for m in metrics_history)
        total_accepted = sum(m.accepted_interactions for m in metrics_history)
        avg_toxicity = sum(m.toxicity_rate for m in metrics_history) / len(
            metrics_history
        )
        avg_net_welfare = sum(
            m.net_social_welfare for m in metrics_history
        ) / len(metrics_history)

        if not args.quiet:
            print(f"Total interactions: {total_interactions}")
            print(f"Accepted:          {total_accepted}")
            print(f"Avg toxicity:      {avg_toxicity:.4f}")
            print(f"Avg surplus:       {sum(m.total_welfare for m in metrics_history) / len(metrics_history):.2f}")
            print(f"Avg net welfare:   {avg_net_welfare:.2f}  (surplus - externalities)")

            # Contract screening summary
            last_contract = metrics_history[-1].contract_metrics
            if last_contract:
                print()
                print("Contract Screening Metrics:")
                print(f"  Separation quality:  {last_contract.get('separation_quality', 0):.4f}")
                print(f"  Infiltration rate:   {last_contract.get('infiltration_rate', 0):.4f}")
                print(f"  Welfare delta:       {last_contract.get('welfare_delta', 0):.4f}")
                print(f"  Attack displacement: {last_contract.get('attack_displacement', 0):.4f}")
                pool_q = last_contract.get("pool_avg_quality", {})
                if pool_q:
                    print("  Pool avg quality:")
                    for pool, q in sorted(pool_q.items()):
                        print(f"    {pool}: {q:.4f}")
                pool_sizes = last_contract.get("pool_sizes", {})
                if pool_sizes:
                    print("  Pool sizes:")
                    for pool, n in sorted(pool_sizes.items()):
                        print(f"    {pool}: {n}")

            print()

    # Export if requested
    if want_rich_export and aggregator is not None:
        history = aggregator.end_simulation()

        if args.export_json:
            from swarm.analysis.export import export_to_json

            safe_json = _safe_export_path(args.export_json, "--export-json")
            path = export_to_json(history, str(safe_json), include_events=True)
            print(f"Exported JSON: {path}")

        if args.export_csv:
            from swarm.analysis.export import export_to_csv

            safe_csv = _safe_export_path(args.export_csv, "--export-csv")
            paths = export_to_csv(history, str(safe_csv), prefix=scenario.scenario_id)
            for kind, path in paths.items():
                print(f"Exported CSV ({kind}): {path}")

        if args.export_dolt:
            try:
                from swarm.analysis.dolt_export import export_to_dolt

                # Resolve Dolt dir relative to repo root (swarm package parent)
                dolt_dir = Path(args.export_dolt)
                if not dolt_dir.is_absolute():
                    repo_root = Path(__file__).resolve().parent.parent
                    dolt_dir = repo_root / dolt_dir
                # Derive a run_dir name from scenario + seed
                seed = scenario.orchestrator_config.seed
                run_dir_name = f"{scenario.scenario_id}_seed{seed}"
                export_to_dolt(history, run_dir_name, str(dolt_dir))
            except Exception as exc:
                print(f"Warning: Dolt export failed: {exc}")

    # Optional standard plot bundle (toxicity/welfare + selection geometry).
    if args.plots is not None:
        if args.plots:
            plots_dir = _safe_export_path(args.plots, "--plots")
        elif args.export_json:
            plots_dir = _safe_export_path(args.export_json, "--export-json").parent
        elif args.export_csv:
            plots_dir = _safe_export_path(args.export_csv, "--export-csv")
        else:
            seed = scenario.orchestrator_config.seed
            plots_dir = Path("runs") / f"{scenario.scenario_id}_seed{seed}"

        from swarm.analysis.run_plots import write_run_plots

        written = write_run_plots(
            metrics_history, plots_dir, scenario_id=scenario.scenario_id,
        )
        for p in written:
            print(f"Plot: {p}")

    # Check success criteria
    if scenario.success_criteria and not args.quiet:
        _check_criteria(scenario.success_criteria, metrics_history)

    return 0


def _check_criteria(criteria: dict, metrics_history: list) -> None:
    """Print success criteria results."""
    print("Success Criteria:")
    print("-" * 60)

    avg_toxicity = (
        sum(m.toxicity_rate for m in metrics_history) / len(metrics_history)
        if metrics_history
        else 0
    )

    all_passed = True

    if "min_epochs" in criteria:
        passed = len(metrics_history) >= criteria["min_epochs"]
        tag = "[PASS]" if passed else "[FAIL]"
        print(f"  {tag} Epochs: {len(metrics_history)} >= {criteria['min_epochs']}")
        all_passed = all_passed and passed

    if "min_agents" in criteria:
        # Can't check without orchestrator reference; skip
        pass

    if "toxicity_threshold" in criteria:
        passed = avg_toxicity <= criteria["toxicity_threshold"]
        tag = "[PASS]" if passed else "[FAIL]"
        print(
            f"  {tag} Toxicity: {avg_toxicity:.4f} <= {criteria['toxicity_threshold']}"
        )
        all_passed = all_passed and passed

    if "min_net_social_welfare" in criteria:
        avg_net_welfare = (
            sum(m.net_social_welfare for m in metrics_history) / len(metrics_history)
            if metrics_history
            else 0
        )
        passed = avg_net_welfare >= criteria["min_net_social_welfare"]
        tag = "[PASS]" if passed else "[FAIL]"
        print(
            f"  {tag} Net social welfare: {avg_net_welfare:.4f} >= "
            f"{criteria['min_net_social_welfare']}"
        )
        all_passed = all_passed and passed

    # Contract screening criteria (use last epoch's contract metrics)
    last_contract = None
    if metrics_history:
        last_contract = metrics_history[-1].contract_metrics

    if "min_separation_quality" in criteria:
        val = last_contract.get("separation_quality", 0.0) if last_contract else 0.0
        passed = val >= criteria["min_separation_quality"]
        tag = "[PASS]" if passed else "[FAIL]"
        print(f"  {tag} Separation quality: {val:.4f} >= {criteria['min_separation_quality']}")
        all_passed = all_passed and passed

    if "max_infiltration_rate" in criteria:
        val = last_contract.get("infiltration_rate", 1.0) if last_contract else 1.0
        passed = val <= criteria["max_infiltration_rate"]
        tag = "[PASS]" if passed else "[FAIL]"
        print(f"  {tag} Infiltration rate: {val:.4f} <= {criteria['max_infiltration_rate']}")
        all_passed = all_passed and passed

    if "min_welfare_delta" in criteria:
        val = last_contract.get("welfare_delta", 0.0) if last_contract else 0.0
        passed = val >= criteria["min_welfare_delta"]
        tag = "[PASS]" if passed else "[FAIL]"
        print(f"  {tag} Welfare delta: {val:.4f} >= {criteria['min_welfare_delta']}")
        all_passed = all_passed and passed

    if "min_attack_displacement" in criteria:
        val = last_contract.get("attack_displacement", 0.0) if last_contract else 0.0
        passed = val >= criteria["min_attack_displacement"]
        tag = "[PASS]" if passed else "[FAIL]"
        print(f"  {tag} Attack displacement: {val:.4f} >= {criteria['min_attack_displacement']}")
        all_passed = all_passed and passed

    print()
    print(f"Result: {'ALL CRITERIA PASSED' if all_passed else 'SOME CRITERIA FAILED'}")


def cmd_evolve(args: argparse.Namespace) -> int:
    """Run evolutionary governance search."""
    from swarm.analysis.evolver import EvolverConfig, run_evolution
    from swarm.scenarios.loader import load_scenario

    scenario_path = Path(args.scenario)
    if not scenario_path.exists():
        print(f"Error: scenario file not found: {scenario_path}", file=sys.stderr)
        return 1

    scenario = load_scenario(scenario_path)

    # Disable log paths for evolution runs
    scenario.orchestrator_config.log_path = None
    scenario.orchestrator_config.log_events = False

    config = EvolverConfig(
        base_scenario=scenario,
        n_iterations=args.iterations,
        num_parents_per_iteration=args.parents,
        eval_epochs=args.eval_epochs,
        eval_steps=args.eval_steps,
        final_eval_epochs=args.final_eval_epochs,
        final_eval_steps=args.final_eval_steps,
        use_llm_mutator=args.use_llm,
        llm_model=args.llm_model,
        seed_base=args.seed,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        resume_from=Path(args.resume_from) if args.resume_from else None,
    )

    print("=" * 60)
    print("Evolutionary Governance Search")
    print("=" * 60)
    print(f"Scenario:    {scenario.scenario_id}")
    print(f"Iterations:  {config.n_iterations}")
    print(f"Eval:        {config.eval_epochs} epochs x {config.eval_steps} steps")
    print(f"LLM mutator: {'yes' if config.use_llm_mutator else 'no'}")
    print()

    run_evolution(config)
    return 0


def cmd_autoresearch(args: argparse.Namespace) -> int:
    """Run an automated objective -> mutate -> evaluate loop."""
    from swarm.analysis.autoresearch import cmd_autoresearch as _run_autoresearch

    return _run_autoresearch(args)


def cmd_list(args: argparse.Namespace) -> int:
    """List available scenario files."""
    scenarios_dir = Path(args.dir)
    if not scenarios_dir.is_dir():
        print(f"Error: directory not found: {scenarios_dir}", file=sys.stderr)
        return 1

    files = sorted(scenarios_dir.glob("*.yaml"))
    if not files:
        print(f"No YAML scenarios found in {scenarios_dir}")
        return 0

    print(f"Available scenarios in {scenarios_dir}/:")
    for f in files:
        print(f"  {f}")

    return 0


def cmd_agents(args: argparse.Namespace) -> int:
    """List registered agent types (subclasses of BaseAgent)."""
    import inspect

    from swarm import agents as agents_pkg
    from swarm.agents.base import BaseAgent

    discovered: list[tuple[str, type]] = []
    seen: set[type] = set()
    for name in getattr(agents_pkg, "__all__", []):
        obj = getattr(agents_pkg, name, None)
        if (
            obj is not None
            and inspect.isclass(obj)
            and issubclass(obj, BaseAgent)
            and obj is not BaseAgent
            and obj not in seen
        ):
            discovered.append((name, obj))
            seen.add(obj)

    discovered.sort(key=lambda item: item[0].lower())

    if not discovered:
        print("No agent types discovered.", file=sys.stderr)
        return 1

    if args.verbose:
        for name, cls in discovered:
            doc = inspect.getdoc(cls) or "(no docstring)"
            print(f"{name}")
            print(f"  module: {cls.__module__}")
            for line in doc.splitlines():
                print(f"  {line}" if line else "")
            print()
        return 0

    name_width = max(len(name) for name, _ in discovered)
    print(f"{'AGENT'.ljust(name_width)}  DESCRIPTION")
    print(f"{'-' * name_width}  {'-' * 11}")
    for name, cls in discovered:
        doc = inspect.getdoc(cls) or "(no docstring)"
        first_line = doc.splitlines()[0]
        print(f"{name.ljust(name_width)}  {first_line}")

    return 0


def cmd_sandbox(args: argparse.Namespace) -> int:
    """Handle sandbox subcommands."""
    from swarm.bridges.worktree.__main__ import (
        cmd_create,
        cmd_destroy,
        cmd_exec,
        cmd_gc,
        cmd_status,
    )
    from swarm.bridges.worktree.__main__ import (
        cmd_list as sandbox_list,
    )

    handlers = {
        "create": cmd_create,
        "destroy": cmd_destroy,
        "list": sandbox_list,
        "exec": cmd_exec,
        "status": cmd_status,
        "gc": cmd_gc,
    }

    subcmd = args.sandbox_cmd
    if subcmd in handlers:
        return int(handlers[subcmd](args))

    print(f"Unknown sandbox subcommand: {subcmd}", file=sys.stderr)
    return 1


def cmd_agentrxiv(args: argparse.Namespace) -> int:
    """Handle AgentRxiv subcommands."""
    subcmd = args.agentrxiv_cmd

    if subcmd == "start":
        return _agentrxiv_start(args)
    elif subcmd == "search":
        return _agentrxiv_search(args)
    elif subcmd == "submit":
        return _agentrxiv_submit(args)
    elif subcmd == "status":
        return _agentrxiv_status(args)
    else:
        print(f"Unknown agentrxiv subcommand: {subcmd}", file=sys.stderr)
        return 1


def _agentrxiv_start(args: argparse.Namespace) -> int:
    """Start the AgentRxiv server."""
    from swarm.research.agentrxiv_server import AgentRxivServer, AgentRxivServerError

    port = getattr(args, "port", 5000) or 5000
    uploads_dir = getattr(args, "uploads_dir", "./agentrxiv_papers") or "./agentrxiv_papers"

    print(f"Starting AgentRxiv server on port {port}...")

    try:
        server = AgentRxivServer(port=port, uploads_dir=uploads_dir)
        server.start(wait=True)
        print(f"AgentRxiv server running at {server.base_url}")
        print("Press Ctrl+C to stop...")

        try:
            import time
            while server.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            server.stop()
            print("Server stopped.")
        return 0

    except AgentRxivServerError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _agentrxiv_search(args: argparse.Namespace) -> int:
    """Search papers on AgentRxiv."""
    from swarm.research.platforms import AgentRxivClient

    base_url = getattr(args, "url", None)
    client = AgentRxivClient(base_url=base_url)

    if not client.health_check():
        print("Error: AgentRxiv server not running", file=sys.stderr)
        print("Start it with: python -m swarm agentrxiv start", file=sys.stderr)
        return 1

    query = args.query
    limit = getattr(args, "limit", 5) or 5

    print(f"Searching AgentRxiv for: {query}")
    print("-" * 60)

    result = client.search(query, limit=limit)

    if not result.papers:
        print("No papers found.")
        return 0

    for i, paper in enumerate(result.papers, 1):
        print(f"\n{i}. {paper.title}")
        print(f"   ID: {paper.paper_id}")
        if paper.abstract:
            abstract = paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract
            print(f"   Abstract: {abstract}")

    print(f"\nTotal results: {result.total_count}")
    return 0


def _agentrxiv_submit(args: argparse.Namespace) -> int:
    """Submit a paper to AgentRxiv."""
    from swarm.research.platforms import AgentRxivClient, Paper

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}", file=sys.stderr)
        return 1

    base_url = getattr(args, "url", None)
    client = AgentRxivClient(base_url=base_url)

    if not client.health_check():
        print("Error: AgentRxiv server not running", file=sys.stderr)
        print("Start it with: python -m swarm agentrxiv start", file=sys.stderr)
        return 1

    title = getattr(args, "title", None) or pdf_path.stem.replace("_", " ")

    print(f"Uploading: {pdf_path}")
    print(f"Title: {title}")

    paper = Paper(title=title, abstract="")
    result = client.submit(paper, pdf_path=str(pdf_path))

    if result.success:
        print(f"Success! Paper ID: {result.paper_id}")
        print("Triggering index update...")
        client.trigger_update()
        print("Done.")
        return 0
    else:
        print(f"Error: {result.message}", file=sys.stderr)
        return 1


def _agentrxiv_status(args: argparse.Namespace) -> int:
    """Check AgentRxiv server status."""
    from swarm.research.platforms import AgentRxivClient

    base_url = getattr(args, "url", None)
    client = AgentRxivClient(base_url=base_url)

    print(f"Checking AgentRxiv at {client.base_url}...")

    if client.health_check():
        print("Status: RUNNING")
        return 0
    else:
        print("Status: NOT RUNNING")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m swarm",
        description="Distributional AGI Safety Sandbox",
    )
    subparsers = parser.add_subparsers(dest="command")

    # run
    run_parser = subparsers.add_parser("run", help="Run a simulation scenario")
    run_parser.add_argument("scenario", help="Path to YAML scenario file")
    run_parser.add_argument(
        "--seed", type=int, default=None, help="Override random seed"
    )
    run_parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of epochs"
    )
    run_parser.add_argument(
        "--steps", type=int, default=None, help="Override steps per epoch"
    )
    run_parser.add_argument(
        "--export-json", metavar="PATH", help="Export results to JSON file"
    )
    run_parser.add_argument(
        "--export-csv", metavar="DIR", help="Export results to CSV directory"
    )
    run_parser.add_argument(
        "--export-dolt",
        nargs="?",
        const="runs/dolt_runs",
        default=None,
        metavar="DIR",
        help="Export results to Dolt repo (default: runs/dolt_runs)",
    )
    run_parser.add_argument(
        "--prompt-audit",
        metavar="PATH",
        help="Write LLM prompt/response audit JSONL to PATH (hashes system prompt by default)",
    )
    run_parser.add_argument(
        "--prompt-audit-include-system",
        action="store_true",
        help="Include full system prompt text in the audit log (more sensitive)",
    )
    run_parser.add_argument(
        "--plots",
        nargs="?",
        const="",
        default=None,
        metavar="DIR",
        help=(
            "Write the standard plot bundle (toxicity/welfare + selection "
            "geometry) to DIR/plots/. With no argument, derives a default "
            "from --export-json/--export-csv or runs/<scenario>_seed<seed>."
        ),
    )
    run_parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress progress output"
    )

    # list
    list_parser = subparsers.add_parser("list", help="List available scenarios")
    list_parser.add_argument(
        "--dir", default="scenarios", help="Scenarios directory (default: scenarios)"
    )

    # agents
    agents_parser = subparsers.add_parser(
        "agents", help="List registered agent types"
    )
    agents_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show the full docstring for each agent (default: first line only)",
    )

    # sandbox
    sandbox_parser = subparsers.add_parser("sandbox", help="Manage worktree sandboxes for agents")
    sandbox_subparsers = sandbox_parser.add_subparsers(dest="sandbox_cmd")

    def _add_sandbox_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--repo", default=None, help="Path to the main git repository")
        p.add_argument("--root", default=None, help="Sandbox root directory")

    # sandbox create
    sb_create = sandbox_subparsers.add_parser("create", help="Create a sandbox for an agent")
    sb_create.add_argument("agent_id", help="Agent identifier")
    sb_create.add_argument("--branch", default=None, help="Branch to check out")
    _add_sandbox_common(sb_create)

    # sandbox destroy
    sb_destroy = sandbox_subparsers.add_parser("destroy", help="Destroy an agent's sandbox")
    sb_destroy.add_argument("agent_id", help="Agent identifier")
    _add_sandbox_common(sb_destroy)

    # sandbox list
    sb_list = sandbox_subparsers.add_parser("list", help="List active sandboxes")
    _add_sandbox_common(sb_list)

    # sandbox exec
    sb_exec = sandbox_subparsers.add_parser("exec", help="Execute a command in a sandbox")
    sb_exec.add_argument("agent_id", help="Agent identifier")
    sb_exec.add_argument("exec_cmd", nargs=argparse.REMAINDER, help="Command to execute (after --)")
    _add_sandbox_common(sb_exec)

    # sandbox status
    sb_status = sandbox_subparsers.add_parser("status", help="Show boundary metrics and agent stats")
    sb_status.add_argument("agent_id", nargs="?", default=None, help="Optional agent ID for per-agent stats")
    _add_sandbox_common(sb_status)

    # sandbox gc
    sb_gc = sandbox_subparsers.add_parser("gc", help="Garbage-collect stale sandboxes")
    _add_sandbox_common(sb_gc)

    # agentrxiv
    arxiv_parser = subparsers.add_parser("agentrxiv", help="Manage AgentRxiv local preprint server")
    arxiv_subparsers = arxiv_parser.add_subparsers(dest="agentrxiv_cmd")

    # agentrxiv start
    arxiv_start = arxiv_subparsers.add_parser("start", help="Start local AgentRxiv server")
    arxiv_start.add_argument("--port", type=int, default=5000, help="Port to run server on (default: 5000)")
    arxiv_start.add_argument("--uploads-dir", default="./agentrxiv_papers", help="Directory for PDF storage")

    # agentrxiv search
    arxiv_search = arxiv_subparsers.add_parser("search", help="Search papers on AgentRxiv")
    arxiv_search.add_argument("query", help="Search query")
    arxiv_search.add_argument("--limit", type=int, default=5, help="Maximum results (default: 5)")
    arxiv_search.add_argument("--url", help="AgentRxiv server URL (default: http://127.0.0.1:5000)")

    # agentrxiv submit
    arxiv_submit = arxiv_subparsers.add_parser("submit", help="Submit a PDF to AgentRxiv")
    arxiv_submit.add_argument("pdf", help="Path to PDF file")
    arxiv_submit.add_argument("--title", help="Paper title (default: derived from filename)")
    arxiv_submit.add_argument("--url", help="AgentRxiv server URL (default: http://127.0.0.1:5000)")

    # agentrxiv status
    arxiv_status = arxiv_subparsers.add_parser("status", help="Check AgentRxiv server status")
    arxiv_status.add_argument("--url", help="AgentRxiv server URL (default: http://127.0.0.1:5000)")

    # autoresearch
    autoresearch_parser = subparsers.add_parser("autoresearch", help="Run autoresearch governance loop")
    autoresearch_parser.add_argument("--objective", required=True, help="Path to objective file (yaml/json/markdown)")
    autoresearch_parser.add_argument("--scenario", required=True, help="Path to YAML scenario file")
    autoresearch_parser.add_argument("--iterations", type=int, default=10, help="Number of loop iterations")
    autoresearch_parser.add_argument("--eval-epochs", type=int, default=3, help="Epochs per candidate evaluation")
    autoresearch_parser.add_argument("--eval-steps", type=int, default=5, help="Steps per epoch in candidate evaluation")
    autoresearch_parser.add_argument("--seeds", default="7,11,19", help="Comma-separated seed panel")
    autoresearch_parser.add_argument("--random-seed", type=int, default=42, help="RNG seed for mutation proposals")
    autoresearch_parser.add_argument("--export-root", default="runs/autoresearch", help="Directory for autoresearch outputs")
    autoresearch_parser.add_argument("--auto-commit", action="store_true", help="Commit summary artifact at end of run")

    # evolve
    evolve_parser = subparsers.add_parser("evolve", help="Evolve governance configurations")
    evolve_parser.add_argument("scenario", help="Path to YAML scenario file")
    evolve_parser.add_argument(
        "--iterations", type=int, default=20, help="Number of evolution iterations (default: 20)"
    )
    evolve_parser.add_argument(
        "--eval-epochs", type=int, default=3, help="Epochs per evaluation (default: 3)"
    )
    evolve_parser.add_argument(
        "--eval-steps", type=int, default=5, help="Steps per epoch in evaluation (default: 5)"
    )
    evolve_parser.add_argument(
        "--final-eval-epochs", type=int, default=10, help="Epochs for final validation (default: 10)"
    )
    evolve_parser.add_argument(
        "--final-eval-steps", type=int, default=10, help="Steps per epoch for final validation (default: 10)"
    )
    evolve_parser.add_argument(
        "--use-llm", action="store_true", help="Use LLM-guided mutator (requires ANTHROPIC_API_KEY)"
    )
    evolve_parser.add_argument(
        "--llm-model", default="claude-sonnet-4-20250514", help="LLM model for mutations"
    )
    evolve_parser.add_argument(
        "--output-dir", metavar="DIR", help="Output directory for results"
    )
    evolve_parser.add_argument(
        "--resume-from", metavar="PATH", help="Resume from a snapshot .pkl file"
    )
    evolve_parser.add_argument(
        "--seed", type=int, default=42, help="Base random seed (default: 42)"
    )
    evolve_parser.add_argument(
        "--parents", type=int, default=3, help="Parents per iteration (default: 3)"
    )

    args = parser.parse_args()

    # Sanitized, opt-out telemetry envelope around command dispatch. Best-effort
    # and PII-free (see swarm/telemetry.py); SWARM_TELEMETRY=off disables it.
    import sys
    import time

    from swarm.telemetry import sanitize_command, track

    subcommand = args.command or "help"
    track(
        "Command Invoked",
        {"subcommand": subcommand, "command": sanitize_command(sys.argv, subcommand)},
    )
    started = time.monotonic()
    try:
        exit_code = _dispatch(args, parser, sandbox_parser, arxiv_parser)
    except Exception as exc:
        track(
            "Command Error",
            {"subcommand": subcommand, "error_type": type(exc).__name__},
            is_error=True,
        )
        raise
    track(
        "Command Completed",
        {
            "subcommand": subcommand,
            "exit_code": exit_code,
            "duration_ms": int((time.monotonic() - started) * 1000),
        },
    )
    return exit_code


def _dispatch(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    sandbox_parser: argparse.ArgumentParser,
    arxiv_parser: argparse.ArgumentParser,
) -> int:
    """Route a parsed command to its handler, returning the process exit code."""
    if args.command == "run":
        return cmd_run(args)
    elif args.command == "evolve":
        return cmd_evolve(args)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "agents":
        return cmd_agents(args)
    elif args.command == "autoresearch":
        return cmd_autoresearch(args)
    elif args.command == "sandbox":
        if args.sandbox_cmd:
            # Strip leading "--" from exec remainder
            if args.sandbox_cmd == "exec":
                raw = getattr(args, "exec_cmd", [])
                if raw and raw[0] == "--":
                    args.exec_cmd = raw[1:]
            return cmd_sandbox(args)
        else:
            sandbox_parser.print_help()
            return 0
    elif args.command == "agentrxiv":
        if args.agentrxiv_cmd:
            return cmd_agentrxiv(args)
        else:
            arxiv_parser.print_help()
            return 0
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
