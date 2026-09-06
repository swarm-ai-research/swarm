"""Rich CLI for the SWARM Worktree Sandbox Bridge.

Usage:
    python -m swarm.bridges.worktree create <agent_id> [--branch BRANCH] [--repo PATH] [--root PATH]
    python -m swarm.bridges.worktree destroy <agent_id>
    python -m swarm.bridges.worktree list
    python -m swarm.bridges.worktree exec <agent_id> -- <cmd...>
    python -m swarm.bridges.worktree status [agent_id]
    python -m swarm.bridges.worktree gc
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from swarm.agentgit.bundle import build_bundle, write_bundle
from swarm.agentgit.policy import AgentGitPolicy
from swarm.bridges.worktree.bridge import WorktreeBridge
from swarm.bridges.worktree.config import WorktreeConfig

console = Console()


def _make_bridge(
    repo: Optional[str] = None,
    root: Optional[str] = None,
) -> WorktreeBridge:
    """Construct a WorktreeBridge and discover existing sandboxes."""
    kwargs: dict[str, Any] = {}
    if repo is not None:
        kwargs["repo_path"] = repo
    if root is not None:
        kwargs["sandbox_root"] = root
    config = WorktreeConfig(**kwargs)
    bridge = WorktreeBridge(config)
    # Discover sandboxes that already exist on disk so the CLI works
    # across invocations without persistent state.
    bridge._sandbox_mgr.discover_existing()
    return bridge


# ── Commands ────────────────────────────────────────────────────────


def cmd_create(args: argparse.Namespace) -> int:
    """Create a sandbox for an agent."""
    bridge = _make_bridge(
        repo=getattr(args, "repo", None),
        root=getattr(args, "root", None),
    )
    branch = getattr(args, "branch", None)

    with console.status(f"[bold green]Creating sandbox for {args.agent_id}..."):
        try:
            sandbox_id = bridge.create_agent_sandbox(
                args.agent_id, branch=branch,
            )
        except RuntimeError as exc:
            console.print(Panel(
                f"[red bold]Error:[/] {exc}",
                title="Create Failed",
                border_style="red",
            ))
            return 1

    sandbox_path = bridge._sandbox_mgr.get_sandbox_path(sandbox_id)
    branch_display = branch or f"agent/{args.agent_id}/workspace"

    body = Text()
    body.append("Sandbox ID: ", style="bold")
    body.append(f"{sandbox_id}\n")
    body.append("Path:       ", style="bold")
    body.append(f"{sandbox_path}\n")
    body.append("Branch:     ", style="bold")
    body.append(f"{branch_display}\n")
    body.append(".env scrub:  ", style="bold")
    body.append("complete", style="green")

    console.print(Panel(body, title=f"Sandbox Created — {args.agent_id}", border_style="green"))
    return 0


def cmd_destroy(args: argparse.Namespace) -> int:
    """Destroy an agent's sandbox."""
    bridge = _make_bridge(
        repo=getattr(args, "repo", None),
        root=getattr(args, "root", None),
    )

    # Register the discovered sandbox with the bridge's agent mapping
    sandbox_id = f"sandbox-{args.agent_id}"
    existing = bridge._sandbox_mgr.discover_existing()
    if sandbox_id not in existing:
        console.print(Panel(
            f"[red]No sandbox found for agent [bold]{args.agent_id}[/bold][/red]",
            title="Destroy Failed",
            border_style="red",
        ))
        return 1

    bridge._agent_sandboxes[args.agent_id] = sandbox_id
    bridge.destroy_agent_sandbox(args.agent_id)

    console.print(Panel(
        f"[green]Sandbox [bold]{sandbox_id}[/bold] destroyed successfully.[/green]",
        title=f"Sandbox Destroyed — {args.agent_id}",
        border_style="green",
    ))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List all active sandboxes."""
    bridge = _make_bridge(
        repo=getattr(args, "repo", None),
        root=getattr(args, "root", None),
    )
    sandboxes = bridge._sandbox_mgr.list_sandboxes()

    if not sandboxes:
        console.print("[dim]No active sandboxes.[/dim]")
        return 0

    table = Table(title="Active Sandboxes")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Path", style="white")
    table.add_column("Branch", style="magenta")
    table.add_column("Status", style="green")

    from swarm.bridges.worktree.sandbox import _run_git

    for sid, path in sorted(sandboxes.items()):
        branch_result = _run_git(
            ["rev-parse", "--abbrev-ref", "HEAD"],
            cwd=path,
        )
        branch = (
            branch_result.stdout.strip()
            if branch_result and branch_result.returncode == 0
            else "?"
        )
        table.add_row(sid, path, branch, "active")

    console.print(table)
    return 0


def cmd_exec(args: argparse.Namespace) -> int:
    """Execute a command in an agent's sandbox."""
    bridge = _make_bridge(
        repo=getattr(args, "repo", None),
        root=getattr(args, "root", None),
    )

    # Discover and register
    sandbox_id = f"sandbox-{args.agent_id}"
    existing = bridge._sandbox_mgr.discover_existing()
    if sandbox_id not in existing:
        console.print(Panel(
            f"[red]No sandbox found for agent [bold]{args.agent_id}[/bold][/red]",
            title="Exec Failed",
            border_style="red",
        ))
        return 1

    bridge._agent_sandboxes[args.agent_id] = sandbox_id

    command: List[str] = args.exec_cmd
    if not command:
        console.print("[red]No command specified. Usage: exec <agent_id> -- <cmd...>[/red]")
        return 1

    cmd_display = " ".join(command)

    with console.status(f"[bold green]Executing: {cmd_display}..."):
        interaction = bridge.dispatch_command(args.agent_id, command)

    # Determine allowed/denied from the last event
    events = bridge.get_events()
    denied = any(
        e.event_type.value == "command:denied"
        for e in events
        if e.payload.get("command") == command
    )

    if denied:
        deny_reason = ""
        for e in events:
            if (
                e.event_type.value == "command:denied"
                and e.payload.get("command") == command
            ):
                deny_reason = e.payload.get("reason", "policy violation")
                break

        label = Text("DENIED", style="bold red")
        body = Text()
        body.append("Command: ", style="bold")
        body.append(f"{cmd_display}\n")
        body.append("Reason:  ", style="bold")
        body.append(f"{deny_reason}", style="red")

        console.print(Panel(body, title=label, border_style="red"))
        return 1

    # Allowed
    label = Text("ALLOWED", style="bold green")

    body_parts: List[str] = []
    body_parts.append(f"[bold]Command:[/] {cmd_display}")
    body_parts.append(
        f"[bold]Return code:[/] {interaction.metadata.get('return_code', '?')}"
    )

    console.print(Panel("\n".join(body_parts), title=label, border_style="green"))

    # Show stdout if present
    stdout = interaction.metadata.get("stdout", "")
    if stdout:
        console.print(Panel(stdout.rstrip(), title="stdout", border_style="dim"))

    # Show stderr if present
    stderr = interaction.metadata.get("stderr", "")
    if stderr:
        console.print(Panel(stderr.rstrip(), title="stderr", border_style="yellow"))

    # Leakage warning
    has_leakage = any(
        e.event_type.value == "security:leakage_detected"
        for e in events
    )
    if has_leakage:
        console.print(
            Panel(
                "[yellow bold]Leakage detected in command output — content was redacted.[/]",
                title="Leakage Warning",
                border_style="yellow",
            )
        )

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show boundary metrics and agent stats."""
    bridge = _make_bridge(
        repo=getattr(args, "repo", None),
        root=getattr(args, "root", None),
    )
    bridge._sandbox_mgr.discover_existing()

    agent_id = getattr(args, "agent_id", None)

    # Boundary metrics
    metrics = bridge.get_boundary_metrics()
    flows = metrics.get("flows", {})
    leakage = metrics.get("leakage", {})

    boundary_table = Table(title="Boundary Metrics")
    boundary_table.add_column("Metric", style="cyan", no_wrap=True)
    boundary_table.add_column("Value", style="white")

    boundary_table.add_row("Total flows", str(flows.get("total", 0)))
    boundary_table.add_row("Inbound flows", str(flows.get("inbound", 0)))
    boundary_table.add_row("Outbound flows", str(flows.get("outbound", 0)))
    boundary_table.add_row("Blocked flows", str(flows.get("blocked", 0)))
    boundary_table.add_row("Bytes in", str(flows.get("bytes_in", 0)))
    boundary_table.add_row("Bytes out", str(flows.get("bytes_out", 0)))
    boundary_table.add_row("Leakage events", str(leakage.get("total_events", 0)))
    boundary_table.add_row("Leakage blocked", str(leakage.get("blocked", 0)))
    boundary_table.add_row("Max severity", str(leakage.get("max_severity", 0.0)))

    console.print(boundary_table)

    # Per-agent stats if requested
    if agent_id:
        stats = bridge.get_agent_stats(agent_id)
        agent_table = Table(title=f"Agent Stats — {agent_id}")
        agent_table.add_column("Stat", style="cyan", no_wrap=True)
        agent_table.add_column("Value", style="white")
        for key, val in stats.items():
            agent_table.add_row(str(key), str(val))
        console.print(agent_table)

    # Policy summary
    policy = metrics.get("policy", {})
    if policy:
        policy_table = Table(title="Policy Summary")
        policy_table.add_column("Policy", style="cyan", no_wrap=True)
        policy_table.add_column("Value", style="white")
        for key, val in policy.items():
            policy_table.add_row(str(key), str(val))
        console.print(policy_table)

    return 0


def cmd_attest(args: argparse.Namespace) -> int:
    """Create an AgentGit provenance bundle for an agent sandbox."""
    bridge = _make_bridge(
        repo=getattr(args, "repo", None),
        root=getattr(args, "root", None),
    )

    sandbox_id = f"sandbox-{args.agent_id}"
    existing = bridge._sandbox_mgr.discover_existing()
    sandbox_path = existing.get(sandbox_id)
    if sandbox_path is None:
        console.print(Panel(
            f"[red]No sandbox found for agent [bold]{args.agent_id}[/bold][/red]",
            title="Attest Failed",
            border_style="red",
        ))
        return 1

    policy = AgentGitPolicy.from_yaml(Path(args.policy))
    bundle = build_bundle(
        repo=Path(sandbox_path),
        task_id=args.task,
        agent_id=args.agent_id,
        policy=policy,
        base_ref=args.base,
        check_results=_parse_checks(args.check),
        signing_key=args.signing_key or os.environ.get("AGENTGIT_SIGNING_KEY"),
        signer_id=args.signer_id,
    )
    write_bundle(bundle, Path(args.output))

    passed = bundle["policy"]["passed"]
    status = Text("PASS", style="bold green") if passed else Text("FAIL", style="bold red")
    body = Text()
    body.append("Agent:      ", style="bold")
    body.append(f"{args.agent_id}\n")
    body.append("Task:       ", style="bold")
    body.append(f"{args.task}\n")
    body.append("Sandbox:    ", style="bold")
    body.append(f"{sandbox_path}\n")
    body.append("Bundle:     ", style="bold")
    body.append(f"{args.output}\n")
    body.append("Changed:    ", style="bold")
    body.append(str(bundle["git"]["totals"]["changed_files"]))
    console.print(Panel(body, title=status, border_style="green" if passed else "red"))
    return 0 if passed or args.warn_only else 1


def cmd_gc(args: argparse.Namespace) -> int:
    """Run garbage collection on stale sandboxes."""
    bridge = _make_bridge(
        repo=getattr(args, "repo", None),
        root=getattr(args, "root", None),
    )
    bridge._sandbox_mgr.discover_existing()

    events = bridge.gc()

    if not events:
        console.print("[dim]No stale sandboxes to collect.[/dim]")
        return 0

    table = Table(title="Garbage Collected Sandboxes")
    table.add_column("Sandbox ID", style="cyan")
    table.add_column("Status", style="green")

    for event in events:
        table.add_row(event.sandbox_id or "?", "collected")

    console.print(table)
    console.print(f"[green]{len(events)} sandbox(es) collected.[/green]")
    return 0


# ── Argparse ────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the worktree sandbox CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m swarm.bridges.worktree",
        description="Rich CLI for the SWARM Worktree Sandbox Bridge",
    )
    subparsers = parser.add_subparsers(dest="subcmd")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--repo", default=None, help="Path to the main git repository")
        p.add_argument("--root", default=None, help="Sandbox root directory")

    # create
    create_p = subparsers.add_parser("create", help="Create a sandbox for an agent")
    create_p.add_argument("agent_id", help="Agent identifier")
    create_p.add_argument("--branch", default=None, help="Branch to check out")
    add_common(create_p)

    # destroy
    destroy_p = subparsers.add_parser("destroy", help="Destroy an agent's sandbox")
    destroy_p.add_argument("agent_id", help="Agent identifier")
    add_common(destroy_p)

    # list
    list_p = subparsers.add_parser("list", help="List active sandboxes")
    add_common(list_p)

    # exec
    exec_p = subparsers.add_parser("exec", help="Execute a command in a sandbox")
    exec_p.add_argument("agent_id", help="Agent identifier")
    exec_p.add_argument(
        "exec_cmd", nargs=argparse.REMAINDER,
        help="Command to execute (after --)",
    )
    add_common(exec_p)

    # status
    status_p = subparsers.add_parser("status", help="Show boundary metrics and agent stats")
    status_p.add_argument(
        "agent_id", nargs="?", default=None,
        help="Optional agent ID for per-agent stats",
    )
    add_common(status_p)

    # attest
    attest_p = subparsers.add_parser(
        "attest",
        help="Create an AgentGit provenance bundle for a sandbox",
    )
    attest_p.add_argument("agent_id", help="Agent identifier")
    attest_p.add_argument("--task", required=True, help="Delegated task identifier")
    attest_p.add_argument("--policy", required=True, help="AgentGit policy YAML")
    attest_p.add_argument("--base", default="HEAD", help="Base ref to diff against")
    attest_p.add_argument(
        "--output",
        default=".agentgit/provenance.json",
        help="Output provenance bundle path",
    )
    attest_p.add_argument(
        "--check",
        action="append",
        default=[],
        metavar="NAME=pass|fail",
        help="Record a required check result; may be repeated",
    )
    attest_p.add_argument("--signing-key", default=None, help="Hex Ed25519 seed (64 hex chars)")
    attest_p.add_argument(
        "--signer-id", default=None, help="Deprecated and ignored (signer_id is the key's DID)"
    )
    attest_p.add_argument(
        "--warn-only",
        action="store_true",
        help="Write bundle but return success even when policy fails",
    )
    add_common(attest_p)

    # gc
    gc_p = subparsers.add_parser("gc", help="Garbage-collect stale sandboxes")
    add_common(gc_p)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Strip leading "--" from exec remainder
    if args.subcmd == "exec":
        raw: List[str] = getattr(args, "exec_cmd", [])
        if raw and raw[0] == "--":
            args.exec_cmd = raw[1:]

    handlers = {
        "create": cmd_create,
        "destroy": cmd_destroy,
        "list": cmd_list,
        "exec": cmd_exec,
        "status": cmd_status,
        "attest": cmd_attest,
        "gc": cmd_gc,
    }

    subcmd = args.subcmd
    if subcmd in handlers:
        return handlers[subcmd](args)

    parser.print_help()
    return 0


def _parse_checks(raw_checks: List[str]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for raw in raw_checks:
        if "=" not in raw:
            raise SystemExit(f"Invalid --check value {raw!r}; expected NAME=pass|fail")
        name, value = raw.split("=", 1)
        normalised = value.strip().lower()
        if normalised not in {"pass", "passed", "true", "fail", "failed", "false"}:
            raise SystemExit(f"Invalid check result {value!r} for {name!r}")
        checks[name.strip()] = normalised in {"pass", "passed", "true"}
    return checks


if __name__ == "__main__":
    sys.exit(main())
