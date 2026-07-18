"""Command line interface for AgentGit MVP."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from swarm.agentgit.bundle import (
    DEFAULT_DEV_SIGNING_KEY,
    build_bundle,
    load_bundle,
    verify_bundle,
    write_bundle,
)
from swarm.agentgit.coordination import DEFAULT_DB_PATH, CoordinationBoard
from swarm.agentgit.memory import KINDS, SCOPES, MemoryEntry, MemoryStore
from swarm.agentgit.policy import AgentGitPolicy, gate_bundle
from swarm.agentgit.review import VERDICT_BLOCK, run_review_panel
from swarm.agentgit.store import (
    DEFAULT_LOG_PATH,
    DEFAULT_NOTES_REF,
    list_noted_commits,
    read_log,
    read_notes,
    store_bundle,
    verify_log,
)


def cmd_attest(args: argparse.Namespace) -> int:
    policy = AgentGitPolicy.from_yaml(Path(args.policy))
    checks = _parse_checks(args.check)

    reviews = None
    if args.review:
        synthesis = run_review_panel(Path(args.repo), base_ref=args.base)
        reviews = synthesis.to_review_dicts()
        print(synthesis.format_text())

    bundle = build_bundle(
        repo=Path(args.repo),
        task_id=args.task,
        agent_id=args.agent,
        policy=policy,
        base_ref=args.base,
        check_results=checks,
        signing_key=args.signing_key or os.environ.get("AGENTGIT_SIGNING_KEY"),
        signer_id=args.signer_id,
        reviews=reviews,
    )
    write_bundle(bundle, Path(args.output))

    # Durable stores record failing attestations too: a policy failure is
    # itself provenance worth keeping tamper-evident.
    if args.store:
        summary = store_bundle(
            bundle,
            repo=Path(args.repo),
            backends=list(dict.fromkeys(args.store)),
            log_path=Path(args.log_path) if args.log_path else None,
            notes_ref=args.notes_ref,
            commit=args.store_commit,
        )
        for backend, info in summary.items():
            print(f"stored [{backend}] {info}")

    passed = bundle["policy"]["passed"]
    status = "PASS" if passed else "FAIL"
    print(
        f"{status} agentgit attestation: {args.output} "
        f"({bundle['git']['totals']['changed_files']} changed files)"
    )
    return 0 if passed or args.warn_only else 1


def cmd_history(args: argparse.Namespace) -> int:
    """Inspect and verify durably stored provenance (event log + git notes)."""

    repo = Path(args.repo)
    signing_key = args.signing_key or os.environ.get("AGENTGIT_SIGNING_KEY")
    exit_code = 0

    log_path = Path(args.log_path) if args.log_path else repo / DEFAULT_LOG_PATH
    entries = read_log(log_path)
    print(f"log: {log_path} ({len(entries)} entries)")
    for entry in entries:
        bundle = entry.get("bundle", {})
        print(
            f"- seq={entry.get('seq')} task={bundle.get('task', {}).get('task_id')} "
            f"agent={bundle.get('agent', {}).get('agent_id')} "
            f"policy_passed={bundle.get('policy', {}).get('passed')}"
        )
    if args.verify:
        # Match cmd_verify's key handling: fall back to the dev key so bundle
        # signatures inside the log are always checked under --verify.
        ok, errors = verify_log(
            log_path, signing_key=signing_key or DEFAULT_DEV_SIGNING_KEY
        )
        print(f"{'PASS' if ok else 'FAIL'} log chain verify: {log_path}")
        for error in errors:
            print(f"- {error}")
        if not ok:
            exit_code = 1

    commits = [args.commit] if args.commit else list_noted_commits(repo, notes_ref=args.notes_ref)
    for commit in commits:
        bundles = read_notes(repo, commit, notes_ref=args.notes_ref)
        print(f"notes[{args.notes_ref}] {commit}: {len(bundles)} bundle(s)")
        for bundle in bundles:
            note_ok = True
            note_errors: list[str] = []
            if args.verify:
                note_ok, note_errors = verify_bundle(
                    bundle, signing_key=signing_key, require_policy_pass=False
                )
            state = "" if not args.verify else (" [verified]" if note_ok else " [INVALID]")
            print(
                f"- task={bundle.get('task', {}).get('task_id')} "
                f"agent={bundle.get('agent', {}).get('agent_id')} "
                f"policy_passed={bundle.get('policy', {}).get('passed')}{state}"
            )
            for error in note_errors:
                print(f"  - {error}")
            if args.verify and not note_ok:
                exit_code = 1
    return exit_code


def cmd_verify(args: argparse.Namespace) -> int:
    bundle = load_bundle(Path(args.bundle))
    ok, errors = verify_bundle(
        bundle,
        signing_key=args.signing_key or os.environ.get("AGENTGIT_SIGNING_KEY"),
        require_policy_pass=not args.allow_policy_fail,
    )
    if ok:
        print(f"PASS agentgit verify: {args.bundle}")
        return 0

    print(f"FAIL agentgit verify: {args.bundle}")
    for error in errors:
        print(f"- {error}")
    return 1


def cmd_gate(args: argparse.Namespace) -> int:
    """Enforce a CI/org-owned policy against an already-attested bundle."""
    bundle = load_bundle(Path(args.bundle))

    # A signing key must be supplied explicitly when gating. verify_bundle falls
    # back to the public DEFAULT_DEV_SIGNING_KEY when given None, so an
    # unconfigured CI job would otherwise accept any dev-key-signed bundle as
    # authentic and run the org policy against attacker-chosen facts — the gate
    # fails open. Require the key here and fail closed if it is missing.
    signing_key = args.signing_key or os.environ.get("AGENTGIT_SIGNING_KEY")
    if not signing_key:
        print(
            f"FAIL agentgit gate: {args.bundle} "
            "(no signing key; set --signing-key or AGENTGIT_SIGNING_KEY)"
        )
        return 1

    # The gate reads facts out of the bundle, so the bundle must be authentic
    # first: verify the signature (not the bundle's own policy — CI applies its
    # own) and fail closed on any tampering/malformed input.
    verify_ok, verify_errors = verify_bundle(
        bundle,
        signing_key=signing_key,
        require_policy_pass=False,
    )
    if not verify_ok:
        print(f"FAIL agentgit gate: {args.bundle} (bundle failed verification)")
        for error in verify_errors:
            print(f"- {error}")
        return 1

    policy = AgentGitPolicy.from_yaml(Path(args.policy))
    # Checks are CI-authoritative at gate time: the bundle's own ``checks`` are
    # agent-supplied and ignored, so a check-based rule can't be defeated by an
    # agent self-attesting a passing result. Unsupplied checks fail closed.
    trusted_checks = _parse_checks(args.check)
    ok, decisions = gate_bundle(
        bundle,
        policy,
        trusted_overrides=args.override,
        trusted_checks=trusted_checks,
    )
    status = "PASS" if ok else "FAIL"
    print(f"{status} agentgit gate: {args.bundle} (policy {args.policy})")
    for decision in decisions:
        if not decision.passed:
            label = "WARN" if decision.severity == "warning" else "FAIL"
            print(f"- [{label}] [{decision.policy_id}] {decision.reason}")
    return 0 if ok else 1


def cmd_review(args: argparse.Namespace) -> int:
    """Run the reviewer panel and print the synthesized outcome."""

    import json

    synthesis = run_review_panel(Path(args.repo), base_ref=args.base)
    if args.json:
        print(json.dumps(synthesis.to_dict(), indent=2, sort_keys=True))
    else:
        print(synthesis.format_text())
    blocked = synthesis.verdict == VERDICT_BLOCK
    return 0 if not blocked or args.warn_only else 1


def cmd_coord(args: argparse.Namespace) -> int:
    """Machine-speed coordination: claim/lock/propose/respond/conflicts."""

    agent = args.agent or os.environ.get("SESSION_ID")
    if not agent:
        print("coord: no agent identity; pass --agent or set SESSION_ID")
        return 1

    with CoordinationBoard(args.db) as board:
        if args.action == "claim":
            result = board.claim(agent, args.target)
            if result.ok:
                print(f"CLAIMED {args.target} (claim {result.claim_id})")
                return 0
            print(f"HELD {args.target} by {result.holder}")
            return 1
        if args.action == "yield":
            ok = board.yield_claim(agent, args.target)
            print(f"{'YIELDED' if ok else 'NOT-HELD'} {args.target}")
            return 0 if ok else 1
        if args.action == "done":
            ok = board.complete(agent, args.target)
            print(f"{'DONE' if ok else 'NOT-HELD'} {args.target}")
            return 0 if ok else 1
        if args.action == "lock":
            lock_result = board.lock(agent, args.target, reason=args.reason)
            if lock_result.ok:
                print(f"LOCKED {lock_result.resource} (lock {lock_result.lock_id})")
                return 0
            for c in lock_result.conflicts:
                print(f"CONFLICT {c['resource']} held by {c['agent']}: {c['reason']}")
            return 1
        if args.action == "release":
            ok = board.release(agent, args.target)
            print(f"{'RELEASED' if ok else 'NOT-HELD'} {args.target}")
            return 0 if ok else 1
        if args.action == "propose":
            proposal_id = board.propose(
                agent, args.kind, args.target, task_id=args.task, to_agent=args.to
            )
            print(f"PROPOSED {args.kind} #{proposal_id} -> {args.to}")
            return 0
        if args.action == "respond":
            ok = board.respond(
                agent, int(args.target), accept=args.accept, response=args.reason
            )
            print(f"{'RESPONDED' if ok else 'NOT-OPEN'} #{args.target}")
            return 0 if ok else 1
        if args.action == "conflicts":
            conflicts = board.detect_conflicts(agent, args.paths or [args.target])
            for c in conflicts:
                print(
                    f"CONFLICT {c['resource']} held by {c['agent']} "
                    f"(overlaps: {', '.join(c['paths'])})"
                )
            if not conflicts:
                print("no conflicting work detected")
            return 1 if conflicts else 0
        # status
        for claim in board.active_claims():
            print(f"claim: {claim['task_id']} by {claim['agent']} since {claim['ts']}")
        for lock in board.active_locks():
            print(f"lock: {lock['resource']} by {lock['agent']}: {lock['reason']}")
        for prop in board.open_proposals(agent):
            print(
                f"proposal #{prop['id']} [{prop['kind']}] from {prop['agent']} "
                f"to {prop['to_agent']}: {prop['body'][:80]}"
            )
        return 0


def _format_memory(entry: MemoryEntry) -> str:
    flag = " [RETIRED]" if entry.retired else ""
    return (
        f"[{entry.scope}/{entry.kind}] {entry.subject}: {entry.body} "
        f"(id={entry.id} v{entry.version} by {entry.author}){flag}"
    )


def cmd_memory(args: argparse.Namespace) -> int:
    """Structured, versioned, scoped operational memory."""

    agent = args.agent or os.environ.get("SESSION_ID") or "local"
    store = MemoryStore(
        Path(args.repo),
        agent=agent,
        home=Path(args.home) if args.home else None,
    )
    if args.action == "remember":
        if not args.subject or not args.body:
            print("memory remember: a subject argument and --body are required")
            return 1
        entry = store.remember(
            scope=args.scope,
            kind=args.kind,
            subject=args.subject,
            body=args.body,
            author=agent,
            entry_id=args.id,
        )
        print(f"REMEMBERED {_format_memory(entry)}")
        return 0
    if args.action == "recall":
        entries = store.recall(args.subject or None, kind=args.kind_filter)
        for entry in entries:
            print(_format_memory(entry))
        if not entries:
            print("no memories found")
        return 0
    if args.action == "retire":
        if not args.id:
            print("memory retire: --id is required")
            return 1
        ok = store.retire(
            scope=args.scope, entry_id=args.id, author=agent, reason=args.reason
        )
        print(f"{'RETIRED' if ok else 'NOT-FOUND'} {args.id}")
        return 0 if ok else 1
    # history
    if not args.id:
        print("memory history: --id is required")
        return 1
    versions = store.history(args.id, scope=args.scope)
    for entry in versions:
        print(f"v{entry.version} @ {entry.updated_at}: {_format_memory(entry)}")
    if not versions:
        print("no memories found")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m swarm.agentgit",
        description="Create task-scoped provenance bundles for agent-authored git diffs.",
    )
    subparsers = parser.add_subparsers(dest="subcmd")

    attest = subparsers.add_parser("attest", help="Evaluate and sign the current diff")
    attest.add_argument("--repo", default=".", help="Git repository path")
    attest.add_argument("--task", required=True, help="Delegated task identifier")
    attest.add_argument("--agent", required=True, help="Agent identity")
    attest.add_argument("--policy", required=True, help="AgentGit policy YAML")
    attest.add_argument("--base", default="HEAD", help="Base ref to diff against")
    attest.add_argument(
        "--output",
        default=".agentgit/provenance.json",
        help="Output provenance bundle path",
    )
    attest.add_argument(
        "--check",
        action="append",
        default=[],
        metavar="NAME=pass|fail",
        help="Record a required check result; may be repeated",
    )
    attest.add_argument(
        "--signing-key",
        default=None,
        help="Hex HMAC key. Defaults to AGENTGIT_SIGNING_KEY or a dev key.",
    )
    attest.add_argument("--signer-id", default="agentgit-local")
    attest.add_argument(
        "--warn-only",
        action="store_true",
        help="Write bundle but return success even when policy fails",
    )
    attest.add_argument(
        "--store",
        action="append",
        default=[],
        choices=["log", "git-notes"],
        help=(
            "Also store the bundle durably: 'log' appends to the hash-chained "
            "event log; 'git-notes' attaches it to the attested commit. May be "
            "repeated."
        ),
    )
    attest.add_argument(
        "--log-path",
        default=None,
        help=f"Event log path (default: <repo>/{DEFAULT_LOG_PATH})",
    )
    attest.add_argument(
        "--notes-ref",
        default=DEFAULT_NOTES_REF,
        help=f"Git notes ref for provenance (default: {DEFAULT_NOTES_REF})",
    )
    attest.add_argument(
        "--store-commit",
        default=None,
        help=(
            "Commit to attach the git note to (default: the bundle's recorded "
            "head commit)"
        ),
    )
    attest.add_argument(
        "--review",
        action="store_true",
        help=(
            "Run the multi-reviewer panel over the diff and record its "
            "synthesized outcome in the bundle's provenance reviews"
        ),
    )
    attest.set_defaults(func=cmd_attest)

    verify = subparsers.add_parser("verify", help="Verify an AgentGit bundle")
    verify.add_argument("bundle", help="Path to provenance bundle JSON")
    verify.add_argument(
        "--signing-key",
        default=None,
        help="Hex HMAC key. Defaults to AGENTGIT_SIGNING_KEY or a dev key.",
    )
    verify.add_argument(
        "--allow-policy-fail",
        action="store_true",
        help="Verify hash/signature even when policy failed",
    )
    verify.set_defaults(func=cmd_verify)

    gate = subparsers.add_parser(
        "gate",
        help="Enforce a CI-owned policy against a bundle's recorded facts",
    )
    gate.add_argument("--bundle", required=True, help="Path to provenance bundle JSON")
    gate.add_argument("--policy", required=True, help="CI/org-owned AgentGit policy YAML")
    gate.add_argument(
        "--signing-key",
        default=None,
        help=(
            "Hex HMAC key for bundle verification (required; falls back to "
            "AGENTGIT_SIGNING_KEY). The gate fails closed if neither is set so it "
            "never accepts a dev-key-signed bundle as authentic."
        ),
    )
    gate.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="RULE_ID",
        help="CI-trusted override: pass a blocking rule's id; may be repeated",
    )
    gate.add_argument(
        "--check",
        action="append",
        default=[],
        metavar="NAME=pass|fail",
        help=(
            "CI-authoritative check result for check-based rules; the bundle's "
            "own checks are ignored at gate time. Unsupplied checks fail closed. "
            "May be repeated."
        ),
    )
    gate.set_defaults(func=cmd_gate)

    history = subparsers.add_parser(
        "history",
        help="Inspect/verify durably stored provenance (event log + git notes)",
    )
    history.add_argument("--repo", default=".", help="Git repository path")
    history.add_argument(
        "--log-path",
        default=None,
        help=f"Event log path (default: <repo>/{DEFAULT_LOG_PATH})",
    )
    history.add_argument(
        "--notes-ref",
        default=DEFAULT_NOTES_REF,
        help=f"Git notes ref for provenance (default: {DEFAULT_NOTES_REF})",
    )
    history.add_argument(
        "--commit",
        default=None,
        help="Show only notes attached to this commit (default: all noted commits)",
    )
    history.add_argument(
        "--verify",
        action="store_true",
        help="Verify the log hash chain and every stored bundle's signature",
    )
    history.add_argument(
        "--signing-key",
        default=None,
        help="Hex HMAC key. Defaults to AGENTGIT_SIGNING_KEY or a dev key.",
    )
    history.set_defaults(func=cmd_history)

    review = subparsers.add_parser(
        "review",
        help="Run the multi-reviewer panel over the current diff",
    )
    review.add_argument("--repo", default=".", help="Git repository path")
    review.add_argument("--base", default="HEAD", help="Base ref to diff against")
    review.add_argument(
        "--json",
        action="store_true",
        help="Emit the full synthesis as JSON instead of the text summary",
    )
    review.add_argument(
        "--warn-only",
        action="store_true",
        help="Print the synthesis but return success even on a block verdict",
    )
    review.set_defaults(func=cmd_review)

    coord = subparsers.add_parser(
        "coord",
        help="Machine-speed coordination: claim/lock/propose/respond/conflicts",
    )
    coord.add_argument(
        "action",
        choices=[
            "claim", "yield", "done", "lock", "release",
            "propose", "respond", "conflicts", "status",
        ],
    )
    coord.add_argument(
        "target",
        nargs="?",
        default="",
        help=(
            "Task id (claim/yield/done), resource (lock/release/conflicts), "
            "proposal body (propose), or proposal id (respond)"
        ),
    )
    coord.add_argument(
        "paths",
        nargs="*",
        help="Additional paths to check (conflicts action)",
    )
    coord.add_argument(
        "--agent",
        default=None,
        help="Agent identity (default: $SESSION_ID)",
    )
    coord.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"Coordination database path (default: {DEFAULT_DB_PATH})",
    )
    coord.add_argument(
        "--kind",
        default="plan",
        choices=["plan", "review", "delegate"],
        help="Proposal kind (propose action)",
    )
    coord.add_argument("--task", default="", help="Related task id (propose action)")
    coord.add_argument(
        "--to", default="#swarm", help="Addressee agent (propose action)"
    )
    coord.add_argument(
        "--reason", default="", help="Lock reason / response text"
    )
    coord.add_argument(
        "--accept",
        action="store_true",
        help="Accept the proposal (respond action; omit to reject)",
    )
    coord.set_defaults(func=cmd_coord)

    memory = subparsers.add_parser(
        "memory",
        help="Structured, versioned, repo/org/agent-scoped operational memory",
    )
    memory.add_argument("action", choices=["remember", "recall", "retire", "history"])
    memory.add_argument(
        "subject",
        nargs="?",
        default="",
        help="Path/module/topic the memory is about (required for remember; "
        "optional recall filter)",
    )
    memory.add_argument("--repo", default=".", help="Git repository path")
    memory.add_argument(
        "--scope",
        default="repo",
        choices=list(SCOPES),
        help="Memory scope (remember/retire/history; default: repo)",
    )
    memory.add_argument(
        "--kind",
        default="context",
        choices=list(KINDS),
        help="Memory kind (remember; default: context)",
    )
    memory.add_argument(
        "--kind-filter",
        default=None,
        choices=list(KINDS),
        help="Only recall memories of this kind",
    )
    memory.add_argument("--body", default="", help="Memory body text (remember)")
    memory.add_argument("--id", default=None, help="Memory id (update/retire/history)")
    memory.add_argument("--reason", default="", help="Retire reason")
    memory.add_argument(
        "--agent", default=None, help="Agent identity (default: $SESSION_ID or 'local')"
    )
    memory.add_argument(
        "--home",
        default=None,
        help="Base dir for org/agent memory files (default: ~)",
    )
    memory.set_defaults(func=cmd_memory)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    result: int = args.func(args)
    return result


def _parse_checks(raw_checks: List[str]) -> Dict[str, bool]:
    checks: Dict[str, bool] = {}
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
