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
