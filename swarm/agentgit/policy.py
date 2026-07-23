"""Task-scoped policy evaluation for agent-authored git changes."""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import yaml

from swarm.agentgit.git import GitSnapshot

# Manifest / lockfiles whose changes count as dependency changes. Defined here
# (not in bundle.py) so both the bundle and the policy engine share one source;
# bundle.py imports it.
DEPENDENCY_FILENAMES = {
    "requirements.txt",
    "pyproject.toml",
    "poetry.lock",
    "Pipfile",
    "Pipfile.lock",
    "setup.py",
    "setup.cfg",
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.toml",
    "Cargo.lock",
    "go.mod",
    "go.sum",
    "Gemfile",
    "Gemfile.lock",
}


@dataclass(frozen=True)
class PolicyDecision:
    """A single pass/fail result emitted into provenance bundles."""

    policy_id: str
    passed: bool
    reason: str
    severity: str = "error"
    details: Dict[str, Any] = field(default_factory=dict)


def _overridden_rules(overrides: Optional[Sequence[Dict[str, Any]]]) -> Set[str]:
    """Rule ids unblocked by a recorded human override (`{rule: <id>, ...}`)."""

    return {o["rule"] for o in (overrides or []) if o.get("rule")}


@dataclass(frozen=True)
class PolicyFacts:
    """The facts a policy evaluates, from either a live diff or a bundle.

    Building this from both a ``GitSnapshot`` (attest time) and a bundle dict
    (CI gate time) lets one rule engine enforce identically in both places.
    """

    changed_paths: List[str]
    added_lines: int
    changed_files: int
    checks: Dict[str, bool]
    dependency_paths: List[str]
    overridden_rules: Set[str]
    # Trust score in [0, 1] for the acting reputation key (see
    # swarm.agentgit.reputation). None = no track record; ``trust_below``
    # rules treat unknown as below-threshold (fail-closed).
    agent_trust: Optional[float] = None

    @classmethod
    def from_snapshot(
        cls,
        snapshot: GitSnapshot,
        check_results: Optional[Dict[str, bool]] = None,
        overrides: Optional[Sequence[Dict[str, Any]]] = None,
        agent_trust: Optional[float] = None,
    ) -> "PolicyFacts":
        paths = [file.path for file in snapshot.changed_files]
        return cls(
            changed_paths=paths,
            added_lines=snapshot.total_additions,
            changed_files=len(snapshot.changed_files),
            checks=dict(check_results or {}),
            dependency_paths=[p for p in paths if _is_dependency_file(p)],
            overridden_rules=_overridden_rules(overrides),
            agent_trust=agent_trust,
        )

    @classmethod
    def from_bundle(
        cls,
        bundle: Dict[str, Any],
        trusted_overrides: Optional[Iterable[str]] = None,
        trusted_checks: Optional[Dict[str, bool]] = None,
        agent_trust: Optional[float] = None,
    ) -> "PolicyFacts":
        """Build facts from a bundle for CI gating.

        The bundle is agent-supplied, so two of its fields are NOT trusted here:

        - ``provenance.overrides`` — an agent could otherwise pre-populate an
          override for the exact CI rule meant to catch it.
        - ``checks`` — the agent authored these too, so a check-based rule (e.g.
          ``when: {check_failed: pytest} then deny``) would be defeated by an
          agent that simply self-attests ``checks={"pytest": true}``.

        At gate time both must come from a CI-controlled source
        (``trusted_overrides`` / ``trusted_checks``); neither is honored by
        default. Path/dependency/line facts are derived from the signed diff and
        remain authoritative. Unknown checks fail closed (deny-by-default).
        """

        git = bundle.get("git", {})
        files = git.get("changed_files", [])
        totals = git.get("totals", {})
        paths = [f.get("path", "") for f in files]
        return cls(
            changed_paths=paths,
            added_lines=int(totals.get("additions", 0)),
            changed_files=int(totals.get("changed_files", len(files))),
            checks=dict(trusted_checks or {}),
            # Like checks/overrides, trust is never read out of the
            # (agent-supplied) bundle: it comes from a CI-side reputation
            # ledger via the caller, or stays None (unknown).
            agent_trust=agent_trust,
            # Derive dependency facts from the signed ``git.changed_files`` (the
            # same source ``from_snapshot`` uses), NOT ``provenance``: the
            # provenance block is only folded into the signed payload for schema
            # v1, so trusting ``provenance.dependency_changes`` would let a v0
            # bundle (or a tampered/absent provenance) silently dodge every
            # ``dependency_changed`` rule. The diff is authoritative in both.
            dependency_paths=[p for p in paths if _is_dependency_file(p)],
            overridden_rules=set(trusted_overrides or []),
        )


_RULE_ACTIONS = {"deny", "require_check", "require_review"}
_RULE_SEVERITIES = {"error", "warning"}

# Condition keys understood by ``ConditionalRule._matches`` and the type each
# expects. Validated at parse time so a typo'd key (which would silently drop a
# rule's scope and fire it on every change) or a wrong-typed value (e.g. a
# non-hashable ``check_failed: [pytest]`` that would crash at evaluate time) is
# caught when a CI-owned policy file loads, not mid-gate.
_CONDITION_TYPES: Dict[str, type | tuple[type, ...]] = {
    "paths_match": (list, tuple),
    "dependency_changed": bool,
    "added_lines_gt": int,
    "changed_files_gt": int,
    "check_failed": str,
    "check_passed": str,
    "trust_below": (int, float),
}


def _validate_when(rule_id: str, when: Dict[str, Any]) -> None:
    """Reject unknown condition keys and wrong-typed values at parse time."""

    for key, value in when.items():
        if key not in _CONDITION_TYPES:
            raise ValueError(
                f"rule {rule_id!r}: unknown condition {key!r}; "
                f"expected one of {sorted(_CONDITION_TYPES)}"
            )
        expected = _CONDITION_TYPES[key]
        # ``bool`` is a subclass of ``int``; keep the numeric thresholds strictly
        # numeric so ``added_lines_gt: true`` / ``trust_below: true`` are
        # flagged rather than silently meaning 1.
        numeric_only = expected is int or (
            isinstance(expected, tuple) and int in expected and bool not in expected
        )
        if numeric_only and isinstance(value, bool):
            raise ValueError(
                f"rule {rule_id!r}: condition {key!r} expects a number, got bool"
            )
        if not isinstance(value, expected):
            raise ValueError(
                f"rule {rule_id!r}: condition {key!r} expects "
                f"{_type_name(expected)}, got {type(value).__name__}"
            )
        # ``value`` is a list/tuple here (validated above); the literal-tuple
        # isinstance lets mypy narrow it to an iterable before the element check.
        if key == "paths_match" and isinstance(value, (list, tuple)):
            if not all(isinstance(p, str) for p in value):
                raise ValueError(
                    f"rule {rule_id!r}: condition 'paths_match' expects a list of strings"
                )


def _type_name(expected: type | tuple[type, ...]) -> str:
    if isinstance(expected, tuple):
        return " or ".join(t.__name__ for t in expected)
    return expected.__name__


@dataclass(frozen=True)
class ConditionalRule:
    """A declarative ``when <condition> then <action>`` policy rule."""

    id: str
    when: Dict[str, Any] = field(default_factory=dict)
    action: str = "deny"  # deny | require_check | require_review
    check: Optional[str] = None
    severity: str = "error"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConditionalRule":
        rule_id = data["id"]
        action = data.get("action", "deny")
        if action not in _RULE_ACTIONS:
            raise ValueError(
                f"rule {rule_id!r}: invalid action {action!r}; "
                f"expected one of {sorted(_RULE_ACTIONS)}"
            )
        severity = data.get("severity", "error")
        if severity not in _RULE_SEVERITIES:
            raise ValueError(
                f"rule {rule_id!r}: invalid severity {severity!r}; "
                f"expected one of {sorted(_RULE_SEVERITIES)}"
            )
        check = data.get("check")
        if action == "require_check" and not check:
            raise ValueError(
                f"rule {rule_id!r}: action 'require_check' requires a non-empty 'check'"
            )
        when = dict(data.get("when", {}))
        _validate_when(rule_id, when)
        return cls(
            id=rule_id,
            when=when,
            action=action,
            check=check,
            severity=severity,
        )

    def _matches(self, facts: PolicyFacts) -> bool:
        """AND of every present condition; an empty ``when`` always matches."""

        when = self.when
        if "paths_match" in when and not any(
            _matches_any(path, when["paths_match"]) for path in facts.changed_paths
        ):
            return False
        if when.get("dependency_changed") and not facts.dependency_paths:
            return False
        if "added_lines_gt" in when and not facts.added_lines > when["added_lines_gt"]:
            return False
        if (
            "changed_files_gt" in when
            and not facts.changed_files > when["changed_files_gt"]
        ):
            return False
        if "check_failed" in when and facts.checks.get(when["check_failed"], False):
            return False
        if "check_passed" in when and not facts.checks.get(when["check_passed"], False):
            return False
        if "trust_below" in when:
            # Unknown trust (no track record) counts as below-threshold:
            # fail-closed, because "no history" must not outrank "bad
            # history". A known score at/above the threshold doesn't match.
            trust = facts.agent_trust
            if trust is not None and trust >= when["trust_below"]:
                return False
        return True

    def evaluate(self, facts: PolicyFacts) -> PolicyDecision:
        policy_id = f"rule:{self.id}"
        if not self._matches(facts):
            return PolicyDecision(
                policy_id=policy_id,
                passed=True,
                reason="Rule condition not met",
                severity=self.severity,
                details={"fired": False},
            )

        overridden = self.id in facts.overridden_rules
        if self.action == "require_check":
            satisfied = facts.checks.get(self.check or "", False)
            passed = satisfied or overridden
            reason = (
                f"Required check '{self.check}' passed"
                if satisfied
                else f"Required check '{self.check}' missing or failed"
            )
        elif self.action == "require_review":
            passed = overridden
            reason = "Rule requires human review"
        else:  # deny
            passed = overridden
            reason = "Rule denies this change"

        details: Dict[str, Any] = {"fired": True, "action": self.action, "when": self.when}
        if self.check:
            details["check"] = self.check
        if overridden:
            passed = True
            details["overridden"] = True
            reason = f"{reason} (satisfied by human override)"
        return PolicyDecision(
            policy_id=policy_id,
            passed=passed,
            reason=reason,
            severity=self.severity,
            details=details,
        )


@dataclass(frozen=True)
class AgentGitPolicy:
    """Declarative limits for one delegated agent task."""

    allowed_paths: List[str] = field(default_factory=list)
    denied_paths: List[str] = field(default_factory=list)
    max_changed_files: int | None = None
    max_added_lines: int | None = None
    required_checks: List[str] = field(default_factory=list)
    require_human_review: bool = False
    rules: List[ConditionalRule] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> "AgentGitPolicy":
        data = yaml.safe_load(path.read_text()) or {}
        return cls(
            allowed_paths=list(data.get("allowed_paths", [])),
            denied_paths=list(data.get("denied_paths", [])),
            max_changed_files=data.get("max_changed_files"),
            max_added_lines=data.get("max_added_lines"),
            required_checks=list(data.get("required_checks", [])),
            require_human_review=bool(data.get("require_human_review", False)),
            rules=[ConditionalRule.from_dict(r) for r in data.get("rules", [])],
        )

    def evaluate(
        self,
        snapshot: GitSnapshot,
        check_results: Dict[str, bool] | None = None,
        overrides: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[PolicyDecision]:
        """Evaluate repository state against this task policy."""

        facts = PolicyFacts.from_snapshot(snapshot, check_results or {}, overrides)
        return self.evaluate_facts(facts)

    def evaluate_facts(self, facts: PolicyFacts) -> List[PolicyDecision]:
        """Evaluate fixed constraints + conditional rules against facts.

        Shared by attest-time (`evaluate`) and CI-gate (`gate_bundle`) paths so
        the verdict is identical whether computed from a live diff or a bundle.
        """

        decisions = self._fixed_decisions(facts)
        if self.require_human_review:
            decisions.append(
                PolicyDecision(
                    policy_id="human-review",
                    passed=False,
                    reason="Policy requires human review before merge",
                    severity="warning",
                )
            )
        decisions.extend(rule.evaluate(facts) for rule in self.rules)
        return decisions

    def _fixed_decisions(self, facts: PolicyFacts) -> List[PolicyDecision]:
        return [
            self._evaluate_allowed_paths(facts.changed_paths),
            self._evaluate_denied_paths(facts.changed_paths),
            self._evaluate_max_changed_files(facts.changed_files),
            self._evaluate_max_added_lines(facts.added_lines),
            self._evaluate_required_checks(facts.checks),
        ]

    def _evaluate_allowed_paths(self, paths: List[str]) -> PolicyDecision:
        if not self.allowed_paths:
            return PolicyDecision(
                policy_id="allowed-paths",
                passed=True,
                reason="No allowed_paths constraint configured",
            )

        violations = [
            path for path in paths if not _matches_any(path, self.allowed_paths)
        ]
        return PolicyDecision(
            policy_id="allowed-paths",
            passed=not violations,
            reason=(
                "All changed files are within allowed paths"
                if not violations
                else "Changed files outside allowed paths"
            ),
            details={"violations": violations, "allowed_paths": self.allowed_paths},
        )

    def _evaluate_denied_paths(self, paths: List[str]) -> PolicyDecision:
        violations = [path for path in paths if _matches_any(path, self.denied_paths)]
        return PolicyDecision(
            policy_id="denied-paths",
            passed=not violations,
            reason=(
                "No changed files match denied paths"
                if not violations
                else "Changed files match denied paths"
            ),
            details={"violations": violations, "denied_paths": self.denied_paths},
        )

    def _evaluate_max_changed_files(self, count: int) -> PolicyDecision:
        if self.max_changed_files is None:
            return PolicyDecision(
                policy_id="max-changed-files",
                passed=True,
                reason="No max_changed_files constraint configured",
                details={"changed_files": count},
            )
        return PolicyDecision(
            policy_id="max-changed-files",
            passed=count <= self.max_changed_files,
            reason=f"Changed file count is {count}/{self.max_changed_files}",
            details={"changed_files": count, "limit": self.max_changed_files},
        )

    def _evaluate_max_added_lines(self, added: int) -> PolicyDecision:
        if self.max_added_lines is None:
            return PolicyDecision(
                policy_id="max-added-lines",
                passed=True,
                reason="No max_added_lines constraint configured",
                details={"added_lines": added},
            )
        return PolicyDecision(
            policy_id="max-added-lines",
            passed=added <= self.max_added_lines,
            reason=f"Added line count is {added}/{self.max_added_lines}",
            details={"added_lines": added, "limit": self.max_added_lines},
        )

    def _evaluate_required_checks(
        self, check_results: Dict[str, bool]
    ) -> PolicyDecision:
        missing = [
            check
            for check in self.required_checks
            if check not in check_results or not check_results[check]
        ]
        return PolicyDecision(
            policy_id="required-checks",
            passed=not missing,
            reason=(
                "All required checks passed"
                if not missing
                else "Required checks are missing or failed"
            ),
            details={"missing_or_failed": missing, "checks": check_results},
        )


def decisions_passed(decisions: Iterable[PolicyDecision]) -> bool:
    return all(decision.passed for decision in decisions if decision.severity != "warning")


def gate_bundle(
    bundle: Dict[str, Any],
    policy: "AgentGitPolicy",
    *,
    trusted_overrides: Optional[Iterable[str]] = None,
    trusted_checks: Optional[Dict[str, bool]] = None,
    agent_trust: Optional[float] = None,
) -> tuple[bool, List[PolicyDecision]]:
    """Enforce a (CI/org-owned) policy against an already-attested bundle.

    Reads the bundle's *recorded* facts (changed files, totals, dependency
    changes) and evaluates ``policy`` against them. This judges what the agent
    actually did against what CI allows, independent of whatever policy the
    agent self-attested with. Returns ``(ok, decisions)``.

    Callers should verify the bundle's signature before gating so the facts are
    authentic. Two inputs are deliberately not read from the (agent-supplied)
    bundle and must come from a CI-controlled source instead:
    ``trusted_overrides`` (rule ids a human approver vouches for) and
    ``trusted_checks`` (authoritative CI check results). Both default to empty,
    so check-based rules fail closed unless CI supplies the result — see
    ``from_bundle``.
    """

    facts = PolicyFacts.from_bundle(
        bundle,
        trusted_overrides=trusted_overrides,
        trusted_checks=trusted_checks,
        agent_trust=agent_trust,
    )
    decisions = policy.evaluate_facts(facts)
    return decisions_passed(decisions), decisions


def _is_dependency_file(path: str) -> bool:
    return path.rsplit("/", 1)[-1] in DEPENDENCY_FILENAMES


def _matches_any(path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch(path, pattern) for pattern in patterns)
