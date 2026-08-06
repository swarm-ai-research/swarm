"""Tests for the Prime Agent bridge.

Covers session-file parsing (the real v3 JSONL entry shapes), RLM spawn
detection in ipython cells, harness drift tracking, refinement policy
adjudication, observable mapping, and delegation-tree credit linking.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from swarm.bridges.prime_agent import (
    BASE_SYSTEM_PROMPT_ID,
    GATED_PROGRESS,
    REFINEMENT_CUSTOM_TYPE,
    GateResult,
    HarnessEditRecord,
    HarnessRefinementPolicy,
    HarnessTracker,
    PolicyDecision,
    PrimeAgentBridge,
    PrimeAgentBridgeConfig,
    PrimeAgentClient,
    PrimeAgentClientConfig,
    PrimeAgentEvent,
    PrimeAgentEventType,
    RefinementPolicyConfig,
    RefinementRecord,
    RlmSpawnRecord,
    SessionOutcome,
    build_session_tree,
    evidence_kinds,
    has_concrete_evidence,
)
from swarm.governance.config import GovernanceConfig

# ---------------------------------------------------------------------------
# Fixture builders — shapes mirror Prime Agent's session v3 format
# ---------------------------------------------------------------------------


def _usage(total: int = 100, cost: float = 0.01) -> Dict[str, Any]:
    return {
        "input": total // 2,
        "output": total // 2,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": total,
        "cost": {
            "input": cost / 2,
            "output": cost / 2,
            "cacheRead": 0.0,
            "cacheWrite": 0.0,
            "total": cost,
        },
    }


class SessionBuilder:
    """Builds a Prime Agent session JSONL file entry by entry."""

    def __init__(
        self,
        session_id: str = "sess-root",
        parent_session: Optional[str] = None,
    ) -> None:
        header: Dict[str, Any] = {
            "type": "session",
            "version": 3,
            "id": session_id,
            "timestamp": "2026-08-01T10:00:00.000Z",
            "cwd": "/repo",
        }
        if parent_session:
            header["parentSession"] = parent_session
        self.lines: List[Dict[str, Any]] = [header]
        self._counter = 0
        self._last_id: Optional[str] = None
        self._minute = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"e{self._counter:07d}"

    def _timestamp(self) -> str:
        self._minute += 1
        return f"2026-08-01T10:{self._minute:02d}:00.000Z"

    def _entry(self, entry: Dict[str, Any], parent_id: Optional[str] = None) -> str:
        entry_id = self._next_id()
        entry["id"] = entry_id
        entry["parentId"] = parent_id if parent_id is not None else self._last_id
        entry["timestamp"] = self._timestamp()
        self.lines.append(entry)
        self._last_id = entry_id
        return entry_id

    def user(self, text: str = "do the thing") -> str:
        return self._entry(
            {"type": "message", "message": {"role": "user", "content": text}}
        )

    def assistant(
        self,
        text: str = "working",
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        stop_reason: str = "stop",
        model: str = "claude-sonnet-4-5",
        tokens: int = 100,
    ) -> str:
        content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        for call in tool_calls or []:
            content.append(
                {
                    "type": "toolCall",
                    "id": call["id"],
                    "name": call.get("name", "ipython"),
                    "arguments": {"code": call.get("code", "")},
                }
            )
        return self._entry(
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": content,
                    "api": "messages",
                    "provider": "anthropic",
                    "model": model,
                    "usage": _usage(tokens),
                    "stopReason": stop_reason,
                },
            }
        )

    def tool_result(
        self, call_id: str, is_error: bool = False, tool_name: str = "ipython"
    ) -> str:
        return self._entry(
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": call_id,
                    "toolName": tool_name,
                    "content": [{"type": "text", "text": "output"}],
                    "isError": is_error,
                },
            }
        )

    def bash(self, command: str, exit_code: int = 0) -> str:
        return self._entry(
            {
                "type": "message",
                "message": {
                    "role": "bashExecution",
                    "command": command,
                    "output": "",
                    "exitCode": exit_code,
                    "cancelled": False,
                    "truncated": False,
                },
            }
        )

    def compaction(self, tokens_before: int = 50_000) -> str:
        return self._entry(
            {
                "type": "compaction",
                "summary": "earlier work summarized",
                "firstKeptEntryId": "e0000001",
                "tokensBefore": tokens_before,
            }
        )

    def refinement(
        self,
        refinement_id: str = "ref-1",
        rationale: str = "`pytest tests/test_x.py` failed twice with the same error",
        edits: Optional[List[Dict[str, Any]]] = None,
        scope: str = "local",
        rollback_of: Optional[str] = None,
    ) -> str:
        if edits is None:
            edits = [
                {
                    "action": "create",
                    "kind": "memory",
                    "id": "mem-1",
                    "title": "check git status first",
                    "content": "Always run git status before committing.",
                    "applied": True,
                    "reason": "observed in this trajectory",
                }
            ]
        data: Dict[str, Any] = {
            "id": refinement_id,
            "summary": "persist a lesson",
            "rationale": rationale,
            "expectedOutcome": "fewer repeated failures",
            "appliedEdits": edits,
            "harnessStatePath": "/home/u/.prime/agent/harness/harness_state.json",
            "scope": scope,
        }
        if rollback_of:
            data["rollbackOf"] = rollback_of
        return self._entry(
            {
                "type": "custom",
                "customType": REFINEMENT_CUSTOM_TYPE,
                "data": data,
            }
        )

    def child_usage(self, target_id: str, tokens: int = 500) -> str:
        return self._entry(
            {
                "type": "child_usage_attributed",
                "targetId": target_id,
                "childUsage": _usage(tokens),
                "aggregateUsage": _usage(tokens + 100),
            }
        )

    def write(self, directory: Path, name: Optional[str] = None) -> Path:
        session_id = self.lines[0]["id"]
        path = directory / f"{name or session_id}.jsonl"
        path.write_text(
            "\n".join(json.dumps(line) for line in self.lines) + "\n",
            encoding="utf-8",
        )
        return path


@pytest.fixture
def simple_session(tmp_path: Path) -> Path:
    builder = SessionBuilder()
    builder.user("fix the failing test")
    call = "call_1"
    builder.assistant(tool_calls=[{"id": call, "code": "print('hi')"}])
    builder.tool_result(call)
    builder.assistant(text="done", stop_reason="stop")
    return builder.write(tmp_path)


# ---------------------------------------------------------------------------
# Client: parsing
# ---------------------------------------------------------------------------


class TestClientParsing:
    def test_parses_header_and_messages(self, simple_session: Path):
        traj = PrimeAgentClient().parse_session(str(simple_session))

        assert traj.session_id == "sess-root"
        assert traj.version == 3
        assert traj.cwd == "/repo"
        assert traj.user_messages == 1
        assert traj.assistant_messages == 2
        assert traj.turns == 2
        assert traj.outcome is SessionOutcome.COMPLETED
        assert traj.models == ["claude-sonnet-4-5"]
        assert traj.duration_seconds > 0

    def test_tool_errors_counted(self, tmp_path: Path):
        builder = SessionBuilder()
        builder.user()
        builder.assistant(
            tool_calls=[
                {"id": "c1", "code": "1/0"},
                {"id": "c2", "code": "print(1)"},
            ]
        )
        builder.tool_result("c1", is_error=True)
        builder.tool_result("c2", is_error=False)
        path = builder.write(tmp_path)

        traj = PrimeAgentClient().parse_session(str(path))
        assert len(traj.tool_calls) == 2
        assert traj.tool_errors == 1

    def test_bash_execution_nonzero_exit_is_error(self, tmp_path: Path):
        builder = SessionBuilder()
        builder.user()
        builder.bash("npm run check", exit_code=1)
        builder.assistant(stop_reason="stop")
        path = builder.write(tmp_path)

        traj = PrimeAgentClient().parse_session(str(path))
        assert traj.tool_errors == 1

    def test_usage_and_child_usage_kept_separate(self, tmp_path: Path):
        builder = SessionBuilder()
        builder.user()
        target = builder.assistant(tokens=200)
        builder.child_usage(target, tokens=1000)
        path = builder.write(tmp_path)

        traj = PrimeAgentClient().parse_session(str(path))
        assert traj.usage.total_tokens == 200
        assert traj.child_usage.total_tokens == 1000

    def test_compaction_recorded(self, tmp_path: Path):
        builder = SessionBuilder()
        builder.user()
        builder.compaction(tokens_before=42_000)
        builder.assistant()
        path = builder.write(tmp_path)

        traj = PrimeAgentClient().parse_session(str(path))
        assert len(traj.compactions) == 1
        assert traj.compactions[0].tokens_before == 42_000

    @pytest.mark.parametrize(
        "stop_reason,expected",
        [
            ("stop", SessionOutcome.COMPLETED),
            ("length", SessionOutcome.TRUNCATED),
            ("aborted", SessionOutcome.ABORTED),
            ("error", SessionOutcome.ERRORED),
            ("weird", SessionOutcome.UNKNOWN),
        ],
    )
    def test_stop_reason_maps_to_outcome(
        self, tmp_path: Path, stop_reason: str, expected: SessionOutcome
    ):
        builder = SessionBuilder()
        builder.user()
        builder.assistant(stop_reason=stop_reason)
        path = builder.write(tmp_path, name=f"s-{stop_reason}")

        traj = PrimeAgentClient().parse_session(str(path))
        assert traj.outcome is expected

    def test_malformed_lines_are_skipped(self, tmp_path: Path):
        path = tmp_path / "broken.jsonl"
        path.write_text(
            '{"type":"session","version":3,"id":"s1","cwd":"/repo"}\n'
            "not json at all\n"
            '{"type":"message","id":"e1","parentId":null,'
            '"timestamp":"2026-08-01T10:00:00.000Z",'
            '"message":{"role":"user","content":"hi"}}\n',
            encoding="utf-8",
        )
        traj = PrimeAgentClient().parse_session(str(path))
        assert traj.user_messages == 1

    def test_empty_file_raises(self, tmp_path: Path):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError):
            PrimeAgentClient().parse_session(str(path))

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            PrimeAgentClient().parse_session(str(tmp_path / "nope.jsonl"))

    def test_discover_sessions_on_missing_root_is_empty(self, tmp_path: Path):
        client = PrimeAgentClient(
            PrimeAgentClientConfig(session_root=str(tmp_path / "absent"))
        )
        assert client.discover_sessions() == []

    def test_discover_sessions_lists_files(self, simple_session: Path):
        client = PrimeAgentClient(
            PrimeAgentClientConfig(session_root=str(simple_session.parent))
        )
        assert simple_session in client.discover_sessions()


class TestContextPath:
    def test_abandoned_branch_included_by_default(self, tmp_path: Path):
        """A refinement on an abandoned branch still mutated durable state."""
        builder = SessionBuilder()
        root = builder.user()
        builder.refinement("ref-branch")  # child of root, then abandoned
        builder.assistant(stop_reason="stop")
        # Restart from root, creating a second branch that becomes the leaf.
        builder._entry(
            {"type": "message", "message": {"role": "user", "content": "retry"}},
            parent_id=root,
        )
        builder.assistant(text="second branch", stop_reason="stop")
        path = builder.write(tmp_path)

        default = PrimeAgentClient().parse_session(str(path))
        assert len(default.refinements) == 1
        assert default.entries_on_context_path < default.entries_total

        scoped = PrimeAgentClient(
            PrimeAgentClientConfig(context_path_only=True)
        ).parse_session(str(path))
        assert scoped.refinements == []


# ---------------------------------------------------------------------------
# Client: RLM spawn detection
# ---------------------------------------------------------------------------


class TestRlmSpawnDetection:
    def _spawns(self, code: str, tmp_path: Path) -> List[RlmSpawnRecord]:
        builder = SessionBuilder()
        builder.user()
        builder.assistant(tool_calls=[{"id": "c1", "code": code}])
        builder.tool_result("c1")
        path = builder.write(tmp_path, name="spawn")
        return PrimeAgentClient().parse_session(str(path)).rlm_spawns

    def test_detects_awaited_spawn_with_name(self, tmp_path: Path):
        spawns = self._spawns(
            'handle = await rlm("Review the auth flow", name="auth-reviewer")',
            tmp_path,
        )
        assert len(spawns) == 1
        assert spawns[0].name == "auth-reviewer"
        assert spawns[0].prompt == "Review the auth flow"

    def test_detects_multiple_spawns_in_one_cell(self, tmp_path: Path):
        spawns = self._spawns(
            'a = await rlm("Review the public API", name="api-reviewer")\n'
            'b = await rlm("Review test coverage", name="test-reviewer")\n',
            tmp_path,
        )
        assert [s.name for s in spawns] == ["api-reviewer", "test-reviewer"]

    def test_registry_calls_are_not_spawns(self, tmp_path: Path):
        spawns = self._spawns(
            "children = await rlm.list_subagents()\n"
            "await rlm.delete_subagent(children[0])\n"
            "await rlm.host_request({'kind': 'goal'})\n",
            tmp_path,
        )
        assert spawns == []

    def test_parentheses_inside_prompt_do_not_truncate(self, tmp_path: Path):
        spawns = self._spawns(
            'await rlm("Audit foo(bar) and baz(qux)", name="auditor")', tmp_path
        )
        assert len(spawns) == 1
        assert spawns[0].name == "auditor"
        assert "foo(bar)" in spawns[0].prompt

    def test_triple_quoted_prompt(self, tmp_path: Path):
        spawns = self._spawns(
            'await rlm("""Multi\nline\ntask""", name="long")', tmp_path
        )
        assert len(spawns) == 1
        assert "Multi" in spawns[0].prompt

    def test_unterminated_call_is_not_counted(self, tmp_path: Path):
        assert self._spawns('await rlm("truncated cell', tmp_path) == []

    def test_fanout_from_single_cell(self, tmp_path: Path):
        builder = SessionBuilder()
        builder.user()
        code = "\n".join(
            f'await rlm("task {i}", name="w{i}")' for i in range(5)
        )
        builder.assistant(tool_calls=[{"id": "c1", "code": code}])
        path = builder.write(tmp_path, name="fanout")

        traj = PrimeAgentClient().parse_session(str(path))
        assert len(traj.rlm_spawns) == 5
        assert traj.max_spawn_fanout == 5


# ---------------------------------------------------------------------------
# Harness: evidence heuristic and drift tracking
# ---------------------------------------------------------------------------


class TestEvidenceHeuristic:
    @pytest.mark.parametrize(
        "text",
        [
            "swarm/core/proxy.py raised on the second call",
            "running `npm run check` produced a type error",
            "the tool call failed with a traceback",
            'the model replied "cannot resolve module foo" twice',
            "turn 4 repeated the same edit",
        ],
    )
    def test_concrete_evidence_accepted(self, text: str):
        assert has_concrete_evidence(text)
        assert evidence_kinds(text)

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "It seems better to be more careful in general.",
            "This is a good lesson to remember going forward.",
        ],
    )
    def test_vague_rationale_rejected(self, text: str):
        assert not has_concrete_evidence(text)

    def test_custom_predicate_overrides_default(self):
        tracker = HarnessTracker(evidence_predicate=lambda r: False)
        refinement = RefinementRecord(
            refinement_id="r1",
            rationale="swarm/core/proxy.py raised on the second call",
        )
        assert not tracker.is_evidence_backed(refinement)


class TestHarnessTracker:
    def _refinement(self, **kwargs: Any) -> RefinementRecord:
        defaults: Dict[str, Any] = {
            "refinement_id": "r1",
            "rationale": "`pytest` failed with the same error twice",
            "edits": [
                HarnessEditRecord(
                    action="create", kind="memory", entry_id="m1", content_chars=100
                )
            ],
        }
        defaults.update(kwargs)
        return RefinementRecord(**defaults)

    def test_entry_counts_by_kind(self):
        tracker = HarnessTracker()
        tracker.record_refinement(
            "a",
            self._refinement(
                edits=[
                    HarnessEditRecord(action="create", kind="memory", entry_id="m1"),
                    HarnessEditRecord(action="create", kind="skill", entry_id="s1"),
                    HarnessEditRecord(action="update", kind="memory", entry_id="m1"),
                ]
            ),
        )
        state = tracker.get_state("a")
        assert state.entries_by_kind["memory"] == 1
        assert state.entries_by_kind["skill"] == 1
        assert state.total_entries == 2

    def test_delete_reduces_count_and_never_goes_negative(self):
        tracker = HarnessTracker()
        tracker.record_refinement(
            "a",
            self._refinement(
                edits=[
                    HarnessEditRecord(action="delete", kind="memory", entry_id="m9")
                ]
            ),
        )
        assert tracker.get_state("a").total_entries == 0

    def test_unapplied_edit_does_not_move_counts(self):
        tracker = HarnessTracker()
        tracker.record_refinement(
            "a",
            self._refinement(
                edits=[
                    HarnessEditRecord(
                        action="create",
                        kind="memory",
                        entry_id="m1",
                        applied=False,
                        error="conflict",
                    )
                ]
            ),
        )
        state = tracker.get_state("a")
        assert state.total_entries == 0
        assert state.failed_edits == 1

    def test_unsupported_refinement_rate(self):
        tracker = HarnessTracker()
        tracker.record_refinement("a", self._refinement(refinement_id="r1"))
        tracker.record_refinement(
            "a", self._refinement(refinement_id="r2", rationale="feels right")
        )
        state = tracker.get_state("a")
        assert state.refinements == 2
        assert state.refinements_with_evidence == 1
        assert state.unsupported_refinement_rate == pytest.approx(0.5)

    def test_rollback_churn(self):
        tracker = HarnessTracker()
        tracker.record_refinement("a", self._refinement(refinement_id="r1"))
        tracker.record_refinement(
            "a", self._refinement(refinement_id="r2", rollback_of="r1")
        )
        assert tracker.get_state("a").rollback_churn == pytest.approx(0.5)

    def test_base_prompt_attempt_flagged(self):
        tracker = HarnessTracker()
        tracker.record_refinement(
            "a",
            self._refinement(
                edits=[
                    HarnessEditRecord(
                        action="update",
                        kind="prompt",
                        entry_id=BASE_SYSTEM_PROMPT_ID,
                    )
                ]
            ),
        )
        assert tracker.get_state("a").base_prompt_attempts == 1

    def test_growth_rate_uses_turns(self, tmp_path: Path):
        builder = SessionBuilder()
        builder.user()
        builder.assistant()
        builder.refinement("ref-1")
        builder.assistant()
        path = builder.write(tmp_path, name="growth")

        traj = PrimeAgentClient().parse_session(str(path))
        tracker = HarnessTracker()
        state = tracker.update("a", traj)
        assert state.turns_observed == 2
        assert state.growth_rate == pytest.approx(0.5)

    def test_drift_score_bounded(self):
        tracker = HarnessTracker()
        for i in range(10):
            tracker.record_refinement(
                "a",
                self._refinement(
                    refinement_id=f"r{i}", rationale="", rollback_of=f"r{i - 1}"
                ),
            )
        assert 0.0 <= tracker.get_state("a").drift_score <= 1.0

    def test_reset(self):
        tracker = HarnessTracker()
        tracker.record_refinement("a", self._refinement())
        tracker.reset("a")
        assert tracker.get_state("a").refinements == 0


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def _enabled_policy(**policy_kwargs: Any) -> HarnessRefinementPolicy:
    return HarnessRefinementPolicy(
        governance_config=GovernanceConfig(self_evolution_enabled=True),
        policy_config=RefinementPolicyConfig(**policy_kwargs),
    )


class TestRefinementPolicy:
    def _refinement(self, **kwargs: Any) -> RefinementRecord:
        defaults: Dict[str, Any] = {
            "refinement_id": "r1",
            "rationale": "`pytest tests/test_x.py` failed with a traceback",
            "edits": [
                HarnessEditRecord(
                    action="create", kind="memory", entry_id="m1", content_chars=100
                )
            ],
        }
        defaults.update(kwargs)
        return RefinementRecord(**defaults)

    def test_disabled_governance_approves_everything(self):
        policy = HarnessRefinementPolicy()  # self_evolution_enabled defaults False
        tracker = HarnessTracker()
        refinement = self._refinement(
            edits=[
                HarnessEditRecord(
                    action="update", kind="prompt", entry_id=BASE_SYSTEM_PROMPT_ID
                )
            ]
        )
        state = tracker.record_refinement("a", refinement)
        result = policy.evaluate_refinement(refinement, state)
        assert result.decision is PolicyDecision.APPROVE

    def test_base_prompt_edit_denied(self):
        policy = _enabled_policy()
        refinement = self._refinement(
            edits=[
                HarnessEditRecord(
                    action="update", kind="prompt", entry_id=BASE_SYSTEM_PROMPT_ID
                )
            ]
        )
        state = policy.tracker.record_refinement("a", refinement)
        result = policy.evaluate_refinement(refinement, state)
        assert result.decision is PolicyDecision.DENY
        assert "base system prompt" in result.reason

    def test_growth_rate_cap_denies(self):
        policy = _enabled_policy()
        refinement = self._refinement()
        state = policy.tracker.record_refinement("a", refinement)
        state.turns_observed = 1  # 1 entry / 1 turn = 1.0 >> default 0.1
        result = policy.evaluate_refinement(refinement, state)
        assert result.decision is PolicyDecision.DENY
        assert "growth rate" in result.reason

    def test_entry_count_cap_denies(self):
        policy = _enabled_policy()
        refinement = self._refinement()
        state = policy.tracker.record_refinement("a", refinement)
        state.entries_by_kind["memory"] = 999
        state.turns_observed = 100_000  # keep growth rate under its own cap
        result = policy.evaluate_refinement(refinement, state)
        assert result.decision is PolicyDecision.DENY
        assert "entry count" in result.reason

    def test_unsupported_refinement_warns_by_default(self):
        policy = _enabled_policy()
        refinement = self._refinement(rationale="this feels like a good habit")
        state = policy.tracker.record_refinement("a", refinement)
        state.turns_observed = 100
        result = policy.evaluate_refinement(refinement, state)
        assert result.decision is PolicyDecision.WARN
        assert "no concrete evidence" in result.reason

    def test_unsupported_refinement_denied_when_required(self):
        policy = _enabled_policy(require_evidence=True)
        refinement = self._refinement(rationale="this feels like a good habit")
        state = policy.tracker.record_refinement("a", refinement)
        state.turns_observed = 100
        result = policy.evaluate_refinement(refinement, state)
        assert result.decision is PolicyDecision.DENY

    def test_global_scope_denied_for_negative_reputation(self):
        policy = _enabled_policy()
        refinement = self._refinement(scope="global")
        state = policy.tracker.record_refinement("a", refinement)
        state.turns_observed = 100
        result = policy.evaluate_refinement(refinement, state, reputation=-0.5)
        assert result.decision is PolicyDecision.DENY
        assert "cross-session" in result.reason

    def test_global_scope_allowed_for_positive_reputation(self):
        policy = _enabled_policy()
        refinement = self._refinement(scope="global")
        state = policy.tracker.record_refinement("a", refinement)
        state.turns_observed = 100
        result = policy.evaluate_refinement(refinement, state, reputation=1.0)
        assert result.decision is PolicyDecision.APPROVE

    def test_oversized_refinement_warns(self):
        policy = _enabled_policy(max_refinement_chars=50)
        refinement = self._refinement(
            edits=[
                HarnessEditRecord(
                    action="create", kind="memory", entry_id="m1", content_chars=500
                )
            ]
        )
        state = policy.tracker.record_refinement("a", refinement)
        state.turns_observed = 100
        result = policy.evaluate_refinement(refinement, state)
        assert result.decision is PolicyDecision.WARN
        assert "small-update budget" in result.reason

    def test_oversized_refinement_denied_for_negative_reputation(self):
        policy = _enabled_policy(max_refinement_chars=50)
        refinement = self._refinement(
            edits=[
                HarnessEditRecord(
                    action="create", kind="memory", entry_id="m1", content_chars=500
                )
            ]
        )
        state = policy.tracker.record_refinement("a", refinement)
        state.turns_observed = 100
        result = policy.evaluate_refinement(refinement, state, reputation=-0.2)
        assert result.decision is PolicyDecision.DENY

    def test_evidence_backed_within_limits_approves(self):
        policy = _enabled_policy()
        refinement = self._refinement()
        state = policy.tracker.record_refinement("a", refinement)
        state.turns_observed = 100
        result = policy.evaluate_refinement(refinement, state)
        assert result.decision is PolicyDecision.APPROVE
        assert result.governance_cost == 0.0


class TestSpawnPolicy:
    def test_depth_cap_denies(self):
        policy = _enabled_policy(max_recursion_depth=1)
        state = policy.tracker.get_state("a")
        result = policy.evaluate_spawn(RlmSpawnRecord(depth=1), state)
        assert result.decision is PolicyDecision.DENY
        assert "recursion depth" in result.reason

    def test_depth_within_cap_approves(self):
        policy = _enabled_policy(max_recursion_depth=2)
        state = policy.tracker.get_state("a")
        result = policy.evaluate_spawn(RlmSpawnRecord(depth=1), state)
        assert result.decision is PolicyDecision.APPROVE

    def test_fanout_cap_denies(self):
        policy = _enabled_policy(max_spawn_fanout=2)
        state = policy.tracker.get_state("a")
        result = policy.evaluate_spawn(RlmSpawnRecord(depth=0), state, fanout=5)
        assert result.decision is PolicyDecision.DENY
        assert "fan-out" in result.reason

    def test_total_children_cap_denies(self):
        policy = _enabled_policy(max_children_total=3)
        state = policy.tracker.get_state("a")
        state.children_spawned = 10
        result = policy.evaluate_spawn(RlmSpawnRecord(depth=0), state)
        assert result.decision is PolicyDecision.DENY

    def test_disabled_governance_approves(self):
        policy = HarnessRefinementPolicy()
        state = policy.tracker.get_state("a")
        result = policy.evaluate_spawn(RlmSpawnRecord(depth=99), state, fanout=99)
        assert result.decision is PolicyDecision.APPROVE


class TestCircuitBreaker:
    def test_base_prompt_attempt_trips_breaker(self):
        policy = _enabled_policy()
        state = policy.tracker.get_state("a")
        state.base_prompt_attempts = 1
        assert policy.should_circuit_break(state)

    def test_failed_edit_flood_trips_breaker(self):
        policy = _enabled_policy(max_failed_edits=2)
        state = policy.tracker.get_state("a")
        state.failed_edits = 3
        assert policy.should_circuit_break(state)

    def test_clean_state_does_not_trip(self):
        policy = _enabled_policy()
        assert not policy.should_circuit_break(policy.tracker.get_state("a"))

    def test_disabled_governance_never_trips(self):
        policy = HarnessRefinementPolicy()
        state = policy.tracker.get_state("a")
        state.base_prompt_attempts = 5
        assert not policy.should_circuit_break(state)

    def test_drift_penalty_zero_below_threshold(self):
        policy = _enabled_policy()
        state = policy.tracker.get_state("a")
        assert policy.compute_drift_penalty(state) == 0.0

    def test_drift_penalty_scales_above_threshold(self):
        policy = HarnessRefinementPolicy(
            governance_config=GovernanceConfig(
                self_evolution_enabled=True,
                self_evolution_divergence_threshold=0.0,
            )
        )
        state = policy.tracker.get_state("a")
        state.refinements = 4
        state.refinements_with_evidence = 0
        state.rollbacks = 4
        state.turns_observed = 1
        state.entries_by_kind["memory"] = 1
        penalty = policy.compute_drift_penalty(state)
        assert 0.0 < penalty <= 1.0


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


class TestBridgeScoring:
    def test_scores_a_session(self, simple_session: Path):
        bridge = PrimeAgentBridge()
        interaction = bridge.analyze_session(str(simple_session))

        assert 0.0 <= interaction.p <= 1.0
        assert -1.0 <= interaction.v_hat <= 1.0
        assert interaction.counterparty == "sess-root"
        assert interaction.metadata["bridge"] == "prime_agent"
        assert interaction.metadata["outcome"] == "completed"

    def test_ungated_completion_is_a_weak_positive(self, simple_session: Path):
        """A clean stop is not verification, so it must not read as success.

        Asserts the progress signal the bridge controls, not an absolute p:
        ``ProxyComputer`` treats zero rework/rejections/misuse as +1 apiece, so
        any clean session starts high regardless of what a bridge contributes.
        The claim under test is that an ungated stop earns the weak 0.3 rather
        than the 0.8 a verified gate earns.
        """
        bridge = PrimeAgentBridge()
        interaction = bridge.analyze_session(str(simple_session))

        assert interaction.metadata["gate_result"] == GateResult.ABSENT.value
        assert interaction.task_progress_delta == pytest.approx(0.3)
        assert interaction.task_progress_delta < GATED_PROGRESS[GateResult.PASSED]

    def test_passing_gate_beats_ungated_completion(self, tmp_path: Path):
        def build(exit_code: int, name: str) -> Path:
            builder = SessionBuilder()
            builder.user()
            builder.bash("npm run check", exit_code=exit_code)
            builder.assistant(stop_reason="stop")
            return builder.write(tmp_path, name=name)

        passing = build(0, "gate-pass")
        failing = build(1, "gate-fail")
        config = PrimeAgentBridgeConfig(
            client_config=PrimeAgentClientConfig(gate_command="npm run check")
        )

        pass_p = PrimeAgentBridge(config).analyze_session(str(passing)).p
        fail_p = PrimeAgentBridge(config).analyze_session(str(failing)).p
        ungated_p = PrimeAgentBridge().analyze_session(str(passing)).p

        assert pass_p > ungated_p > fail_p

    def test_error_outcome_scores_below_completion(self, tmp_path: Path):
        def build(stop_reason: str, name: str) -> Path:
            builder = SessionBuilder()
            builder.user()
            builder.assistant(stop_reason=stop_reason)
            return builder.write(tmp_path, name=name)

        ok = PrimeAgentBridge().analyze_session(str(build("stop", "ok")))
        bad = PrimeAgentBridge().analyze_session(str(build("error", "bad")))
        assert ok.p > bad.p

    def test_failed_harness_edits_count_as_verifier_rejections(
        self, tmp_path: Path
    ):
        builder = SessionBuilder()
        builder.user()
        builder.assistant()
        builder.refinement(
            "ref-1",
            edits=[
                {
                    "action": "create",
                    "kind": "memory",
                    "id": "m1",
                    "applied": False,
                    "error": "conflicting baseline",
                },
                {
                    "action": "update",
                    "kind": "prompt",
                    "id": BASE_SYSTEM_PROMPT_ID,
                    "applied": False,
                    "error": "immutable",
                },
            ],
        )
        path = builder.write(tmp_path, name="rejections")

        interaction = PrimeAgentBridge().analyze_session(str(path))
        # 2 failed edits + 1 base-prompt attempt
        assert interaction.verifier_rejections == 3

    def test_policy_denials_become_tool_misuse_flags(self, tmp_path: Path):
        builder = SessionBuilder()
        builder.user()
        builder.assistant()
        builder.refinement(
            "ref-1",
            edits=[
                {
                    "action": "update",
                    "kind": "prompt",
                    "id": BASE_SYSTEM_PROMPT_ID,
                    "content": "rewrite everything",
                    "applied": True,
                }
            ],
        )
        path = builder.write(tmp_path, name="denied")

        bridge = PrimeAgentBridge(
            PrimeAgentBridgeConfig(
                governance_config=GovernanceConfig(self_evolution_enabled=True)
            )
        )
        interaction = bridge.analyze_session(str(path))
        assert interaction.tool_misuse_flags >= 1
        assert interaction.metadata["policy_denials"]

    def test_circuit_break_rejects_the_interaction(self, tmp_path: Path):
        builder = SessionBuilder()
        builder.user()
        builder.assistant()
        builder.refinement(
            "ref-1",
            edits=[
                {
                    "action": "update",
                    "kind": "prompt",
                    "id": BASE_SYSTEM_PROMPT_ID,
                    "applied": True,
                }
            ],
        )
        path = builder.write(tmp_path, name="breaker")

        bridge = PrimeAgentBridge(
            PrimeAgentBridgeConfig(
                governance_config=GovernanceConfig(self_evolution_enabled=True)
            )
        )
        interaction = bridge.analyze_session(str(path))
        assert interaction.accepted is False
        assert interaction.metadata["circuit_broken"] is True

    def test_drift_state_exposed(self, tmp_path: Path):
        builder = SessionBuilder()
        builder.user()
        builder.assistant()
        builder.refinement("ref-1")
        path = builder.write(tmp_path, name="drift")

        bridge = PrimeAgentBridge()
        bridge.analyze_session(str(path))
        state = bridge.get_drift_state("sess-root")
        assert state.refinements == 1
        assert state.refinements_with_evidence == 1

    def test_bridge_events_recorded(self, tmp_path: Path):
        builder = SessionBuilder()
        builder.user()
        builder.assistant(tool_calls=[{"id": "c1", "code": 'await rlm("go")'}])
        builder.tool_result("c1")
        builder.refinement("ref-1")
        path = builder.write(tmp_path, name="events")

        bridge = PrimeAgentBridge()
        bridge.analyze_session(str(path))
        kinds = {e.event_type for e in bridge.get_bridge_events()}
        assert PrimeAgentEventType.SESSION_STARTED in kinds
        assert PrimeAgentEventType.SESSION_COMPLETED in kinds
        assert PrimeAgentEventType.RLM_SPAWNED in kinds
        assert PrimeAgentEventType.HARNESS_REFINED in kinds

    def test_reputation_gates_global_refinement(self, tmp_path: Path):
        builder = SessionBuilder()
        builder.user()
        builder.assistant()
        builder.refinement("ref-1", scope="global")
        path = builder.write(tmp_path, name="global")

        bridge = PrimeAgentBridge(
            PrimeAgentBridgeConfig(
                governance_config=GovernanceConfig(self_evolution_enabled=True)
            )
        )
        bridge.update_agent_reputation("sess-root", -1.0)
        interaction = bridge.analyze_session(str(path))
        assert interaction.tool_misuse_flags >= 1

    def test_analyze_directory(self, tmp_path: Path):
        for i in range(3):
            builder = SessionBuilder(session_id=f"s{i}")
            builder.user()
            builder.assistant()
            builder.write(tmp_path)

        interactions = PrimeAgentBridge().analyze_directory(str(tmp_path))
        assert len(interactions) == 3

    def test_analyze_directory_rejects_non_directory(self, simple_session: Path):
        with pytest.raises(NotADirectoryError):
            PrimeAgentBridge().analyze_directory(str(simple_session))

    def test_metrics_include_harness_aggregates(self, tmp_path: Path):
        builder = SessionBuilder()
        builder.user()
        builder.assistant()
        builder.refinement("ref-1", rationale="vague feeling")
        builder.write(tmp_path)

        bridge = PrimeAgentBridge()
        bridge.analyze_directory(str(tmp_path))
        metrics = bridge.get_metrics()
        assert metrics["n_sessions"] == 1
        assert metrics["total_refinements"] == 1
        assert metrics["unsupported_refinement_rate"] == pytest.approx(1.0)
        assert 0.0 <= metrics["toxicity_rate"] <= 1.0


class TestDelegationTree:
    def _tree(self, tmp_path: Path) -> Path:
        parent = SessionBuilder(session_id="parent")
        parent.user()
        parent.assistant(
            tool_calls=[
                {"id": "c1", "code": 'await rlm("sub task", name="child-1")'}
            ]
        )
        parent.tool_result("c1")
        parent_path = parent.write(tmp_path)

        child = SessionBuilder(
            session_id="child", parent_session=str(parent_path)
        )
        child.user()
        child.assistant()
        child.write(tmp_path)
        return tmp_path

    def test_depth_assignment(self, tmp_path: Path):
        self._tree(tmp_path)
        client = PrimeAgentClient()
        trajectories = client.parse_sessions(
            [str(p) for p in sorted(tmp_path.glob("*.jsonl"))]
        )
        build_session_tree(trajectories)
        by_id = {t.session_id: t for t in trajectories}
        assert by_id["parent"].depth == 0
        assert by_id["child"].depth == 1

    def test_child_interaction_links_to_parent(self, tmp_path: Path):
        self._tree(tmp_path)
        bridge = PrimeAgentBridge()
        interactions = bridge.analyze_session_tree(str(tmp_path))

        assert len(interactions) == 2
        parent_i = next(i for i in interactions if i.counterparty == "parent")
        child_i = next(i for i in interactions if i.counterparty == "child")
        assert parent_i.causal_parents == []
        assert child_i.causal_parents == [parent_i.interaction_id]
        assert child_i.initiator == "parent"

    def test_orphan_child_is_scored_as_a_root(self, tmp_path: Path):
        orphan = SessionBuilder(
            session_id="orphan", parent_session="/elsewhere/gone.jsonl"
        )
        orphan.user()
        orphan.assistant()
        orphan.write(tmp_path)

        interactions = PrimeAgentBridge().analyze_session_tree(str(tmp_path))
        assert len(interactions) == 1
        assert interactions[0].causal_parents == []

    def test_empty_directory_returns_nothing(self, tmp_path: Path):
        assert PrimeAgentBridge().analyze_session_tree(str(tmp_path)) == []


class TestEventSerialization:
    def test_event_roundtrip(self):
        event = PrimeAgentEvent(
            event_type=PrimeAgentEventType.HARNESS_REFINED,
            agent_id="a",
            payload={"k": 1},
        )
        restored = PrimeAgentEvent.from_dict(event.to_dict())
        assert restored.event_type is event.event_type
        assert restored.agent_id == "a"
        assert restored.payload == {"k": 1}

    def test_refinement_roundtrip_from_upstream_shape(self):
        data = {
            "id": "ref-9",
            "summary": "s",
            "rationale": "r",
            "expectedOutcome": "e",
            "scope": "global",
            "rollbackOf": "ref-8",
            "harnessStatePath": "/h.json",
            "appliedEdits": [
                {
                    "action": "create",
                    "kind": "skill",
                    "id": "sk1",
                    "title": "t",
                    "applied": True,
                    "after": {"content": "abcd"},
                }
            ],
        }
        record = RefinementRecord.from_dict(data, entry_id="e1")
        assert record.refinement_id == "ref-9"
        assert record.is_rollback
        assert record.scope == "global"
        assert record.edits[0].content_chars == 4
        assert record.net_entry_delta == 1
        assert "e" in record.evidence_text

    def test_trajectory_to_dict_is_json_serializable(self, simple_session: Path):
        traj = PrimeAgentClient().parse_session(str(simple_session))
        json.dumps(traj.to_dict())


class TestInvariants:
    """p must stay in [0, 1] across the whole observable range."""

    @pytest.mark.parametrize("stop_reason", ["stop", "length", "aborted", "error"])
    @pytest.mark.parametrize("n_errors", [0, 5, 50])
    def test_p_stays_a_probability(
        self, tmp_path: Path, stop_reason: str, n_errors: int
    ):
        builder = SessionBuilder()
        builder.user()
        calls = [{"id": f"c{i}", "code": "boom"} for i in range(n_errors)]
        builder.assistant(tool_calls=calls, stop_reason=stop_reason)
        for call in calls:
            builder.tool_result(call["id"], is_error=True)
        path = builder.write(tmp_path, name=f"inv-{stop_reason}-{n_errors}")

        interaction = PrimeAgentBridge().analyze_session(str(path))
        assert 0.0 <= interaction.p <= 1.0
        assert -1.0 <= interaction.v_hat <= 1.0
