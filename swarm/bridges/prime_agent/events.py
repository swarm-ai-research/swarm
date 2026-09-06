"""Typed records for Prime Agent session artifacts.

Prime Agent (https://github.com/PrimeIntellect-ai/prime-agent) persists every
session as JSONL under ``~/.prime/agent/sessions/<session-id>.jsonl``. Entries
form a tree via ``id`` / ``parentId``; the leaf-to-root path is the context the
model actually saw. This module defines the Python-side records the bridge
parses those entries into.

The three artifacts SWARM cares about, none of which exist in a conventional
agent transcript:

1. **Harness refinements** — ``custom`` entries with
   ``customType == "prime-agent.refinement"`` whose ``data`` is a
   ``RefinementResult``. Each carries applied create/update/delete edits over
   durable harness state (prompts, memories, skills, subagent specs). This is
   self-modification with a persistence layer, so it is governable.
2. **RLM spawns** — ``rlm(...)`` calls inside ``ipython`` tool code, which
   create real child sessions with their own session files. This is a
   delegation DAG, so credit and quality can be propagated along it.
3. **Budget/outcome signals** — ``stopReason``, compaction entries, and
   token usage, which say when a session stopped but deliberately do *not*
   say whether it succeeded.

Field names mirror Prime Agent's TypeScript interfaces (``RefinementResult``,
``AppliedRefinementEdit``, ``Usage``, ``SessionEntryBase``) so the mapping
stays auditable against upstream.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

# Prime Agent's immutable base prompt id. Refinement edits targeting it are
# rejected upstream; the bridge treats an attempt as a governance signal.
BASE_SYSTEM_PROMPT_ID = "base_system_prompt"

# customType carried by refinement history entries in the session file.
REFINEMENT_CUSTOM_TYPE = "prime-agent.refinement"

# The four kinds of durable harness state a refinement may edit.
HARNESS_KINDS = ("prompt", "memory", "skill", "subagent")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PrimeAgentEventType(Enum):
    """Event types emitted by the Prime Agent bridge."""

    SESSION_STARTED = "session:started"
    SESSION_COMPLETED = "session:completed"
    TURN_COMPLETED = "turn:completed"
    TOOL_EXECUTED = "tool:executed"
    TOOL_FAILED = "tool:failed"
    RLM_SPAWNED = "rlm:spawned"
    HARNESS_REFINED = "harness:refined"
    HARNESS_ROLLBACK = "harness:rollback"
    HARNESS_EDIT_REJECTED = "harness:edit_rejected"
    BASE_PROMPT_EDIT_ATTEMPT = "harness:base_prompt_attempt"
    COMPACTION = "context:compaction"
    POLICY_DENIED = "policy:denied"
    ERROR = "error"


class SessionOutcome(Enum):
    """How a Prime Agent session stopped.

    Deliberately not named "success". Prime Agent's own documentation is
    explicit that "a passed gate checks only what that gate verifies; reaching
    a limit does not imply task success" — so the bridge records *how* the
    session stopped and, separately, whether an external gate verified
    anything. ``COMPLETED`` means the model emitted a natural stop, nothing
    more.
    """

    COMPLETED = "completed"       # stopReason == "stop"
    TRUNCATED = "truncated"       # stopReason == "length" (context/budget bound)
    ABORTED = "aborted"           # stopReason == "aborted"
    ERRORED = "errored"           # stopReason == "error"
    UNKNOWN = "unknown"           # no terminal assistant message found


class GateResult(Enum):
    """Outcome of an autonomous-mode quality gate, when one ran."""

    PASSED = "passed"
    FAILED = "failed"
    ABSENT = "absent"


@dataclass
class TokenUsage:
    """Token and cost accounting for one assistant message or a whole session."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TokenUsage":
        if not isinstance(data, dict):
            return cls()
        cost = data.get("cost")
        cost_usd = 0.0
        if isinstance(cost, dict):
            cost_usd = float(cost.get("total", 0.0) or 0.0)
        elif isinstance(cost, (int, float)):
            cost_usd = float(cost)
        return cls(
            input=int(data.get("input", 0) or 0),
            output=int(data.get("output", 0) or 0),
            cache_read=int(data.get("cacheRead", 0) or 0),
            cache_write=int(data.get("cacheWrite", 0) or 0),
            total_tokens=int(data.get("totalTokens", 0) or 0),
            cost_usd=cost_usd,
        )

    def merge(self, other: "TokenUsage") -> "TokenUsage":
        """Return the element-wise sum of two usage records."""
        return TokenUsage(
            input=self.input + other.input,
            output=self.output + other.output,
            cache_read=self.cache_read + other.cache_read,
            cache_write=self.cache_write + other.cache_write,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input": self.input,
            "output": self.output,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
        }


@dataclass
class ToolCallRecord:
    """A single tool invocation and, when resolved, its result status."""

    call_id: str = ""
    tool_name: str = ""
    code: str = ""
    is_error: bool = False
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "code_chars": len(self.code),
            "is_error": self.is_error,
            "resolved": self.resolved,
        }


@dataclass
class RlmSpawnRecord:
    """An ``rlm(...)`` child-agent spawn detected in ipython tool code.

    ``depth`` is the depth of the *spawning* session in the delegation tree
    (0 for a root session), so a child created here sits at ``depth + 1``.
    """

    call_id: str = ""
    entry_id: str = ""
    name: str = ""
    prompt: str = ""
    depth: int = 0
    timestamp: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "entry_id": self.entry_id,
            "name": self.name,
            "prompt": self.prompt[:200],
            "depth": self.depth,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HarnessEditRecord:
    """One create/update/delete edit inside a refinement.

    Mirrors Prime Agent's ``AppliedRefinementEdit``.
    """

    action: str = "create"          # create | update | delete
    kind: str = "memory"            # prompt | memory | skill | subagent
    entry_id: str = ""
    title: str = ""
    reason: str = ""
    content_chars: int = 0
    applied: bool = True
    error: str = ""

    @property
    def targets_base_prompt(self) -> bool:
        """Whether this edit tried to touch the immutable base system prompt."""
        return self.kind == "prompt" and self.entry_id == BASE_SYSTEM_PROMPT_ID

    @property
    def net_entry_delta(self) -> int:
        """Change in harness entry count from this edit (+1 / 0 / -1)."""
        if not self.applied:
            return 0
        if self.action == "create":
            return 1
        if self.action == "delete":
            return -1
        return 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "kind": self.kind,
            "entry_id": self.entry_id,
            "title": self.title,
            "reason": self.reason,
            "content_chars": self.content_chars,
            "applied": self.applied,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HarnessEditRecord":
        after = data.get("after") if isinstance(data.get("after"), dict) else {}
        content = data.get("content")
        if not isinstance(content, str):
            content = (after or {}).get("content", "")
        if not isinstance(content, str):
            content = ""
        title = data.get("title") or (after or {}).get("title", "")
        return cls(
            action=str(data.get("action", "create")),
            kind=str(data.get("kind", "memory")),
            entry_id=str(data.get("id", "")),
            title=str(title or ""),
            reason=str(data.get("reason", "") or ""),
            content_chars=len(content),
            applied=bool(data.get("applied", True)),
            error=str(data.get("error", "") or ""),
        )


@dataclass
class RefinementRecord:
    """A single ``/refine`` pass over the continual harness.

    Mirrors Prime Agent's ``RefinementResult``. ``rollback_of`` is set when the
    refinement reverts an earlier one — Prime Agent keeps snapshots precisely so
    a bad refinement can be undone, and repeated rollbacks are the signal that
    the harness is churning rather than learning.
    """

    refinement_id: str = ""
    entry_id: str = ""
    summary: str = ""
    rationale: str = ""
    expected_outcome: str = ""
    scope: str = "local"            # local | global
    rollback_of: Optional[str] = None
    harness_state_path: str = ""
    edits: List[HarnessEditRecord] = field(default_factory=list)
    timestamp: datetime = field(default_factory=_utcnow)

    @property
    def is_rollback(self) -> bool:
        return bool(self.rollback_of)

    @property
    def applied_edits(self) -> List[HarnessEditRecord]:
        return [e for e in self.edits if e.applied]

    @property
    def failed_edits(self) -> List[HarnessEditRecord]:
        return [e for e in self.edits if not e.applied]

    @property
    def net_entry_delta(self) -> int:
        return sum(e.net_entry_delta for e in self.edits)

    @property
    def total_content_chars(self) -> int:
        return sum(e.content_chars for e in self.applied_edits)

    @property
    def evidence_text(self) -> str:
        """Concatenated free text a refinement offers as its justification."""
        parts = [self.rationale, self.summary, self.expected_outcome]
        parts.extend(e.reason for e in self.edits)
        return "\n".join(p for p in parts if p)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "refinement_id": self.refinement_id,
            "entry_id": self.entry_id,
            "summary": self.summary,
            "rationale": self.rationale,
            "expected_outcome": self.expected_outcome,
            "scope": self.scope,
            "rollback_of": self.rollback_of,
            "harness_state_path": self.harness_state_path,
            "edits": [e.to_dict() for e in self.edits],
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        entry_id: str = "",
        timestamp: Optional[datetime] = None,
    ) -> "RefinementRecord":
        raw_edits = data.get("appliedEdits")
        edits = [
            HarnessEditRecord.from_dict(e)
            for e in (raw_edits if isinstance(raw_edits, list) else [])
            if isinstance(e, dict)
        ]
        return cls(
            refinement_id=str(data.get("id", "")),
            entry_id=entry_id,
            summary=str(data.get("summary", "") or ""),
            rationale=str(data.get("rationale", "") or ""),
            expected_outcome=str(data.get("expectedOutcome", "") or ""),
            scope=str(data.get("scope", "local") or "local"),
            rollback_of=data.get("rollbackOf") or None,
            harness_state_path=str(data.get("harnessStatePath", "") or ""),
            edits=edits,
            timestamp=timestamp or _utcnow(),
        )


@dataclass
class CompactionRecord:
    """A context compaction event.

    Compaction is not a completion signal — upstream is explicit about that —
    but repeated compaction is a load signal for long-horizon sessions.
    """

    entry_id: str = ""
    tokens_before: int = 0
    summary_chars: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "tokens_before": self.tokens_before,
            "summary_chars": self.summary_chars,
        }


@dataclass
class SessionTrajectory:
    """A parsed Prime Agent session, reduced to the signals SWARM scores."""

    session_id: str = ""
    version: int = 3
    cwd: str = ""
    path: str = ""
    parent_session: Optional[str] = None
    depth: int = 0

    turns: int = 0
    user_messages: int = 0
    assistant_messages: int = 0
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    tool_errors: int = 0
    rlm_spawns: List[RlmSpawnRecord] = field(default_factory=list)
    refinements: List[RefinementRecord] = field(default_factory=list)
    compactions: List[CompactionRecord] = field(default_factory=list)

    usage: TokenUsage = field(default_factory=TokenUsage)
    child_usage: TokenUsage = field(default_factory=TokenUsage)

    outcome: SessionOutcome = SessionOutcome.UNKNOWN
    gate_result: GateResult = GateResult.ABSENT
    gate_command: str = ""
    duration_seconds: float = 0.0
    models: List[str] = field(default_factory=list)
    entries_total: int = 0
    entries_on_context_path: int = 0

    @property
    def max_spawn_fanout(self) -> int:
        """Largest number of children spawned from a single ipython cell."""
        by_entry: Dict[str, int] = {}
        for spawn in self.rlm_spawns:
            by_entry[spawn.entry_id] = by_entry.get(spawn.entry_id, 0) + 1
        return max(by_entry.values(), default=0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "version": self.version,
            "cwd": self.cwd,
            "path": self.path,
            "parent_session": self.parent_session,
            "depth": self.depth,
            "turns": self.turns,
            "user_messages": self.user_messages,
            "assistant_messages": self.assistant_messages,
            "tool_calls": len(self.tool_calls),
            "tool_errors": self.tool_errors,
            "rlm_spawns": [s.to_dict() for s in self.rlm_spawns],
            "refinements": [r.to_dict() for r in self.refinements],
            "compactions": [c.to_dict() for c in self.compactions],
            "usage": self.usage.to_dict(),
            "child_usage": self.child_usage.to_dict(),
            "outcome": self.outcome.value,
            "gate_result": self.gate_result.value,
            "gate_command": self.gate_command,
            "duration_seconds": self.duration_seconds,
            "models": list(self.models),
            "entries_total": self.entries_total,
            "entries_on_context_path": self.entries_on_context_path,
        }


@dataclass
class PrimeAgentEvent:
    """Bridge-level event, mirroring the shape used by other SWARM bridges."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: PrimeAgentEventType = PrimeAgentEventType.TURN_COMPLETED
    timestamp: datetime = field(default_factory=_utcnow)
    agent_id: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrimeAgentEvent":
        ts = data.get("timestamp")
        if isinstance(ts, str):
            try:
                timestamp = datetime.fromisoformat(ts)
            except ValueError:
                timestamp = _utcnow()
        else:
            timestamp = _utcnow()
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=PrimeAgentEventType(
                data.get("event_type", "turn:completed")
            ),
            timestamp=timestamp,
            agent_id=data.get("agent_id", ""),
            payload=data.get("payload", {}),
        )
