"""Parser for Prime Agent session files.

Prime Agent writes one JSONL file per session under
``~/.prime/agent/sessions/<session-id>.jsonl``. The first line is a
``SessionHeader``; every later line is an entry with ``id`` / ``parentId``
forming a tree, so branching happens in place rather than by forking files.

``PrimeAgentClient`` is offline-only by design: it reads files Prime Agent has
already written and never launches or attaches to an agent. Prime Agent
executes model-generated Python with the user's permissions and is explicitly
*not* a security sandbox, so a governance layer that shells out to it would be
claiming an isolation boundary that does not exist. Scoring a transcript after
the fact makes no such claim.

Parsing scope
-------------
By default the client reads **every** entry in the file, not only the
leaf-to-root context path that Prime Agent's ``buildSessionContext()`` feeds
back to the model. A refinement applied on a branch that was later abandoned
still mutated durable harness state on disk, so it is still a self-modification
SWARM should see. ``PrimeAgentClientConfig.context_path_only`` restricts
parsing to the surviving path when you specifically want "what the model saw";
either way ``entries_on_context_path`` is reported so the divergence is visible.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from swarm.bridges.prime_agent.events import (
    REFINEMENT_CUSTOM_TYPE,
    CompactionRecord,
    GateResult,
    RefinementRecord,
    RlmSpawnRecord,
    SessionOutcome,
    SessionTrajectory,
    TokenUsage,
    ToolCallRecord,
)

logger = logging.getLogger(__name__)

DEFAULT_SESSION_ROOT = "~/.prime/agent/sessions"

# `rlm(` spawns a child agent. `rlm.list_subagents(` / `rlm.delete_subagent(` /
# `rlm.host_request(` are registry and host calls, not spawns — the mandatory
# `(` immediately after the name (modulo whitespace) excludes them.
RLM_SPAWN_PATTERN = re.compile(r"\brlm\s*\(")

# `name="..."` keyword inside an rlm call.
RLM_NAME_PATTERN = re.compile(r"""name\s*=\s*(['"])(.*?)\1""", re.DOTALL)

# Leading string literal (the child's task prompt), including triple-quoted.
RLM_PROMPT_PATTERN = re.compile(
    r"""^\s*(?:f|r|rf|fr)?(\"\"\"|'''|\"|')(.*?)\1""", re.DOTALL
)

_STOP_REASON_TO_OUTCOME = {
    "stop": SessionOutcome.COMPLETED,
    "toolUse": SessionOutcome.COMPLETED,
    "length": SessionOutcome.TRUNCATED,
    "aborted": SessionOutcome.ABORTED,
    "error": SessionOutcome.ERRORED,
}


@dataclass
class PrimeAgentClientConfig:
    """Configuration for the Prime Agent session parser."""

    session_root: str = DEFAULT_SESSION_ROOT
    #: Restrict parsing to the leaf-to-root context path (see module docstring).
    context_path_only: bool = False
    #: Command whose exit status counts as an autonomous-mode quality gate.
    #: Unset means no gate ran and the bridge reports ``GateResult.ABSENT``
    #: rather than inferring success from the session merely finishing.
    gate_command: str = ""
    #: Cap on stored prompt text per spawn record.
    max_prompt_chars: int = 2000
    #: Skip files larger than this (bytes). 0 disables the check.
    max_file_bytes: int = 64 * 1024 * 1024


class PrimeAgentClient:
    """Reads Prime Agent session JSONL into :class:`SessionTrajectory`."""

    def __init__(self, config: Optional[PrimeAgentClientConfig] = None) -> None:
        self.config = config or PrimeAgentClientConfig()

    # --- Discovery ---

    def discover_sessions(self, root: Optional[str] = None) -> List[Path]:
        """List session files under ``root`` (default: the configured root).

        Returns an empty list when the directory does not exist, so callers can
        probe a machine that has never run Prime Agent without special-casing.
        """
        root_path = Path(root or self.config.session_root).expanduser()
        if not root_path.is_dir():
            logger.debug("Prime Agent session root not found: %s", root_path)
            return []
        return sorted(root_path.glob("*.jsonl"))

    # --- Parsing ---

    def parse_session(self, path: str) -> SessionTrajectory:
        """Parse one session file into a trajectory.

        Args:
            path: Path to a Prime Agent ``*.jsonl`` session file.

        Returns:
            SessionTrajectory with harness refinements, RLM spawns, tool
            outcomes, usage, and the terminal stop reason.

        Raises:
            FileNotFoundError: if the file does not exist.
            ValueError: if the file contains no parsable JSON lines.
        """
        file_path = Path(path).expanduser()
        if not file_path.is_file():
            raise FileNotFoundError(f"Session file not found: {file_path}")
        if (
            self.config.max_file_bytes
            and file_path.stat().st_size > self.config.max_file_bytes
        ):
            raise ValueError(
                f"Session file exceeds max_file_bytes: {file_path} "
                f"({file_path.stat().st_size} bytes)"
            )

        header, entries = self._read_entries(file_path)
        if header is None and not entries:
            raise ValueError(f"No parsable JSON lines in {file_path}")

        context_ids = self._context_path_ids(entries)
        scanned = (
            [e for e in entries if e.get("id") in context_ids]
            if self.config.context_path_only
            else entries
        )

        trajectory = SessionTrajectory(
            session_id=str((header or {}).get("id", file_path.stem)),
            version=int((header or {}).get("version", 1) or 1),
            cwd=str((header or {}).get("cwd", "")),
            path=str(file_path),
            parent_session=(header or {}).get("parentSession") or None,
            entries_total=len(entries),
            entries_on_context_path=len(context_ids),
        )

        self._populate(trajectory, scanned, context_ids)
        return trajectory

    def parse_sessions(self, paths: Iterable[str]) -> List[SessionTrajectory]:
        """Parse many sessions, skipping (and logging) unparsable files."""
        out: List[SessionTrajectory] = []
        for p in paths:
            try:
                out.append(self.parse_session(str(p)))
            except (OSError, ValueError):
                logger.exception("Failed to parse Prime Agent session %s", p)
        return out

    # --- Internals ---

    def _read_entries(
        self, file_path: Path
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split a session file into its header and its tree entries."""
        header: Optional[Dict[str, Any]] = None
        entries: List[Dict[str, Any]] = []
        with file_path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed JSON at %s:%d", file_path, lineno
                    )
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("type") == "session":
                    if header is None:
                        header = obj
                    continue
                entries.append(obj)
        return header, entries

    def _context_path_ids(self, entries: List[Dict[str, Any]]) -> Set[str]:
        """Ids on the leaf-to-root path — the context the model actually saw.

        The leaf is the last entry in file order that no other entry claims as
        a parent. Walking up ``parentId`` from there reproduces the path
        ``buildSessionContext()`` uses, without needing to replay compaction.
        """
        by_id: Dict[str, Dict[str, Any]] = {}
        parented: Set[str] = set()
        for entry in entries:
            entry_id = entry.get("id")
            if isinstance(entry_id, str):
                by_id[entry_id] = entry
            parent_id = entry.get("parentId")
            if isinstance(parent_id, str):
                parented.add(parent_id)

        leaf_id: Optional[str] = None
        for entry in reversed(entries):
            entry_id = entry.get("id")
            if isinstance(entry_id, str) and entry_id not in parented:
                leaf_id = entry_id
                break

        path_ids: Set[str] = set()
        cursor = leaf_id
        while isinstance(cursor, str) and cursor in by_id and cursor not in path_ids:
            path_ids.add(cursor)
            parent = by_id[cursor].get("parentId")
            cursor = parent if isinstance(parent, str) else None
        return path_ids

    def _populate(
        self,
        trajectory: SessionTrajectory,
        entries: List[Dict[str, Any]],
        context_ids: Set[str],
    ) -> None:
        """Fill a trajectory from the scanned entries."""
        timestamps: List[datetime] = []
        pending_calls: Dict[str, ToolCallRecord] = {}
        models: List[str] = []
        last_stop_reason: Optional[str] = None

        for entry in entries:
            ts = _parse_timestamp(entry.get("timestamp"))
            if ts is not None:
                timestamps.append(ts)
            entry_type = entry.get("type")
            entry_id = str(entry.get("id", ""))

            if entry_type == "message":
                stop_reason = self._handle_message(
                    trajectory, entry, entry_id, ts, pending_calls, models
                )
                if stop_reason is not None and (
                    not self.config.context_path_only or entry_id in context_ids
                ):
                    last_stop_reason = stop_reason
            elif entry_type == "compaction":
                summary = entry.get("summary")
                trajectory.compactions.append(
                    CompactionRecord(
                        entry_id=entry_id,
                        tokens_before=int(entry.get("tokensBefore", 0) or 0),
                        summary_chars=len(summary) if isinstance(summary, str) else 0,
                    )
                )
            elif entry_type == "custom":
                if entry.get("customType") == REFINEMENT_CUSTOM_TYPE:
                    data = entry.get("data")
                    if isinstance(data, dict):
                        trajectory.refinements.append(
                            RefinementRecord.from_dict(data, entry_id, ts)
                        )
            elif entry_type == "child_usage_attributed":
                trajectory.child_usage = trajectory.child_usage.merge(
                    TokenUsage.from_dict(entry.get("childUsage"))
                )

        trajectory.tool_calls = list(pending_calls.values())
        trajectory.tool_errors = sum(1 for c in trajectory.tool_calls if c.is_error)
        trajectory.turns = trajectory.assistant_messages
        trajectory.models = list(dict.fromkeys(models))
        trajectory.outcome = _STOP_REASON_TO_OUTCOME.get(
            last_stop_reason or "", SessionOutcome.UNKNOWN
        )
        if timestamps:
            trajectory.duration_seconds = max(
                0.0, (max(timestamps) - min(timestamps)).total_seconds()
            )
        self._resolve_gate(trajectory)

    def _handle_message(
        self,
        trajectory: SessionTrajectory,
        entry: Dict[str, Any],
        entry_id: str,
        ts: Optional[datetime],
        pending_calls: Dict[str, ToolCallRecord],
        models: List[str],
    ) -> Optional[str]:
        """Process one ``message`` entry. Returns its stopReason, if any."""
        message = entry.get("message")
        if not isinstance(message, dict):
            return None
        role = message.get("role")

        if role == "user":
            trajectory.user_messages += 1
            return None

        if role == "assistant":
            trajectory.assistant_messages += 1
            trajectory.usage = trajectory.usage.merge(
                TokenUsage.from_dict(message.get("usage"))
            )
            model = message.get("model")
            if isinstance(model, str) and model:
                models.append(model)
            for block in _content_blocks(message):
                if block.get("type") != "toolCall":
                    continue
                call = ToolCallRecord(
                    call_id=str(block.get("id", "")),
                    tool_name=str(block.get("name", "")),
                    code=_tool_call_code(block),
                )
                pending_calls[call.call_id] = call
                if call.tool_name == "ipython":
                    trajectory.rlm_spawns.extend(
                        self._extract_rlm_spawns(
                            call.code, call.call_id, entry_id, trajectory.depth, ts
                        )
                    )
            stop_reason = message.get("stopReason")
            return stop_reason if isinstance(stop_reason, str) else None

        if role == "toolResult":
            call_id = str(message.get("toolCallId", ""))
            resolved_call = pending_calls.get(call_id)
            if resolved_call is None:
                # A result whose call is off the scanned path still tells us
                # whether that execution failed.
                resolved_call = ToolCallRecord(
                    call_id=call_id,
                    tool_name=str(message.get("toolName", "")),
                )
                pending_calls[call_id] = resolved_call
            resolved_call.is_error = bool(message.get("isError", False))
            resolved_call.resolved = True
            return None

        if role == "bashExecution":
            command = str(message.get("command", ""))
            exit_code = message.get("exitCode")
            call = ToolCallRecord(
                call_id=f"bash:{entry_id}",
                tool_name="bash",
                code=command,
                is_error=bool(exit_code not in (0, None)),
                resolved=True,
            )
            pending_calls[call.call_id] = call
            return None

        return None

    def _extract_rlm_spawns(
        self,
        code: str,
        call_id: str,
        entry_id: str,
        depth: int,
        ts: Optional[datetime],
    ) -> List[RlmSpawnRecord]:
        """Find ``rlm(...)`` child spawns inside one ipython cell."""
        spawns: List[RlmSpawnRecord] = []
        for match in RLM_SPAWN_PATTERN.finditer(code):
            args = _balanced_args(code, match.end() - 1)
            if args is None:
                continue
            name_match = RLM_NAME_PATTERN.search(args)
            prompt_match = RLM_PROMPT_PATTERN.match(args)
            prompt = prompt_match.group(2) if prompt_match else ""
            record = RlmSpawnRecord(
                call_id=call_id,
                entry_id=entry_id,
                name=name_match.group(2) if name_match else "",
                prompt=prompt[: self.config.max_prompt_chars],
                depth=depth,
            )
            if ts is not None:
                record.timestamp = ts
            spawns.append(record)
        return spawns

    def _resolve_gate(self, trajectory: SessionTrajectory) -> None:
        """Set the gate result from the configured gate command, if any.

        With no configured gate the result stays ``ABSENT``: a session that
        stopped cleanly has not thereby been verified, and the bridge does not
        pretend otherwise.
        """
        gate = self.config.gate_command.strip()
        if not gate:
            return
        trajectory.gate_command = gate
        matches = [
            c
            for c in trajectory.tool_calls
            if gate in c.code and c.resolved
        ]
        if not matches:
            return
        trajectory.gate_result = (
            GateResult.FAILED if matches[-1].is_error else GateResult.PASSED
        )


# --- Module-level helpers ---


def _content_blocks(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize a message's ``content`` to a list of block dicts."""
    content = message.get("content")
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def _tool_call_code(block: Dict[str, Any]) -> str:
    """Extract the source text from a toolCall block's arguments."""
    args = block.get("arguments")
    if not isinstance(args, dict):
        return ""
    for key in ("code", "command", "input"):
        value = args.get(key)
        if isinstance(value, str):
            return value
    return ""


def _balanced_args(code: str, open_paren_index: int) -> Optional[str]:
    """Return the text between a call's parentheses, respecting nesting.

    Quote-aware so that a paren inside a prompt string does not terminate the
    argument list early. Returns None when the call is unterminated (a
    truncated cell), in which case the spawn is not counted.
    """
    depth = 0
    quote: Optional[str] = None
    i = open_paren_index
    start = open_paren_index + 1
    while i < len(code):
        ch = code[i]
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if code.startswith(quote, i):
                i += len(quote)
                quote = None
                continue
            i += 1
            continue
        if ch in "\"'":
            for candidate in ('"""', "'''", '"', "'"):
                if code.startswith(candidate, i):
                    quote = candidate
                    i += len(candidate)
                    break
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return code[start:i]
        i += 1
    return None


def _parse_timestamp(raw: Any) -> Optional[datetime]:
    """Parse an ISO-8601 or Unix-ms timestamp into an aware datetime."""
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(raw / 1000.0, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(raw, str) or not raw:
        return None
    text = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


@dataclass
class SessionTree:
    """A parent session and its transitively linked children.

    Prime Agent records a child's origin in its header's ``parentSession``
    field (an absolute path to the parent's session file), which is what makes
    the delegation DAG recoverable from disk alone.
    """

    roots: List[SessionTrajectory] = field(default_factory=list)
    children: Dict[str, List[SessionTrajectory]] = field(default_factory=dict)

    def depth_of(self, trajectory: SessionTrajectory) -> int:
        return trajectory.depth


def build_session_tree(
    trajectories: List[SessionTrajectory],
) -> SessionTree:
    """Link parsed sessions into a delegation tree and assign depths.

    Sets ``depth`` in place on each trajectory: 0 for a root session, parent + 1
    for a child. Sessions whose ``parentSession`` points outside the supplied
    set are treated as roots, so partial exports still analyze cleanly.
    """
    by_path: Dict[str, SessionTrajectory] = {}
    for traj in trajectories:
        resolved = str(Path(traj.path).expanduser())
        by_path[resolved] = traj

    tree = SessionTree()
    for traj in trajectories:
        parent_path = traj.parent_session
        resolved_parent = (
            str(Path(parent_path).expanduser()) if parent_path else None
        )
        if resolved_parent and resolved_parent in by_path:
            tree.children.setdefault(resolved_parent, []).append(traj)
        else:
            tree.roots.append(traj)

    def assign(traj: SessionTrajectory, depth: int, seen: Set[str]) -> None:
        resolved = str(Path(traj.path).expanduser())
        if resolved in seen:
            logger.warning("Cycle in parentSession links at %s", resolved)
            return
        seen.add(resolved)
        traj.depth = depth
        for spawn in traj.rlm_spawns:
            spawn.depth = depth
        for child in tree.children.get(resolved, []):
            assign(child, depth + 1, seen)

    for root in tree.roots:
        assign(root, 0, set())
    return tree
