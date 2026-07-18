"""Structured, versioned, scoped operational memory for agents.

Code, issues, and PRs don't capture operational knowledge: "module X is
fragile because...", "never use lib Y here", "deploys fail if the migration
order changes", "the previous agent failed this task because...". This module
gives agents a durable place for exactly that.

Design:

- **Structured**: every memory is a typed :class:`MemoryEntry` — a ``kind``
  (fragility, constraint, failure, practice, context), a ``subject`` (the
  path/module/topic it is about), and a body.
- **Versioned**: storage is an append-only JSONL stream; updating a memory
  appends a new version, never rewrites history. The current state of an id
  is its latest record; the full history stays replayable (repo event-log
  invariants). Retiring a memory appends a tombstone.
- **Scoped**: three scopes with separate backing files —
  ``repo`` (``.agentgit/memory.jsonl`` inside the repository, so it travels
  with clones and is reviewable in PRs), ``org``
  (``~/.agentgit/org-memory.jsonl``, shared across repos), and ``agent``
  (``~/.agentgit/agent-memory/<agent>.jsonl``, private operational notes).

Recall is subject-aware: a memory about ``swarm/core`` surfaces when an
agent asks about ``swarm/core/proxy.py`` — so ``recall_for_paths`` with a
diff's changed paths gives "what should I know before touching these files".
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

MEMORY_SCHEMA_V1 = "agentgit.memory.v1"

SCOPES = ("repo", "org", "agent")
# Recall returns the most specific scope first: an agent's own note about a
# module outranks the org-wide one.
_SCOPE_SPECIFICITY = {"agent": 0, "repo": 1, "org": 2}

KINDS = ("fragility", "constraint", "failure", "practice", "context")

DEFAULT_REPO_MEMORY = ".agentgit/memory.jsonl"


@dataclass(frozen=True)
class MemoryEntry:
    """One version of one operational memory."""

    id: str
    scope: str  # repo | org | agent
    kind: str  # fragility | constraint | failure | practice | context
    subject: str  # path/module/topic the memory is about
    body: str
    author: str  # agent that wrote this version
    version: int
    created_at: str
    updated_at: str
    retired: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"schema": MEMORY_SCHEMA_V1, **asdict(self)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        fields = {k: v for k, v in data.items() if k != "schema"}
        return cls(**fields)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:64] or "memory"


def _normalize_subject(subject: str) -> str:
    return subject.replace("\\", "/").rstrip("/")


def _subject_matches(subject: str, query: str) -> bool:
    """A memory applies when its subject is the query, a parent of it, or
    a child of it (asking about a module surfaces file-level notes too)."""

    subject = _normalize_subject(subject)
    query = _normalize_subject(query)
    return (
        subject == query
        or query.startswith(subject + "/")
        or subject.startswith(query + "/")
    )


class MemoryStore:
    """Append-only, versioned operational memory across repo/org/agent scopes.

    ``repo_root`` anchors the repo-scoped file; ``home`` anchors org and agent
    files (defaults to the user's home, injectable for tests). ``agent`` is
    required only to write or read agent-scoped memories.
    """

    def __init__(
        self,
        repo_root: Path,
        *,
        agent: Optional[str] = None,
        home: Optional[Path] = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.agent = agent
        base = Path(home) if home is not None else Path.home()
        self._paths: Dict[str, Path] = {
            "repo": self.repo_root / DEFAULT_REPO_MEMORY,
            "org": base / ".agentgit" / "org-memory.jsonl",
        }
        if agent:
            self._paths["agent"] = (
                base / ".agentgit" / "agent-memory" / f"{_slugify(agent)}.jsonl"
            )

    # -- write ---------------------------------------------------------------
    def remember(
        self,
        *,
        scope: str,
        kind: str,
        subject: str,
        body: str,
        author: str,
        entry_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """Record a memory (new id) or append a new version (existing id)."""

        if scope not in SCOPES:
            raise ValueError(f"invalid scope {scope!r}; expected one of {SCOPES}")
        if kind not in KINDS:
            raise ValueError(f"invalid kind {kind!r}; expected one of {KINDS}")
        if scope == "agent" and "agent" not in self._paths:
            raise ValueError("agent-scoped memory requires MemoryStore(agent=...)")

        now = datetime.now(timezone.utc).isoformat()
        subject = _normalize_subject(subject)
        entry_id = entry_id or f"{kind}-{_slugify(subject)}"
        prior = self._current(scope).get(entry_id)
        entry = MemoryEntry(
            id=entry_id,
            scope=scope,
            kind=kind,
            subject=subject,
            body=body,
            author=author,
            version=(prior.version + 1) if prior else 1,
            created_at=prior.created_at if prior else now,
            updated_at=now,
            details=dict(details or {}),
        )
        self._append(scope, entry)
        return entry

    def retire(self, *, scope: str, entry_id: str, author: str, reason: str = "") -> bool:
        """Append a tombstone version; history stays intact and replayable."""

        prior = self._current(scope).get(entry_id)
        if prior is None or prior.retired:
            return False
        entry = MemoryEntry(
            id=prior.id,
            scope=prior.scope,
            kind=prior.kind,
            subject=prior.subject,
            body=prior.body,
            author=author,
            version=prior.version + 1,
            created_at=prior.created_at,
            updated_at=datetime.now(timezone.utc).isoformat(),
            retired=True,
            details={**prior.details, "retire_reason": reason},
        )
        self._append(scope, entry)
        return True

    # -- read ----------------------------------------------------------------
    def recall(
        self,
        subject: Optional[str] = None,
        *,
        kind: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
    ) -> List[MemoryEntry]:
        """Current (non-retired) memories, most specific scope first.

        ``subject`` matches hierarchically (see :func:`_subject_matches`);
        ``None`` returns everything in the requested scopes.
        """

        results: List[MemoryEntry] = []
        for scope in scopes or [s for s in SCOPES if s in self._paths]:
            if scope not in self._paths:
                continue
            for entry in self._current(scope).values():
                if entry.retired:
                    continue
                if kind is not None and entry.kind != kind:
                    continue
                if subject is not None and not _subject_matches(entry.subject, subject):
                    continue
                results.append(entry)
        return sorted(
            results,
            key=lambda e: (_SCOPE_SPECIFICITY[e.scope], e.subject, e.id),
        )

    def recall_for_paths(self, paths: Iterable[str]) -> List[MemoryEntry]:
        """Everything an agent should know before touching ``paths``.

        Feed it a diff's changed paths (e.g. ``[f.path for f in
        snapshot.changed_files]``) to surface fragility notes, constraints,
        and prior failures scoped to the code about to change.
        """

        seen: Dict[tuple[str, str], MemoryEntry] = {}
        for path in paths:
            for entry in self.recall(path):
                seen[(entry.scope, entry.id)] = entry
        return sorted(
            seen.values(),
            key=lambda e: (_SCOPE_SPECIFICITY[e.scope], e.subject, e.id),
        )

    def history(self, entry_id: str, *, scope: str) -> List[MemoryEntry]:
        """All versions of a memory, oldest first (append order)."""

        return [e for e in self._read(scope) if e.id == entry_id]

    # -- internals -----------------------------------------------------------
    def _append(self, scope: str, entry: MemoryEntry) -> None:
        path = self._paths[scope]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")

    def _read(self, scope: str) -> List[MemoryEntry]:
        path = self._paths.get(scope)
        if path is None or not path.exists():
            return []
        entries: List[MemoryEntry] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(MemoryEntry.from_dict(json.loads(line)))
        return entries

    def _current(self, scope: str) -> Dict[str, MemoryEntry]:
        current: Dict[str, MemoryEntry] = {}
        for entry in self._read(scope):
            current[entry.id] = entry  # later versions overwrite: last wins
        return current
