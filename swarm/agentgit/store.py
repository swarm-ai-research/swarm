"""Durable storage for AgentGit provenance bundles.

The MVP wrote each bundle as a loose JSON file under ``.agentgit/``, which is
easy to lose and easy to rewrite silently. This module adds two complementary
durable backends:

- **Append-only event log** (``.agentgit/log.jsonl``): every attested bundle
  becomes one hash-chained JSONL entry. Each entry commits to the previous
  entry's hash, so truncating, reordering, or editing history breaks the chain
  and is detected by :func:`verify_log`.
- **Git notes** (``refs/notes/agentgit``): the bundle is attached to the
  commit it attests, so provenance travels with git history itself (fetch or
  push the notes ref alongside branches) instead of living beside it.

Both backends store the bundle verbatim, so the existing receipt signature and
identity checks in :func:`swarm.agentgit.bundle.verify_bundle` keep working on
stored copies unchanged.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from swarm.agentgit.bundle import verify_bundle

LOG_SCHEMA_V1 = "agentgit.log.v1"
DEFAULT_LOG_PATH = ".agentgit/log.jsonl"
DEFAULT_NOTES_REF = "refs/notes/agentgit"
GENESIS_HASH = "0" * 64


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bundle_digest(bundle: Dict[str, Any]) -> str:
    """Stable content digest of a bundle (canonical-JSON SHA-256)."""

    return _sha256(_canonical_json(bundle))


# ---------------------------------------------------------------------------
# append-only hash-chained event log
# ---------------------------------------------------------------------------
def _entry_hash(entry: Dict[str, Any]) -> str:
    hashed = {k: v for k, v in entry.items() if k != "entry_hash"}
    return _sha256(_canonical_json(hashed))


def append_to_log(
    bundle: Dict[str, Any],
    log_path: Path,
) -> Dict[str, Any]:
    """Append ``bundle`` to the hash-chained event log and return the entry.

    The log is append-only: existing entries are never rewritten. Each entry's
    ``prev_hash`` is the previous entry's ``entry_hash`` (or a genesis marker),
    making any in-place edit or truncation detectable.
    """

    entries = read_log(log_path)
    prev_hash = entries[-1]["entry_hash"] if entries else GENESIS_HASH
    entry: Dict[str, Any] = {
        "schema": LOG_SCHEMA_V1,
        "seq": len(entries),
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "prev_hash": prev_hash,
        "bundle_sha256": bundle_digest(bundle),
        "bundle": bundle,
    }
    entry["entry_hash"] = _entry_hash(entry)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(_canonical_json(entry) + "\n")
    return entry


def read_log(log_path: Path) -> List[Dict[str, Any]]:
    """Read all log entries (empty list when the log does not exist)."""

    if not log_path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def verify_log(
    log_path: Path,
    *,
    signing_key: str | None = None,
) -> Tuple[bool, List[str]]:
    """Verify the log's hash chain (and, with a key, each embedded bundle).

    Chain verification detects tampering with *stored* entries: edits, drops,
    and reorders all change some ``entry_hash``/``prev_hash`` pair. Passing
    ``signing_key`` additionally re-verifies every embedded bundle's receipt
    signature, catching a forged entry that was re-chained consistently.
    """

    errors: List[str] = []
    prev_hash = GENESIS_HASH
    for i, entry in enumerate(read_log(log_path)):
        prefix = f"entry {i}"
        if entry.get("schema") != LOG_SCHEMA_V1:
            errors.append(f"{prefix}: unsupported schema {entry.get('schema')!r}")
            continue
        if entry.get("seq") != i:
            errors.append(f"{prefix}: seq {entry.get('seq')!r} != position {i}")
        if entry.get("prev_hash") != prev_hash:
            errors.append(f"{prefix}: prev_hash does not chain to previous entry")
        if entry.get("entry_hash") != _entry_hash(entry):
            errors.append(f"{prefix}: entry_hash does not match entry contents")
        bundle = entry.get("bundle", {})
        if entry.get("bundle_sha256") != bundle_digest(bundle):
            errors.append(f"{prefix}: bundle_sha256 does not match embedded bundle")
        if signing_key is not None:
            ok, bundle_errors = verify_bundle(
                bundle, signing_key=signing_key, require_policy_pass=False
            )
            if not ok:
                errors.extend(f"{prefix}: bundle: {err}" for err in bundle_errors)
        # Chain onto the recorded hash so a single corrupted entry yields one
        # localized error instead of cascading down the rest of the log.
        prev_hash = entry.get("entry_hash", prev_hash)
    return not errors, errors


# ---------------------------------------------------------------------------
# git notes
# ---------------------------------------------------------------------------
def attach_note(
    repo: Path,
    bundle: Dict[str, Any],
    *,
    commit: str | None = None,
    notes_ref: str = DEFAULT_NOTES_REF,
) -> str:
    """Attach ``bundle`` as a git note on ``commit`` and return the commit sha.

    Defaults to the bundle's recorded head commit — the commit the attested
    diff was built on. Notes are *appended* (JSONL), so re-attesting the same
    commit accumulates bundles rather than overwriting earlier provenance.
    """

    target = commit or bundle.get("git", {}).get("head_commit")
    if not target:
        raise ValueError(
            "bundle has no head_commit and no explicit commit was given; "
            "cannot attach a git note"
        )
    resolved = _git(repo, ["rev-parse", "--verify", f"{target}^{{commit}}"])
    _git(
        repo,
        ["notes", f"--ref={notes_ref}", "append", "-m", _canonical_json(bundle), resolved],
    )
    return resolved


def read_notes(
    repo: Path,
    commit: str,
    *,
    notes_ref: str = DEFAULT_NOTES_REF,
) -> List[Dict[str, Any]]:
    """Read all bundles attached to ``commit`` (empty list when none)."""

    try:
        raw = _git(repo, ["notes", f"--ref={notes_ref}", "show", commit])
    except subprocess.CalledProcessError:
        return []
    bundles: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        if line.strip():
            bundles.append(json.loads(line))
    return bundles


def list_noted_commits(
    repo: Path,
    *,
    notes_ref: str = DEFAULT_NOTES_REF,
) -> List[str]:
    """List commit shas that have provenance notes attached."""

    try:
        raw = _git(repo, ["notes", f"--ref={notes_ref}", "list"])
    except subprocess.CalledProcessError:
        return []
    # `git notes list` prints "<note-blob-sha> <annotated-commit-sha>" lines.
    return [line.split()[1] for line in raw.splitlines() if line.strip()]


def _git(repo: Path, args: List[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# combined storage entrypoint used by the CLI
# ---------------------------------------------------------------------------
def store_bundle(
    bundle: Dict[str, Any],
    *,
    repo: Path,
    backends: List[str],
    log_path: Optional[Path] = None,
    notes_ref: str = DEFAULT_NOTES_REF,
    commit: str | None = None,
) -> Dict[str, Any]:
    """Store ``bundle`` in each requested backend (``log``, ``git-notes``).

    Returns a summary dict describing where the bundle was stored.
    """

    summary: Dict[str, Any] = {}
    for backend in backends:
        if backend == "log":
            path = log_path or (repo / DEFAULT_LOG_PATH)
            entry = append_to_log(bundle, path)
            summary["log"] = {"path": str(path), "seq": entry["seq"]}
        elif backend == "git-notes":
            sha = attach_note(repo, bundle, commit=commit, notes_ref=notes_ref)
            summary["git-notes"] = {"ref": notes_ref, "commit": sha}
        else:
            raise ValueError(f"unknown provenance store backend {backend!r}")
    return summary
