"""Git inspection helpers for AgentGit provenance bundles."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class ChangedFile:
    """A single changed path with line-count metadata from ``git diff``."""

    path: str
    status: str
    additions: int = 0
    deletions: int = 0


@dataclass(frozen=True)
class GitSnapshot:
    """Minimal repository state needed for an agent provenance bundle."""

    repo_path: str
    base_ref: str
    head_ref: str
    base_commit: Optional[str]
    head_commit: Optional[str]
    branch: Optional[str]
    changed_files: List[ChangedFile]

    @property
    def total_additions(self) -> int:
        return sum(f.additions for f in self.changed_files)

    @property
    def total_deletions(self) -> int:
        return sum(f.deletions for f in self.changed_files)


def collect_snapshot(repo: Path, base_ref: str = "HEAD") -> GitSnapshot:
    """Collect changed-file metadata for ``repo`` relative to ``base_ref``.

    The snapshot includes both tracked and untracked files so agents cannot
    hide generated files outside the normal diff.
    """

    repo = repo.resolve()
    head_ref = "WORKTREE"
    changed = _collect_tracked_changes(repo, base_ref)
    changed.extend(_collect_untracked_changes(repo))
    changed = sorted(_dedupe(changed), key=lambda item: item.path)

    return GitSnapshot(
        repo_path=str(repo),
        base_ref=base_ref,
        head_ref=head_ref,
        base_commit=_git_optional(repo, ["rev-parse", base_ref]),
        head_commit=_git_optional(repo, ["rev-parse", "HEAD"]),
        branch=_git_optional(repo, ["rev-parse", "--abbrev-ref", "HEAD"]),
        changed_files=changed,
    )


def _collect_tracked_changes(repo: Path, base_ref: str) -> List[ChangedFile]:
    status = _git(repo, ["diff", "--name-status", base_ref, "--"])
    numstat = _git(repo, ["diff", "--numstat", base_ref, "--"])
    line_counts = _parse_numstat(numstat)

    files: List[ChangedFile] = []
    for raw in status.splitlines():
        if not raw.strip():
            continue
        parts = raw.split("\t")
        status_code = parts[0]
        path = parts[-1]
        additions, deletions = line_counts.get(path, (0, 0))
        files.append(
            ChangedFile(
                path=_normalize_path(path),
                status=status_code,
                additions=additions,
                deletions=deletions,
            )
        )
    return files


def _collect_untracked_changes(repo: Path) -> List[ChangedFile]:
    output = _git(repo, ["ls-files", "--others", "--exclude-standard"])
    files: List[ChangedFile] = []
    for raw in output.splitlines():
        path = _normalize_path(raw)
        full_path = repo / path
        files.append(
            ChangedFile(
                path=path,
                status="??",
                additions=_count_lines(full_path),
                deletions=0,
            )
        )
    return files


def _parse_numstat(output: str) -> dict[str, tuple[int, int]]:
    counts: dict[str, tuple[int, int]] = {}
    for raw in output.splitlines():
        parts = raw.split("\t")
        if len(parts) < 3:
            continue
        added, deleted, path = parts[0], parts[1], parts[-1]
        counts[_normalize_path(path)] = (_parse_count(added), _parse_count(deleted))
    return counts


def _parse_count(value: str) -> int:
    # "-" is git's marker for binary files in --numstat output.
    if value == "-":
        return 0
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Unexpected numstat count {value!r}") from exc


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    # Surface read errors rather than silently returning 0: an unreadable
    # file must not let a line-limit policy pass by undercounting.
    with path.open("rb") as f:
        return sum(1 for _ in f)


def _dedupe(files: List[ChangedFile]) -> List[ChangedFile]:
    by_path: dict[str, ChangedFile] = {}
    for file in files:
        by_path[file.path] = file
    return list(by_path.values())


def _normalize_path(path: str) -> str:
    return Path(path).as_posix()


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


def _git_optional(repo: Path, args: List[str]) -> Optional[str]:
    try:
        value = _git(repo, args)
    except subprocess.CalledProcessError:
        return None
    return value or None
