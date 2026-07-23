#!/usr/bin/env python3
"""Point the GasTown bridge at the LIVE repo and see what it picks up.

Bypasses BeadsClient (which expects a 'beads' table) and reads directly
from the real .beads/beads.db 'issues' table, then feeds the data through
the GasTownMapper + GitObserver pipeline.

Enhanced: extracts per-issue git stats by correlating commit messages with
issue IDs and titles for differentiated signal.
"""

import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add repo root to path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from swarm.bridges.gastown.mapper import GasTownMapper  # noqa: E402
from swarm.core.proxy import ProxyComputer  # noqa: E402


def load_issues(db_path: str) -> list[dict]:
    """Read all issues from the real beads DB."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, title, status, assignee, priority, issue_type, "
        "created_at, updated_at, closed_at FROM issues ORDER BY updated_at"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_git_log(workspace: str) -> list[dict]:
    """Get full commit log with timestamps and diffs."""
    result = subprocess.run(
        ["git", "log", "--format=%H|%aI|%s", "--shortstat"],
        capture_output=True, text=True, cwd=workspace, timeout=10,
    )
    if result.returncode != 0:
        return []

    commits = []
    lines = result.stdout.strip().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if "|" in line and line.count("|") >= 2:
            parts = line.split("|", 2)
            sha, ts, msg = parts[0], parts[1], parts[2]
            files_changed = 0
            insertions = 0
            deletions = 0
            # Next non-empty line might be shortstat
            if i + 1 < len(lines):
                stat_line = lines[i + 1].strip()
                if "file" in stat_line or "insertion" in stat_line:
                    m = re.search(r"(\d+) file", stat_line)
                    if m:
                        files_changed = int(m.group(1))
                    m = re.search(r"(\d+) insertion", stat_line)
                    if m:
                        insertions = int(m.group(1))
                    m = re.search(r"(\d+) deletion", stat_line)
                    if m:
                        deletions = int(m.group(1))
                    i += 1
            commits.append({
                "sha": sha, "timestamp": ts, "message": msg,
                "files_changed": files_changed,
                "insertions": insertions, "deletions": deletions,
            })
        i += 1
    return commits


def correlate_commits_to_issue(
    issue: dict, commits: list[dict]
) -> list[dict]:
    """Find commits likely related to an issue by keyword matching."""
    title = issue["title"].lower()
    issue_id = issue["id"].lower()

    # Extract keywords from title (3+ char words)
    keywords = [w for w in re.findall(r"\w+", title) if len(w) >= 3]

    # Time window: created_at to closed_at (or now)
    try:
        created = datetime.fromisoformat(issue["created_at"])
    except (ValueError, TypeError):
        created = datetime(2020, 1, 1, tzinfo=timezone.utc)
    if issue.get("closed_at"):
        try:
            closed = datetime.fromisoformat(issue["closed_at"])
        except (ValueError, TypeError):
            closed = datetime.now(timezone.utc)
    else:
        closed = datetime.now(timezone.utc)

    matched = []
    for c in commits:
        try:
            ct = datetime.fromisoformat(c["timestamp"])
        except (ValueError, TypeError):
            continue

        # Must be within issue lifetime
        if ct < created or ct > closed:
            continue

        msg = c["message"].lower()

        # Direct ID match
        if issue_id in msg:
            matched.append(c)
            continue

        # Keyword overlap (need 2+ keyword matches)
        hits = sum(1 for kw in keywords if kw in msg)
        if hits >= 2:
            matched.append(c)

    return matched


def build_git_stats(matched_commits: list[dict]) -> dict:
    """Build git_stats dict from correlated commits."""
    n = len(matched_commits)
    files = sum(c["files_changed"] for c in matched_commits)
    ci_failures = sum(1 for c in matched_commits if "[ci-fail]" in c["message"])

    # Review iterations: count amend/fixup commits
    rework = sum(
        1 for c in matched_commits
        if any(k in c["message"].lower() for k in ["fixup", "amend", "fix:", "fix "])
    )

    # Time to merge: first commit to last commit
    hours = None
    if n >= 2:
        try:
            timestamps = [
                datetime.fromisoformat(c["timestamp"]) for c in matched_commits
            ]
            hours = (max(timestamps) - min(timestamps)).total_seconds() / 3600.0
        except (ValueError, TypeError):
            pass

    return {
        "commit_count": n,
        "files_changed": files,
        "review_iterations": rework,
        "ci_failures": ci_failures,
        "time_to_merge_hours": hours,
    }


def main() -> None:
    workspace = str(REPO)
    db_path = str(REPO / ".beads" / "beads.db")

    print("=" * 70)
    print("GasTown Bridge — Live Probe (enhanced)")
    print(f"Workspace: {workspace}")
    print(f"DB:        {db_path}")
    print("=" * 70)

    # 1. Load real issues
    issues = load_issues(db_path)
    print(f"\nFound {len(issues)} issues in .beads/beads.db")

    # 2. Load full git log
    commits = get_git_log(workspace)
    print(f"Found {len(commits)} commits in git log\n")

    # 3. Set up the mapper
    proxy = ProxyComputer(sigmoid_k=2.0)
    mapper = GasTownMapper(proxy=proxy)

    # 4. Process each issue with correlated git stats
    print("-" * 90)
    print(
        f"{'ID':<20} {'Status':<6} {'Commits':>7} {'Files':>6} "
        f"{'Rework':>6} {'Hours':>7} {'p':>6} {'v_hat':>7}  Title"
    )
    print("-" * 90)

    interactions = []
    for issue in issues:
        # Correlate commits
        matched = correlate_commits_to_issue(issue, commits)
        git_stats = build_git_stats(matched)

        bead = {
            "id": issue["id"],
            "status": "done" if issue["status"] == "closed" else issue["status"],
            "title": issue["title"],
            "assignee": issue.get("assignee") or "unassigned",
        }

        interaction = mapper.map_bead_completion(
            bead=bead,
            git_stats=git_stats,
            agent_id=issue.get("assignee") or "unassigned",
        )
        interactions.append(interaction)

        short_id = issue["id"].replace("distributional-agi-safety-", "")
        status = {"closed": "done", "in_progress": "wip"}.get(
            issue["status"], issue["status"]
        )
        hours_str = (
            f"{git_stats['time_to_merge_hours']:.1f}"
            if git_stats["time_to_merge_hours"] is not None
            else "-"
        )
        title = issue["title"][:35]

        print(
            f"  {short_id:<18} {status:<6} {git_stats['commit_count']:>7} "
            f"{git_stats['files_changed']:>6} {git_stats['review_iterations']:>6} "
            f"{hours_str:>7} {interaction.p:>6.3f} {interaction.v_hat:>7.3f}  {title}"
        )

    # 5. Aggregate stats
    print("\n" + "=" * 70)
    print("AGGREGATE STATS")
    print("=" * 70)

    ps = [i.p for i in interactions]
    v_hats = [i.v_hat for i in interactions]

    done = [i for i in interactions if i.metadata.get("bead_status") == "done"]
    wip = [i for i in interactions if i.metadata.get("bead_status") != "done"]

    print(f"  Total interactions:    {len(interactions)}")
    print(f"  Completed (done):      {len(done)}")
    print(f"  In-progress/open:      {len(wip)}")
    print(f"  Mean p:                {sum(ps)/len(ps):.4f}")
    print(f"  Min p:                 {min(ps):.4f}")
    print(f"  Max p:                 {max(ps):.4f}")
    print(f"  Std p:                 {(sum((x - sum(ps)/len(ps))**2 for x in ps) / len(ps))**0.5:.4f}")
    print(f"  Mean v_hat:            {sum(v_hats)/len(v_hats):.4f}")

    if done:
        done_ps = [i.p for i in done]
        print(f"  Mean p (done):         {sum(done_ps)/len(done_ps):.4f}")
    if wip:
        wip_ps = [i.p for i in wip]
        print(f"  Mean p (wip):          {sum(wip_ps)/len(wip_ps):.4f}")

    # Toxicity: E[1-p | accepted]
    accepted = [i for i in interactions if i.accepted]
    if accepted:
        toxicity = sum(1.0 - i.p for i in accepted) / len(accepted)
        print(f"  Toxicity (E[1-p]):     {toxicity:.4f}")

    # Quality gap: E[p|done] - E[p|wip]
    if done and wip:
        gap = (sum(i.p for i in done) / len(done)) - (
            sum(i.p for i in wip) / len(wip)
        )
        print(f"  Quality gap:           {gap:+.4f}")

    # Per-issue detail: which issues look best/worst
    print("\n" + "=" * 70)
    print("RANKING (by p)")
    print("=" * 70)
    ranked = sorted(
        zip(interactions, issues, strict=True), key=lambda x: x[0].p, reverse=True
    )
    for i, (inter, iss) in enumerate(ranked, 1):
        short_id = iss["id"].replace("distributional-agi-safety-", "")
        print(f"  {i:>2}. p={inter.p:.3f}  v_hat={inter.v_hat:+.3f}  [{short_id}] {iss['title'][:50]}")

    print()


if __name__ == "__main__":
    main()
