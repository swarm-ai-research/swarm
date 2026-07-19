#!/usr/bin/env python3
"""Prompt improvement proposal generator for Paperclip agents.

Analyzes PerformanceTracker logs and self-eval history to propose
prompt/tool modifications. Proposals are written to a proposals/ dir
inside the agent's home directory for human review before adoption.

Requires 20+ heartbeats of data before generating proposals.

Reference: Zhang et al., Hyperagents (arXiv:2603.19461) Section 4 --
metacognitive self-modification via structured proposals.

Usage:
    python scripts/propose_improvements.py --agent-home agents/ceo
    python scripts/propose_improvements.py --agent-home agents/ceo --tracker-log agents/ceo/performance.jsonl
    python scripts/propose_improvements.py --agent-home agents/ceo --dry-run  # preview without writing

Environment variables (auto-injected by Paperclip adapter):
    PAPERCLIP_API_URL, PAPERCLIP_API_KEY, PAPERCLIP_AGENT_ID, PAPERCLIP_COMPANY_ID
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_HEARTBEATS = 20
BLOCKER_FREQ_THRESHOLD = 0.25  # above this = problem
COMPLETION_RATE_THRESHOLD = 0.6  # below this = problem
REVIEW_FAIL_THRESHOLD = 0.3  # above this = problem
TIME_TO_CLOSE_TREND_THRESHOLD = 20.0  # >20% slower = problem


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Proposal:
    """A single prompt/tool improvement proposal."""

    proposal_id: str
    agent_id: str
    category: str  # "prompt", "tool", "workflow", "heartbeat"
    title: str
    rationale: str  # what data pattern triggered this
    suggested_change: str  # the actual proposed modification
    evidence: dict[str, Any] = field(default_factory=dict)
    priority: str = "medium"  # "high", "medium", "low"
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_yaml_str(self) -> str:
        """Render as human-readable YAML."""
        # Build a dict with all proposal fields plus metadata
        data = {
            "proposal_id": self.proposal_id,
            "agent_id": self.agent_id,
            "category": self.category,
            "priority": self.priority,
            "created_at": self.created_at,
            "title": self.title,
            "rationale": self.rationale,
            "suggested_change": self.suggested_change,
            "evidence": self.evidence,
            "status": "pending",
            "reviewed_by": None,
            "reviewed_at": None,
        }
        # Use yaml.dump for safe escaping and proper YAML formatting
        return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def load_tracker_events(log_path: Path) -> list[dict]:
    """Load all events from a PerformanceTracker JSONL file."""
    if not log_path.exists():
        return []
    events: list[dict] = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def count_heartbeats(events: list[dict]) -> int:
    """Count heartbeat events."""
    return sum(1 for e in events if e.get("event_type") == "heartbeat")


def compute_metrics(events: list[dict]) -> dict[str, Any]:
    """Compute aggregate metrics from tracker events."""
    started = sum(1 for e in events if e.get("event_type") == "task_started")
    completed = sum(1 for e in events if e.get("event_type") == "task_completed")
    blocked = sum(1 for e in events if e.get("event_type") == "task_blocked")
    review_passed = sum(1 for e in events if e.get("event_type") == "review_passed")
    review_failed = sum(1 for e in events if e.get("event_type") == "review_failed")
    heartbeats = count_heartbeats(events)

    completion_rate = completed / started if started > 0 else 0.0
    blocker_freq = blocked / started if started > 0 else 0.0
    total_reviews = review_passed + review_failed
    review_fail_rate = review_failed / total_reviews if total_reviews > 0 else 0.0

    # Time-to-close trend: compare first half vs second half
    start_times: dict[str, str] = {}
    close_durations: list[float] = []
    for ev in events:
        etype = ev.get("event_type", "")
        task_id = ev.get("metadata", {}).get("task_id", "")
        if etype == "task_started" and task_id:
            start_times[task_id] = ev["timestamp"]
        elif etype == "task_completed" and task_id and task_id in start_times:
            t0 = datetime.fromisoformat(start_times[task_id])
            t1 = datetime.fromisoformat(ev["timestamp"])
            close_durations.append((t1 - t0).total_seconds())

    ttc_trend_pct = 0.0
    if len(close_durations) >= 4:
        mid = len(close_durations) // 2
        avg_first = sum(close_durations[:mid]) / mid
        avg_second = sum(close_durations[mid:]) / len(close_durations[mid:])
        if avg_first > 0:
            ttc_trend_pct = ((avg_second - avg_first) / avg_first) * 100

    # Blocker reasons
    blocker_reasons: dict[str, int] = {}
    for ev in events:
        if ev.get("event_type") == "task_blocked":
            reason = ev.get("metadata", {}).get("reason", "unspecified")
            if not reason:
                reason = "unspecified"
            blocker_reasons[reason] = blocker_reasons.get(reason, 0) + 1

    return {
        "tasks_started": started,
        "tasks_completed": completed,
        "tasks_blocked": blocked,
        "heartbeats": heartbeats,
        "completion_rate": completion_rate,
        "blocker_freq": blocker_freq,
        "review_fail_rate": review_fail_rate,
        "ttc_trend_pct": ttc_trend_pct,
        "blocker_reasons": blocker_reasons,
        "avg_time_to_close_s": (
            sum(close_durations) / len(close_durations) if close_durations else None
        ),
    }


def generate_proposals(agent_id: str, metrics: dict[str, Any]) -> list[Proposal]:
    """Generate proposals based on detected patterns."""
    proposals: list[Proposal] = []
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    # 1. High blocker frequency
    if metrics["blocker_freq"] > BLOCKER_FREQ_THRESHOLD:
        top_reasons = sorted(
            metrics["blocker_reasons"].items(), key=lambda x: x[1], reverse=True,
        )[:3]
        reasons_str = ", ".join(f"{r} ({c}x)" for r, c in top_reasons)

        proposals.append(Proposal(
            proposal_id=f"prop-{ts}-blockers",
            agent_id=agent_id,
            category="workflow",
            title="Reduce blocker frequency",
            rationale=(
                f"Blocker frequency is {metrics['blocker_freq']:.0%} "
                f"(threshold: {BLOCKER_FREQ_THRESHOLD:.0%}). "
                f"Top reasons: {reasons_str}."
            ),
            suggested_change=(
                "Add a pre-work checklist to the heartbeat that verifies "
                "dependencies are met and required resources are available "
                "before checkout. Consider adding a 'readiness check' step "
                "between checkout and work execution."
            ),
            evidence={
                "blocker_freq": round(metrics["blocker_freq"], 3),
                "tasks_blocked": metrics["tasks_blocked"],
                "tasks_started": metrics["tasks_started"],
                "top_reasons": reasons_str,
            },
            priority="high",
        ))

    # 2. Low completion rate
    if (
        metrics["tasks_started"] >= 5
        and metrics["completion_rate"] < COMPLETION_RATE_THRESHOLD
    ):
        proposals.append(Proposal(
            proposal_id=f"prop-{ts}-completion",
            agent_id=agent_id,
            category="prompt",
            title="Improve task completion rate",
            rationale=(
                f"Completion rate is {metrics['completion_rate']:.0%} "
                f"(threshold: {COMPLETION_RATE_THRESHOLD:.0%}). "
                f"{metrics['tasks_started']} started, "
                f"{metrics['tasks_completed']} completed."
            ),
            suggested_change=(
                "Add explicit scope-bounding to the agent prompt: before "
                "starting work, the agent should estimate complexity and "
                "break down tasks that exceed a single heartbeat's capacity. "
                "Consider adding a 'will this fit in one heartbeat?' gate."
            ),
            evidence={
                "completion_rate": round(metrics["completion_rate"], 3),
                "tasks_started": metrics["tasks_started"],
                "tasks_completed": metrics["tasks_completed"],
            },
            priority="high",
        ))

    # 3. High review failure rate
    if metrics["review_fail_rate"] > REVIEW_FAIL_THRESHOLD:
        proposals.append(Proposal(
            proposal_id=f"prop-{ts}-review",
            agent_id=agent_id,
            category="prompt",
            title="Reduce review rejection rate",
            rationale=(
                f"Review fail rate is {metrics['review_fail_rate']:.0%} "
                f"(threshold: {REVIEW_FAIL_THRESHOLD:.0%})."
            ),
            suggested_change=(
                "Add a self-review step before marking work as done. "
                "The agent should re-read its output against the task "
                "description and check for completeness, correctness, "
                "and adherence to project conventions."
            ),
            evidence={
                "review_fail_rate": round(metrics["review_fail_rate"], 3),
            },
            priority="medium",
        ))

    # 4. Time-to-close trending up
    if metrics["ttc_trend_pct"] > TIME_TO_CLOSE_TREND_THRESHOLD:
        proposals.append(Proposal(
            proposal_id=f"prop-{ts}-slowdown",
            agent_id=agent_id,
            category="workflow",
            title="Address increasing time-to-close trend",
            rationale=(
                f"Average time-to-close has increased by "
                f"{metrics['ttc_trend_pct']:.0f}% comparing recent tasks "
                f"to earlier ones (threshold: {TIME_TO_CLOSE_TREND_THRESHOLD:.0f}%)."
            ),
            suggested_change=(
                "Investigate root cause: is task complexity increasing, "
                "or is the agent spending more time on non-essential steps? "
                "Consider adding time-boxing guidance to the heartbeat "
                "prompt and flagging tasks that exceed expected duration."
            ),
            evidence={
                "ttc_trend_pct": round(metrics["ttc_trend_pct"], 1),
                "avg_time_to_close_s": metrics.get("avg_time_to_close_s"),
            },
            priority="medium",
        ))

    # 5. If metrics look healthy, propose a positive reinforcement note
    if not proposals and metrics["heartbeats"] >= MIN_HEARTBEATS:
        proposals.append(Proposal(
            proposal_id=f"prop-{ts}-healthy",
            agent_id=agent_id,
            category="prompt",
            title="Performance is healthy - no changes needed",
            rationale=(
                f"After {metrics['heartbeats']} heartbeats, all metrics "
                f"are within acceptable ranges. Completion rate: "
                f"{metrics['completion_rate']:.0%}, blocker freq: "
                f"{metrics['blocker_freq']:.0%}."
            ),
            suggested_change="No changes recommended at this time.",
            evidence={
                "completion_rate": round(metrics["completion_rate"], 3),
                "blocker_freq": round(metrics["blocker_freq"], 3),
                "review_fail_rate": round(metrics["review_fail_rate"], 3),
                "heartbeats": metrics["heartbeats"],
            },
            priority="low",
        ))

    return proposals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate prompt/tool improvement proposals from self-eval data",
    )
    parser.add_argument(
        "--agent-home", type=str, required=True,
        help="Path to agent home directory (e.g., agents/ceo)",
    )
    parser.add_argument(
        "--tracker-log", type=str, default=None,
        help="Path to PerformanceTracker JSONL log (default: <agent-home>/performance.jsonl)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview proposals without writing files",
    )
    parser.add_argument(
        "--min-heartbeats", type=int, default=MIN_HEARTBEATS,
        help=f"Minimum heartbeats required (default: {MIN_HEARTBEATS})",
    )
    args = parser.parse_args()

    agent_home = Path(args.agent_home)
    if not agent_home.is_dir():
        print(f"Error: Agent home directory not found: {agent_home}", file=sys.stderr)
        sys.exit(1)

    # Derive agent_id from directory name
    agent_id = agent_home.name

    # Load tracker log
    tracker_log = Path(args.tracker_log) if args.tracker_log else agent_home / "performance.jsonl"
    events = load_tracker_events(tracker_log)

    if not events:
        print(f"No tracker events found at {tracker_log}")
        print("Agents need to accumulate performance data before proposals can be generated.")
        sys.exit(0)

    heartbeat_count = count_heartbeats(events)
    print(f"Found {len(events)} events ({heartbeat_count} heartbeats) for agent '{agent_id}'")

    if heartbeat_count < args.min_heartbeats:
        print(
            f"Need {args.min_heartbeats} heartbeats before generating proposals "
            f"(have {heartbeat_count}). Exiting.",
        )
        sys.exit(0)

    # Compute metrics and generate proposals
    metrics = compute_metrics(events)
    proposals = generate_proposals(agent_id, metrics)

    if not proposals:
        print("No proposals generated (all metrics nominal).")
        sys.exit(0)

    print(f"\nGenerated {len(proposals)} proposal(s):\n")

    # Write proposals
    proposals_dir = agent_home / "proposals"
    if not args.dry_run:
        proposals_dir.mkdir(parents=True, exist_ok=True)

    for prop in proposals:
        print(f"  [{prop.priority.upper()}] {prop.title}")
        print(f"    {prop.rationale[:120]}...")
        print()

        if not args.dry_run:
            filename = f"{prop.proposal_id}.yaml"
            filepath = proposals_dir / filename
            filepath.write_text(prop.to_yaml_str())
            print(f"    -> Written to {filepath}")

    if args.dry_run:
        print("\n(dry run -- no files written)")
    else:
        print(f"\nProposals written to {proposals_dir}/")
        print("Review and update status field: pending -> approved | rejected | superseded")


if __name__ == "__main__":
    main()
