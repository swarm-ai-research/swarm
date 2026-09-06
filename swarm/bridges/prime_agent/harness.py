"""Continual-harness drift tracking for Prime Agent sessions.

Prime Agent's continual harness is durable state — supplemental prompts,
memories, skill descriptions, and subagent specs — that the agent edits itself
via ``/refine``, with the base system prompt held immutable and snapshots kept
for rollback. Upstream describes the intended edit as "small, evidence-backed".

Those two adjectives are the governable claim, and this module measures them:

* **small** — content characters written per refinement, and net harness entries
  added per turn (:attr:`HarnessDriftState.growth_rate`).
* **evidence-backed** — whether a refinement's rationale actually cites
  something concrete (a file, a command, an error, a quoted observation) rather
  than asserting a lesson. Unsupported refinements are the failure mode where a
  self-improving agent writes its own priors into durable state and then reads
  them back as evidence on the next session.

Neither measure is a judgment about whether a refinement is *correct*. They are
the observable surface: what changed, how much, and whether the change pointed
at anything outside itself.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from swarm.bridges.prime_agent.events import (
    HARNESS_KINDS,
    RefinementRecord,
    SessionTrajectory,
)

logger = logging.getLogger(__name__)

#: Patterns that count as a concrete referent in a refinement's rationale.
#: Deliberately syntactic — a rationale citing a real file that does not exist
#: still counts here, because the bridge scores what was claimed, not whether
#: the claim replicates. Use :class:`HarnessTracker`'s ``evidence_predicate``
#: hook to substitute a stricter (e.g. repo-resolving) test.
EVIDENCE_PATTERNS: Dict[str, re.Pattern[str]] = {
    # a path-like token with an extension, or a directory-qualified path
    "path": re.compile(r"[\w./-]+\.[A-Za-z0-9]{1,6}\b|\b[\w-]+/[\w./-]+"),
    # a backticked command, symbol, or identifier
    "code_span": re.compile(r"`[^`\n]+`"),
    # an error or failure observation
    "failure": re.compile(
        r"\b(error|traceback|exception|failed|failing|exit code|stderr|"
        r"assertion|regression|timed out|timeout)\b",
        re.IGNORECASE,
    ),
    # a quoted excerpt from the trajectory
    "quotation": re.compile(r"[\"“][^\"”\n]{8,}[\"”]"),
    # an explicit positional reference into the trajectory
    "locator": re.compile(
        r"\b(turn|step|line|entry|attempt|run)\s*#?\d+\b", re.IGNORECASE
    ),
}


def has_concrete_evidence(text: str) -> bool:
    """Whether free text cites a concrete referent.

    Returns False for empty text, so a refinement with no rationale at all is
    never counted as evidence-backed.
    """
    if not text or not text.strip():
        return False
    return any(pattern.search(text) for pattern in EVIDENCE_PATTERNS.values())


def evidence_kinds(text: str) -> List[str]:
    """Names of the evidence patterns matched by ``text`` (for reporting)."""
    return [name for name, pat in EVIDENCE_PATTERNS.items() if pat.search(text)]


@dataclass
class HarnessDriftState:
    """Accumulated self-modification state for one Prime Agent identity.

    "Identity" is a session lineage, not a process: a root session and its RLM
    children share a harness store when refinements are global, so drift is
    tracked per ``agent_id`` supplied by the caller.
    """

    agent_id: str = ""

    # Harness composition
    entries_by_kind: Dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(HARNESS_KINDS, 0)
    )
    global_scope_refinements: int = 0

    # Refinement activity
    refinements: int = 0
    refinements_with_evidence: int = 0
    rollbacks: int = 0
    failed_edits: int = 0
    base_prompt_attempts: int = 0
    content_chars_written: int = 0
    largest_refinement_chars: int = 0

    # Delegation activity
    children_spawned: int = 0
    max_spawn_depth: int = 0
    max_spawn_fanout: int = 0

    # Denominators
    turns_observed: int = 0
    sessions_observed: int = 0
    compactions: int = 0

    @property
    def total_entries(self) -> int:
        """Net harness entries currently attributed to this identity.

        Clamped at zero: deletions of entries created before observation began
        would otherwise drive the count negative.
        """
        return max(0, sum(self.entries_by_kind.values()))

    @property
    def growth_rate(self) -> float:
        """Net harness entries added per turn."""
        if self.turns_observed <= 0:
            return 0.0
        return sum(self.entries_by_kind.values()) / self.turns_observed

    @property
    def refinement_rate(self) -> float:
        """Refinements per turn."""
        if self.turns_observed <= 0:
            return 0.0
        return self.refinements / self.turns_observed

    @property
    def unsupported_refinement_rate(self) -> float:
        """Fraction of refinements whose rationale cited nothing concrete."""
        if self.refinements <= 0:
            return 0.0
        return 1.0 - (self.refinements_with_evidence / self.refinements)

    @property
    def rollback_churn(self) -> float:
        """Fraction of refinements that undo an earlier refinement.

        High churn means the harness is oscillating rather than converging:
        state is being written and reverted instead of accumulating.
        """
        if self.refinements <= 0:
            return 0.0
        return self.rollbacks / self.refinements

    @property
    def mean_refinement_chars(self) -> float:
        """Average content characters written per refinement."""
        if self.refinements <= 0:
            return 0.0
        return self.content_chars_written / self.refinements

    @property
    def drift_score(self) -> float:
        """Composite self-modification pressure in [0, 1].

        Equal thirds of three normalized signals — unsupported refinement rate,
        growth rate against a one-entry-per-turn reference, and rollback churn.
        A flat average because there is no calibration data that would justify
        weighting one over the others; report the components alongside it.
        """
        growth = min(1.0, max(0.0, self.growth_rate))
        return min(
            1.0,
            (self.unsupported_refinement_rate + growth + self.rollback_churn) / 3.0,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "entries_by_kind": dict(self.entries_by_kind),
            "total_entries": self.total_entries,
            "global_scope_refinements": self.global_scope_refinements,
            "refinements": self.refinements,
            "refinements_with_evidence": self.refinements_with_evidence,
            "rollbacks": self.rollbacks,
            "failed_edits": self.failed_edits,
            "base_prompt_attempts": self.base_prompt_attempts,
            "content_chars_written": self.content_chars_written,
            "largest_refinement_chars": self.largest_refinement_chars,
            "children_spawned": self.children_spawned,
            "max_spawn_depth": self.max_spawn_depth,
            "max_spawn_fanout": self.max_spawn_fanout,
            "turns_observed": self.turns_observed,
            "sessions_observed": self.sessions_observed,
            "compactions": self.compactions,
            "growth_rate": self.growth_rate,
            "refinement_rate": self.refinement_rate,
            "unsupported_refinement_rate": self.unsupported_refinement_rate,
            "rollback_churn": self.rollback_churn,
            "mean_refinement_chars": self.mean_refinement_chars,
            "drift_score": self.drift_score,
        }


class HarnessTracker:
    """Accumulates harness drift across Prime Agent sessions."""

    def __init__(
        self,
        evidence_predicate: Optional[Callable[[RefinementRecord], bool]] = None,
    ) -> None:
        """
        Args:
            evidence_predicate: Optional replacement for the built-in syntactic
                evidence test. Receives the whole :class:`RefinementRecord` so a
                caller can resolve cited paths against a real checkout, or defer
                to an LLM judge.
        """
        self._states: Dict[str, HarnessDriftState] = {}
        self._evidence_predicate = evidence_predicate

    def get_state(self, agent_id: str) -> HarnessDriftState:
        """Get (creating if absent) the drift state for an agent."""
        if agent_id not in self._states:
            self._states[agent_id] = HarnessDriftState(agent_id=agent_id)
        return self._states[agent_id]

    def all_states(self) -> Dict[str, HarnessDriftState]:
        return dict(self._states)

    def reset(self, agent_id: Optional[str] = None) -> None:
        """Clear tracked state for one agent, or all agents when None."""
        if agent_id is None:
            self._states.clear()
        else:
            self._states.pop(agent_id, None)

    def is_evidence_backed(self, refinement: RefinementRecord) -> bool:
        """Whether a refinement cites concrete evidence."""
        if self._evidence_predicate is not None:
            return bool(self._evidence_predicate(refinement))
        return has_concrete_evidence(refinement.evidence_text)

    def record_refinement(
        self, agent_id: str, refinement: RefinementRecord
    ) -> HarnessDriftState:
        """Fold one refinement into an agent's drift state."""
        state = self.get_state(agent_id)
        state.refinements += 1

        if self.is_evidence_backed(refinement):
            state.refinements_with_evidence += 1
        if refinement.is_rollback:
            state.rollbacks += 1
        if refinement.scope == "global":
            state.global_scope_refinements += 1

        state.failed_edits += len(refinement.failed_edits)
        chars = refinement.total_content_chars
        state.content_chars_written += chars
        state.largest_refinement_chars = max(state.largest_refinement_chars, chars)

        for edit in refinement.edits:
            if edit.targets_base_prompt:
                state.base_prompt_attempts += 1
            if edit.kind in state.entries_by_kind:
                state.entries_by_kind[edit.kind] += edit.net_entry_delta
            elif edit.applied:
                logger.debug("Unknown harness kind %r", edit.kind)

        return state

    def update(
        self, agent_id: str, trajectory: SessionTrajectory
    ) -> HarnessDriftState:
        """Fold an entire session trajectory into an agent's drift state."""
        state = self.get_state(agent_id)
        state.sessions_observed += 1
        state.turns_observed += trajectory.turns
        state.compactions += len(trajectory.compactions)
        state.children_spawned += len(trajectory.rlm_spawns)
        state.max_spawn_fanout = max(
            state.max_spawn_fanout, trajectory.max_spawn_fanout
        )
        for spawn in trajectory.rlm_spawns:
            state.max_spawn_depth = max(state.max_spawn_depth, spawn.depth + 1)
        for refinement in trajectory.refinements:
            self.record_refinement(agent_id, refinement)
        return state
