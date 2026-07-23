"""Track-record reputation for agent-authored changes.

Wires the existing :class:`~swarm.agents.performance_tracker.PerformanceTracker`
(append-only JSONL event substrate) into agentgit outcomes. A track record
attaches not to a bare agent name but to a :class:`ReputationKey` — agent
identity + runtime + toolchain + prompt/config version — because "claude on
runtime X with config v3" and "claude on runtime Y with config v7" are
different actors with different failure modes.

Recorded outcomes per key:

- attestations (policy pass/fail, review-panel verdicts, review burden,
  security findings) — derived automatically from a provenance bundle;
- merge / rollback / test-breakage / security-incident outcomes — supplied
  later by CI or the operator, when the ground truth is known;
- domains of competence — top-level paths the key has shipped changes in.

``track_record()`` folds the event stream into rates plus a trust score in
``[0, 1]``, and the policy engine consumes that score through the
``trust_below`` rule condition (see :mod:`swarm.agentgit.policy`):
low reputation → restricted scope.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from swarm.agents.performance_tracker import PerformanceEvent, PerformanceTracker

DEFAULT_LEDGER_PATH = ".agentgit/reputation.jsonl"

# agentgit-specific event types, layered on the PerformanceEvent substrate.
EVENT_ATTESTED = "agentgit_attested"
EVENT_MERGED = "agentgit_merged"
EVENT_ROLLED_BACK = "agentgit_rolled_back"
EVENT_TESTS_BROKEN = "agentgit_tests_broken"
EVENT_SECURITY_INCIDENT = "agentgit_security_incident"

# Trust-score severity weights: how many "failure units" each bad outcome
# contributes to the Beta-style estimate below. A security incident is worse
# than a rollback is worse than a broken test run.
_FAILURE_WEIGHTS = {
    EVENT_ROLLED_BACK: 2.0,
    EVENT_TESTS_BROKEN: 1.0,
    EVENT_SECURITY_INCIDENT: 3.0,
}


@dataclass(frozen=True)
class ReputationKey:
    """The exact actor a track record attaches to.

    Identity alone is not enough: the same agent under a different runtime,
    toolchain, or prompt/config version behaves differently, so each
    combination accrues its own record. ``key_id`` is a stable digest used
    as the ledger's agent identifier.
    """

    agent_id: str
    did: str = ""
    model: str = ""
    runtime: str = ""
    toolchain: str = ""
    config_version: str = ""

    @property
    def key_id(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True)
        return f"{self.agent_id}@{hashlib.sha256(canonical.encode()).hexdigest()[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_bundle(cls, bundle: Dict[str, Any]) -> "ReputationKey":
        """Derive the key from a provenance bundle.

        Runtime facts come from ``provenance.environment`` — the block the
        enriched bundle (issue 8ll9) records model/runtime/toolchain under.
        Missing fields stay empty: a sparse key still accrues a record, it
        is just a coarser actor.
        """

        env = bundle.get("provenance", {}).get("environment", {}) or {}
        return cls(
            agent_id=str(bundle.get("agent", {}).get("agent_id", "") or ""),
            did=str(bundle.get("identity", {}).get("did", "") or ""),
            model=str(env.get("model", "") or ""),
            runtime=str(env.get("runtime", "") or ""),
            toolchain=str(env.get("toolchain", "") or ""),
            config_version=str(env.get("config_version", "") or ""),
        )


@dataclass
class TrackRecord:
    """Aggregated track record for one reputation key."""

    key_id: str
    key: Dict[str, Any]
    attestations: int = 0
    policy_passes: int = 0
    merges: int = 0
    merge_attempts: int = 0
    rollbacks: int = 0
    test_breakages: int = 0
    security_incidents: int = 0
    security_findings: int = 0
    review_findings: int = 0
    domains: Dict[str, int] = field(default_factory=dict)
    trust: float = 0.5

    @property
    def policy_pass_rate(self) -> float:
        return self.policy_passes / self.attestations if self.attestations else 0.0

    @property
    def merge_success_rate(self) -> float:
        return self.merges / self.merge_attempts if self.merge_attempts else 0.0

    @property
    def rollback_rate(self) -> float:
        return self.rollbacks / self.merges if self.merges else 0.0

    @property
    def review_burden(self) -> float:
        """Mean panel findings per attestation — how much reviewer attention
        this actor's changes consume."""
        return self.review_findings / self.attestations if self.attestations else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "policy_pass_rate": self.policy_pass_rate,
            "merge_success_rate": self.merge_success_rate,
            "rollback_rate": self.rollback_rate,
            "review_burden": self.review_burden,
        }


class ReputationLedger:
    """Append-only reputation ledger over the PerformanceTracker substrate.

    One JSONL file holds every key's events (``PerformanceEvent.agent_id``
    carries the ``key_id``); per-key trackers share it. History is never
    rewritten — the record is replayable like every other agentgit log.
    """

    def __init__(self, path: Path | str = DEFAULT_LEDGER_PATH) -> None:
        self.path = Path(path)

    def _tracker(self, key_id: str) -> PerformanceTracker:
        return PerformanceTracker(key_id, self.path)

    # -- recording -----------------------------------------------------------
    def record_attestation(self, bundle: Dict[str, Any]) -> ReputationKey:
        """Fold one attested bundle into the ledger.

        Everything derivable is derived: policy verdict, panel review burden
        and blocking security findings (from ``provenance.reviews``), and
        the top-level domains the diff touched. Failing attestations are
        recorded too — they are exactly the track record's job.
        """

        key = ReputationKey.from_bundle(bundle)
        reviews = bundle.get("provenance", {}).get("reviews", []) or []
        review_findings = sum(len(r.get("findings", []) or []) for r in reviews)
        security_findings = sum(
            1
            for r in reviews
            for f in (r.get("findings", []) or [])
            if r.get("reviewer") == "security" and f.get("severity") == "blocking"
        )
        domains = sorted(
            {
                f.get("path", "").split("/", 1)[0]
                for f in bundle.get("git", {}).get("changed_files", [])
                if f.get("path")
            }
        )
        self._tracker(key.key_id)._append(
            EVENT_ATTESTED,
            {
                "task_id": bundle.get("task", {}).get("task_id", ""),
                "repo": bundle.get("git", {}).get("repo_path", ""),
                "policy_passed": bool(bundle.get("policy", {}).get("passed", False)),
                "review_findings": review_findings,
                "security_findings": security_findings,
                "domains": domains,
                "key": key.to_dict(),
            },
        )
        return key

    def record_outcome(
        self,
        key_id: str,
        outcome: str,
        *,
        task_id: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a ground-truth outcome (merge/rollback/breakage/incident).

        These arrive after attestation, from CI or the operator — the agent
        never self-attests its own merge success.
        """

        valid = {EVENT_MERGED, EVENT_ROLLED_BACK, EVENT_TESTS_BROKEN, EVENT_SECURITY_INCIDENT}
        if outcome not in valid:
            raise ValueError(f"invalid outcome {outcome!r}; expected one of {sorted(valid)}")
        self._tracker(key_id)._append(
            outcome, {"task_id": task_id, **(details or {})}
        )

    # -- reading -------------------------------------------------------------
    def _events(self, key_id: Optional[str] = None) -> List[PerformanceEvent]:
        if not self.path.exists():
            return []
        events: List[PerformanceEvent] = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                event = PerformanceEvent.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if key_id is None or event.agent_id == key_id:
                events.append(event)
        return events

    def key_ids(self) -> List[str]:
        return sorted({e.agent_id for e in self._events()})

    def keys_for_agent(self, agent_id: str) -> List[str]:
        """Every reputation key a bare agent name has accrued records under."""
        return [k for k in self.key_ids() if k.split("@", 1)[0] == agent_id]

    def track_record(self, key_id: str) -> TrackRecord:
        """Fold the key's events into rates + a trust score.

        Trust is a Beta-style smoothed estimate: ``(successes + 1) /
        (successes + failures + 2)`` — an unknown actor starts at 0.5 and
        moves with evidence. Successes = merges + policy passes; failures =
        policy fails + severity-weighted rollbacks / test breakages /
        security incidents. Deterministic, replayable from the ledger.
        """

        record = TrackRecord(key_id=key_id, key={})
        successes = 0.0
        failures = 0.0
        for event in self._events(key_id):
            meta = event.metadata or {}
            if event.event_type == EVENT_ATTESTED:
                record.attestations += 1
                record.review_findings += int(meta.get("review_findings", 0) or 0)
                record.security_findings += int(meta.get("security_findings", 0) or 0)
                if not record.key:
                    record.key = dict(meta.get("key", {}) or {})
                for domain in meta.get("domains", []) or []:
                    record.domains[domain] = record.domains.get(domain, 0) + 1
                if meta.get("policy_passed"):
                    record.policy_passes += 1
                    successes += 0.5  # attest-time signal, weaker than a merge
                else:
                    failures += 0.5
            elif event.event_type == EVENT_MERGED:
                record.merges += 1
                record.merge_attempts += 1
                successes += 1.0
            elif event.event_type == EVENT_ROLLED_BACK:
                record.rollbacks += 1
                failures += _FAILURE_WEIGHTS[EVENT_ROLLED_BACK]
            elif event.event_type == EVENT_TESTS_BROKEN:
                record.test_breakages += 1
                failures += _FAILURE_WEIGHTS[EVENT_TESTS_BROKEN]
            elif event.event_type == EVENT_SECURITY_INCIDENT:
                record.security_incidents += 1
                failures += _FAILURE_WEIGHTS[EVENT_SECURITY_INCIDENT]
        record.trust = (successes + 1.0) / (successes + failures + 2.0)
        return record

    def trust_for_bundle(self, bundle: Dict[str, Any]) -> Optional[float]:
        """Trust score for the bundle's reputation key.

        ``None`` when the key has no history at all — the policy engine's
        ``trust_below`` condition treats unknown as below-threshold
        (fail-closed): no track record is not the same as a good one.
        """

        key = ReputationKey.from_bundle(bundle)
        if not self._events(key.key_id):
            return None
        return self.track_record(key.key_id).trust
