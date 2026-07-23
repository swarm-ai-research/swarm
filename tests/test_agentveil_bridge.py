"""Integration + mitigation tests for AgentVeilBridge (v1, mock-only).

Covers the orchestrator wiring (client + mapper + policy + EventLog) and the
failure-mode contracts the bridge is responsible for end-to-end:

  - End-to-end flow: trust-check → admission → interaction → write-back.
  - ATTESTATION_SUBMITTED logging: every write-back is mirrored to the core
    EventLog *and* the in-memory AgentVeilEvent audit list (D4).
  - D5 (p invariant): malformed / out-of-range trust responses still yield a
    SoftInteraction with p ∈ [0, 1].
  - C4 (write-back amplification): uncertain-band suppression, plus the tier
    amplification cap surfacing on the produced observables.
  - C5 (collusion rate limit): one attestation per (initiator, counterparty)
    pair per epoch; counters rotate on epoch change.
  - C6 (dispute flooding): negative-attestation quota per initiator per epoch.
  - D6 (deterministic replay): two independent bridges fed identical fixtures
    and inputs produce identical decision content and a replayable core log.
  - E3 (privacy): attestation payloads carry only the sign + canonical
    SHA-256(interaction_id || ±sign) hash, never the raw p.
  - A1 (registry down): fail-closed circuit breaker denies before any call.
"""

from __future__ import annotations

import hashlib

import pytest

from swarm.bridges.agentveil import (
    AgentVeilBridge,
    AgentVeilConfig,
    AgentVeilEventType,
    AttestationEvent,
    AVPClient,
)
from swarm.logging.event_log import EventLog
from swarm.models.events import EventType

ORCH = "did:key:zOrchestrator00000000000000000000000000000"


def _did(name: str) -> str:
    # Pad to a realistic-looking did:key length; uniqueness is by name.
    return f"did:key:z6Mk{name}" + "0" * (40 - len(name))


def _client(**tiers: dict) -> AVPClient:
    """Build a mock client whose fixtures are keyed by the helper DIDs."""
    return AVPClient(fixtures={_did(k): v for k, v in tiers.items()})


def _elite() -> dict:
    return {
        "tier": "elite",
        "allowed": True,
        "risk_level": "low",
        "confidence": 0.95,
        "reputation_score": 0.9,
        "attestation_count": 20,
    }


def _newcomer_allowed() -> dict:
    # Admitted (min_tier=newcomer) but maps to a mid-band p → uncertain band.
    return {
        "tier": "newcomer",
        "allowed": True,
        "risk_level": "high",
        "confidence": 0.5,
    }


class TestEndToEndFlow:
    def test_trusted_agent_produces_interaction_and_positive_attestation(self):
        bridge = AgentVeilBridge(
            AgentVeilConfig(min_tier="basic"),
            client=_client(alice=_elite()),
        )
        ix = bridge.process_interaction(
            initiator_did=ORCH, counterparty_did=_did("alice"), epoch=0
        )

        assert ix is not None
        assert ix.accepted is True
        assert 0.0 <= ix.p <= 1.0
        assert ix.p >= 0.7  # elite + high confidence → positive band
        assert ix.metadata["bridge"] == "agentveil"
        assert ix.metadata["tier"] == "elite"

        submitted = bridge.get_events_by_type(
            AgentVeilEventType.ATTESTATION_SUBMITTED
        )
        assert len(submitted) == 1
        assert submitted[0].payload["outcome_sign"] == 1
        # Client actually received the attestation.
        assert len(bridge.client.submitted_attestations) == 1

    def test_admission_denied_returns_none_no_interaction(self):
        bridge = AgentVeilBridge(
            AgentVeilConfig(min_tier="trusted"),  # elite passes; basic fails
            client=_client(
                bob={"tier": "basic", "allowed": True, "confidence": 0.8}
            ),
        )
        result = bridge.process_interaction(
            initiator_did=ORCH, counterparty_did=_did("bob"), epoch=0
        )
        assert result is None
        assert bridge.get_interactions() == []
        denied = bridge.get_events_by_type(AgentVeilEventType.TRUST_DENIED)
        assert len(denied) == 1
        assert "below_min" in denied[0].payload["reason"]
        # No attestation attempted on a denied admission.
        assert bridge.client.submitted_attestations == []

    def test_registry_denied_flag_denies_admission(self):
        bridge = AgentVeilBridge(
            AgentVeilConfig(min_tier="newcomer"),
            client=_client(
                mal={"tier": "elite", "allowed": False, "confidence": 0.9}
            ),
        )
        result = bridge.process_interaction(
            initiator_did=ORCH, counterparty_did=_did("mal"), epoch=0
        )
        assert result is None
        denied = bridge.get_events_by_type(AgentVeilEventType.TRUST_DENIED)
        assert len(denied) == 1
        assert denied[0].payload["reason"] == "admission_gate_registry_denied"


class TestAttestationLogging:
    """D4: the write-back is never hidden — it lands in both logs."""

    def test_core_event_log_records_attestation_submitted(self, tmp_path):
        log = EventLog(tmp_path / "log.jsonl")
        bridge = AgentVeilBridge(
            AgentVeilConfig(min_tier="basic"),
            client=_client(alice=_elite()),
            event_log=log,
        )
        ix = bridge.process_interaction(
            initiator_did=ORCH, counterparty_did=_did("alice"), epoch=0
        )

        events = list(log.replay())
        types = [e.event_type for e in events]
        assert EventType.INTERACTION_COMPLETED in types
        assert EventType.ATTESTATION_SUBMITTED in types

        att = next(
            e for e in events if e.event_type == EventType.ATTESTATION_SUBMITTED
        )
        assert att.interaction_id == ix.interaction_id
        assert att.payload["outcome_sign"] == 1
        assert att.payload["bridge"] == "agentveil"

    def test_attestation_payload_omits_raw_p_and_uses_canonical_hash(self, tmp_path):
        # E3: only sign + SHA-256(interaction_id || ±sign), never raw p.
        log = EventLog(tmp_path / "log.jsonl")
        bridge = AgentVeilBridge(
            AgentVeilConfig(min_tier="basic"),
            client=_client(alice=_elite()),
            event_log=log,
        )
        ix = bridge.process_interaction(
            initiator_did=ORCH, counterparty_did=_did("alice"), epoch=0
        )
        att = next(
            e for e in log.replay()
            if e.event_type == EventType.ATTESTATION_SUBMITTED
        )
        assert "p" not in att.payload
        expected = hashlib.sha256(
            f"{ix.interaction_id}||+1".encode()
        ).hexdigest()
        assert att.payload["evidence_hash"] == expected
        # The in-memory audit event carries the same hash.
        submitted = bridge.get_events_by_type(
            AgentVeilEventType.ATTESTATION_SUBMITTED
        )
        assert submitted[0].payload["evidence_hash"] == expected
        assert "p" not in submitted[0].payload

    def test_works_without_event_log(self):
        # event_log is optional; the in-memory audit trail still records.
        bridge = AgentVeilBridge(
            AgentVeilConfig(min_tier="basic"), client=_client(alice=_elite())
        )
        bridge.process_interaction(
            initiator_did=ORCH, counterparty_did=_did("alice"), epoch=0
        )
        assert len(
            bridge.get_events_by_type(AgentVeilEventType.ATTESTATION_SUBMITTED)
        ) == 1


class TestD5PInvariant:
    """Malformed trust responses must not produce p outside [0, 1]."""

    @pytest.mark.parametrize(
        "fixture",
        [
            {"tier": "elite", "allowed": True, "confidence": 9.9},  # out of range
            {"tier": "elite", "allowed": True, "confidence": -5.0},
            {"tier": "trusted", "allowed": True, "confidence": 100.0},
            {"tier": "newcomer", "allowed": True, "confidence": 0.5},
        ],
    )
    def test_p_stays_in_unit_interval(self, fixture):
        bridge = AgentVeilBridge(
            AgentVeilConfig(min_tier="newcomer"),
            client=_client(x=fixture),
        )
        ix = bridge.process_interaction(
            initiator_did=ORCH, counterparty_did=_did("x"), epoch=0
        )
        assert ix is not None
        assert 0.0 <= ix.p <= 1.0
        assert -1.0 <= ix.v_hat <= 1.0


class TestC4WritebackAmplification:
    def test_uncertain_band_suppresses_attestation(self):
        bridge = AgentVeilBridge(
            AgentVeilConfig(min_tier="newcomer"),
            client=_client(mid=_newcomer_allowed()),
        )
        ix = bridge.process_interaction(
            initiator_did=ORCH, counterparty_did=_did("mid"), epoch=0
        )
        # Confirm the interaction really lands in the uncertain band so the
        # suppression below is testing C4 and not some other gate.
        assert 0.3 <= ix.p < 0.7, f"expected uncertain-band p, got {ix.p}"

        assert bridge.get_events_by_type(
            AgentVeilEventType.ATTESTATION_SUBMITTED
        ) == []
        suppressed = bridge.get_events_by_type(
            AgentVeilEventType.ATTESTATION_SUPPRESSED
        )
        assert len(suppressed) == 1
        assert suppressed[0].payload["suppression_reason"] == "uncertain_band"
        assert bridge.client.submitted_attestations == []

    def test_tier_amplification_cap_applied_to_observables(self):
        # C4: elite tier ordinal (1.0) is capped at max_tier_v_hat_contribution.
        cap = 0.6
        bridge = AgentVeilBridge(
            AgentVeilConfig(min_tier="basic", max_tier_v_hat_contribution=cap),
            client=_client(alice=_elite()),
        )
        ix = bridge.process_interaction(
            initiator_did=ORCH, counterparty_did=_did("alice"), epoch=0
        )
        assert ix.task_progress_delta == pytest.approx(cap)

    def test_suppressed_writeback_logged_to_core_log(self, tmp_path):
        log = EventLog(tmp_path / "log.jsonl")
        bridge = AgentVeilBridge(
            AgentVeilConfig(min_tier="newcomer"),
            client=_client(mid=_newcomer_allowed()),
            event_log=log,
        )
        bridge.process_interaction(
            initiator_did=ORCH, counterparty_did=_did("mid"), epoch=0
        )
        types = [e.event_type for e in log.replay()]
        assert EventType.ATTESTATION_SUPPRESSED in types
        assert EventType.ATTESTATION_SUBMITTED not in types


class TestC5PerPairRateLimit:
    def test_second_same_pair_same_epoch_suppressed(self):
        bridge = AgentVeilBridge(
            AgentVeilConfig(min_tier="basic"),
            client=_client(alice=_elite()),
        )
        bridge.process_interaction(
            initiator_did=ORCH, counterparty_did=_did("alice"), epoch=0
        )
        bridge.process_interaction(
            initiator_did=ORCH, counterparty_did=_did("alice"), epoch=0
        )
        assert len(
            bridge.get_events_by_type(AgentVeilEventType.ATTESTATION_SUBMITTED)
        ) == 1
        suppressed = bridge.get_events_by_type(
            AgentVeilEventType.ATTESTATION_SUPPRESSED
        )
        assert len(suppressed) == 1
        assert suppressed[0].payload["suppression_reason"] == "per_pair_rate_limit"

    def test_new_epoch_resets_pair_limit(self):
        bridge = AgentVeilBridge(
            AgentVeilConfig(min_tier="basic"),
            client=_client(alice=_elite()),
        )
        bridge.process_interaction(
            initiator_did=ORCH, counterparty_did=_did("alice"), epoch=0
        )
        bridge.process_interaction(
            initiator_did=ORCH, counterparty_did=_did("alice"), epoch=1
        )
        assert len(
            bridge.get_events_by_type(AgentVeilEventType.ATTESTATION_SUBMITTED)
        ) == 2
        assert bridge.current_epoch == 1


class TestC6NegativeAttestationQuota:
    """Force a negative write-back via thresholds, then exhaust the quota."""

    def _negative_config(self) -> AgentVeilConfig:
        # A newcomer-allowed interaction maps to p ≈ 0.65. Putting BOTH
        # thresholds above that p (uncertain band = [0.70, 0.99)) leaves p
        # below the band, so the write-back gate passes and resolves to a
        # *negative* attestation (sign = -1, since p < positive_threshold).
        return AgentVeilConfig(
            min_tier="newcomer",
            writeback_positive_threshold=0.99,
            writeback_negative_threshold=0.70,
        )

    def test_negative_attestation_has_sign_minus_one(self):
        bridge = AgentVeilBridge(
            self._negative_config(), client=_client(mid=_newcomer_allowed())
        )
        ix = bridge.process_interaction(
            initiator_did=ORCH, counterparty_did=_did("mid"), epoch=0
        )
        submitted = bridge.get_events_by_type(
            AgentVeilEventType.ATTESTATION_SUBMITTED
        )
        assert len(submitted) == 1
        assert submitted[0].payload["outcome_sign"] == -1
        expected = hashlib.sha256(
            f"{ix.interaction_id}||-1".encode()
        ).hexdigest()
        assert submitted[0].payload["evidence_hash"] == expected

    def test_sixth_negative_from_same_initiator_suppressed(self):
        # Six distinct counterparties (per-pair limit allows one each); the
        # quota of 5 negatives/initiator/epoch trips on the sixth.
        fixtures = {f"c{i}": _newcomer_allowed() for i in range(6)}
        bridge = AgentVeilBridge(self._negative_config(), client=_client(**fixtures))
        for i in range(6):
            bridge.process_interaction(
                initiator_did=ORCH, counterparty_did=_did(f"c{i}"), epoch=0
            )
        submitted = bridge.get_events_by_type(
            AgentVeilEventType.ATTESTATION_SUBMITTED
        )
        suppressed = bridge.get_events_by_type(
            AgentVeilEventType.ATTESTATION_SUPPRESSED
        )
        assert len(submitted) == 5
        assert len(suppressed) == 1
        assert (
            suppressed[0].payload["suppression_reason"]
            == "negative_attestation_quota"
        )


class TestD6Reproducibility:
    """Mock mode: identical fixtures + inputs → identical decision content."""

    def _run(self, log_path):
        log = EventLog(log_path)
        bridge = AgentVeilBridge(
            AgentVeilConfig(min_tier="newcomer"),
            client=_client(alice=_elite(), mid=_newcomer_allowed()),
            event_log=log,
        )
        for did in ("alice", "mid"):
            bridge.process_interaction(
                initiator_did=ORCH, counterparty_did=_did(did), epoch=0
            )
        return bridge, log

    def test_decision_content_is_deterministic(self, tmp_path):
        b1, log1 = self._run(tmp_path / "a.jsonl")
        b2, log2 = self._run(tmp_path / "b.jsonl")

        # Decision content (p, v_hat, sign) reproduces; only uuid/timestamp
        # identity metadata differs between runs.
        p1 = [(round(i.v_hat, 12), round(i.p, 12)) for i in b1.get_interactions()]
        p2 = [(round(i.v_hat, 12), round(i.p, 12)) for i in b2.get_interactions()]
        assert p1 == p2

        t1 = [e.event_type for e in b1.get_events()]
        t2 = [e.event_type for e in b2.get_events()]
        assert t1 == t2

        signs1 = [
            e.payload.get("outcome_sign")
            for e in b1.get_events()
            if e.event_type
            in (
                AgentVeilEventType.ATTESTATION_SUBMITTED,
                AgentVeilEventType.ATTESTATION_SUPPRESSED,
            )
        ]
        signs2 = [
            e.payload.get("outcome_sign")
            for e in b2.get_events()
            if e.event_type
            in (
                AgentVeilEventType.ATTESTATION_SUBMITTED,
                AgentVeilEventType.ATTESTATION_SUPPRESSED,
            )
        ]
        assert signs1 == signs2

    def test_core_log_replays_as_core_events(self, tmp_path):
        # The on-disk log contains only core Events (replayable invariant).
        _, log = self._run(tmp_path / "a.jsonl")
        types = [e.event_type for e in log.replay()]
        assert types  # non-empty
        assert all(isinstance(t, EventType) for t in types)


class TestCircuitBreaker:
    def test_fail_closed_denies_after_threshold_failures(self):
        bridge = AgentVeilBridge(
            AgentVeilConfig(min_tier="basic", fail_mode="closed"),
            client=_client(alice=_elite()),
        )
        for _ in range(5):
            bridge.record_registry_failure()
        assert bridge.circuit_open is True
        result = bridge.process_interaction(
            initiator_did=ORCH, counterparty_did=_did("alice"), epoch=0
        )
        assert result is None
        assert bridge.get_events_by_type(
            AgentVeilEventType.CIRCUIT_BREAKER_TRIPPED
        )

    def test_fail_open_allows_with_warning(self):
        bridge = AgentVeilBridge(
            AgentVeilConfig(min_tier="basic", fail_mode="open"),
            client=_client(alice=_elite()),
        )
        for _ in range(5):
            bridge.record_registry_failure()
        assert bridge.circuit_open is False
        result = bridge.process_interaction(
            initiator_did=ORCH, counterparty_did=_did("alice"), epoch=0
        )
        assert result is not None

    def test_success_resets_failure_count(self):
        bridge = AgentVeilBridge(
            AgentVeilConfig(min_tier="basic", fail_mode="closed"),
            client=_client(alice=_elite()),
        )
        for _ in range(4):
            bridge.record_registry_failure()
        bridge.record_registry_success()
        assert bridge.circuit_open is False


class TestAuxiliaryFlows:
    def test_ingest_incoming_attestation_produces_interaction(self):
        bridge = AgentVeilBridge(AgentVeilConfig(), client=_client())
        att = AttestationEvent(
            interaction_id="ext-1", outcome_sign=-1, evidence_hash="x"
        )
        ix = bridge.ingest_attestation(
            att, initiator_id=ORCH, counterparty_id=_did("carol")
        )
        assert ix.interaction_id == "ext-1"
        assert 0.0 <= ix.p <= 1.0
        assert ix.p < 0.5  # negative incoming attestation
        assert ix.accepted is False
        assert bridge.get_interactions() == [ix]

    def test_fetch_reputation_records_event(self):
        bridge = AgentVeilBridge(
            AgentVeilConfig(), client=_client(alice=_elite())
        )
        snap = bridge.fetch_reputation(_did("alice"))
        assert snap.tier == "elite"
        assert bridge.get_events_by_type(
            AgentVeilEventType.REPUTATION_FETCHED
        )
