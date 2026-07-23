"""Regression + mitigation tests for AVPMapper.

Covers:
  - Codex P2 fix: negative attestations populate penalty fields so v_hat
    tracks the attestation's sign.
  - D5 (p invariant): all observable fields land in [-1, +1]; downstream
    SoftInteraction.p remains in [0, 1].
  - C4 (tier amplification): tier contribution is capped at
    max_tier_v_hat_contribution; no single tier can drive v_hat past
    that ceiling.
  - D1 (no double-counting): reputation_change_to_update always returns
    r_delta = 0.0 in v1; tier influences p exclusively through the
    mapper, not via payoff-side reputation.
  - D2 (categorical → ordinal): tier strings map to a fixed ordinal
    ladder; unknown tiers fall back to 0.0; attestation outcome_sign
    is read as continuous ±1 rather than parsed from a tier label.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from swarm.bridges.agentveil.config import AgentVeilConfig
from swarm.bridges.agentveil.mapper import AVPMapper
from swarm.core.proxy import ProxyComputer


def _att(outcome_sign: int) -> SimpleNamespace:
    return SimpleNamespace(
        outcome_sign=outcome_sign,
        interaction_id="",
        evidence_hash="",
    )


class TestAttestationOutcomeSignFidelity:
    """Negative attestations must produce v_hat < 0 / p < 0.5."""

    def setup_method(self) -> None:
        self.mapper = AVPMapper()
        self.computer = ProxyComputer()

    def _map(self, sign: int):
        return self.mapper.attestation_received_to_interaction(
            _att(sign),
            initiator_id="a",
            counterparty_id="b",
            computer=self.computer,
        )

    def test_negative_attestation_records_as_negative(self):
        ix = self._map(-1)
        assert ix.v_hat < 0, f"expected v_hat<0, got {ix.v_hat}"
        assert ix.p < 0.5, f"expected p<0.5, got {ix.p}"
        assert ix.accepted is False

    def test_positive_attestation_records_as_positive(self):
        ix = self._map(+1)
        assert ix.v_hat > 0
        assert ix.p > 0.5
        assert ix.accepted is True

    def test_negative_attestation_penalty_fields_populated(self):
        """Mitigation: the penalty fields *must* be non-zero so the proxy
        weighting can't read them as 'good'. This is the load-bearing
        invariant for the bug."""
        ix = self._map(-1)
        assert ix.rework_count > 0
        assert ix.verifier_rejections > 0
        assert ix.tool_misuse_flags > 0
        assert ix.counterparty_engagement_delta < 0

    def test_positive_attestation_no_penalty_fields(self):
        """Conversely, positive attestations should not carry penalty signals."""
        ix = self._map(+1)
        assert ix.rework_count == 0
        assert ix.verifier_rejections == 0
        assert ix.tool_misuse_flags == 0

    @pytest.mark.parametrize("sign", [-1, +1])
    def test_p_in_valid_range(self, sign):
        ix = self._map(sign)
        assert 0.0 <= ix.p <= 1.0


# ---------------------------------------------------------------- D5

class TestD5PInvariant:
    """D5: all observable fields stay in [-1, +1]; p stays in [0, 1]."""

    def setup_method(self):
        self.mapper = AVPMapper()
        self.computer = ProxyComputer()

    @pytest.mark.parametrize("tier", ["newcomer", "basic", "trusted", "elite", "unknown_tier_xyz"])
    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("allowed", [True, False])
    def test_observables_in_valid_range(self, tier, confidence, allowed):
        decision = SimpleNamespace(allowed=allowed, tier=tier, confidence=confidence)
        obs = self.mapper.trust_decision_to_observables(decision, peer_id="p")
        assert -1.0 <= obs.task_progress_delta <= 1.0
        assert -1.0 <= obs.counterparty_engagement_delta <= 1.0
        assert obs.rework_count >= 0
        assert obs.verifier_rejections >= 0
        assert obs.tool_misuse_flags >= 0

    def test_observables_resist_out_of_range_confidence(self):
        # Even if a caller hands an out-of-range confidence (contract
        # violation), the mapper clamps engagement to [-1, +1].
        decision = SimpleNamespace(allowed=True, tier="basic", confidence=999.0)
        obs = self.mapper.trust_decision_to_observables(decision, peer_id="p")
        assert -1.0 <= obs.counterparty_engagement_delta <= 1.0

    @pytest.mark.parametrize("sign", [-1, 0, +1])
    def test_attestation_p_in_unit_interval(self, sign):
        att = _att(sign)
        ix = self.mapper.attestation_received_to_interaction(
            att, initiator_id="a", counterparty_id="b", computer=self.computer,
        )
        assert 0.0 <= ix.p <= 1.0
        assert -1.0 <= ix.v_hat <= 1.0


# ---------------------------------------------------------------- C4

class TestC4TierAmplificationCap:
    """C4: tier contribution is capped at max_tier_v_hat_contribution;
    no single tier can saturate v_hat above that ceiling."""

    def test_elite_tier_capped_at_default_06(self):
        # Elite ordinal is +1.0; with default cap 0.6, observable should be 0.6.
        m = AVPMapper()
        decision = SimpleNamespace(allowed=True, tier="elite", confidence=1.0)
        obs = m.trust_decision_to_observables(decision, peer_id="p")
        assert obs.task_progress_delta == pytest.approx(0.6, abs=1e-9)

    def test_newcomer_tier_capped_at_negative_cap(self):
        # Newcomer ordinal is -0.5; with cap 0.6, |signal| < cap, no clamp.
        m = AVPMapper()
        decision = SimpleNamespace(allowed=True, tier="newcomer", confidence=1.0)
        obs = m.trust_decision_to_observables(decision, peer_id="p")
        assert obs.task_progress_delta == pytest.approx(-0.5, abs=1e-9)

    def test_cap_override_reduces_signal(self):
        # Lower cap should pull elite signal down.
        cfg = AgentVeilConfig(max_tier_v_hat_contribution=0.3)
        m = AVPMapper(config=cfg)
        decision = SimpleNamespace(allowed=True, tier="elite", confidence=1.0)
        obs = m.trust_decision_to_observables(decision, peer_id="p")
        assert obs.task_progress_delta == pytest.approx(0.3, abs=1e-9)

    def test_tier_contribution_alone_bounded_by_cap(self):
        # The C4 invariant pins the tier signal itself, not the final
        # v_hat (other signals also contribute). With a zero-signal
        # decision (allowed=False to neutralise the verifier reward,
        # confidence=0 to neutralise engagement), v_hat's positive
        # component comes only from tier — and that component should
        # not exceed weight(task_progress) * cap.
        cfg = AgentVeilConfig(max_tier_v_hat_contribution=0.6)
        m = AVPMapper(config=cfg)
        decision = SimpleNamespace(allowed=False, tier="elite", confidence=0.0)
        obs = m.trust_decision_to_observables(decision, peer_id="p")
        # The cap is enforced at the observable boundary:
        assert obs.task_progress_delta == pytest.approx(0.6, abs=1e-9)
        # And the cap-relaxed limit on v_hat-from-tier holds:
        # |contribution| ≤ weights.task_progress * cap.
        pc = ProxyComputer()
        tier_only_contribution = pc.weights.task_progress * obs.task_progress_delta
        assert abs(tier_only_contribution) <= cfg.max_tier_v_hat_contribution + 1e-9

    def test_doubling_cap_doubles_tier_signal(self):
        # Linearity check: cap controls the signal linearly until hitting
        # the natural ordinal ceiling (1.0).
        low = AVPMapper(config=AgentVeilConfig(max_tier_v_hat_contribution=0.3))
        high = AVPMapper(config=AgentVeilConfig(max_tier_v_hat_contribution=0.6))
        decision = SimpleNamespace(allowed=True, tier="elite", confidence=0.0)
        s_low = low.trust_decision_to_observables(decision, peer_id="p").task_progress_delta
        s_high = high.trust_decision_to_observables(decision, peer_id="p").task_progress_delta
        assert s_high == pytest.approx(2 * s_low, abs=1e-9)


# ---------------------------------------------------------------- D1

class TestD1NoReputationDoubleCounting:
    """D1: reputation_change_to_update returns r_delta=0.0 in v1.
    Tier influences p through the mapper, not payoff-side reputation."""

    def test_r_delta_is_zero(self):
        m = AVPMapper()
        snap = SimpleNamespace(score=0.95, tier="elite", attestation_count=42)
        upd = m.reputation_change_to_update(snap, agent_id="a")
        assert upd["r_delta"] == 0.0

    def test_r_delta_zero_regardless_of_score(self):
        m = AVPMapper()
        for score in (-1.0, 0.0, 0.5, 0.99, 1.0):
            snap = SimpleNamespace(score=score, tier="elite", attestation_count=1)
            assert m.reputation_change_to_update(snap, agent_id="a")["r_delta"] == 0.0

    def test_update_carries_source_tag(self):
        m = AVPMapper()
        snap = SimpleNamespace(score=0.5, tier="basic", attestation_count=1)
        upd = m.reputation_change_to_update(snap, agent_id="alice")
        assert upd["agent_id"] == "alice"
        assert upd["source"] == "agentveil"


# ---------------------------------------------------------------- D2

class TestD2CategoricalToOrdinal:
    """D2: tier strings map to a fixed ordinal ladder; unknown tiers
    fall back to 0.0; attestation outcome_sign is continuous, not
    parsed from a categorical tier."""

    @pytest.mark.parametrize("tier,expected", [
        ("newcomer", -0.5),
        ("basic", 0.0),
        ("trusted", 0.5),
        ("elite", 0.6),  # capped from 1.0 by C4 default
    ])
    def test_ordinal_ladder_is_monotone(self, tier, expected):
        m = AVPMapper()
        decision = SimpleNamespace(allowed=True, tier=tier, confidence=1.0)
        obs = m.trust_decision_to_observables(decision, peer_id="p")
        assert obs.task_progress_delta == pytest.approx(expected, abs=1e-9)

    def test_unknown_tier_falls_back_to_zero(self, caplog):
        m = AVPMapper()
        decision = SimpleNamespace(allowed=True, tier="archmage", confidence=1.0)
        with caplog.at_level("WARNING"):
            obs = m.trust_decision_to_observables(decision, peer_id="p")
        assert obs.task_progress_delta == 0.0
        # And the operator gets warned so the misconfiguration is visible.
        assert any("Unknown tier" in r.message and "archmage" in r.message
                   for r in caplog.records)

    def test_attestation_outcome_sign_is_continuous_not_categorical(self):
        # D2's other half: attestations carry numeric outcome_sign,
        # never a tier string. The mapper's task_progress should scale
        # linearly with outcome_sign.
        m = AVPMapper()
        pc = ProxyComputer()
        att_pos = m.attestation_received_to_interaction(
            _att(+1), initiator_id="a", counterparty_id="b", computer=pc,
        )
        att_zero = m.attestation_received_to_interaction(
            _att(0), initiator_id="a", counterparty_id="b", computer=pc,
        )
        att_neg = m.attestation_received_to_interaction(
            _att(-1), initiator_id="a", counterparty_id="b", computer=pc,
        )
        # Strict monotonicity in outcome_sign across all three cases.
        assert att_neg.v_hat < att_zero.v_hat < att_pos.v_hat
        assert att_neg.p < att_zero.p < att_pos.p
