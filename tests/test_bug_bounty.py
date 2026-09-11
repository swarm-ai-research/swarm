"""Tests for the RLVR bug-bounty mechanism (LessWrong T2bzBkJuBeNNgzhbh).

Covers the three pieces that have to agree for the exploit-or-report choice
to mean anything: the shared registry, the hunter's decision rule, and the
governance lever that verifies, pays, and patches.
"""

import random

import pytest

from swarm.agents.bug_hunter import BugHunterAgent
from swarm.env.state import EnvState
from swarm.governance.bug_bounty import BugBountyLever
from swarm.governance.config import GovernanceConfig
from swarm.models.interaction import SoftInteraction
from swarm.redteam.bug_channels import (
    BountyTerms,
    BugChannel,
    BugRegistry,
    default_channels,
)


def _registry() -> BugRegistry:
    return BugRegistry.with_default_channels()


def _report(channel_id: str, initiator: str = "hunter_1") -> SoftInteraction:
    return SoftInteraction(
        initiator=initiator,
        counterparty="honest_1",
        accepted=True,
        p=0.6,
        metadata={"bug_report": {"channel_id": channel_id}},
    )


class TestBugChannel:
    def test_rejects_unknown_offset_keys(self):
        with pytest.raises(ValueError, match="unknown offset keys"):
            BugChannel(channel_id="bad", offsets={"not_an_observable": 1.0})

    def test_rejects_out_of_range_severity(self):
        with pytest.raises(ValueError, match="severity"):
            BugChannel(channel_id="bad", offsets={}, severity=1.5)

    def test_default_channels_are_distinct_and_valid(self):
        channels = default_channels()
        assert len({c.channel_id for c in channels}) == len(channels)
        assert all(c.offsets for c in channels)


class TestBugRegistry:
    def test_channels_start_open(self):
        registry = _registry()
        assert len(registry.open_channels()) == len(registry.channels)
        assert registry.is_open("progress_inflation")

    def test_patch_closes_channel_at_landing_epoch(self):
        registry = _registry()
        registry.schedule_patch("progress_inflation", land_epoch=3, current_epoch=1)

        assert registry.apply_patches(2) == []
        assert registry.is_open("progress_inflation")

        assert registry.apply_patches(3) == ["progress_inflation"]
        assert not registry.is_open("progress_inflation")

    def test_stall_is_counted_in_channel_epochs(self):
        registry = _registry()
        registry.schedule_patch("verifier_mute", land_epoch=5, current_epoch=2)
        assert registry.stall_epochs == 3

    def test_patched_channel_cannot_be_rescheduled(self):
        registry = _registry()
        registry.schedule_patch("verifier_mute", land_epoch=1, current_epoch=0)
        registry.apply_patches(1)
        registry.schedule_patch("verifier_mute", land_epoch=9, current_epoch=5)
        assert "verifier_mute" not in registry.pending_patches

    def test_exploit_on_patched_channel_does_not_land(self):
        registry = _registry()
        registry.schedule_patch("engagement_farm", land_epoch=1, current_epoch=0)
        registry.apply_patches(1)

        assert registry.record_exploit("engagement_farm") is False
        assert registry.exploit_attempts == 1
        assert registry.exploits_landed == 0

    def test_report_share_counts_reports_against_landed_exploits(self):
        registry = _registry()
        registry.record_exploit("progress_inflation")
        registry.reports_valid = 3
        assert registry.summary()["report_share"] == pytest.approx(0.75)


class TestHunterEconomics:
    def _hunter(self, **config) -> BugHunterAgent:
        hunter = BugHunterAgent("hunter_1", config=config, rng=random.Random(0))
        hunter.set_registry(_registry())
        return hunter

    def test_no_bounty_means_reporting_is_worthless(self):
        hunter = self._hunter()
        assert hunter.report_value() == 0.0
        assert hunter.exploit_value() > 0.0

    def test_bounty_above_exploit_value_flips_the_choice(self):
        hunter = self._hunter(exploit_gain=0.35, expected_uses=6.0)
        exploit_value = hunter.exploit_value()
        hunter.registry.terms = BountyTerms(
            enabled=True, amount=exploit_value + 1.0, reputation=0.0
        )
        assert hunter.report_value() > hunter.exploit_value()

    def test_learned_detection_rate_erodes_exploit_value(self):
        hunter = self._hunter(audit_penalty=2.0)
        before = hunter.exploit_value()
        for _ in range(20):
            hunter.update_from_outcome(
                SoftInteraction(
                    initiator="hunter_1",
                    counterparty="honest_1",
                    c_a=1.5,
                    metadata={"exploit_channel": "progress_inflation"},
                ),
                payoff=-1.0,
            )
        assert hunter.exploit_value() < before
        assert hunter.exploit_value() < 0.0

    def test_costless_exploits_teach_the_hunter_nothing_to_fear(self):
        hunter = self._hunter()
        for _ in range(20):
            hunter.update_from_outcome(
                SoftInteraction(
                    initiator="hunter_1",
                    counterparty="honest_1",
                    c_a=0.0,
                    metadata={"exploit_channel": "progress_inflation"},
                ),
                payoff=1.0,
            )
        assert hunter.get_strategy_report()["p_detect"] == pytest.approx(0.0)

    def test_exploit_value_falls_as_channels_get_patched(self):
        hunter = self._hunter()
        before = hunter.exploit_value()
        registry = hunter.registry
        for channel_id in list(registry.channels)[:2]:
            registry.schedule_patch(channel_id, land_epoch=1, current_epoch=0)
        registry.apply_patches(1)
        assert hunter.exploit_value() < before


class TestHunterSignals:
    def _hunter(self, **config) -> BugHunterAgent:
        hunter = BugHunterAgent("hunter_1", config=config, rng=random.Random(1))
        hunter.set_registry(_registry())
        return hunter

    def test_offsets_apply_only_while_exploiting(self):
        hunter = self._hunter()
        assert hunter.get_signal_manipulation() == {}

        hunter._exploit_action("honest_1", "progress_inflation")
        offsets = hunter.get_signal_manipulation()
        assert offsets == dict(
            hunter.registry.get("progress_inflation").offsets
        )

    def test_patched_channel_yields_no_offsets_but_counts_the_attempt(self):
        hunter = self._hunter()
        registry = hunter.registry
        registry.schedule_patch("progress_inflation", land_epoch=1, current_epoch=0)
        registry.apply_patches(1)

        hunter._exploit_action("honest_1", "progress_inflation")
        assert hunter.get_signal_manipulation() == {}
        assert registry.exploit_attempts == 1
        assert registry.exploits_landed == 0

    def test_exploit_is_harmful_in_ground_truth(self):
        hunter = self._hunter()
        hunter._exploit_action("honest_1", "verifier_mute")
        assert hunter.draw_ground_truth() == -1

    def test_ground_truth_draw_is_cleared_after_one_interaction(self):
        hunter = self._hunter()
        hunter._exploit_action("honest_1", "verifier_mute")
        assert hunter.draw_ground_truth() == -1
        assert hunter.draw_ground_truth() in (1, -1)
        assert hunter._active_exploit is None


class TestBugBountyLever:
    def _lever(self, **overrides) -> BugBountyLever:
        # A perfect verifier by default, so each test isolates one failure
        # mode by relaxing exactly the knob it is about.
        settings = {
            "bug_bounty_enabled": True,
            "bug_bounty_amount": 3.0,
            "bug_bounty_reputation": 0.2,
            "bug_bounty_patch_delay_epochs": 1,
            "bug_bounty_verifier_sensitivity": 1.0,
            "bug_bounty_verifier_specificity": 1.0,
            "bug_bounty_false_report_penalty": 2.0,
        }
        settings.update(overrides)
        config = GovernanceConfig(**settings)
        lever = BugBountyLever(config, seed=7)
        lever.attach_registry(_registry())
        return lever

    def test_attach_publishes_terms_to_agents(self):
        lever = self._lever()
        terms = lever.registry.terms
        assert terms.enabled and terms.amount == 3.0
        assert terms.false_report_penalty == 2.0

    def test_disabled_lever_is_inert(self):
        lever = BugBountyLever(GovernanceConfig(), seed=1)
        lever.attach_registry(_registry())
        effect = lever.on_interaction(_report("progress_inflation"), EnvState())
        assert effect.resource_deltas == {}
        assert lever.registry.reports_submitted == 0

    def test_verified_report_pays_and_schedules_a_patch(self):
        lever = self._lever()
        state = EnvState(current_epoch=4)

        effect = lever.on_interaction(_report("progress_inflation"), state)

        assert effect.resource_deltas["hunter_1"] == 3.0
        assert effect.reputation_deltas["hunter_1"] == pytest.approx(0.2)
        assert lever.registry.pending_patches["progress_inflation"] == 5
        assert lever.registry.bounty_paid == 3.0
        assert lever.registry.bounty_farmed == 0.0

    def test_patch_lands_on_epoch_start(self):
        lever = self._lever()
        lever.on_interaction(_report("progress_inflation"), EnvState(current_epoch=4))

        lever.on_epoch_start(EnvState(current_epoch=5), 5)
        assert not lever.registry.is_open("progress_inflation")

    def test_channel_stays_exploitable_until_the_patch_lands(self):
        lever = self._lever(bug_bounty_patch_delay_epochs=3)
        lever.on_interaction(_report("verifier_mute"), EnvState(current_epoch=1))

        lever.on_epoch_start(EnvState(current_epoch=2), 2)
        assert lever.registry.is_open("verifier_mute")
        assert lever.registry.record_exploit("verifier_mute") is True

    def test_duplicate_report_verifies_but_pays_nothing(self):
        lever = self._lever()
        state = EnvState(current_epoch=0)
        lever.on_interaction(_report("engagement_farm", "hunter_1"), state)

        effect = lever.on_interaction(_report("engagement_farm", "hunter_2"), state)

        assert effect.resource_deltas == {}
        assert effect.details["outcome"] == "duplicate"
        assert lever.registry.bounty_paid == 3.0

    def test_caught_fabrication_is_penalised(self):
        lever = self._lever()
        effect = lever.on_interaction(_report("phantom_1"), EnvState())

        assert effect.cost_a == 2.0
        assert effect.reputation_deltas["hunter_1"] == -2.0
        assert lever.registry.reports_fabricated == 1
        assert lever.registry.bounty_paid == 0.0

    def test_missed_fabrication_pays_a_bounty_for_nothing(self):
        lever = self._lever(bug_bounty_verifier_specificity=0.0)
        effect = lever.on_interaction(_report("phantom_1"), EnvState())

        assert effect.resource_deltas["hunter_1"] == 3.0
        assert lever.registry.bounty_farmed == 3.0
        assert lever.registry.summary()["channels_patched"] == 0.0

    def test_missed_genuine_report_leaves_the_channel_open(self):
        lever = self._lever(bug_bounty_verifier_sensitivity=0.0)
        effect = lever.on_interaction(_report("progress_inflation"), EnvState())

        assert effect.details["outcome"] == "missed_report"
        assert effect.resource_deltas == {}
        assert lever.registry.is_open("progress_inflation")

    def test_ordinary_interactions_are_ignored(self):
        lever = self._lever()
        plain = SoftInteraction(initiator="honest_1", counterparty="honest_2", p=0.7)
        effect = lever.on_interaction(plain, EnvState())
        assert effect.resource_deltas == {}
        assert lever.registry.reports_submitted == 0

    def test_malformed_report_payload_is_ignored(self):
        lever = self._lever()
        bad = SoftInteraction(
            initiator="hunter_1",
            counterparty="honest_1",
            metadata={"bug_report": "progress_inflation"},
        )
        effect = lever.on_interaction(bad, EnvState())
        assert effect.resource_deltas == {}
        assert lever.registry.reports_submitted == 0


class TestScenarioWiring:
    """End-to-end: the arms in scenarios/rlvr_bounty/ must actually wire up."""

    def _run(self, arm: str, epochs: int = 6):
        from pathlib import Path

        from swarm.scenarios.loader import build_orchestrator, load_scenario

        scenario = load_scenario(Path(f"scenarios/rlvr_bounty/{arm}.yaml"))
        scenario.orchestrator_config.n_epochs = epochs
        scenario.orchestrator_config.seed = 42
        orchestrator = build_orchestrator(scenario)
        orchestrator.run()
        hunters = [
            a
            for a in orchestrator.get_all_agents()
            if isinstance(a, BugHunterAgent)
        ]
        return orchestrator, hunters

    def test_hunters_share_one_registry(self):
        _, hunters = self._run("bounty", epochs=1)
        assert len(hunters) == 4
        assert len({id(h.registry) for h in hunters}) == 1

    def test_lever_sees_the_same_registry_as_the_hunters(self):
        orchestrator, hunters = self._run("bounty", epochs=1)
        levers = [
            lever
            for lever in orchestrator.governance_engine._levers
            if isinstance(lever, BugBountyLever)
        ]
        assert len(levers) == 1
        assert levers[0].registry is hunters[0].registry

    def test_without_a_bounty_hunters_exploit_and_nothing_is_patched(self):
        _, hunters = self._run("laissez_faire")
        registry = hunters[0].registry
        assert registry.exploits_landed > 0
        assert registry.patched == {}

    def test_bounty_arm_patches_channels_instead_of_exploiting_them(self):
        _, hunters = self._run("bounty")
        registry = hunters[0].registry
        assert registry.reports_valid > 0
        assert registry.patched
        assert registry.exploits_landed == 0

    def test_proxy_audits_never_charge_a_successful_exploit(self):
        """The hack raises p, and a p-conditioned audit looks for low p."""
        _, hunters = self._run("punish_proxy")
        assert all(h.get_strategy_report()["p_detect"] == 0.0 for h in hunters)
        assert hunters[0].registry.exploits_landed > 0

    def test_certificate_punishment_deters_without_patching(self):
        """Deterrence and repair are different achievements."""
        _, hunters = self._run("punish_certificate", epochs=10)
        registry = hunters[0].registry
        assert any(h.get_strategy_report()["p_detect"] > 0 for h in hunters)
        assert sum(h.finds_withheld for h in hunters) > 0
        assert registry.patched == {}

    def test_exploits_are_recorded_as_harmful_ground_truth(self):
        """The reward-hacking signature: p reads high, ground truth is -1."""
        collected = []

        from pathlib import Path

        from swarm.scenarios.loader import build_orchestrator, load_scenario

        scenario = load_scenario(Path("scenarios/rlvr_bounty/laissez_faire.yaml"))
        scenario.orchestrator_config.n_epochs = 4
        scenario.orchestrator_config.seed = 42
        orchestrator = build_orchestrator(scenario)
        orchestrator.on_interaction_complete(
            lambda interaction, a, b: collected.append(interaction)
        )
        orchestrator.run()

        exploits = [i for i in collected if "exploit_channel" in i.metadata]
        assert exploits
        assert all(i.ground_truth == -1 for i in exploits)
        assert all(0.0 <= i.p <= 1.0 for i in collected)
        # The whole problem in one assertion: the proxy rates the harmful
        # interactions above the honest ones.
        honest = [i for i in collected if "exploit_channel" not in i.metadata]
        assert sum(i.p for i in exploits) / len(exploits) > sum(
            i.p for i in honest
        ) / len(honest)
