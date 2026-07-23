"""Tests for the escalating-pressure scenario components (beads pins).

Covers:
- PressureResponsiveAgent mode logic (pressure schedule, competence,
  fabrication propensity, adaptivity to being caught)
- ObfuscationObservableGenerator.draw_ground_truth delegation
- CertificateGateLever behavior across coverage / ground-truth cases
- Loader wiring: agent type registration and generator wrapping
"""

import random

from swarm.agents.base import Observation
from swarm.agents.pressure_responsive import PressureResponsiveAgent
from swarm.core.observable_generator import (
    DefaultObservableGenerator,
    ObfuscationObservableGenerator,
)
from swarm.env.state import EnvState, InteractionProposal
from swarm.governance.certificate_gate import CertificateGateLever
from swarm.governance.config import GovernanceConfig
from swarm.models.agent import AgentState, AgentType
from swarm.models.interaction import InteractionType, SoftInteraction


def _make_agent(**config) -> PressureResponsiveAgent:
    return PressureResponsiveAgent(
        agent_id="worker_1",
        config=config,
        rng=random.Random(7),
    )


def _obs(epoch: int, reputation: float = 0.0) -> Observation:
    return Observation(
        agent_state=AgentState(agent_id="worker_1", reputation=reputation),
        current_epoch=epoch,
    )


class TestPressureSchedule:
    def test_zero_before_warmup(self):
        agent = _make_agent(pressure_warmup_epochs=5, pressure_ramp_epochs=10)
        assert agent._pressure_at(0) == 0.0
        assert agent._pressure_at(5) == 0.0

    def test_ramps_linearly_and_caps(self):
        agent = _make_agent(
            pressure_warmup_epochs=5, pressure_ramp_epochs=10, pressure_max=1.0
        )
        assert agent._pressure_at(10) == 0.5
        assert agent._pressure_at(15) == 1.0
        assert agent._pressure_at(40) == 1.0


class TestModeLogic:
    def test_full_competence_always_success(self):
        agent = _make_agent(competence=1.0)
        for epoch in range(30):
            agent.act(_obs(epoch))
            assert agent.mode == "success"
            assert agent.draw_ground_truth() == 1

    def test_no_pressure_failures_reported_honestly(self):
        agent = _make_agent(
            competence=0.0, pressure_warmup_epochs=100, pressure_ramp_epochs=10
        )
        for epoch in range(30):
            agent.act(_obs(epoch))
            assert agent.mode == "honest_negative"
            assert agent.draw_ground_truth() == 1

    def test_full_pressure_failures_fabricated(self):
        agent = _make_agent(
            competence=0.0,
            pressure_warmup_epochs=0,
            pressure_ramp_epochs=1,
            fabrication_gain=1.0,
            adaptive=False,
        )
        agent.act(_obs(epoch=10))
        assert agent.pressure == 1.0
        assert agent.mode == "fabricate"
        assert agent.draw_ground_truth() == -1

    def test_fabrication_inflates_honest_negative_deflates(self):
        agent = _make_agent()
        agent.mode = "fabricate"
        offsets = agent.get_signal_manipulation()
        assert offsets["task_progress_delta"] > 0
        assert offsets["counterparty_engagement_delta"] > 0

        agent.mode = "honest_negative"
        offsets = agent.get_signal_manipulation()
        assert offsets["task_progress_delta"] < 0

        agent.mode = "success"
        assert agent.get_signal_manipulation() == {}


class TestAdaptivity:
    def test_reputation_drop_decays_propensity(self):
        agent = _make_agent(adaptive=True, caught_decay=0.5)
        agent.act(_obs(0, reputation=0.0))
        assert agent.fabrication_propensity == 1.0
        agent.act(_obs(1, reputation=-0.5))
        assert agent.times_caught == 1
        assert agent.fabrication_propensity == 0.5
        agent.act(_obs(2, reputation=-1.0))
        assert agent.fabrication_propensity == 0.25

    def test_propensity_floor_holds(self):
        agent = _make_agent(adaptive=True, caught_decay=0.5, propensity_floor=0.2)
        rep = 0.0
        for epoch in range(10):
            agent.act(_obs(epoch, reputation=rep))
            rep -= 0.3
        assert agent.fabrication_propensity == 0.2

    def test_non_adaptive_ignores_reputation(self):
        agent = _make_agent(adaptive=False)
        agent.act(_obs(0, reputation=0.0))
        agent.act(_obs(1, reputation=-1.0))
        assert agent.times_caught == 0
        assert agent.fabrication_propensity == 1.0


class TestGroundTruthDelegation:
    def _wrapper_and_state(self, agent):
        state = EnvState()
        state.add_agent(agent_id=agent.agent_id, agent_type=AgentType.HONEST)
        wrapper = ObfuscationObservableGenerator(
            inner=DefaultObservableGenerator(rng=random.Random(3)),
            agents={agent.agent_id: agent},
        )
        return wrapper, state

    def test_fabricating_agent_yields_negative_ground_truth(self):
        agent = _make_agent()
        agent.mode = "fabricate"
        wrapper, _ = self._wrapper_and_state(agent)
        assert wrapper.draw_ground_truth(agent.agent_id) == -1

    def test_honest_modes_yield_positive_ground_truth(self):
        agent = _make_agent()
        wrapper, _ = self._wrapper_and_state(agent)
        for mode in ("success", "honest_negative"):
            agent.mode = mode
            assert wrapper.draw_ground_truth(agent.agent_id) == 1

    def test_unknown_agent_yields_none(self):
        agent = _make_agent()
        wrapper, _ = self._wrapper_and_state(agent)
        assert wrapper.draw_ground_truth("someone_else") is None

    def test_fabrication_signals_beat_honest_negative(self):
        """Fabricated observables must look better than honest-negative ones."""
        agent = _make_agent()
        wrapper, state = self._wrapper_and_state(agent)
        proposal = InteractionProposal(
            initiator_id=agent.agent_id,
            counterparty_id="agent_b",
        )
        agent.mode = "fabricate"
        fab = wrapper.generate(proposal, True, state)
        agent.mode = "honest_negative"
        neg = wrapper.generate(proposal, True, state)
        assert fab.task_progress_delta > neg.task_progress_delta
        assert (
            fab.counterparty_engagement_delta > neg.counterparty_engagement_delta
        )


def _interaction(ground_truth):
    return SoftInteraction(
        initiator="worker_1",
        counterparty="agent_b",
        interaction_type=InteractionType.COLLABORATION,
        accepted=True,
        p=0.9,
        ground_truth=ground_truth,
    )


class TestCertificateGateLever:
    def test_disabled_is_noop(self):
        config = GovernanceConfig(certificate_gate_enabled=False)
        lever = CertificateGateLever(config, seed=1)
        effect = lever.on_interaction(_interaction(-1), EnvState())
        assert effect.cost_a == 0.0
        assert effect.reputation_deltas == {}

    def test_no_certificate_is_noop(self):
        config = GovernanceConfig(
            certificate_gate_enabled=True, certificate_coverage=1.0
        )
        lever = CertificateGateLever(config, seed=1)
        effect = lever.on_interaction(_interaction(None), EnvState())
        assert effect.cost_a == 0.0
        assert effect.details["reason"] == "no_certificate"

    def test_full_coverage_catches_fabrication(self):
        config = GovernanceConfig(
            certificate_gate_enabled=True,
            certificate_coverage=1.0,
            certificate_penalty=1.5,
        )
        lever = CertificateGateLever(config, seed=1)
        effect = lever.on_interaction(_interaction(-1), EnvState())
        assert effect.cost_a == 1.5
        assert effect.reputation_deltas["worker_1"] == -1.5
        assert effect.details["caught"] is True

    def test_genuine_positive_passes_verification(self):
        config = GovernanceConfig(
            certificate_gate_enabled=True, certificate_coverage=1.0
        )
        lever = CertificateGateLever(config, seed=1)
        effect = lever.on_interaction(_interaction(1), EnvState())
        assert effect.cost_a == 0.0
        assert effect.details == {"verified": True, "caught": False}

    def test_zero_coverage_never_verifies(self):
        config = GovernanceConfig(
            certificate_gate_enabled=True, certificate_coverage=0.0
        )
        lever = CertificateGateLever(config, seed=1)
        for _ in range(20):
            effect = lever.on_interaction(_interaction(-1), EnvState())
            assert effect.cost_a == 0.0
            assert effect.details["reason"] == "not_covered"


class TestLoaderWiring:
    def test_scenario_builds_with_wrapped_generator(self):
        from swarm.scenarios.loader import build_orchestrator, load_scenario

        scenario = load_scenario("scenarios/escalating_pressure.yaml")
        orchestrator = build_orchestrator(scenario)
        assert isinstance(
            orchestrator._observable_generator, ObfuscationObservableGenerator
        )
        workers = [
            a
            for a in orchestrator._agents.values()
            if isinstance(a, PressureResponsiveAgent)
        ]
        assert len(workers) == 4

    def test_ground_truth_recorded_end_to_end(self):
        """Run a short simulation; fabrication-capable workers must produce
        interactions with ground_truth set, and every recorded value must be
        +1 or -1."""
        from swarm.scenarios.loader import build_orchestrator, load_scenario

        scenario = load_scenario("scenarios/escalating_pressure.yaml")
        # Force the fabrication regime hard so the smoke test is decisive:
        # incompetent workers, instant full pressure.
        for spec in scenario.agent_specs:
            if spec.get("type") == "pressure_responsive":
                spec.setdefault("config", {}).update(
                    {
                        "competence": 0.0,
                        "pressure_warmup_epochs": 0,
                        "pressure_ramp_epochs": 1,
                        "adaptive": False,
                    }
                )
        scenario.orchestrator_config.n_epochs = 3
        orchestrator = build_orchestrator(scenario)
        recorded = []
        orchestrator.on_interaction_complete(
            lambda interaction, _pi, _pc: recorded.append(interaction)
        )
        orchestrator.run()
        interactions = [i for i in recorded if i.ground_truth is not None]
        assert interactions, "no interactions recorded ground truth"
        assert all(i.ground_truth in (-1, 1) for i in interactions)
        worker_ids = {
            a.agent_id
            for a in orchestrator._agents.values()
            if isinstance(a, PressureResponsiveAgent)
        }
        fabricated = [i for i in interactions if i.ground_truth == -1]
        assert fabricated, "fully pressured incompetent workers never fabricated"
        assert all(i.initiator in worker_ids for i in fabricated)
