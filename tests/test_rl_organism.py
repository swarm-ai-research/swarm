"""Tests for the RL organism agent and effort-driven observables (bead boll)."""

import random

import pytest

from swarm.agents.base import ActionType, Observation
from swarm.agents.base import InteractionProposal as AgentProposal
from swarm.agents.rl_organism import (
    SIGNAL_PREFIX,
    RLOrganismAgent,
    RLOrganismConfig,
)
from swarm.core.observable_generator import (
    DefaultObservableGenerator,
    EffortObservableGenerator,
)
from swarm.core.proxy import ProxyComputer
from swarm.env.state import EnvState, InteractionProposal
from swarm.models.agent import AgentState, AgentType
from swarm.models.interaction import InteractionType, SoftInteraction


def make_agent(capacity: int = 2, seed: int = 7, **cfg) -> RLOrganismAgent:
    config = {"capacity": capacity, **cfg}
    return RLOrganismAgent(
        agent_id="rl_organism_1", config=config, rng=random.Random(seed)
    )


def make_observation(agent: RLOrganismAgent, partners=("a", "b"), epoch=0) -> Observation:
    return Observation(
        agent_state=AgentState(agent_id=agent.agent_id, agent_type=agent.agent_type),
        current_epoch=epoch,
        visible_agents=[
            {"agent_id": p, "agent_type": "honest", "reputation": 0.0, "resources": 100}
            for p in partners
        ],
    )


class TestEffortObservableGenerator:
    def test_p_monotone_in_effort_and_bounded(self):
        gen = EffortObservableGenerator(rng=random.Random(0))
        proxy = ProxyComputer()
        state = EnvState()
        means = []
        for effort in (0.1, 0.5, 0.9):
            ps = []
            for _ in range(500):
                prop = InteractionProposal(
                    initiator_id="x", counterparty_id="y", metadata={"effort": effort}
                )
                _, p = proxy.compute_labels(gen.generate(prop, True, state))
                assert 0.0 <= p <= 1.0
                ps.append(p)
            means.append(sum(ps) / len(ps))
        assert means[0] < means[1] < means[2]
        # Low effort must be near/below break-even, high effort clearly above.
        assert means[0] < 0.45
        assert means[2] > 0.70

    def test_fallback_without_effort_key(self):
        state = EnvState()
        state.add_agent("x", name="x", agent_type=AgentType.HONEST)
        inner = DefaultObservableGenerator(rng=random.Random(1))
        gen = EffortObservableGenerator(inner=inner, rng=random.Random(1))
        prop = InteractionProposal(initiator_id="x", counterparty_id="y")
        obs = gen.generate(prop, True, state)
        # Honest-type default signals: no misuse flags ever.
        assert obs.tool_misuse_flags == 0

    def test_effort_clamped(self):
        gen = EffortObservableGenerator(rng=random.Random(2))
        state = EnvState()
        prop = InteractionProposal(
            initiator_id="x", counterparty_id="y", metadata={"effort": 7.0}
        )
        obs = gen.generate(prop, True, state)
        assert -1.0 <= obs.task_progress_delta <= 1.0


class TestRLOrganismConfig:
    def test_capacity_validation(self):
        with pytest.raises(ValueError):
            RLOrganismConfig.from_dict({"capacity": 5})

    def test_defaults(self):
        cfg = RLOrganismConfig.from_dict({})
        assert cfg.capacity == 2
        assert len(cfg.efforts) == 2
        assert len(cfg.transfers) == 3


class TestStateFeatures:
    def test_capacity_gates_state_size(self):
        obs_sizes = {}
        for cap in range(4):
            agent = make_agent(capacity=cap)
            obs = make_observation(agent)
            state = agent._partner_state("a", obs)
            obs_sizes[cap] = len(state)
        assert obs_sizes == {0: 0, 1: 1, 2: 2, 3: 3}

    def test_state_ignores_agent_type(self):
        """The emergence claim dies if the learner can see type labels."""
        agent = make_agent(capacity=3)
        obs1 = make_observation(agent)
        obs2 = make_observation(agent)
        for a in obs2.visible_agents:
            a["agent_type"] = "rl_organism"  # relabel every partner
        assert agent._partner_state("a", obs1) == agent._partner_state("a", obs2)

    def test_signal_extraction_takes_latest(self):
        agent = make_agent(capacity=3)
        obs = make_observation(agent)
        obs.visible_posts = [
            {"author_id": "a", "content": f"{SIGNAL_PREFIX}0", "created_at": "2026-01-01T00:00:00"},
            {"author_id": "a", "content": f"{SIGNAL_PREFIX}1", "created_at": "2026-01-02T00:00:00"},
            {"author_id": "b", "content": "hello", "created_at": "2026-01-03T00:00:00"},
        ]
        signals = agent._latest_signals(obs)
        assert signals == {"a": 1}


class TestLearning:
    def _propose_and_learn(self, agent, obs, accepted, payoff_fn, n=200):
        """Drive propose->outcome loops; returns arms chosen in final 50."""
        chosen = []
        for i in range(n):
            action = None
            while action is None or action.action_type != ActionType.PROPOSE_INTERACTION:
                action = agent.act(obs)
            nonce = action.metadata["rl_nonce"]
            pending = agent._pending_proposes[nonce]
            effort = agent.rl_config.efforts[pending.arm[0]]
            tau = agent.rl_config.transfers[pending.arm[1]]
            if i >= n - 50:
                chosen.append(pending.arm)
            interaction = SoftInteraction(
                interaction_id=f"i{i}",
                initiator=agent.agent_id,
                counterparty=action.counterparty_id,
                interaction_type=InteractionType.TRADE,
                accepted=accepted,
                p=0.77 if effort > 0.5 else 0.35,
                tau=tau,
                metadata={"rl_nonce": nonce, "effort": effort},
            )
            agent.update_from_outcome(interaction, payoff_fn(effort, tau))
        return chosen

    def test_learns_high_effort_when_extraction_unprofitable(self):
        agent = make_agent(capacity=0, signal_enabled=False, epsilon_decay=0.99)
        obs = make_observation(agent, partners=("a",))

        # Reward mirrors the real engine with theta=0.5 but transfers refused
        # (tau arms pay nothing): high effort should dominate.
        def payoff(effort, tau):
            p = 0.77 if effort > 0.5 else 0.35
            s_soft = p * 2.0 - (1 - p) * 1.0
            return 0.5 * s_soft

        chosen = self._propose_and_learn(agent, obs, accepted=True, payoff_fn=payoff)
        high_effort = sum(1 for arm in chosen if arm[0] == 1)
        assert high_effort > 40  # >80% of late choices

    def test_learns_extraction_when_it_pays(self):
        agent = make_agent(capacity=0, signal_enabled=False, epsilon_decay=0.99)
        obs = make_observation(agent, partners=("a",))

        def payoff(effort, tau):
            p = 0.77 if effort > 0.5 else 0.35
            s_soft = p * 2.0 - (1 - p) * 1.0
            return 0.5 * s_soft - tau  # negative tau = extraction income

        chosen = self._propose_and_learn(agent, obs, accepted=True, payoff_fn=payoff)
        # With acceptance held fixed, the optimum is the most negative
        # transfer (effort tradeoffs only exist in the full environment).
        extractive = sum(1 for arm in chosen if arm[1] == 0)
        assert extractive > 40

    def test_accept_learning_rejects_toxic_offers(self):
        agent = make_agent(capacity=1, signal_enabled=False, epsilon_decay=0.99)
        obs = make_observation(agent, partners=("pred",))
        decisions = []
        for i in range(300):
            proposal = AgentProposal(
                proposal_id=f"p{i}",
                initiator_id="pred",
                counterparty_id=agent.agent_id,
                interaction_type=InteractionType.TRADE,
                offered_transfer=-0.4,
            )
            accepted = agent.accept_interaction(proposal, obs)
            if i >= 250:
                decisions.append(accepted)
            interaction = SoftInteraction(
                interaction_id=f"p{i}",
                initiator="pred",
                counterparty=agent.agent_id,
                interaction_type=InteractionType.TRADE,
                accepted=accepted,
                p=0.35,
                tau=-0.4,
            )
            agent.update_from_outcome(interaction, -0.37 if accepted else 0.0)
        assert sum(decisions) < len(decisions) * 0.3

    def test_pending_maps_bounded(self):
        agent = make_agent(capacity=0, signal_enabled=False)
        obs = make_observation(agent, partners=("a",))
        for _ in range(1200):
            action = agent.act(obs)
            del action
        assert len(agent._pending_proposes) <= 500


class TestDeterminism:
    def test_same_seed_same_trajectory(self):
        actions1, actions2 = [], []
        for actions in (actions1, actions2):
            agent = make_agent(capacity=2, seed=123)
            obs = make_observation(agent, epoch=0)
            for i in range(30):
                obs.current_epoch = i // 10
                a = agent.act(obs)
                actions.append((a.action_type, a.counterparty_id, a.content))
        assert actions1 == actions2


class TestLoaderIntegration:
    def test_registered_in_agent_types(self):
        from swarm.scenarios.loader import AGENT_TYPES

        assert AGENT_TYPES["rl_organism"] is RLOrganismAgent

    def test_scenario_smoke_run(self):
        """Two epochs end-to-end through the real loader + orchestrator."""
        from swarm.scenarios.loader import build_orchestrator, load_scenario

        scenario = load_scenario("scenarios/rl_emergence.yaml")
        scenario.orchestrator_config.n_epochs = 2
        scenario.orchestrator_config.steps_per_epoch = 5
        orch = build_orchestrator(scenario)
        from swarm.core.observable_generator import EffortObservableGenerator

        assert isinstance(orch._observable_generator, EffortObservableGenerator)
        # completed_interactions is cleared at epoch boundaries — accumulate
        # via the completion callback, as the sweep script does.
        interactions = []
        orch.on_interaction_complete(
            lambda interaction, pi, pc: interactions.append(interaction)
        )
        orch.run()
        assert interactions, "no interactions completed in smoke run"
        for i in interactions:
            assert 0.0 <= i.p <= 1.0
        rl_initiated = [
            i for i in interactions if i.metadata.get("effort") is not None
        ]
        assert rl_initiated, "rl_organism never proposed in smoke run"
