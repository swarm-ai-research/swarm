"""Tests for the CrewAI-backed SWARM agent adapter.

These tests verify the adapter layer without requiring the real ``crewai``
package.  The crew is mocked so that tests stay fast and deterministic.

Security-related tests are grouped under ``TestSecurity*`` classes.
"""

import json
import random
from unittest.mock import MagicMock

import pytest

from swarm.agents.base import Action, ActionType, Observation
from swarm.agents.crewai_adapter import (
    _ACTION_KIND_MAP,
    _BUILTIN_PROFILE_NAMES,
    _VALID_KINDS,
    DEFAULT_CREW_TIMEOUT_SECONDS,
    MAX_CONTENT_LENGTH,
    MAX_DELIBERATION_MEMORY,
    MAX_ID_LENGTH,
    MAX_METADATA_KEYS,
    MAX_METADATA_VALUE_LENGTH,
    MAX_RATIONALE_LENGTH,
    MAX_RAW_TRACE_LENGTH,
    MAX_STAGED_ACTIONS,
    CrewAgentRole,
    CrewAIToolAdapter,
    CrewBackedAgent,
    CrewConfig,
    SwarmActionSchema,
    _sanitize_crew_metadata,
    get_profile_agents,
    register_crew_profile,
)
from swarm.models.agent import AgentState, AgentType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_observation() -> Observation:
    """Minimal valid observation with IDs for validation tests."""
    return Observation(
        agent_state=AgentState(
            agent_id="test_agent",
            reputation=1.0,
            resources=100.0,
        ),
        current_epoch=0,
        current_step=0,
        can_post=True,
        can_interact=True,
        can_vote=True,
        can_claim_task=True,
        visible_posts=[{"post_id": "p1", "content": "hello"}],
        visible_agents=[{"agent_id": "other_1", "reputation": 0.5}],
        available_tasks=[{"task_id": "t1", "difficulty": "easy"}],
        pending_proposals=[{"proposal_id": "prop_1", "initiator_id": "other_1"}],
        ecosystem_metrics={"toxicity": 0.1},
        available_bounties=[{"bounty_id": "b1", "reward_amount": 10.0}],
    )


@pytest.fixture()
def crew_config() -> CrewConfig:
    return CrewConfig(
        crew_profile="general_v1",
        role_name="Test Agent",
        model="gpt-4o-mini",
        enable_trace=True,
    )


@pytest.fixture()
def agent(crew_config: CrewConfig) -> CrewBackedAgent:
    return CrewBackedAgent(
        agent_id="crew_test_1",
        crew_config=crew_config,
        rng=random.Random(42),
    )


# ---------------------------------------------------------------------------
# SwarmActionSchema tests
# ---------------------------------------------------------------------------


class TestSwarmActionSchema:
    def test_valid_schema(self):
        schema = SwarmActionSchema(
            kind="post",
            content="Hello world",
            confidence=0.9,
            rationale="Testing",
        )
        assert schema.kind == "post"
        assert schema.confidence == 0.9

    def test_defaults(self):
        schema = SwarmActionSchema(kind="noop")
        assert schema.content == ""
        assert schema.target_id == ""
        assert schema.confidence == 0.7
        assert schema.metadata == {}

    def test_json_schema_generation(self):
        schema = SwarmActionSchema.model_json_schema()
        assert "kind" in schema["properties"]
        assert "content" in schema["properties"]

    def test_valid_kinds_list(self):
        assert "noop" in _VALID_KINDS
        assert "post" in _VALID_KINDS
        assert "claim_task" in _VALID_KINDS

    def test_action_kind_map_completeness(self):
        """Every valid kind string maps to an ActionType."""
        for kind in _VALID_KINDS:
            assert kind in _ACTION_KIND_MAP

    def test_invalid_kind_rejected(self):
        """Fix #7: Unknown kind strings are rejected at parse time."""
        with pytest.raises(ValueError, match="Invalid action kind"):
            SwarmActionSchema(kind="totally_bogus")

    def test_content_max_length(self):
        """Fix #3: Content exceeding MAX_CONTENT_LENGTH is rejected."""
        with pytest.raises(ValueError):
            SwarmActionSchema(kind="post", content="x" * (MAX_CONTENT_LENGTH + 1))

    def test_rationale_max_length(self):
        """Fix #3: Rationale exceeding MAX_RATIONALE_LENGTH is rejected."""
        with pytest.raises(ValueError):
            SwarmActionSchema(kind="noop", rationale="r" * (MAX_RATIONALE_LENGTH + 1))

    def test_target_id_max_length(self):
        """Fix #3: target_id exceeding MAX_ID_LENGTH is rejected."""
        with pytest.raises(ValueError):
            SwarmActionSchema(kind="noop", target_id="i" * (MAX_ID_LENGTH + 1))

    def test_counterparty_id_max_length(self):
        """Fix #3: counterparty_id exceeding MAX_ID_LENGTH is rejected."""
        with pytest.raises(ValueError):
            SwarmActionSchema(kind="noop", counterparty_id="c" * (MAX_ID_LENGTH + 1))


# ---------------------------------------------------------------------------
# CrewAIToolAdapter tests
# ---------------------------------------------------------------------------


class TestCrewAIToolAdapter:
    def test_no_observation(self):
        adapter = CrewAIToolAdapter()
        assert adapter.get_agent_state() == {}
        assert adapter.get_visible_posts() == []
        assert adapter.get_available_tasks() == []
        assert adapter.get_pending_proposals() == []
        assert adapter.get_visible_agents() == []
        assert adapter.get_ecosystem_metrics() == {}
        assert adapter.get_available_bounties() == []

    def test_with_observation(self, default_observation: Observation):
        adapter = CrewAIToolAdapter()
        adapter.update_observation(default_observation)

        state = adapter.get_agent_state()
        assert state["agent_id"] == "test_agent"
        assert state["resources"] == 100.0

        posts = adapter.get_visible_posts()
        assert len(posts) == 1
        assert posts[0]["post_id"] == "p1"

        tasks = adapter.get_available_tasks()
        assert len(tasks) == 1

        agents = adapter.get_visible_agents()
        assert len(agents) == 1

        metrics = adapter.get_ecosystem_metrics()
        assert metrics["toxicity"] == 0.1

    def test_stage_action(self):
        adapter = CrewAIToolAdapter()
        result = adapter.stage_action({"kind": "post", "content": "test"})
        assert "staged" in result
        assert len(adapter.get_staged_actions()) == 1

    def test_update_observation_clears_staged(self, default_observation: Observation):
        adapter = CrewAIToolAdapter()
        adapter.stage_action({"kind": "post"})
        assert len(adapter.get_staged_actions()) == 1

        adapter.update_observation(default_observation)
        assert len(adapter.get_staged_actions()) == 0

    def test_stage_action_cap(self):
        """Fix #14: stage_action rejects beyond MAX_STAGED_ACTIONS."""
        adapter = CrewAIToolAdapter()
        for i in range(MAX_STAGED_ACTIONS):
            result = adapter.stage_action({"kind": "post", "i": i})
            assert "staged" in result

        # Next one should be rejected
        result = adapter.stage_action({"kind": "post", "i": "overflow"})
        assert "rejected" in result
        assert len(adapter.get_staged_actions()) == MAX_STAGED_ACTIONS


# ---------------------------------------------------------------------------
# Crew profile tests
# ---------------------------------------------------------------------------


class TestCrewProfiles:
    def test_get_default_profiles(self):
        for profile_name in ("market_team_v1", "audit_team_v1", "general_v1"):
            agents = get_profile_agents(profile_name)
            assert len(agents) >= 2
            assert all(isinstance(a, CrewAgentRole) for a in agents)

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown crew profile"):
            get_profile_agents("nonexistent_profile")

    def test_register_custom_profile(self):
        custom_agents = [
            CrewAgentRole(role="CustomRole", goal="Do something"),
        ]
        register_crew_profile("test_custom_safe", custom_agents)
        retrieved = get_profile_agents("test_custom_safe")
        assert len(retrieved) == 1
        assert retrieved[0].role == "CustomRole"

    def test_cannot_overwrite_builtin_profile(self):
        """Fix #9: Built-in profiles are protected from overwrite."""
        for name in _BUILTIN_PROFILE_NAMES:
            with pytest.raises(ValueError, match="Cannot overwrite built-in"):
                register_crew_profile(
                    name, [CrewAgentRole(role="Evil", goal="Hijack")]
                )


# ---------------------------------------------------------------------------
# CrewConfig tests
# ---------------------------------------------------------------------------


class TestCrewConfig:
    def test_defaults(self):
        config = CrewConfig()
        assert config.crew_profile == "general_v1"
        assert config.temperature == 0.7
        assert config.enable_trace is True
        assert config.timeout == DEFAULT_CREW_TIMEOUT_SECONDS

    def test_custom_values(self):
        config = CrewConfig(
            crew_profile="audit_team_v1",
            role_name="Auditor",
            model="claude-sonnet-4-5-20250929",
            temperature=0.3,
            timeout=60.0,
        )
        assert config.crew_profile == "audit_team_v1"
        assert config.role_name == "Auditor"
        assert config.timeout == 60.0


# ---------------------------------------------------------------------------
# Metadata sanitization tests
# ---------------------------------------------------------------------------


class TestSanitizeCrewMetadata:
    def test_allows_safe_types(self):
        raw = {"key_str": "value", "key_int": 42, "key_float": 3.14, "key_bool": True}
        result = _sanitize_crew_metadata(raw)
        assert result == raw

    def test_drops_nested_dicts(self):
        raw = {"safe": "ok", "nested": {"inner": "dropped"}}
        result = _sanitize_crew_metadata(raw)
        assert "safe" in result
        assert "nested" not in result

    def test_drops_lists(self):
        raw = {"safe": 1, "items": [1, 2, 3]}
        result = _sanitize_crew_metadata(raw)
        assert "items" not in result

    def test_truncates_long_strings(self):
        raw = {"long": "x" * 5000}
        result = _sanitize_crew_metadata(raw)
        assert len(result["long"]) == MAX_METADATA_VALUE_LENGTH

    def test_caps_number_of_keys(self):
        raw = {f"key_{i}": i for i in range(100)}
        result = _sanitize_crew_metadata(raw)
        assert len(result) == MAX_METADATA_KEYS

    def test_empty_input(self):
        assert _sanitize_crew_metadata({}) == {}


# ---------------------------------------------------------------------------
# CrewBackedAgent tests (with mocked crew)
# ---------------------------------------------------------------------------


class TestCrewBackedAgent:
    def test_init(self, agent: CrewBackedAgent):
        assert agent.agent_id == "crew_test_1"
        assert agent.crew_config.crew_profile == "general_v1"
        assert agent._crew is None  # lazy

    def test_agent_type_is_crewai(self, agent: CrewBackedAgent):
        """Fix #15: Agent type should be CREWAI, not HONEST."""
        assert agent.agent_type == AgentType.CREWAI

    def test_build_context(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        ctx = agent._build_context(default_observation)
        assert ctx["epoch"] == 0
        assert ctx["step"] == 0
        assert ctx["agent_id"] == "crew_test_1"
        assert ctx["budget"] == 100.0
        assert ctx["reputation"] == 1.0
        assert ctx["can_post"] is True
        assert "action_schema" in ctx
        assert "valid_action_kinds" in ctx
        assert isinstance(ctx["valid_action_kinds"], list)

    def test_parse_action_valid_json(self, agent: CrewBackedAgent):
        raw = json.dumps({"kind": "post", "content": "Hello!", "confidence": 0.9})
        schema = agent._parse_action(raw)
        assert schema.kind == "post"
        assert schema.content == "Hello!"
        assert schema.confidence == 0.9

    def test_parse_action_json_in_code_block(self, agent: CrewBackedAgent):
        raw = '```json\n{"kind": "vote", "target_id": "p1"}\n```'
        schema = agent._parse_action(raw)
        assert schema.kind == "vote"
        assert schema.target_id == "p1"

    def test_parse_action_fallback_noop(self, agent: CrewBackedAgent):
        raw = "This is not valid JSON at all."
        schema = agent._parse_action(raw)
        assert schema.kind == "noop"
        assert "Parse failure" in schema.rationale

    def test_parse_action_invalid_kind_fallback_noop(self, agent: CrewBackedAgent):
        """Fix #7: An invalid kind in valid JSON falls back to noop."""
        raw = json.dumps({"kind": "hacky_exploit"})
        schema = agent._parse_action(raw)
        assert schema.kind == "noop"

    def test_schema_to_action(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        # Must collect valid IDs first
        agent._collect_valid_ids(default_observation)
        schema = SwarmActionSchema(
            kind="post",
            content="Hello world",
            confidence=0.85,
            rationale="Want to contribute",
        )
        action = agent._schema_to_action(schema)
        assert isinstance(action, Action)
        assert action.action_type == ActionType.POST
        assert action.content == "Hello world"
        assert action.agent_id == "crew_test_1"
        assert action.metadata["confidence"] == 0.85

    def test_act_with_mocked_crew(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        """Full act() cycle with a mocked crew."""
        mock_crew = MagicMock()
        mock_result = MagicMock()
        mock_result.raw = json.dumps(
            {
                "kind": "post",
                "content": "Crew decided to post",
                "confidence": 0.92,
                "rationale": "Contributing useful info",
            }
        )
        mock_crew.kickoff.return_value = mock_result
        agent._crew = mock_crew

        action = agent.act(default_observation)

        assert action.action_type == ActionType.POST
        assert action.content == "Crew decided to post"
        assert action.metadata["_crew_trace"]["confidence"] == 0.92
        assert action.metadata["_crew_trace"]["crew_profile"] == "general_v1"

        # Verify deliberation memory was recorded
        history = agent.get_deliberation_history()
        assert len(history) == 1
        assert history[0]["action_kind"] == "post"

    def test_act_crew_failure_falls_back_to_noop(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        """When the crew raises an exception, agent returns noop."""
        mock_crew = MagicMock()
        mock_crew.kickoff.side_effect = RuntimeError("Crew exploded")
        agent._crew = mock_crew

        action = agent.act(default_observation)
        assert action.action_type == ActionType.NOOP

    def test_act_unparseable_output_falls_back_to_noop(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        """When crew output isn't valid JSON, agent returns noop with trace."""
        mock_crew = MagicMock()
        mock_result = MagicMock()
        mock_result.raw = "I think we should do something but I forgot the format"
        mock_crew.kickoff.return_value = mock_result
        agent._crew = mock_crew

        action = agent.act(default_observation)
        assert action.action_type == ActionType.NOOP
        # Trace should still be attached
        assert "_crew_trace" in action.metadata

    def test_accept_interaction_trust_based(self, agent: CrewBackedAgent):
        """accept_interaction uses counterparty trust."""
        from swarm.agents.base import InteractionProposal

        proposal = InteractionProposal(
            initiator_id="other_agent",
            counterparty_id=agent.agent_id,
        )
        obs = Observation()

        # Unknown agent: trust = 0.5, threshold = 0.4 -> accept
        assert agent.accept_interaction(proposal, obs) is True

        # Set low trust
        agent._counterparty_memory["other_agent"] = 0.1
        assert agent.accept_interaction(proposal, obs) is False

    def test_propose_interaction(self, agent: CrewBackedAgent):
        obs = Observation()

        # Unknown agent -> trust 0.5 -> propose
        proposal = agent.propose_interaction(obs, "counterparty_1")
        assert proposal is not None
        assert proposal.initiator_id == agent.agent_id
        assert proposal.counterparty_id == "counterparty_1"

        # Low-trust agent -> None
        agent._counterparty_memory["low_trust"] = 0.1
        assert agent.propose_interaction(obs, "low_trust") is None

    def test_trace_disabled(
        self, default_observation: Observation
    ):
        """When enable_trace=False, no _crew_trace in metadata."""
        config = CrewConfig(enable_trace=False)
        agent = CrewBackedAgent(
            agent_id="no_trace",
            crew_config=config,
        )

        mock_crew = MagicMock()
        mock_result = MagicMock()
        mock_result.raw = json.dumps({"kind": "noop"})
        mock_crew.kickoff.return_value = mock_result
        agent._crew = mock_crew

        action = agent.act(default_observation)
        assert "_crew_trace" not in action.metadata

    def test_repr(self, agent: CrewBackedAgent):
        r = repr(agent)
        assert "CrewBackedAgent" in r
        assert "crew_test_1" in r
        assert "general_v1" in r


# ---------------------------------------------------------------------------
# Security: ID validation (Fix #1)
# ---------------------------------------------------------------------------


class TestSecurityIDValidation:
    def test_valid_target_id_accepted(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        """target_id present in observation is accepted."""
        agent._collect_valid_ids(default_observation)
        schema = SwarmActionSchema(kind="vote", target_id="p1")
        action = agent._schema_to_action(schema)
        assert action.target_id == "p1"

    def test_invalid_target_id_rejected(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        """target_id NOT in observation is rejected to empty string."""
        agent._collect_valid_ids(default_observation)
        schema = SwarmActionSchema(kind="vote", target_id="fake_post_999")
        action = agent._schema_to_action(schema)
        assert action.target_id == ""

    def test_valid_counterparty_id_accepted(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        """counterparty_id present in visible_agents is accepted."""
        agent._collect_valid_ids(default_observation)
        schema = SwarmActionSchema(
            kind="propose_interaction", counterparty_id="other_1"
        )
        action = agent._schema_to_action(schema)
        assert action.counterparty_id == "other_1"

    def test_invalid_counterparty_id_rejected(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        """counterparty_id NOT in visible_agents is rejected to empty string."""
        agent._collect_valid_ids(default_observation)
        schema = SwarmActionSchema(
            kind="propose_interaction", counterparty_id="secret_admin_agent"
        )
        action = agent._schema_to_action(schema)
        assert action.counterparty_id == ""

    def test_task_id_in_available_tasks(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        agent._collect_valid_ids(default_observation)
        schema = SwarmActionSchema(kind="claim_task", target_id="t1")
        action = agent._schema_to_action(schema)
        assert action.target_id == "t1"

    def test_bounty_id_accepted(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        agent._collect_valid_ids(default_observation)
        schema = SwarmActionSchema(kind="place_bid", target_id="b1")
        action = agent._schema_to_action(schema)
        assert action.target_id == "b1"

    def test_proposal_id_accepted(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        agent._collect_valid_ids(default_observation)
        schema = SwarmActionSchema(
            kind="accept_interaction", target_id="prop_1"
        )
        action = agent._schema_to_action(schema)
        assert action.target_id == "prop_1"

    def test_empty_ids_pass_through(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        """Empty IDs (default) should remain empty, not rejected."""
        agent._collect_valid_ids(default_observation)
        schema = SwarmActionSchema(kind="noop")
        action = agent._schema_to_action(schema)
        assert action.target_id == ""
        assert action.counterparty_id == ""


# ---------------------------------------------------------------------------
# Security: Metadata namespace isolation (Fix #2)
# ---------------------------------------------------------------------------


class TestSecurityMetadataNamespacing:
    def test_crew_metadata_is_namespaced(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        """Crew-supplied metadata lives under 'crew_metadata' key."""
        agent._collect_valid_ids(default_observation)
        schema = SwarmActionSchema(
            kind="noop",
            metadata={"custom_key": "custom_val"},
        )
        action = agent._schema_to_action(schema)
        assert "crew_metadata" in action.metadata
        assert action.metadata["crew_metadata"]["custom_key"] == "custom_val"
        # Should NOT have crew keys at top level
        assert "custom_key" not in action.metadata

    def test_crew_cannot_inject_internal_keys(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        """Crew metadata with keys like 'bid_id' won't pollute top-level."""
        agent._collect_valid_ids(default_observation)
        schema = SwarmActionSchema(
            kind="noop",
            metadata={
                "bid_id": "attacker_bid",
                "reward_amount": 99999,
                "_crew_trace": {"fake": True},
            },
        )
        action = agent._schema_to_action(schema)
        # These should only be inside crew_metadata, not at root
        assert action.metadata.get("bid_id") is None
        assert action.metadata.get("reward_amount") is None
        # _crew_trace in crew_metadata is dropped (nested dict)
        assert "_crew_trace" not in action.metadata["crew_metadata"]


# ---------------------------------------------------------------------------
# Security: Timeout (Fix #4)
# ---------------------------------------------------------------------------


class TestSecurityTimeout:
    def test_timeout_raises_on_slow_crew(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        """Fix #4: A crew that exceeds the timeout triggers noop fallback."""
        import time

        mock_crew = MagicMock()

        def slow_kickoff(**kwargs):
            time.sleep(5)
            return MagicMock(raw='{"kind": "post"}')

        mock_crew.kickoff.side_effect = slow_kickoff
        agent._crew = mock_crew
        agent.crew_config = CrewConfig(timeout=0.1)  # 100ms

        action = agent.act(default_observation)
        assert action.action_type == ActionType.NOOP

    def test_kickoff_with_timeout_normal(self, agent: CrewBackedAgent):
        """Normal kickoff within timeout succeeds."""
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "result"
        result = agent._kickoff_with_timeout(mock_crew, {})
        assert result == "result"

    def test_timeout_does_not_block_caller(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        """After timeout, act() returns promptly (no wait for thread)."""
        import time

        mock_crew = MagicMock()

        def slow_kickoff(**kwargs):
            time.sleep(10)
            return MagicMock(raw='{"kind": "post"}')

        mock_crew.kickoff.side_effect = slow_kickoff
        agent._crew = mock_crew
        agent.crew_config = CrewConfig(timeout=0.1)

        start = time.monotonic()
        action = agent.act(default_observation)
        elapsed = time.monotonic() - start

        assert action.action_type == ActionType.NOOP
        # Must return in well under the 10s sleep
        assert elapsed < 2.0

    def test_pool_reused_across_ticks(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        """Thread pool is reused across normal ticks."""
        mock_crew = MagicMock()
        mock_result = MagicMock()
        mock_result.raw = json.dumps({"kind": "noop"})
        mock_crew.kickoff.return_value = mock_result
        agent._crew = mock_crew

        agent.act(default_observation)
        pool_after_first = agent._executor_pool

        agent.act(default_observation)
        pool_after_second = agent._executor_pool

        assert pool_after_first is pool_after_second

    def test_pool_recreated_after_timeout(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        """After a timeout, the old pool is discarded and a fresh one is created."""
        import time

        mock_crew = MagicMock()

        def slow_then_fast(**kwargs):
            if mock_crew.kickoff.call_count <= 1:
                time.sleep(5)
            return MagicMock(raw='{"kind": "noop"}')

        mock_crew.kickoff.side_effect = slow_then_fast
        agent._crew = mock_crew
        agent.crew_config = CrewConfig(timeout=0.1)

        # First call times out
        agent.act(default_observation)
        assert agent._executor_pool is None  # discarded

        # Second call creates a fresh pool
        mock_crew.kickoff.side_effect = None
        mock_result = MagicMock()
        mock_result.raw = json.dumps({"kind": "noop"})
        mock_crew.kickoff.return_value = mock_result
        agent.crew_config = CrewConfig(timeout=120.0)

        action = agent.act(default_observation)
        assert action.action_type == ActionType.NOOP
        assert agent._executor_pool is not None

    def test_timeout_specific_log_message(
        self, agent: CrewBackedAgent, default_observation: Observation, caplog
    ):
        """Timeout produces a distinct warning, not the generic exception log."""
        import logging
        import time

        mock_crew = MagicMock()
        mock_crew.kickoff.side_effect = lambda **kw: time.sleep(5)
        agent._crew = mock_crew
        agent.crew_config = CrewConfig(timeout=0.1)

        with caplog.at_level(logging.WARNING):
            agent.act(default_observation)

        assert any("timed out after" in r.message for r in caplog.records)
        assert not any(
            "deliberation failed" in r.message for r in caplog.records
        )


# ---------------------------------------------------------------------------
# Security: Deliberation memory cap (Fix #6)
# ---------------------------------------------------------------------------


class TestSecurityDeliberationMemory:
    def test_memory_capped(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        """Fix #6: Memory never exceeds MAX_DELIBERATION_MEMORY."""
        mock_crew = MagicMock()
        mock_result = MagicMock()
        mock_result.raw = json.dumps({"kind": "noop"})
        mock_crew.kickoff.return_value = mock_result
        agent._crew = mock_crew

        for i in range(MAX_DELIBERATION_MEMORY + 50):
            obs = Observation(
                agent_state=default_observation.agent_state,
                current_epoch=i,
                current_step=0,
            )
            agent.act(obs)

        assert len(agent._deliberation_memory) == MAX_DELIBERATION_MEMORY

    def test_memory_rationale_truncated(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        """Fix #6: Rationales in memory are truncated to 500 chars."""
        mock_crew = MagicMock()
        mock_result = MagicMock()
        mock_result.raw = json.dumps({
            "kind": "noop",
            "rationale": "x" * 1500,
        })
        mock_crew.kickoff.return_value = mock_result
        agent._crew = mock_crew

        agent.act(default_observation)

        entry = agent._deliberation_memory[-1]
        assert len(entry["rationale"]) == 500


# ---------------------------------------------------------------------------
# Security: Trace truncation (Fix #8)
# ---------------------------------------------------------------------------


class TestSecurityTraceOutput:
    def test_raw_output_truncated_in_trace(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        """Fix #8: raw_output in trace is capped at MAX_RAW_TRACE_LENGTH."""
        mock_crew = MagicMock()
        mock_result = MagicMock()
        # Return valid JSON but with a huge rationale
        mock_result.raw = json.dumps({"kind": "noop"}) + " " * 10_000
        mock_crew.kickoff.return_value = mock_result
        agent._crew = mock_crew

        action = agent.act(default_observation)

        trace = action.metadata.get("_crew_trace", {})
        assert len(trace.get("raw_output", "")) <= MAX_RAW_TRACE_LENGTH

    def test_rationale_truncated_in_trace(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        """Fix #8: rationale in trace is capped at MAX_RATIONALE_LENGTH."""
        mock_crew = MagicMock()
        long_rationale = "r" * 1999  # within schema limit
        mock_result = MagicMock()
        mock_result.raw = json.dumps({
            "kind": "noop",
            "rationale": long_rationale,
        })
        mock_crew.kickoff.return_value = mock_result
        agent._crew = mock_crew

        action = agent.act(default_observation)

        trace = action.metadata.get("_crew_trace", {})
        assert len(trace.get("rationale", "")) <= MAX_RATIONALE_LENGTH


# ---------------------------------------------------------------------------
# Loader integration tests
# ---------------------------------------------------------------------------


class TestLoaderIntegration:
    def test_lazy_import_crewai_classes(self):
        """Verify the lazy import function in the loader works."""
        from swarm.scenarios.loader import _get_crewai_classes

        AgentCls, ConfigCls = _get_crewai_classes()
        assert AgentCls is CrewBackedAgent
        assert ConfigCls is CrewConfig

    def test_create_agents_crewai_type(self):
        """create_agents handles type='crewai_adapter'."""
        from swarm.scenarios.loader import create_agents

        specs = [
            {
                "type": "crewai_adapter",
                "count": 2,
                "params": {
                    "role_name": "Trader",
                    "crew_profile": "market_team_v1",
                },
            }
        ]
        agents = create_agents(specs, seed=99)
        assert len(agents) == 2
        assert all(isinstance(a, CrewBackedAgent) for a in agents)
        assert agents[0].agent_id == "crewai_1"
        assert agents[1].agent_id == "crewai_2"
        assert agents[0].crew_config.role_name == "Trader"
        assert agents[0].crew_config.crew_profile == "market_team_v1"

    def test_create_agents_timeout_from_yaml(self):
        """timeout param in YAML is wired to CrewConfig."""
        from swarm.scenarios.loader import create_agents

        specs = [
            {
                "type": "crewai_adapter",
                "count": 1,
                "params": {
                    "role_name": "Fast Agent",
                    "timeout": 30.0,
                },
            }
        ]
        agents = create_agents(specs, seed=1)
        assert agents[0].crew_config.timeout == 30.0

    def test_create_agents_default_timeout(self):
        """Without timeout in YAML, default is used."""
        from swarm.scenarios.loader import create_agents

        specs = [
            {
                "type": "crewai_adapter",
                "count": 1,
                "params": {"role_name": "Default"},
            }
        ]
        agents = create_agents(specs, seed=1)
        assert agents[0].crew_config.timeout == DEFAULT_CREW_TIMEOUT_SECONDS

    def test_create_agents_mixed(self):
        """CrewAI agents can coexist with scripted agents."""
        from swarm.scenarios.loader import create_agents

        specs = [
            {"type": "honest", "count": 2},
            {
                "type": "crewai_adapter",
                "count": 1,
                "params": {"role_name": "Auditor"},
            },
            {"type": "opportunistic", "count": 1},
        ]
        agents = create_agents(specs, seed=42)
        assert len(agents) == 4
        crew_agents = [a for a in agents if isinstance(a, CrewBackedAgent)]
        assert len(crew_agents) == 1


# ---------------------------------------------------------------------------
# Action kind mapping exhaustiveness
# ---------------------------------------------------------------------------


class TestActionKindMapping:
    @pytest.mark.parametrize("kind", _VALID_KINDS)
    def test_each_kind_maps_to_action_type(self, kind: str):
        action_type = _ACTION_KIND_MAP[kind]
        assert isinstance(action_type, ActionType)

    def test_all_mapped_action_types_are_unique(self):
        """Each kind string maps to a distinct ActionType."""
        values = list(_ACTION_KIND_MAP.values())
        assert len(values) == len(set(values))


class TestCrewMetadataSanitizationParity:
    """Regression for beads 909k: CrewAI sanitizer at parity with Swarms."""

    def test_key_length_capped_and_control_chars_stripped(self):
        from swarm.agents.crewai_adapter import (
            MAX_METADATA_KEY_LENGTH,
            _sanitize_crew_metadata,
        )

        raw = {"k" * 500: "v", "safe\x00key\x1f": "v2", "\x7fnull": "v3"}
        result = _sanitize_crew_metadata(raw)
        for key in result:
            assert len(key) <= MAX_METADATA_KEY_LENGTH
            assert "\x00" not in key and "\x1f" not in key and "\x7f" not in key

    def test_empty_key_after_sanitization_dropped(self):
        from swarm.agents.crewai_adapter import _sanitize_crew_metadata

        assert _sanitize_crew_metadata({"\x00\x01\x02": "value"}) == {}

    def test_non_finite_floats_dropped(self):
        from swarm.agents.crewai_adapter import _sanitize_crew_metadata

        raw = {
            "good": 1.0,
            "nan": float("nan"),
            "inf": float("inf"),
            "ninf": float("-inf"),
            "flag": True,
        }
        result = _sanitize_crew_metadata(raw)
        assert result == {"good": 1.0, "flag": True}
        assert result["flag"] is True  # bool preserved as bool, not coerced

    def test_executor_pool_shutdown_check_is_defensive(self):
        """getattr with default: a pool object without _shutdown must not crash."""
        import concurrent.futures

        from swarm.agents.crewai_adapter import CrewBackedAgent

        agent = CrewBackedAgent.__new__(CrewBackedAgent)
        agent.agent_id = "crew_test"
        agent._executor_pool = None
        pool = agent._get_executor_pool()
        assert isinstance(pool, concurrent.futures.ThreadPoolExecutor)
        # Same live pool reused while not shut down.
        assert agent._get_executor_pool() is pool
        pool.shutdown(wait=False)
        assert agent._get_executor_pool() is not pool
        agent._executor_pool.shutdown(wait=False)
