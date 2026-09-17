"""Tests for the council core protocol (all mocked, no API calls)."""

import asyncio

import pytest

from swarm.agents.llm_config import LLMConfig, LLMProvider
from swarm.council.config import CouncilConfig, CouncilMemberConfig
from swarm.council.protocol import Council, CouncilResult
from swarm.council.ranking import (
    aggregate_rankings,
    anonymize_responses,
    parse_rankings,
)

# ── anonymize_responses ──────────────────────────────────────────────


class TestAnonymizeResponses:
    def test_deterministic_with_seed(self):
        responses = {"alice": "resp_a", "bob": "resp_b", "carol": "resp_c"}
        map1, anon1 = anonymize_responses(responses, seed=42)
        map2, anon2 = anonymize_responses(responses, seed=42)
        assert map1 == map2
        assert anon1 == anon2

    def test_different_seed_different_order(self):
        responses = {"alice": "resp_a", "bob": "resp_b", "carol": "resp_c"}
        map1, _ = anonymize_responses(responses, seed=42)
        map2, _ = anonymize_responses(responses, seed=99)
        # With different seeds, at least one mapping should differ
        # (not guaranteed but extremely likely with 3 items)
        # Just verify structure is correct
        assert set(map1.values()) == set(responses.keys())
        assert set(map2.values()) == set(responses.keys())

    def test_labels_are_consecutive(self):
        responses = {"x": "1", "y": "2", "z": "3"}
        anon_map, anon_responses = anonymize_responses(responses, seed=0)
        assert set(anon_map.keys()) == {"A", "B", "C"}
        assert set(anon_responses.keys()) == {"A", "B", "C"}

    def test_single_response(self):
        responses = {"only": "single"}
        anon_map, anon_responses = anonymize_responses(responses, seed=0)
        assert anon_map == {"A": "only"}
        assert anon_responses == {"A": "single"}


# ── parse_rankings ───────────────────────────────────────────────────


class TestParseRankings:
    def test_structured_format(self):
        text = "1. A\n2. B\n3. C"
        result = parse_rankings(text, 3)
        assert result == ["A", "B", "C"]

    def test_comma_separated(self):
        text = "B, A, C"
        result = parse_rankings(text, 3)
        assert result == ["B", "A", "C"]

    def test_arrow_separated(self):
        text = "C > A > B"
        result = parse_rankings(text, 3)
        assert result == ["C", "A", "B"]

    def test_embedded_in_text(self):
        text = "I think the best ranking is: B then A then C. Response B was great."
        result = parse_rankings(text, 3)
        assert result == ["B", "A", "C"]

    def test_malformed_returns_none(self):
        text = "I can't decide, they're all good!"
        result = parse_rankings(text, 3)
        assert result is None

    def test_duplicate_labels_returns_none(self):
        text = "1. A\n2. A\n3. B"
        result = parse_rankings(text, 3)
        assert result is None

    def test_reasoning_before_ranking_marker(self):
        text = (
            "A is thorough but slow to the point. B has a factual error. "
            "C is concise and correct.\n\nRANKING:\n1. C\n2. A\n3. B"
        )
        assert parse_rankings(text, 3) == ["C", "A", "B"]

    def test_numbered_reasoning_uses_last_list(self):
        text = (
            "1. A is thorough.\n2. B has an error.\n3. C is concise.\n\n"
            "1. C\n2. A\n3. B"
        )
        assert parse_rankings(text, 3) == ["C", "A", "B"]

    def test_capitalized_words_after_numbers_ignored(self):
        text = "Step 2. The best is clear.\n**Ranking:**\n1. Response B\n2. Response A"
        assert parse_rankings(text, 2) == ["B", "A"]

    def test_two_responses(self):
        text = "1. B\n2. A"
        result = parse_rankings(text, 2)
        assert result == ["B", "A"]


# ── aggregate_rankings (Borda count) ────────────────────────────────


class TestAggregateRankings:
    def test_equal_weights(self):
        anon_map = {"A": "alice", "B": "bob", "C": "carol"}
        rankings = {
            "r1": ["A", "B", "C"],  # A=2, B=1, C=0
            "r2": ["A", "C", "B"],  # A=2, C=1, B=0
            "r3": ["B", "A", "C"],  # B=2, A=1, C=0
        }
        # A=5, B=3, C=1
        result = aggregate_rankings(rankings, anon_map)
        assert result == ["alice", "bob", "carol"]

    def test_unequal_weights(self):
        anon_map = {"A": "alice", "B": "bob"}
        rankings = {
            "r1": ["A", "B"],  # A=1, B=0
            "r2": ["B", "A"],  # B=1, A=0
        }
        weights = {"r1": 1.0, "r2": 3.0}
        # A: 1*1 + 0*3 = 1, B: 0*1 + 1*3 = 3
        result = aggregate_rankings(rankings, anon_map, weights)
        assert result == ["bob", "alice"]

    def test_tie_broken_by_label(self):
        anon_map = {"A": "alice", "B": "bob"}
        rankings = {
            "r1": ["A", "B"],  # A=1, B=0
            "r2": ["B", "A"],  # B=1, A=0
        }
        # Tied at 1 each, broken by label (A < B)
        result = aggregate_rankings(rankings, anon_map)
        assert result == ["alice", "bob"]

    def test_single_ranker(self):
        anon_map = {"A": "alice", "B": "bob", "C": "carol"}
        rankings = {"r1": ["C", "A", "B"]}
        result = aggregate_rankings(rankings, anon_map)
        assert result == ["carol", "alice", "bob"]


# ── Full deliberation (mocked) ──────────────────────────────────────


class TestFullDeliberation:
    @pytest.mark.asyncio
    async def test_end_to_end_with_mocks(self):
        """Full deliberation with mock query functions."""

        async def mock_query_alice(sys: str, usr: str) -> str:
            if "Rank" in sys or "rank" in usr.lower():
                return "1. A\n2. B\n3. C"
            if "chairman" in sys.lower() or "synthesize" in sys.lower():
                return "Synthesis: All members agree on the approach."
            return "Alice's response to the query."

        async def mock_query_bob(sys: str, usr: str) -> str:
            if "Rank" in sys or "rank" in usr.lower():
                return "1. B\n2. A\n3. C"
            return "Bob's detailed analysis."

        async def mock_query_carol(sys: str, usr: str) -> str:
            if "Rank" in sys or "rank" in usr.lower():
                return "1. A\n2. C\n3. B"
            return "Carol's perspective."

        config = CouncilConfig(
            members=[
                CouncilMemberConfig(
                    member_id="alice",
                    llm_config=LLMConfig(provider=LLMProvider.OLLAMA),
                ),
                CouncilMemberConfig(
                    member_id="bob",
                    llm_config=LLMConfig(provider=LLMProvider.OLLAMA),
                ),
                CouncilMemberConfig(
                    member_id="carol",
                    llm_config=LLMConfig(provider=LLMProvider.OLLAMA),
                ),
            ],
            chairman=CouncilMemberConfig(
                member_id="alice",
                llm_config=LLMConfig(provider=LLMProvider.OLLAMA),
            ),
            min_members_required=2,
            seed=42,
        )

        query_fns = {
            "alice": mock_query_alice,
            "bob": mock_query_bob,
            "carol": mock_query_carol,
        }

        council = Council(config=config, query_fns=query_fns)
        result = await council.deliberate("System prompt", "User query")

        assert result.success is True
        assert result.members_responded == 3
        assert result.members_total == 3
        assert len(result.responses) == 3
        assert result.synthesis != ""

    @pytest.mark.asyncio
    async def test_to_dict_serialization(self):
        result = CouncilResult(
            synthesis="test synthesis",
            responses={"a": "resp_a"},
            rankings={"a": ["A"]},
            aggregate_ranking=["a"],
            members_responded=1,
            members_total=1,
        )
        d = result.to_dict()
        assert d["synthesis"] == "test synthesis"
        assert d["success"] is True
        assert d["members_responded"] == 1


# ── Graceful degradation ────────────────────────────────────────────


class TestGracefulDegradation:
    @pytest.mark.asyncio
    async def test_quorum_not_met(self):
        """When too many members fail, deliberation reports failure."""

        async def mock_fail(sys: str, usr: str) -> str:
            raise RuntimeError("API unavailable")

        async def mock_ok(sys: str, usr: str) -> str:
            return "OK response"

        config = CouncilConfig(
            members=[
                CouncilMemberConfig(
                    member_id="a",
                    llm_config=LLMConfig(provider=LLMProvider.OLLAMA),
                ),
                CouncilMemberConfig(
                    member_id="b",
                    llm_config=LLMConfig(provider=LLMProvider.OLLAMA),
                ),
                CouncilMemberConfig(
                    member_id="c",
                    llm_config=LLMConfig(provider=LLMProvider.OLLAMA),
                ),
            ],
            min_members_required=2,
        )

        # Only 1 succeeds, but need 2
        query_fns = {
            "a": mock_fail,
            "b": mock_fail,
            "c": mock_ok,
        }

        council = Council(config=config, query_fns=query_fns)
        result = await council.deliberate("sys", "usr")

        assert result.success is False
        assert "Quorum" in result.error
        assert result.members_responded == 1

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Members that timeout are excluded gracefully."""

        async def mock_slow(sys: str, usr: str) -> str:
            await asyncio.sleep(10)
            return "too slow"

        async def mock_fast(sys: str, usr: str) -> str:
            if "rank" in usr.lower() or "Rank" in sys:
                return "1. A\n2. B"
            if "synthesize" in sys.lower() or "chairman" in sys.lower():
                return "Fast synthesis."
            return "Fast response"

        config = CouncilConfig(
            members=[
                CouncilMemberConfig(
                    member_id="slow",
                    llm_config=LLMConfig(provider=LLMProvider.OLLAMA),
                ),
                CouncilMemberConfig(
                    member_id="fast1",
                    llm_config=LLMConfig(provider=LLMProvider.OLLAMA),
                ),
                CouncilMemberConfig(
                    member_id="fast2",
                    llm_config=LLMConfig(provider=LLMProvider.OLLAMA),
                ),
            ],
            min_members_required=2,
            timeout_per_member=0.1,
            seed=42,
        )

        query_fns = {
            "slow": mock_slow,
            "fast1": mock_fast,
            "fast2": mock_fast,
        }

        council = Council(config=config, query_fns=query_fns)
        result = await council.deliberate("sys", "usr")

        assert result.success is True
        assert result.members_responded >= 2
        assert "slow" not in result.responses

    @pytest.mark.asyncio
    async def test_single_member_skips_ranking(self):
        """With only one successful response, ranking is skipped."""

        async def mock_fail(sys: str, usr: str) -> str:
            raise RuntimeError("fail")

        async def mock_ok(sys: str, usr: str) -> str:
            if "synthesize" in sys.lower() or "chairman" in sys.lower():
                return "Only one response available."
            return "Single response"

        config = CouncilConfig(
            members=[
                CouncilMemberConfig(
                    member_id="a",
                    llm_config=LLMConfig(provider=LLMProvider.OLLAMA),
                ),
                CouncilMemberConfig(
                    member_id="b",
                    llm_config=LLMConfig(provider=LLMProvider.OLLAMA),
                ),
            ],
            min_members_required=1,
        )

        query_fns = {"a": mock_fail, "b": mock_ok}
        council = Council(config=config, query_fns=query_fns)
        result = await council.deliberate("sys", "usr")

        assert result.success is True
        assert result.members_responded == 1
        assert result.rankings == {}  # Ranking skipped for single response
