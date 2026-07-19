"""Council deliberation protocol implementation.

Three-stage protocol:
1. Collect: Parallel query to all council members
2. Rank: Each member ranks anonymized peer responses
3. Synthesize: Chairman produces final answer from all data
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from swarm.council.config import CouncilConfig
from swarm.council.prompts import build_ranking_prompt, build_synthesis_prompt
from swarm.council.ranking import (
    aggregate_rankings,
    anonymize_responses,
    parse_rankings,
)

logger = logging.getLogger(__name__)

# Type for an async query function: (system_prompt, user_prompt) -> response_text
QueryFn = Callable[[str, str], Awaitable[str]]


@dataclass
class MemberResponse:
    """Response from a single council member."""

    member_id: str
    response: str
    success: bool = True
    error: Optional[str] = None


@dataclass
class MemberRanking:
    """Ranking from a single council member."""

    member_id: str
    ranking: Optional[List[str]] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class CouncilResult:
    """Result of a full council deliberation."""

    synthesis: str
    responses: Dict[str, str] = field(default_factory=dict)
    rankings: Dict[str, List[str]] = field(default_factory=dict)
    aggregate_ranking: List[str] = field(default_factory=list)
    members_responded: int = 0
    members_total: int = 0
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for event logging."""
        return {
            "synthesis": self.synthesis,
            "responses": self.responses,
            "rankings": self.rankings,
            "aggregate_ranking": self.aggregate_ranking,
            "members_responded": self.members_responded,
            "members_total": self.members_total,
            "success": self.success,
            "error": self.error,
        }


class Council:
    """Multi-LLM council that deliberates using a 3-stage protocol.

    Provider-agnostic: accepts a dict of member_id -> async query functions.
    """

    def __init__(
        self,
        config: CouncilConfig,
        query_fns: Dict[str, QueryFn],
    ):
        """Initialize council.

        Args:
            config: Council configuration
            query_fns: Dict of member_id -> async callable(system_prompt, user_prompt) -> str
        """
        self.config = config
        self.query_fns = query_fns

    async def deliberate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> CouncilResult:
        """Run the full 3-stage council deliberation.

        Args:
            system_prompt: System prompt for the initial query
            user_prompt: User prompt / query

        Returns:
            CouncilResult with synthesis and metadata
        """
        # Stage 1: Collect responses
        responses = await self._stage_collect(system_prompt, user_prompt)

        if len(responses) < self.config.min_members_required:
            return CouncilResult(
                synthesis="",
                responses=responses,
                members_responded=len(responses),
                members_total=len(self.query_fns),
                success=False,
                error=f"Quorum not met: {len(responses)}/{self.config.min_members_required}",
            )

        # Stage 2: Rank (if more than 1 response)
        rankings: Dict[str, List[str]] = {}
        aggregate: List[str] = []

        if len(responses) > 1:
            rankings, aggregate = await self._stage_rank(
                user_prompt, responses
            )

        # Stage 3: Synthesize
        synthesis = await self._stage_synthesize(
            user_prompt, responses, rankings, aggregate
        )

        return CouncilResult(
            synthesis=synthesis,
            responses=responses,
            rankings=rankings,
            aggregate_ranking=aggregate,
            members_responded=len(responses),
            members_total=len(self.query_fns),
            success=True,
        )

    async def _stage_collect(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, str]:
        """Stage 1: Query all members in parallel with timeout."""

        async def _query_member(member_id: str) -> MemberResponse:
            try:
                result = await asyncio.wait_for(
                    self.query_fns[member_id](system_prompt, user_prompt),
                    timeout=self.config.timeout_per_member,
                )
                return MemberResponse(member_id=member_id, response=result)
            except asyncio.TimeoutError:
                logger.warning(f"Council member {member_id} timed out")
                return MemberResponse(
                    member_id=member_id, response="", success=False,
                    error="timeout",
                )
            except Exception as e:
                logger.warning(f"Council member {member_id} failed: {e}")
                return MemberResponse(
                    member_id=member_id, response="", success=False,
                    error=str(e),
                )

        tasks = [_query_member(mid) for mid in self.query_fns]
        results = await asyncio.gather(*tasks)

        return {r.member_id: r.response for r in results if r.success}

    async def _stage_rank(
        self,
        query: str,
        responses: Dict[str, str],
    ) -> tuple[Dict[str, List[str]], List[str]]:
        """Stage 2: Each member ranks anonymized peer responses."""
        anon_map, anon_responses = anonymize_responses(
            responses, seed=self.config.seed
        )

        rank_system, rank_user = build_ranking_prompt(query, anon_responses)
        n = len(anon_responses)

        async def _rank_member(member_id: str) -> MemberRanking:
            if member_id not in self.query_fns:
                return MemberRanking(member_id=member_id, success=False, error="no query fn")
            try:
                text = await asyncio.wait_for(
                    self.query_fns[member_id](rank_system, rank_user),
                    timeout=self.config.timeout_per_member,
                )
                parsed = parse_rankings(text, n)
                if parsed is None:
                    return MemberRanking(
                        member_id=member_id, success=False,
                        error=f"Failed to parse ranking: {text[:100]}",
                    )
                return MemberRanking(member_id=member_id, ranking=parsed)
            except Exception as e:
                logger.warning(f"Ranking failed for {member_id}: {e}")
                return MemberRanking(
                    member_id=member_id, success=False, error=str(e),
                )

        tasks = [_rank_member(mid) for mid in responses]
        results = await asyncio.gather(*tasks)

        rankings = {
            r.member_id: r.ranking
            for r in results
            if r.success and r.ranking is not None
        }

        # Build weight map from config
        weights = {
            m.member_id: m.weight
            for m in self.config.members
            if m.member_id in rankings
        }

        aggregate = aggregate_rankings(rankings, anon_map, weights) if rankings else list(responses.keys())

        return rankings, aggregate

    async def _stage_synthesize(
        self,
        query: str,
        responses: Dict[str, str],
        rankings: Dict[str, List[str]],
        aggregate: List[str],
    ) -> str:
        """Stage 3: Chairman synthesizes final answer."""
        synth_system, synth_user = build_synthesis_prompt(
            query, responses, rankings, aggregate
        )

        chairman_id = (
            self.config.chairman.member_id
            if self.config.chairman
            else next(iter(self.query_fns))
        )

        if chairman_id not in self.query_fns:
            # Fallback to any available member
            logger.warning(
                f"Configured chairman {chairman_id} not in query_fns; "
                f"falling back to {next(iter(self.query_fns))}"
            )
            chairman_id = next(iter(self.query_fns))

        try:
            synthesis = await asyncio.wait_for(
                self.query_fns[chairman_id](synth_system, synth_user),
                timeout=self.config.timeout_per_member * 2,  # Chairman gets more time
            )
            return synthesis
        except Exception as e:
            logger.error(f"Chairman synthesis failed: {e}")
            # Fallback: return the top-ranked response
            if aggregate:
                return responses.get(aggregate[0], "Council deliberation failed.")
            return next(iter(responses.values()), "Council deliberation failed.")
