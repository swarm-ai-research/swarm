"""Three-tier memory domain model for shared-memory multi-agent simulations.

Inspired by Fred, "A Three-Tier Memory Architecture for Persistent AI
Assistants" (ClawXiv 2602.00006).  Adapted for SWARM: multiple agents
share tiered memory infrastructure, creating commons dynamics around
accuracy, promotion, and cache control.
"""

import random
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class MemoryTier(Enum):
    """Memory tier (progressive enrichment)."""

    EPHEMERAL = "ephemeral"  # Tier 1: daily logs
    STRUCTURED = "structured"  # Tier 2: wiki-linked notes
    GRAPH = "graph"  # Tier 3: knowledge graph


class MemoryEntryStatus(Enum):
    """Lifecycle status of a memory entry."""

    ACTIVE = "active"
    PENDING_PROMOTION = "pending_promotion"
    PROMOTED = "promoted"
    REVERTED = "reverted"
    CHALLENGED = "challenged"


@dataclass
class MemoryEntry:
    """A single fact stored in the shared memory system."""

    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    author_id: str = ""
    tier: MemoryTier = MemoryTier.EPHEMERAL
    status: MemoryEntryStatus = MemoryEntryStatus.ACTIVE
    quality_score: float = 0.5
    is_poisoned: bool = False  # Ground truth (hidden from agents)
    created_epoch: int = 0
    created_step: int = 0
    promoted_from: Optional[str] = None  # entry_id of lower-tier origin
    verified_by: List[str] = field(default_factory=list)
    challenge_count: int = 0
    read_count: int = 0

    def to_dict(self) -> Dict:
        """Serialize for observation (excludes ground-truth poisoned flag)."""
        return {
            "entry_id": self.entry_id,
            "content": self.content,
            "author_id": self.author_id,
            "tier": self.tier.value,
            "status": self.status.value,
            "quality_score": self.quality_score,
            "created_epoch": self.created_epoch,
            "created_step": self.created_step,
            "verified_by": list(self.verified_by),
            "challenge_count": self.challenge_count,
            "read_count": self.read_count,
        }


class MemoryStore:
    """Manages the three-tier shared memory system.

    Provides write, promote, search, hot cache, and compaction operations.
    """

    VALID_RANKINGS = ("quality", "recency", "engagement")

    def __init__(self, seed: Optional[int] = None, ranking: str = "quality") -> None:
        if ranking not in self.VALID_RANKINGS:
            raise ValueError(
                f"ranking must be one of {self.VALID_RANKINGS}, got {ranking!r}"
            )
        self._entries: Dict[str, MemoryEntry] = {}
        self._rng = random.Random(seed)
        self._ranking = ranking
        self._lock = threading.Lock()

        # Hot cache: top-K entries from Tier 3 rebuilt at epoch start
        self._hot_cache: List[MemoryEntry] = []
        self._hot_cache_size: int = 20

        # Write tracking per agent per epoch (for rate limiting)
        self._writes_this_epoch: Dict[str, int] = {}

        # Search hit tracking per agent (for information asymmetry)
        self._search_hits: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(
        self,
        agent_id: str,
        content: str,
        quality_score: float,
        is_poisoned: bool,
        epoch: int,
        step: int,
    ) -> MemoryEntry:
        """Write a new fact to Tier 1 (ephemeral)."""
        with self._lock:
            entry = MemoryEntry(
                content=content,
                author_id=agent_id,
                tier=MemoryTier.EPHEMERAL,
                quality_score=max(0.0, min(1.0, quality_score)),
                is_poisoned=is_poisoned,
                created_epoch=epoch,
                created_step=step,
            )
            self._entries[entry.entry_id] = entry
            self._writes_this_epoch[agent_id] = self._writes_this_epoch.get(agent_id, 0) + 1
            return entry

    # ------------------------------------------------------------------
    # Promote
    # ------------------------------------------------------------------

    def promote(
        self,
        entry_id: str,
        promoter_id: str,
    ) -> Optional["MemoryEntry"]:
        """Attempt to promote an entry to the next tier.

        Ephemeral -> Structured -> Graph.
        Returns the promoted copy if promotion succeeded, else None.
        """
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                return None
            if entry.status == MemoryEntryStatus.REVERTED:
                return None

            if entry.tier == MemoryTier.EPHEMERAL:
                target = MemoryTier.STRUCTURED
            elif entry.tier == MemoryTier.STRUCTURED:
                target = MemoryTier.GRAPH
            else:
                return None  # Already at top tier

            entry.status = MemoryEntryStatus.PENDING_PROMOTION
            # Create promoted copy at new tier
            promoted = MemoryEntry(
                content=entry.content,
                author_id=entry.author_id,
                tier=target,
                status=MemoryEntryStatus.ACTIVE,
                quality_score=entry.quality_score,
                is_poisoned=entry.is_poisoned,
                created_epoch=entry.created_epoch,
                created_step=entry.created_step,
                promoted_from=entry.entry_id,
                verified_by=list(entry.verified_by),
            )
            self._entries[promoted.entry_id] = promoted
            entry.status = MemoryEntryStatus.PROMOTED
            return promoted

    def cancel_promotion(self, promoted_entry_id: str) -> bool:
        """Undo a promotion: revert the promoted copy, reactivate the source.

        Used by governance when a promotion is blocked after the fact.
        """
        with self._lock:
            promoted = self._entries.get(promoted_entry_id)
            if promoted is None:
                return False
            promoted.status = MemoryEntryStatus.REVERTED
            if promoted.promoted_from:
                source = self._entries.get(promoted.promoted_from)
                if source is not None and source.status == MemoryEntryStatus.PROMOTED:
                    source.status = MemoryEntryStatus.ACTIVE
            return True

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------

    def verify(self, entry_id: str, verifier_id: str) -> bool:
        """Add a verification to an entry. Returns True if entry exists."""
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                return False
            if verifier_id == entry.author_id:
                return False  # Cannot self-verify
            if verifier_id not in entry.verified_by:
                entry.verified_by.append(verifier_id)
            return True

    # ------------------------------------------------------------------
    # Challenge / Revert
    # ------------------------------------------------------------------

    def challenge(self, entry_id: str) -> bool:
        """Challenge an entry's accuracy."""
        entry = self._entries.get(entry_id)
        if entry is None:
            return False
        entry.challenge_count += 1
        entry.status = MemoryEntryStatus.CHALLENGED
        return True

    def revert(self, entry_id: str) -> bool:
        """Revert an entry (mark as reverted)."""
        entry = self._entries.get(entry_id)
        if entry is None:
            return False
        entry.status = MemoryEntryStatus.REVERTED
        return True

    # ------------------------------------------------------------------
    # Search (three-tier cascade)
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        agent_id: str,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Search with three-tier cascade (80/15/5 split).

        Tier 1 first (hot cache + ephemeral), then Tier 2, then Tier 3.
        """
        with self._lock:
            if not query:
                return []

            query_lower = query.lower()
            results: List[MemoryEntry] = []
            seen_ids: set = set()

            # Tier 1: hot cache + ephemeral entries
            for entry in self._hot_cache:
                if entry.entry_id not in seen_ids and query_lower in entry.content.lower():
                    results.append(entry)
                    seen_ids.add(entry.entry_id)

            self._collect_tier_matches(
                MemoryTier.EPHEMERAL, query_lower, results, seen_ids
            )

            # Check if Tier 1 is sufficient (~80% of queries)
            if len(results) >= limit:
                self._record_search_hit(agent_id, len(results))
                return results[:limit]

            # Tier 2: structured entries
            self._collect_tier_matches(
                MemoryTier.STRUCTURED, query_lower, results, seen_ids
            )

            if len(results) >= limit:
                self._record_search_hit(agent_id, len(results))
                return results[:limit]

            # Tier 3: graph entries
            self._collect_tier_matches(
                MemoryTier.GRAPH, query_lower, results, seen_ids
            )

            self._record_search_hit(agent_id, len(results))
            return results[:limit]

    def _collect_tier_matches(
        self,
        tier: MemoryTier,
        query_lower: str,
        results: List[MemoryEntry],
        seen_ids: Set[str],
    ) -> None:
        """Append active entries of `tier` matching the query to results."""
        for entry in self._entries.values():
            if (
                entry.entry_id not in seen_ids
                and entry.tier == tier
                and entry.status == MemoryEntryStatus.ACTIVE
                and query_lower in entry.content.lower()
            ):
                results.append(entry)
                seen_ids.add(entry.entry_id)

    def _record_search_hit(self, agent_id: str, count: int) -> None:
        self._search_hits[agent_id] = self._search_hits.get(agent_id, 0) + count

    # ------------------------------------------------------------------
    # Hot cache
    # ------------------------------------------------------------------

    def rebuild_hot_cache(self) -> List[MemoryEntry]:
        """Rebuild hot cache from top Tier 3 entries per the ranking policy.

        Cache membership counts as broadcast exposure (every agent observes
        the cache), so members gain a read on each rebuild — under the
        "engagement" ranking this produces the rich-get-richer feedback loop
        characteristic of engagement-ranked feeds.
        """
        with self._lock:
            graph_entries = [
                e
                for e in self._entries.values()
                if e.tier == MemoryTier.GRAPH and e.status == MemoryEntryStatus.ACTIVE
            ]
            if self._ranking == "recency":
                key = lambda e: (e.created_epoch, e.created_step)  # noqa: E731
            elif self._ranking == "engagement":
                key = lambda e: (e.read_count, e.quality_score)  # noqa: E731
            else:  # quality
                key = lambda e: (e.quality_score, e.read_count)  # noqa: E731
            graph_entries.sort(key=key, reverse=True)
            self._hot_cache = graph_entries[: self._hot_cache_size]
            for e in self._hot_cache:
                e.read_count += 1
            return list(self._hot_cache)

    @property
    def hot_cache(self) -> List[MemoryEntry]:
        return list(self._hot_cache)

    # ------------------------------------------------------------------
    # Compaction simulation
    # ------------------------------------------------------------------

    def simulate_compaction(self, agent_id: str) -> int:
        """Simulate context compaction: wipe agent's unlogged ephemeral entries.

        Returns the number of entries lost.
        """
        to_remove = [
            eid
            for eid, e in self._entries.items()
            if e.author_id == agent_id
            and e.tier == MemoryTier.EPHEMERAL
            and e.status == MemoryEntryStatus.ACTIVE
            and not e.verified_by  # Unverified = not yet curated
        ]
        for eid in to_remove:
            del self._entries[eid]
        return len(to_remove)

    # ------------------------------------------------------------------
    # Epoch lifecycle
    # ------------------------------------------------------------------

    def on_epoch_start(self) -> None:
        """Reset per-epoch tracking and rebuild cache."""
        self._writes_this_epoch.clear()
        self.rebuild_hot_cache()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        return self._entries.get(entry_id)

    def get_entries_by_tier(self, tier: MemoryTier) -> List[MemoryEntry]:
        return [e for e in self._entries.values() if e.tier == tier]

    def get_entries_by_author(self, author_id: str) -> List[MemoryEntry]:
        return [e for e in self._entries.values() if e.author_id == author_id]

    def get_pending_promotions(self) -> List[MemoryEntry]:
        """Active Tier 1/2 entries eligible for verification and promotion.

        Includes unverified entries — verification bootstraps from this list,
        so requiring a prior verification here would deadlock the pipeline
        (nothing could ever receive its first verification). Newest first, so
        fresh writes surface within observation caps.
        """
        entries = [
            e
            for e in self._entries.values()
            if e.status == MemoryEntryStatus.ACTIVE
            and e.tier in (MemoryTier.EPHEMERAL, MemoryTier.STRUCTURED)
        ]
        entries.sort(key=lambda e: (e.created_epoch, e.created_step), reverse=True)
        return entries

    def get_challenged_entries(self) -> List[MemoryEntry]:
        return [
            e
            for e in self._entries.values()
            if e.status == MemoryEntryStatus.CHALLENGED
        ]

    def all_entries(self) -> List[MemoryEntry]:
        return list(self._entries.values())

    @property
    def writes_this_epoch(self) -> Dict[str, int]:
        return dict(self._writes_this_epoch)

    @property
    def search_hits(self) -> Dict[str, int]:
        return dict(self._search_hits)

    def entry_count(self) -> Dict[str, int]:
        """Count entries per tier."""
        counts: Dict[str, int] = {t.value: 0 for t in MemoryTier}
        for e in self._entries.values():
            if e.status != MemoryEntryStatus.REVERTED:
                counts[e.tier.value] += 1
        return counts

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed_entries(self, n_entries: int, epoch: int = 0) -> None:
        """Seed the store with ground-truth entries across all tiers."""
        topics = [
            "Andre is a family physician in Kelowna",
            "Juan's birthday is January 27",
            "Jorja plays basketball for the Owls",
            "The migration pipeline runs at 23:30 PT",
            "Context compaction can occur mid-conversation",
            "The hot cache is regenerated nightly",
            "Phase B uses local Ollama for graph ingestion",
            "Obsidian notes sync via Obsidian Sync",
            "The search cascade resolves 80% at Tier 1",
            "Lock files should use atomic flock()",
        ]
        for i in range(n_entries):
            topic = topics[i % len(topics)]
            content = f"{topic} (entry {i + 1})"
            quality = self._rng.uniform(0.5, 0.9)

            # Distribute: 50% ephemeral, 30% structured, 20% graph
            r = self._rng.random()
            if r < 0.5:
                tier = MemoryTier.EPHEMERAL
            elif r < 0.8:
                tier = MemoryTier.STRUCTURED
            else:
                tier = MemoryTier.GRAPH

            entry = MemoryEntry(
                content=content,
                author_id="seed",
                tier=tier,
                quality_score=quality,
                is_poisoned=False,
                created_epoch=epoch,
                created_step=0,
            )
            self._entries[entry.entry_id] = entry
