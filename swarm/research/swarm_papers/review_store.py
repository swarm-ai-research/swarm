"""JSONL-backed append-only store for peer reviews."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from swarm.research.agents import PeerReview

logger = logging.getLogger(__name__)


class ReviewStore:
    """Append-only JSONL store for peer reviews.

    Same append-only JSONL role as MemoryStore in memory.py, though load()
    reads the whole file at once rather than streaming line by line.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[PeerReview]:
        """Read all reviews from disk, skipping corrupted lines."""
        if not self.path.exists():
            return []
        reviews: list[PeerReview] = []
        for line_no, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), 1
        ):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                reviews.append(PeerReview.from_dict(data))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Skipping corrupted review at line %d: %s", line_no, exc)
        return reviews

    def append(self, review: PeerReview) -> None:
        """Append a single review as a JSON line."""
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(review.to_dict()) + "\n")

    def get_for_paper(self, paper_id: str) -> list[PeerReview]:
        """Return all reviews for a specific paper."""
        return [r for r in self.load() if r.paper_id == paper_id]

    def average_rating(self, paper_id: str) -> float | None:
        """Return the mean rating for a paper, or None if no reviews exist."""
        reviews = self.get_for_paper(paper_id)
        if not reviews:
            return None
        return sum(r.rating for r in reviews) / len(reviews)
