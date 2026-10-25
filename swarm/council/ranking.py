"""Ranking utilities for the council protocol.

Stage 2: Each member ranks the anonymized responses from other members.
Rankings are aggregated using Borda count with optional member weights.
"""

import random
import re
from typing import Dict, List, Optional, Tuple


def anonymize_responses(
    responses: Dict[str, str],
    seed: Optional[int] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Anonymize member responses with shuffled labels.

    Args:
        responses: Dict of member_id -> response text
        seed: Random seed for deterministic shuffling

    Returns:
        Tuple of (anon_map, anon_responses) where:
            anon_map: label -> member_id mapping
            anon_responses: label -> response text
    """
    rng = random.Random(seed)
    member_ids = list(responses.keys())
    rng.shuffle(member_ids)

    labels = [chr(65 + i) for i in range(len(member_ids))]  # A, B, C, ...
    anon_map: Dict[str, str] = {}
    anon_responses: Dict[str, str] = {}

    for label, member_id in zip(labels, member_ids, strict=True):
        anon_map[label] = member_id
        anon_responses[label] = responses[member_id]

    return anon_map, anon_responses


def parse_rankings(text: str, n_responses: int) -> Optional[List[str]]:
    """Parse a ranking from LLM text output.

    Tolerates reasoning before the ranking: text after the last "RANKING:"
    marker is parsed, and the last n numbered labels win. Tries structured
    format first (e.g., "1. A\\n2. B\\n3. C"),
    then falls back to regex extraction.

    Args:
        text: LLM response text containing rankings
        n_responses: Expected number of labels

    Returns:
        Ordered list of labels (best first), or None if parsing fails
    """
    labels = [chr(65 + i) for i in range(n_responses)]

    # Reasoning may precede the ranking; read only what follows the last
    # "RANKING:" marker when there is one.
    markers = list(re.finditer(r"ranking\W*:", text, re.IGNORECASE))
    if markers:
        text = text[markers[-1].end():]

    # Try structured format: "1. A", "2. Response B", etc. Take the last
    # n matches so numbered reasoning above the list does not count.
    structured = re.findall(r"\d+[.)]\s*\**(?:Response\s+)?([A-Z])\b", text)
    tail = structured[-n_responses:]
    if len(tail) == n_responses and set(tail) == set(labels):
        return tail

    # Try comma-separated: "A, B, C" or "A > B > C"
    for sep in [r"\s*>\s*", r"\s*,\s*"]:
        parts = re.split(sep, text.strip())
        cleaned = [p.strip().upper() for p in parts if p.strip().upper() in labels]
        if len(cleaned) == n_responses and set(cleaned) == set(labels):
            return cleaned

    # Try extracting all single-letter labels in order
    found = re.findall(r"\b([A-Z])\b", text)
    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for f in found:
        if f in labels and f not in seen:
            seen.add(f)
            ordered.append(f)
    if len(ordered) == n_responses:
        return ordered

    return None


def aggregate_rankings(
    rankings: Dict[str, List[str]],
    anon_map: Dict[str, str],
    weights: Optional[Dict[str, float]] = None,
) -> List[str]:
    """Aggregate rankings using weighted Borda count.

    Args:
        rankings: Dict of ranker_id -> ordered list of labels (best first)
        anon_map: label -> member_id mapping
        weights: Optional dict of ranker_id -> weight (defaults to equal)

    Returns:
        List of member_ids sorted by aggregate score (best first)
    """
    n = len(anon_map)
    scores: Dict[str, float] = dict.fromkeys(anon_map, 0.0)

    for ranker_id, ranking in rankings.items():
        w = weights.get(ranker_id, 1.0) if weights else 1.0
        for position, label in enumerate(ranking):
            # Borda: top gets n-1 points, second gets n-2, etc.
            scores[label] += w * (n - 1 - position)

    # Sort by score descending, then by label for determinism
    sorted_labels = sorted(scores, key=lambda lb: (-scores[lb], lb))
    return [anon_map[label] for label in sorted_labels]
