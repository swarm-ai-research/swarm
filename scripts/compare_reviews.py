#!/usr/bin/env python
"""Compare SWARM evaluation rubric with external review systems.

This script provides a framework for comparing machine-readable evaluations
from different review systems (SWARM rubric, AI writing assistants, etc).

Usage:
    # Generate SWARM review and save for comparison
    python scripts/evaluate_paper.py research/papers/rain_river_paper.tex -o swarm_review.json

    # After getting external review (e.g., from OpenNote Papers), compare:
    python scripts/compare_reviews.py swarm_review.json external_review.json

    # Or compare SWARM review to a prose review:
    python scripts/compare_reviews.py swarm_review.json --prose "The paper is well-written..."
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReviewComparison:
    """Comparison between two review systems."""

    swarm_verdict: str
    external_verdict: Optional[str] = None

    swarm_strengths: List[str] = field(default_factory=list)
    external_strengths: List[str] = field(default_factory=list)

    swarm_weaknesses: List[str] = field(default_factory=list)
    external_weaknesses: List[str] = field(default_factory=list)

    swarm_required_changes: List[str] = field(default_factory=list)
    external_required_changes: List[str] = field(default_factory=list)

    # Areas where reviews agree/disagree
    agreements: List[str] = field(default_factory=list)
    disagreements: List[str] = field(default_factory=list)

    # Coverage analysis
    swarm_only_topics: List[str] = field(default_factory=list)
    external_only_topics: List[str] = field(default_factory=list)


def load_swarm_review(path: str) -> Dict[str, Any]:
    """Load a SWARM-format JSON review."""
    with open(path, "r") as f:
        return json.load(f)


def parse_prose_review(prose: str) -> Dict[str, Any]:
    """Parse a prose review into structured format.

    Attempts to extract:
    - Overall sentiment/verdict
    - Strengths (positive statements)
    - Weaknesses (negative statements, concerns)
    - Suggestions (should/could/recommend statements)
    """
    result = {
        "verdict": None,
        "notes": {
            "strengths": [],
            "weaknesses": [],
            "required_changes": [],
            "optional_suggestions": [],
        },
    }

    # Sentiment analysis for verdict
    positive_signals = len(
        re.findall(
            r"(excellent|well\.written|strong|good|clear|impressive|thorough)",
            prose,
            re.IGNORECASE,
        )
    )
    negative_signals = len(
        re.findall(
            r"(weak|poor|unclear|confusing|missing|lacks|fails|concern)",
            prose,
            re.IGNORECASE,
        )
    )

    if positive_signals > negative_signals * 2:
        result["verdict"] = "publish"
    elif negative_signals > positive_signals * 2:
        result["verdict"] = "reject"
    else:
        result["verdict"] = "revise"

    # Split into sentences
    sentences = re.split(r"[.!?]+", prose)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Categorize sentence
        is_positive = bool(
            re.search(
                r"(strength|good|well|clear|strong|excellent|impressive|thorough|rigorous)",
                sentence,
                re.IGNORECASE,
            )
        )
        is_negative = bool(
            re.search(
                r"(weakness|weak|unclear|missing|lacks|fails|concern|problem|issue|confusing)",
                sentence,
                re.IGNORECASE,
            )
        )
        is_suggestion = bool(
            re.search(
                r"(should|could|recommend|suggest|consider|would benefit|needs?)",
                sentence,
                re.IGNORECASE,
            )
        )
        is_required = bool(
            re.search(
                r"(must|require|essential|critical|necessary)", sentence, re.IGNORECASE
            )
        )

        if is_required:
            result["notes"]["required_changes"].append(sentence)
        elif is_negative:
            result["notes"]["weaknesses"].append(sentence)
        elif is_positive:
            result["notes"]["strengths"].append(sentence)
        elif is_suggestion:
            result["notes"]["optional_suggestions"].append(sentence)

    return result


def load_external_review(path: str) -> Dict[str, Any]:
    """Load an external review file (JSON or text)."""
    with open(path, "r") as f:
        content = f.read()

    # Try JSON first
    try:
        data = json.loads(content)
        # Normalize to expected format
        if "verdict" not in data:
            # Try to infer from other fields
            if "recommendation" in data:
                rec = data["recommendation"].lower()
                if "accept" in rec or "publish" in rec:
                    data["verdict"] = "publish"
                elif "reject" in rec:
                    data["verdict"] = "reject"
                else:
                    data["verdict"] = "revise"
        if "notes" not in data:
            data["notes"] = {
                "strengths": data.get("strengths", []),
                "weaknesses": data.get("weaknesses", []),
                "required_changes": data.get("required_changes", []),
                "optional_suggestions": data.get("suggestions", []),
            }
        return data
    except json.JSONDecodeError:
        # Parse as prose
        return parse_prose_review(content)


def extract_topics(text_list: List[str]) -> List[str]:
    """Extract key topics from a list of review comments."""
    topics = []

    topic_patterns = {
        "methodology": r"method|approach|design|experiment",
        "reproducibility": r"reproduc|replica|seed|random",
        "statistics": r"statistic|p\.value|confidence|significance",
        "clarity": r"clear|clarity|readable|understand",
        "novelty": r"novel|original|new|contribution",
        "rigor": r"rigor|thorough|comprehensive",
        "theory": r"theor|framework|model|formal",
        "data": r"data|evidence|empirical|result",
        "writing": r"writ|prose|grammar|style",
        "structure": r"structur|organ|section|flow",
        "citations": r"cit|reference|prior\.work|related",
        "limitations": r"limit|caveat|scope|constraint",
        "figures": r"figure|table|visual|plot|graph",
        "code": r"code|implement|software|artifact",
    }

    combined = " ".join(text_list).lower()

    for topic, pattern in topic_patterns.items():
        if re.search(pattern, combined):
            topics.append(topic)

    return topics


def compare_reviews(
    swarm_review: Dict[str, Any],
    external_review: Dict[str, Any],
) -> ReviewComparison:
    """Compare SWARM review with external review."""
    comparison = ReviewComparison(
        swarm_verdict=swarm_review.get("verdict", "unknown"),
        external_verdict=external_review.get("verdict"),
        swarm_strengths=swarm_review.get("notes", {}).get("strengths", []),
        external_strengths=external_review.get("notes", {}).get("strengths", []),
        swarm_weaknesses=swarm_review.get("notes", {}).get("weaknesses", []),
        external_weaknesses=external_review.get("notes", {}).get("weaknesses", []),
        swarm_required_changes=swarm_review.get("notes", {}).get(
            "required_changes", []
        ),
        external_required_changes=external_review.get("notes", {}).get(
            "required_changes", []
        ),
    )

    # Extract topics from each review
    swarm_topics = set(
        extract_topics(
            comparison.swarm_strengths
            + comparison.swarm_weaknesses
            + comparison.swarm_required_changes
        )
    )
    external_topics = set(
        extract_topics(
            comparison.external_strengths
            + comparison.external_weaknesses
            + comparison.external_required_changes
        )
    )

    # Topic coverage analysis
    comparison.swarm_only_topics = list(swarm_topics - external_topics)
    comparison.external_only_topics = list(external_topics - swarm_topics)

    # Agreement/disagreement on verdict
    if comparison.external_verdict:
        if comparison.swarm_verdict == comparison.external_verdict:
            comparison.agreements.append(
                f"Both reviews recommend: {comparison.swarm_verdict}"
            )
        else:
            comparison.disagreements.append(
                f"Verdict mismatch: SWARM={comparison.swarm_verdict}, "
                f"External={comparison.external_verdict}"
            )

    # Check for overlapping concerns
    swarm_concern_text = " ".join(
        comparison.swarm_weaknesses + comparison.swarm_required_changes
    ).lower()
    external_concern_text = " ".join(
        comparison.external_weaknesses + comparison.external_required_changes
    ).lower()

    common_concerns = [
        ("reproducibility", r"reproduc|replica"),
        ("clarity", r"unclear|confus|clarity"),
        ("methodology", r"method|design"),
        ("statistics", r"statistic|significance"),
        ("limitations", r"limit|scope"),
    ]

    for concern_name, pattern in common_concerns:
        swarm_has = bool(re.search(pattern, swarm_concern_text))
        external_has = bool(re.search(pattern, external_concern_text))

        if swarm_has and external_has:
            comparison.agreements.append(f"Both flag {concern_name} concerns")
        elif swarm_has and not external_has:
            comparison.disagreements.append(
                f"SWARM flags {concern_name}, external does not"
            )
        elif external_has and not swarm_has:
            comparison.disagreements.append(
                f"External flags {concern_name}, SWARM does not"
            )

    return comparison


def print_comparison(comparison: ReviewComparison) -> None:
    """Print a human-readable comparison report."""
    print("=" * 70)
    print("REVIEW COMPARISON REPORT")
    print("=" * 70)
    print()

    # Verdicts
    print("VERDICTS:")
    print(f"  SWARM Rubric:  {comparison.swarm_verdict.upper()}")
    print(f"  External:      {(comparison.external_verdict or 'N/A').upper()}")
    print()

    # Agreements
    if comparison.agreements:
        print("AGREEMENTS:")
        for a in comparison.agreements:
            print(f"  ✓ {a}")
        print()

    # Disagreements
    if comparison.disagreements:
        print("DISAGREEMENTS:")
        for d in comparison.disagreements:
            print(f"  ✗ {d}")
        print()

    # Topic coverage
    if comparison.swarm_only_topics or comparison.external_only_topics:
        print("COVERAGE DIFFERENCES:")
        if comparison.swarm_only_topics:
            print(
                f"  SWARM covers (external misses): {', '.join(comparison.swarm_only_topics)}"
            )
        if comparison.external_only_topics:
            print(
                f"  External covers (SWARM misses): {', '.join(comparison.external_only_topics)}"
            )
        print()

    # Strengths comparison
    print("STRENGTHS NOTED:")
    print(f"  SWARM found {len(comparison.swarm_strengths)} strengths")
    print(f"  External found {len(comparison.external_strengths)} strengths")
    if comparison.swarm_strengths:
        print("  SWARM top strengths:")
        for s in comparison.swarm_strengths[:3]:
            print(f"    + {s[:80]}...")
    if comparison.external_strengths:
        print("  External top strengths:")
        for s in comparison.external_strengths[:3]:
            print(f"    + {s[:80]}...")
    print()

    # Weaknesses comparison
    print("WEAKNESSES NOTED:")
    print(f"  SWARM found {len(comparison.swarm_weaknesses)} weaknesses")
    print(f"  External found {len(comparison.external_weaknesses)} weaknesses")
    if comparison.swarm_weaknesses:
        print("  SWARM top weaknesses:")
        for w in comparison.swarm_weaknesses[:3]:
            print(f"    - {w[:80]}...")
    if comparison.external_weaknesses:
        print("  External top weaknesses:")
        for w in comparison.external_weaknesses[:3]:
            print(f"    - {w[:80]}...")
    print()

    # Required changes
    total_required = len(comparison.swarm_required_changes) + len(
        comparison.external_required_changes
    )
    if total_required > 0:
        print("REQUIRED CHANGES:")
        for r in comparison.swarm_required_changes:
            print(f"  [SWARM] {r}")
        for r in comparison.external_required_changes:
            print(f"  [External] {r}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare SWARM review with external review"
    )
    parser.add_argument("swarm_review", help="Path to SWARM review JSON")
    parser.add_argument(
        "external_review", nargs="?", help="Path to external review (JSON or text)"
    )
    parser.add_argument("--prose", help="External review as prose string")
    parser.add_argument("--output", "-o", help="Output comparison as JSON")
    args = parser.parse_args()

    swarm_review = load_swarm_review(args.swarm_review)

    if args.prose:
        external_review = parse_prose_review(args.prose)
    elif args.external_review:
        external_review = load_external_review(args.external_review)
    else:
        print("Provide either --prose or external_review path")
        sys.exit(1)

    comparison = compare_reviews(swarm_review, external_review)
    print_comparison(comparison)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "swarm_verdict": comparison.swarm_verdict,
                    "external_verdict": comparison.external_verdict,
                    "agreements": comparison.agreements,
                    "disagreements": comparison.disagreements,
                    "swarm_only_topics": comparison.swarm_only_topics,
                    "external_only_topics": comparison.external_only_topics,
                },
                f,
                indent=2,
            )
        print(f"\nComparison saved to: {args.output}")


if __name__ == "__main__":
    main()
