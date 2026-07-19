"""Prompt templates for the council protocol stages."""

from typing import Dict, List, Optional, Tuple


def build_ranking_prompt(
    query: str,
    anon_responses: Dict[str, str],
) -> Tuple[str, str]:
    """Build prompts for Stage 2 (peer ranking).

    Args:
        query: The original query that members responded to
        anon_responses: Dict of label -> response text

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = (
        "You are a fair and impartial judge. You will be shown responses from "
        "multiple anonymous participants to the same query. Rank them from best "
        "to worst based on quality, accuracy, and usefulness. "
        "Output ONLY the ranking as a numbered list, e.g.:\n"
        "1. A\n2. B\n3. C"
    )

    response_text = ""
    for label in sorted(anon_responses.keys()):
        response_text += f"\n--- Response {label} ---\n{anon_responses[label]}\n"

    user_prompt = (
        f"Original query: {query}\n\n"
        f"Responses:{response_text}\n"
        f"Rank these {len(anon_responses)} responses from best to worst. "
        f"Output ONLY the ranking."
    )

    return system_prompt, user_prompt


def build_synthesis_prompt(
    query: str,
    responses: Dict[str, str],
    rankings: Optional[Dict[str, List[str]]] = None,
    aggregate: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """Build prompts for Stage 3 (chairman synthesis).

    Args:
        query: The original query
        responses: Dict of member_id -> response text
        rankings: Optional dict of ranker_id -> ranking labels
        aggregate: Optional aggregate ranking of member_ids (best first)

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = (
        "You are the chairman of a council of AI experts. You have received "
        "responses from multiple council members and their peer rankings. "
        "Synthesize the best answer by drawing on the strongest points from "
        "each response. When resolving disagreements, prefer the majority view "
        "or the highest-ranked response, but use your judgment. Be concise and actionable."
    )

    response_text = ""
    for member_id, text in responses.items():
        response_text += f"\n--- {member_id} ---\n{text}\n"

    user_prompt = f"Original query: {query}\n\nCouncil responses:{response_text}\n"

    if aggregate:
        user_prompt += (
            f"Aggregate ranking (best first): {', '.join(aggregate)}\n\n"
        )

    user_prompt += (
        "Synthesize the best answer from these responses. "
        "If they agree, consolidate. If they disagree, explain the consensus."
    )

    return system_prompt, user_prompt
