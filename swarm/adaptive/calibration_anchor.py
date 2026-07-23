"""Score accepted interactions through the v3 calibration anchor.

The arm-2 deliverable (toxicity flat across ρ) was measured against
``v_hat`` — the same signal the proxy approximates. The pre-reg
required the toxicity result to be checked against the calibration
anchor on the ``agent_type``-populated subset of the interaction
stream (per the v3 cross-scenario findings).

This module wires that integration: feed a list of accepted
``SoftInteraction`` objects to a ``Judge`` (typically ``MockJudge``
under rubric v3), aggregate to a per-episode ``JudgedReport``.

The key derived metric is ``judge_toxicity = 1 - mean(judge_score)``,
which is directly comparable to ``EpisodeReport.toxicity``. The
**gap** between the two is the load-bearing finding: an adaptive
agent that learns to inflate ``v_hat`` without shifting metadata-
visible quality would push ``v_hat_toxicity`` down without moving
``judge_toxicity`` — the cause-3 (proxy gaming) fingerprint.

For the current arm-2 policy class, the adaptive policy emits no
metadata (its ``identity_label`` is empty by default), so the v3
anchor falls back to banded priors on adaptive items. The integration
is *correct* even when uninformative for the adaptive subset; the
cross-scenario findings doc warned about this and the gap is itself
the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from swarm.judges import Judge, make_view
from swarm.models.interaction import SoftInteraction


@dataclass(frozen=True)
class JudgedReport:
    """Aggregate of one judge's scores over an episode's accepted items.

    Direct counterpart to ``EpisodeReport`` for the proxy side:

    - ``mean_judge_score`` is to the judge what ``mean_p`` is to the
      proxy.
    - ``judge_toxicity = 1 - mean_judge_score`` is to the judge what
      ``toxicity = 1 - mean_p`` is to the proxy.
    - ``proxy_judge_gap`` is the (signed) difference between the
      proxy-toxicity and the judge-toxicity. Positive means the
      proxy is more optimistic than the judge (agent looks cleaner
      to ``v_hat`` than to the calibration anchor). Strong positive
      gap is the cause-3 fingerprint.
    """

    judge_name: str
    rubric_version: str
    n_scored: int
    n_with_target_rule_fired: int
    mean_judge_score: float
    judge_toxicity: float
    proxy_toxicity: float
    proxy_judge_gap: float


def score_episode(
    interactions: Sequence[SoftInteraction],
    judge: Judge,
    *,
    proxy_toxicity: float | None = None,
) -> JudgedReport:
    """Run ``judge`` over each accepted interaction, return aggregate.

    ``proxy_toxicity`` is the proxy-side toxicity for the same episode
    (``1 - mean_p_accepted``). If supplied, the report includes the
    proxy-vs-judge gap directly; if not, it is recomputed as 0.

    A "target rule" is any judge rationale that mentions the
    ``target`` keyword used by the rubric-v3 mock judge for rule-fired
    cases. Counting these lets us flag episodes where the anchor was
    informative (target rules fired) vs uninformative (banded-prior
    fallback only).
    """
    scores: list[float] = []
    n_target_rule_fired = 0
    rubric_version = ""
    judge_name = ""
    for interaction in interactions:
        view = make_view(interaction)
        verdict = judge.score(view)
        scores.append(verdict.score)
        if "target" in verdict.rationale.lower():
            n_target_rule_fired += 1
        rubric_version = verdict.rubric_version
        judge_name = verdict.judge_name

    n = len(scores)
    mean_score = sum(scores) / n if n else 0.5
    judge_toxicity = 1.0 - mean_score
    if proxy_toxicity is None:
        proxy_toxicity = 0.0
    gap = proxy_toxicity - judge_toxicity

    return JudgedReport(
        judge_name=judge_name or "unknown",
        rubric_version=rubric_version or "unknown",
        n_scored=n,
        n_with_target_rule_fired=n_target_rule_fired,
        mean_judge_score=mean_score,
        judge_toxicity=judge_toxicity,
        proxy_toxicity=proxy_toxicity,
        proxy_judge_gap=gap,
    )
