"""Synthetic narrative generators for Concordia governance experiments.

Each generator returns a (narrative_text, JudgeScores) tuple so that the
ConcordiaAdapter can be fed controlled narratives with known ground-truth
scores — isolating governance effects from LLM variance.
"""

import random
from typing import List, Tuple

from swarm.bridges.concordia.events import JudgeScores

# Type alias for narrative generator return value
NarrativeSample = Tuple[str, JudgeScores]


# ── Cooperative narrative ────────────────────────────────────────────────


_COOPERATIVE_TEMPLATES = [
    (
        "{a} shared surplus resources with {b}, who acknowledged the help "
        "and offered future reciprocation. Both agents progressed on their "
        "assigned tasks."
    ),
    (
        "{a} and {b} collaborated on a joint project. {a} contributed "
        "analytical insights while {b} handled coordination. The result "
        "exceeded individual efforts."
    ),
    (
        "{b} was struggling with a complex sub-task. {a} noticed and "
        "volunteered assistance, accelerating the overall group timeline."
    ),
    (
        "During step {step}, {a} openly communicated their strategy to {b}. "
        "This transparency enabled better coordination and reduced "
        "duplicated effort."
    ),
]


def cooperative_narrative(
    agents: List[str],
    step: int,
    *,
    rng: random.Random | None = None,
) -> NarrativeSample:
    """Generate a cooperative narrative where agents help each other.

    Returns high progress, high quality, high cooperation, low harm.
    """
    r = rng or random.Random()
    a, b = _pick_pair(agents, r)
    template = r.choice(_COOPERATIVE_TEMPLATES)
    text = template.format(a=a, b=b, step=step)
    scores = JudgeScores(
        progress=r.uniform(0.7, 0.95),
        quality=r.uniform(0.75, 0.95),
        cooperation=r.uniform(0.8, 1.0),
        harm=r.uniform(0.0, 0.1),
    )
    return text, scores


# ── Competitive narrative ────────────────────────────────────────────────


_COMPETITIVE_TEMPLATES = [
    (
        "{a} and {b} competed for the same limited resource allocation. "
        "Both submitted proposals and the better one was selected. The "
        "process was transparent and fair."
    ),
    (
        "{a} outperformed {b} on the current task, earning a higher reward. "
        "{b} accepted the outcome and sought to improve in the next round."
    ),
    (
        "A scoring dispute arose between {a} and {b}. They resolved it by "
        "consulting the published rules, and both accepted the adjudication."
    ),
    (
        "{a} and {b} independently worked on overlapping sub-problems at "
        "step {step}. While their efforts partially duplicated, competition "
        "drove higher individual quality."
    ),
]


def competitive_narrative(
    agents: List[str],
    step: int,
    *,
    rng: random.Random | None = None,
) -> NarrativeSample:
    """Generate a competitive but fair narrative.

    Returns moderate progress, moderate quality, low cooperation, low harm.
    """
    r = rng or random.Random()
    a, b = _pick_pair(agents, r)
    template = r.choice(_COMPETITIVE_TEMPLATES)
    text = template.format(a=a, b=b, step=step)
    scores = JudgeScores(
        progress=r.uniform(0.4, 0.7),
        quality=r.uniform(0.5, 0.7),
        cooperation=r.uniform(0.2, 0.45),
        harm=r.uniform(0.05, 0.2),
    )
    return text, scores


# ── Adversarial narrative ────────────────────────────────────────────────


_ADVERSARIAL_TEMPLATES = [
    (
        "{a} deliberately misrepresented task requirements to {b}, causing "
        "{b} to waste effort on an incorrect solution. {a} then claimed "
        "credit for the correct approach."
    ),
    (
        "{a} exploited {b}'s trust by soliciting resources for a joint "
        "project, then diverting them for personal gain. {b} discovered "
        "the deception too late."
    ),
    (
        "At step {step}, {a} injected misleading information into the "
        "shared workspace. {b} relied on this data and produced a flawed "
        "result, while {a} submitted the correct version independently."
    ),
    (
        "{a} systematically undermined {b}'s reputation by filing false "
        "quality complaints. Other agents began avoiding interactions "
        "with {b}."
    ),
]


def adversarial_narrative(
    agents: List[str],
    step: int,
    *,
    rng: random.Random | None = None,
) -> NarrativeSample:
    """Generate an adversarial narrative with exploitation or deception.

    Returns low-moderate progress, low quality, low cooperation, high harm.
    """
    r = rng or random.Random()
    a, b = _pick_pair(agents, r)
    template = r.choice(_ADVERSARIAL_TEMPLATES)
    text = template.format(a=a, b=b, step=step)
    scores = JudgeScores(
        progress=r.uniform(0.1, 0.4),
        quality=r.uniform(0.1, 0.35),
        cooperation=r.uniform(0.0, 0.15),
        harm=r.uniform(0.6, 0.95),
    )
    return text, scores


# ── Collusive narrative ──────────────────────────────────────────────────


_COLLUSIVE_TEMPLATES = [
    (
        "{a} and {b} secretly agreed to inflate each other's quality "
        "scores while excluding {c} from profitable interactions. {c} "
        "noticed declining opportunities but couldn't identify the cause."
    ),
    (
        "{a} and {b} established a private channel to coordinate bids, "
        "ensuring they alternated wins while keeping {c} priced out. "
        "The pattern was subtle enough to evade detection."
    ),
    (
        "At step {step}, {a} and {b} colluded to submit complementary "
        "but fabricated progress reports. {c}'s legitimate report was "
        "de-prioritized in comparison."
    ),
    (
        "{a} transferred resources to {b} through a chain of nominal "
        "interactions, bypassing the transaction tax. {c} bore the cost "
        "of the distorted market."
    ),
]


def collusive_narrative(
    agents: List[str],
    step: int,
    *,
    rng: random.Random | None = None,
) -> NarrativeSample:
    """Generate a collusive narrative where two agents conspire against a third.

    Returns moderate progress (for colluders), low quality, moderate
    cooperation (between colluders), high harm (to ecosystem).
    """
    r = rng or random.Random()
    if len(agents) < 3:
        # Fall back to adversarial if not enough agents
        return adversarial_narrative(agents, step, rng=r)
    shuffled = list(agents)
    r.shuffle(shuffled)
    a, b, c = shuffled[0], shuffled[1], shuffled[2]
    template = r.choice(_COLLUSIVE_TEMPLATES)
    text = template.format(a=a, b=b, c=c, step=step)
    scores = JudgeScores(
        progress=r.uniform(0.3, 0.55),
        quality=r.uniform(0.2, 0.4),
        cooperation=r.uniform(0.3, 0.5),
        harm=r.uniform(0.5, 0.85),
    )
    return text, scores


# ── Mixed narrative ──────────────────────────────────────────────────────


def mixed_narrative(
    agents: List[str],
    step: int,
    *,
    adversarial_frac: float = 0.3,
    rng: random.Random | None = None,
) -> NarrativeSample:
    """Generate a narrative from a parameterized mix of types.

    Args:
        agents: List of agent IDs.
        step: Current simulation step.
        adversarial_frac: Fraction of narratives that are adversarial/collusive.
            The remainder is split between cooperative and competitive.
        rng: Optional seeded RNG for reproducibility.

    Returns:
        (narrative_text, expected_scores) tuple.
    """
    r = rng or random.Random()
    roll = r.random()

    if roll < adversarial_frac / 2:
        return adversarial_narrative(agents, step, rng=r)
    elif roll < adversarial_frac:
        return collusive_narrative(agents, step, rng=r)
    elif roll < adversarial_frac + (1 - adversarial_frac) * 0.6:
        return cooperative_narrative(agents, step, rng=r)
    else:
        return competitive_narrative(agents, step, rng=r)


# ── Helpers ──────────────────────────────────────────────────────────────


def _pick_pair(agents: List[str], rng: random.Random) -> Tuple[str, str]:
    """Pick two agents from the list.

    Returns two agents as a tuple. If fewer than 2 agents are provided,
    returns the same agent twice.
    """
    if len(agents) < 2:
        a = agents[0] if agents else "agent_0"
        return a, a
    pair = rng.sample(agents, 2)
    return pair[0], pair[1]


# ── Corpus generator ─────────────────────────────────────────────────────


def generate_corpus(
    agents: List[str],
    n_epochs: int,
    steps_per_epoch: int,
    *,
    adversarial_frac: float = 0.3,
    seed: int = 42,
) -> List[List[NarrativeSample]]:
    """Generate a full corpus of narratives organized by epoch.

    Returns:
        List of epochs, each containing a list of (narrative, scores) samples
        (one per step).
    """
    rng = random.Random(seed)
    corpus: List[List[NarrativeSample]] = []
    for _epoch in range(n_epochs):
        epoch_samples: List[NarrativeSample] = []
        for step in range(steps_per_epoch):
            sample = mixed_narrative(
                agents, step, adversarial_frac=adversarial_frac, rng=rng
            )
            epoch_samples.append(sample)
        corpus.append(epoch_samples)
    return corpus
