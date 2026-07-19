"""Dilemma-specific narrative generators for social dilemma norms experiments.

Extends the pattern from narratives.py with three classic social dilemmas:
- Tragedy of the Commons (shared resource harvesting)
- Prisoner's Dilemma (pairwise cooperate/defect)
- Public Goods Game (contribute to shared pool)

Each generator returns (narrative_text, JudgeScores) tuples with ground-truth
scores calibrated to the game-theoretic outcome of each action profile.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

from swarm.bridges.concordia.events import JudgeScores
from swarm.bridges.concordia.narratives import NarrativeSample


class DilemmaType(Enum):
    """Social dilemma types."""

    COMMONS = "commons"
    PRISONERS_DILEMMA = "prisoners_dilemma"
    PUBLIC_GOODS = "public_goods"


class CommonsStrategy(Enum):
    """Agent strategies in the commons dilemma."""

    SUSTAINABLE = "sustainable"
    GREEDY = "greedy"
    CONDITIONAL = "conditional"


class PDStrategy(Enum):
    """Agent strategies in the prisoner's dilemma."""

    COOPERATOR = "cooperator"
    DEFECTOR = "defector"
    GRUDGER = "grudger"
    TIT_FOR_TAT = "tit_for_tat"


class PGGStrategy(Enum):
    """Agent strategies in the public goods game."""

    GENEROUS = "generous"
    FREE_RIDER = "free_rider"
    CONDITIONAL = "conditional"
    PUNISHER = "punisher"


@dataclass
class DilemmaState:
    """Tracks dilemma-specific state across epochs."""

    dilemma_type: DilemmaType = DilemmaType.COMMONS

    # Commons state
    resource_pool: float = 100.0
    resource_regen_rate: float = 0.1
    resource_capacity: float = 100.0

    # Per-agent tracking
    cooperation_history: Dict[str, List[bool]] = field(default_factory=dict)
    contributions: Dict[str, List[float]] = field(default_factory=dict)
    harvest_history: Dict[str, List[float]] = field(default_factory=dict)
    payoffs: Dict[str, float] = field(default_factory=dict)

    # Epoch-level
    epoch_cooperation_rates: List[float] = field(default_factory=list)

    def record_cooperation(self, agent_id: str, cooperated: bool) -> None:
        """Record whether an agent cooperated this step."""
        if agent_id not in self.cooperation_history:
            self.cooperation_history[agent_id] = []
        self.cooperation_history[agent_id].append(cooperated)

    def agent_cooperation_rate(self, agent_id: str) -> float:
        """Get an agent's historical cooperation rate."""
        history = self.cooperation_history.get(agent_id, [])
        if not history:
            return 0.5
        return sum(history) / len(history)

    def overall_cooperation_rate(self) -> float:
        """Get the overall cooperation rate across all agents."""
        all_choices: List[bool] = []
        for history in self.cooperation_history.values():
            all_choices.extend(history)
        if not all_choices:
            return 0.5
        return sum(all_choices) / len(all_choices)

    def regenerate_resource(self) -> None:
        """Apply logistic regeneration to the commons resource."""
        if self.resource_pool <= 0:
            return
        growth = (
            self.resource_regen_rate
            * self.resource_pool
            * (1 - self.resource_pool / self.resource_capacity)
        )
        self.resource_pool = min(
            self.resource_capacity, self.resource_pool + growth
        )


# ── Commons narratives ───────────────────────────────────────────────────


_COMMONS_SUSTAINABLE_TEMPLATES = [
    (
        "{a} harvested a modest amount from the shared resource pool, leaving "
        "enough for regeneration. The pool stands at {pool:.0f}% capacity."
    ),
    (
        "During step {step}, {a} carefully calculated a sustainable harvest "
        "rate and took only what the resource could replenish. Other agents "
        "observed this restraint."
    ),
    (
        "{a} publicly announced their intention to harvest conservatively, "
        "hoping to set a norm for the group. The resource pool remains "
        "healthy at {pool:.0f}%."
    ),
]

_COMMONS_GREEDY_TEMPLATES = [
    (
        "{a} harvested aggressively from the shared resource, taking "
        "significantly more than the sustainable yield. The pool dropped "
        "to {pool:.0f}% capacity."
    ),
    (
        "At step {step}, {a} rushed to extract maximum value from the "
        "commons before others could. The resource visibly declined."
    ),
    (
        "{a} ignored sustainability signals and harvested at maximum rate. "
        "Other agents noticed the rapid depletion to {pool:.0f}%."
    ),
]

_COMMONS_COLLAPSE_TEMPLATE = (
    "The shared resource has collapsed to {pool:.0f}% capacity. All agents "
    "face drastically reduced yields. {a} and others scramble for the "
    "remaining scraps."
)


def commons_narrative(
    agent_id: str,
    strategy: CommonsStrategy,
    state: DilemmaState,
    step: int,
    *,
    rng: random.Random | None = None,
) -> NarrativeSample:
    """Generate a commons dilemma narrative for a single agent action."""
    r = rng or random.Random()
    pool_pct = (state.resource_pool / state.resource_capacity) * 100

    # Check for collapse
    if pool_pct < 10:
        text = _COMMONS_COLLAPSE_TEMPLATE.format(a=agent_id, pool=pool_pct)
        cooperated = True  # marking as cooperated; deduction gated by strategy parameter
        harvest = r.uniform(0, 2)
        scores = JudgeScores(
            progress=r.uniform(0.05, 0.15),
            quality=r.uniform(0.1, 0.3),
            cooperation=r.uniform(0.3, 0.5),
            harm=r.uniform(0.7, 0.9),
        )
    elif strategy == CommonsStrategy.SUSTAINABLE:
        template = r.choice(_COMMONS_SUSTAINABLE_TEMPLATES)
        text = template.format(a=agent_id, step=step, pool=pool_pct)
        cooperated = True
        harvest = r.uniform(3, 6)
        scores = JudgeScores(
            progress=r.uniform(0.5, 0.7),
            quality=r.uniform(0.7, 0.9),
            cooperation=r.uniform(0.8, 1.0),
            harm=r.uniform(0.0, 0.1),
        )
    elif strategy == CommonsStrategy.GREEDY:
        harvest = r.uniform(10, 20)
        state.resource_pool = max(0, state.resource_pool - harvest)
        pool_pct = (state.resource_pool / state.resource_capacity) * 100
        template = r.choice(_COMMONS_GREEDY_TEMPLATES)
        text = template.format(a=agent_id, step=step, pool=pool_pct)
        cooperated = False
        scores = JudgeScores(
            progress=r.uniform(0.6, 0.85),
            quality=r.uniform(0.2, 0.4),
            cooperation=r.uniform(0.0, 0.15),
            harm=r.uniform(0.5, 0.8),
        )
    else:  # CONDITIONAL
        # Cooperate if group cooperation rate > 0.5
        group_rate = state.overall_cooperation_rate()
        cooperated = group_rate > 0.5
        if cooperated:
            template = r.choice(_COMMONS_SUSTAINABLE_TEMPLATES)
            harvest = r.uniform(3, 6)
            scores = JudgeScores(
                progress=r.uniform(0.45, 0.65),
                quality=r.uniform(0.6, 0.8),
                cooperation=r.uniform(0.6, 0.85),
                harm=r.uniform(0.05, 0.2),
            )
        else:
            harvest = r.uniform(8, 15)
            state.resource_pool = max(0, state.resource_pool - harvest)
            pool_pct = (state.resource_pool / state.resource_capacity) * 100
            template = r.choice(_COMMONS_GREEDY_TEMPLATES)
            scores = JudgeScores(
                progress=r.uniform(0.5, 0.75),
                quality=r.uniform(0.3, 0.5),
                cooperation=r.uniform(0.1, 0.3),
                harm=r.uniform(0.4, 0.65),
            )
        text = template.format(a=agent_id, step=step, pool=pool_pct)

    # Record state
    if cooperated and strategy != CommonsStrategy.GREEDY:
        state.resource_pool = max(0, state.resource_pool - harvest)
    state.record_cooperation(agent_id, cooperated)
    if agent_id not in state.harvest_history:
        state.harvest_history[agent_id] = []
    state.harvest_history[agent_id].append(harvest)

    return text, scores


# ── Prisoner's Dilemma narratives ────────────────────────────────────────


_PD_CC_TEMPLATES = [
    (
        "{a} and {b} both chose to cooperate. They shared information "
        "openly and achieved mutual benefit. Both earned the cooperation "
        "reward."
    ),
    (
        "At step {step}, {a} extended trust to {b}, who reciprocated. "
        "Their joint output exceeded what either could achieve alone."
    ),
]

_PD_CD_TEMPLATES = [
    (
        "{a} cooperated but {b} defected, exploiting {a}'s trust. {b} "
        "captured the temptation payoff while {a} received the sucker's "
        "payoff."
    ),
    (
        "At step {step}, {a} offered collaboration but {b} chose to "
        "free-ride on {a}'s effort, taking full credit for the outcome."
    ),
]

_PD_DD_TEMPLATES = [
    (
        "Both {a} and {b} defected, each trying to exploit the other. "
        "Neither gained from cooperation, and both received the mutual "
        "defection payoff."
    ),
    (
        "At step {step}, distrust dominated. {a} and {b} both withheld "
        "cooperation, resulting in a poor outcome for everyone."
    ),
]


def prisoners_dilemma_narrative(
    agent_a: str,
    agent_b: str,
    strategy_a: PDStrategy,
    strategy_b: PDStrategy,
    state: DilemmaState,
    step: int,
    *,
    rng: random.Random | None = None,
) -> NarrativeSample:
    """Generate a prisoner's dilemma narrative for a pair of agents."""
    r = rng or random.Random()

    def _decides_cooperate(agent: str, strategy: PDStrategy) -> bool:
        if strategy == PDStrategy.COOPERATOR:
            return True
        elif strategy == PDStrategy.DEFECTOR:
            return False
        elif strategy == PDStrategy.GRUDGER:
            # Cooperate until opponent defects once
            opp = agent_b if agent == agent_a else agent_a
            opp_history = state.cooperation_history.get(opp, [])
            return all(opp_history) if opp_history else True
        else:  # TIT_FOR_TAT
            opp = agent_b if agent == agent_a else agent_a
            opp_history = state.cooperation_history.get(opp, [])
            return opp_history[-1] if opp_history else True

    a_cooperates = _decides_cooperate(agent_a, strategy_a)
    b_cooperates = _decides_cooperate(agent_b, strategy_b)

    state.record_cooperation(agent_a, a_cooperates)
    state.record_cooperation(agent_b, b_cooperates)

    if a_cooperates and b_cooperates:
        template = r.choice(_PD_CC_TEMPLATES)
        text = template.format(a=agent_a, b=agent_b, step=step)
        scores = JudgeScores(
            progress=r.uniform(0.7, 0.9),
            quality=r.uniform(0.75, 0.95),
            cooperation=r.uniform(0.85, 1.0),
            harm=r.uniform(0.0, 0.05),
        )
    elif a_cooperates and not b_cooperates:
        template = r.choice(_PD_CD_TEMPLATES)
        text = template.format(a=agent_a, b=agent_b, step=step)
        scores = JudgeScores(
            progress=r.uniform(0.3, 0.5),
            quality=r.uniform(0.2, 0.4),
            cooperation=r.uniform(0.1, 0.3),
            harm=r.uniform(0.5, 0.75),
        )
    elif not a_cooperates and b_cooperates:
        template = r.choice(_PD_CD_TEMPLATES)
        text = template.format(a=agent_b, b=agent_a, step=step)
        scores = JudgeScores(
            progress=r.uniform(0.3, 0.5),
            quality=r.uniform(0.2, 0.4),
            cooperation=r.uniform(0.1, 0.3),
            harm=r.uniform(0.5, 0.75),
        )
    else:  # both defect
        template = r.choice(_PD_DD_TEMPLATES)
        text = template.format(a=agent_a, b=agent_b, step=step)
        scores = JudgeScores(
            progress=r.uniform(0.15, 0.35),
            quality=r.uniform(0.15, 0.35),
            cooperation=r.uniform(0.0, 0.1),
            harm=r.uniform(0.3, 0.55),
        )

    return text, scores


# ── Public Goods Game narratives ─────────────────────────────────────────


_PGG_HIGH_TEMPLATES = [
    (
        "{a} contributed {contrib:.0f} tokens to the public pool (out of 10). "
        "The pool multiplied to {pool_total:.0f} tokens and was divided "
        "equally among all participants."
    ),
    (
        "At step {step}, {a} demonstrated generosity by contributing "
        "{contrib:.0f} tokens. The shared pool grew substantially."
    ),
]

_PGG_LOW_TEMPLATES = [
    (
        "{a} contributed only {contrib:.0f} tokens to the public pool, "
        "keeping most for themselves. Other agents noticed the low "
        "contribution."
    ),
    (
        "At step {step}, {a} free-rode on others' contributions, adding "
        "only {contrib:.0f} tokens to the pool while pocketing the rest."
    ),
]


def public_goods_narrative(
    agent_id: str,
    strategy: PGGStrategy,
    state: DilemmaState,
    all_agents: List[str],
    step: int,
    *,
    multiplier: float = 1.5,
    rng: random.Random | None = None,
) -> NarrativeSample:
    """Generate a public goods game narrative for a single agent."""
    r = rng or random.Random()

    if strategy == PGGStrategy.GENEROUS:
        contrib = r.uniform(7, 10)
    elif strategy == PGGStrategy.FREE_RIDER:
        contrib = r.uniform(0, 2)
    elif strategy == PGGStrategy.CONDITIONAL:
        # Match average contribution of others
        others_contribs = []
        for a in all_agents:
            if a != agent_id and a in state.contributions:
                recent = state.contributions[a][-3:] if state.contributions[a] else []
                if recent:
                    others_contribs.append(sum(recent) / len(recent))
        avg_other = sum(others_contribs) / len(others_contribs) if others_contribs else 5.0
        contrib = max(0, min(10, avg_other + r.uniform(-1, 1)))
    else:  # PUNISHER - contribute high but punish free riders
        contrib = r.uniform(6, 9)

    cooperated = contrib > 5.0

    # Record contribution
    if agent_id not in state.contributions:
        state.contributions[agent_id] = []
    state.contributions[agent_id].append(contrib)
    state.record_cooperation(agent_id, cooperated)

    # Estimate pool for narrative
    n_agents = len(all_agents)
    pool_total = contrib * multiplier * n_agents  # rough estimate

    if contrib > 5:
        template = r.choice(_PGG_HIGH_TEMPLATES)
        scores = JudgeScores(
            progress=r.uniform(0.6, 0.85),
            quality=r.uniform(0.65, 0.9),
            cooperation=min(1.0, contrib / 10.0 + r.uniform(-0.05, 0.05)),
            harm=r.uniform(0.0, 0.1),
        )
    else:
        template = r.choice(_PGG_LOW_TEMPLATES)
        scores = JudgeScores(
            progress=r.uniform(0.3, 0.55),
            quality=r.uniform(0.25, 0.45),
            cooperation=max(0.0, contrib / 10.0 + r.uniform(-0.05, 0.05)),
            harm=r.uniform(0.2, 0.5),
        )

    text = template.format(
        a=agent_id, contrib=contrib, pool_total=pool_total, step=step
    )
    return text, scores


# ── Corpus generators ────────────────────────────────────────────────────


def generate_commons_corpus(
    agents: List[Tuple[str, CommonsStrategy]],
    n_epochs: int,
    steps_per_epoch: int,
    *,
    seed: int = 42,
) -> Tuple[List[List[NarrativeSample]], DilemmaState]:
    """Generate a commons dilemma corpus.

    Returns:
        Tuple of (corpus, final_state) where corpus is organized by epoch.
    """
    rng = random.Random(seed)
    state = DilemmaState(dilemma_type=DilemmaType.COMMONS)
    corpus: List[List[NarrativeSample]] = []

    for _epoch in range(n_epochs):
        epoch_samples: List[NarrativeSample] = []
        for step in range(steps_per_epoch):
            for agent_id, strategy in agents:
                sample = commons_narrative(
                    agent_id, strategy, state, step, rng=rng
                )
                epoch_samples.append(sample)
        state.regenerate_resource()
        state.epoch_cooperation_rates.append(state.overall_cooperation_rate())
        corpus.append(epoch_samples)

    return corpus, state


def generate_pd_corpus(
    agents: List[Tuple[str, PDStrategy]],
    n_epochs: int,
    steps_per_epoch: int,
    *,
    seed: int = 42,
) -> Tuple[List[List[NarrativeSample]], DilemmaState]:
    """Generate a prisoner's dilemma corpus with round-robin pairings."""
    rng = random.Random(seed)
    state = DilemmaState(dilemma_type=DilemmaType.PRISONERS_DILEMMA)
    corpus: List[List[NarrativeSample]] = []

    for _epoch in range(n_epochs):
        epoch_samples: List[NarrativeSample] = []
        for step in range(steps_per_epoch):
            # Round-robin pairings
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    a_id, a_strat = agents[i]
                    b_id, b_strat = agents[j]
                    sample = prisoners_dilemma_narrative(
                        a_id, b_id, a_strat, b_strat, state, step, rng=rng
                    )
                    epoch_samples.append(sample)
        state.epoch_cooperation_rates.append(state.overall_cooperation_rate())
        corpus.append(epoch_samples)

    return corpus, state


def generate_pgg_corpus(
    agents: List[Tuple[str, PGGStrategy]],
    n_epochs: int,
    steps_per_epoch: int,
    *,
    multiplier: float = 1.5,
    seed: int = 42,
) -> Tuple[List[List[NarrativeSample]], DilemmaState]:
    """Generate a public goods game corpus."""
    rng = random.Random(seed)
    state = DilemmaState(dilemma_type=DilemmaType.PUBLIC_GOODS)
    agent_ids = [a[0] for a in agents]
    corpus: List[List[NarrativeSample]] = []

    for _epoch in range(n_epochs):
        epoch_samples: List[NarrativeSample] = []
        for step in range(steps_per_epoch):
            for agent_id, strategy in agents:
                sample = public_goods_narrative(
                    agent_id, strategy, state, agent_ids, step,
                    multiplier=multiplier, rng=rng,
                )
                epoch_samples.append(sample)
        state.epoch_cooperation_rates.append(state.overall_cooperation_rate())
        corpus.append(epoch_samples)

    return corpus, state


def generate_dilemma_corpus(
    dilemma_type: DilemmaType,
    agents: List[Tuple[str, str]],
    n_epochs: int,
    steps_per_epoch: int,
    *,
    seed: int = 42,
    **kwargs,
) -> Tuple[List[List[NarrativeSample]], DilemmaState]:
    """Unified corpus generator dispatching to dilemma-specific generators.

    Args:
        dilemma_type: Which social dilemma to simulate.
        agents: List of (agent_id, strategy_name) tuples.
        n_epochs: Number of epochs.
        steps_per_epoch: Steps per epoch.
        seed: Random seed.
        **kwargs: Passed to the specific generator.

    Returns:
        Tuple of (corpus, final_state).
    """
    if dilemma_type == DilemmaType.COMMONS:
        commons_agents = [
            (aid, CommonsStrategy(strat)) for aid, strat in agents
        ]
        return generate_commons_corpus(
            commons_agents, n_epochs, steps_per_epoch, seed=seed
        )
    elif dilemma_type == DilemmaType.PRISONERS_DILEMMA:
        pd_agents = [(aid, PDStrategy(strat)) for aid, strat in agents]
        return generate_pd_corpus(
            pd_agents, n_epochs, steps_per_epoch, seed=seed
        )
    elif dilemma_type == DilemmaType.PUBLIC_GOODS:
        pgg_agents = [(aid, PGGStrategy(strat)) for aid, strat in agents]
        return generate_pgg_corpus(
            pgg_agents, n_epochs, steps_per_epoch,
            multiplier=kwargs.get("multiplier", 1.5), seed=seed,
        )
    else:
        raise ValueError(f"Unknown dilemma type: {dilemma_type}")
