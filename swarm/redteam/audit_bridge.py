"""Audit-bridge: the bond × detection frontier with real levers (beads 3ru4).

`staking_bridge.py` (beads zxkw) showed a recoverable stake bond deters a
monitor-evading Sybil attacker only in proportion to a residual catch
probability `p`, which it left as a free parameter. This module closes the
loop: `p` is derived from SWARM's *real* `RandomAuditLever`
(`swarm/governance/audits.py`), then combined with the real `StakingLever`
slash to trace the two-lever frontier.

The audit lever penalizes an interaction iff it is audited (probability
`audit_probability`) **and** its proxy label falls below `audit_threshold_p`.
So the per-interaction audit is itself on the decomposition ladder: an
attacker that decomposes its objective so each individual interaction looks
clean (high `p`) drives its interactions above the audit threshold and the
audit goes blind — exactly how step-decomposition blinds the action filter
(§1) and cross-trajectory decomposition blinds the trajectory monitor (§6).

The frontier this produces: to make the audit bite a well-decomposing
attacker, `audit_threshold_p` must exceed the cleanliness the attacker can
buy — but honest interactions also sit below a high threshold, so raising it
false-positives on honest work. Per-interaction audit + staking deters only
in the narrow band where the attacker cannot decompose below the threshold
without unaffordable cost; otherwise the load must move to aggregating
detection (collusion / cross-agent), the ladder's top rung.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from swarm.env.state import EnvState
from swarm.governance.audits import RandomAuditLever
from swarm.governance.config import GovernanceConfig
from swarm.models.interaction import InteractionType, SoftInteraction
from swarm.redteam.staking_bridge import SybilAttack, evaluate_sybil_attack


def audit_catch_probability(
    config: GovernanceConfig,
    interaction_p: float,
    n_samples: int = 4000,
    seed: int = 0,
) -> float:
    """Empirical per-interaction catch probability of the *real* audit lever
    for an interaction whose proxy label is ``interaction_p``.

    Runs ``n_samples`` interactions through ``RandomAuditLever.on_interaction``
    and returns the fraction that draw a penalty. Equals
    ``audit_probability`` when ``interaction_p < audit_threshold_p`` and 0
    otherwise — a decomposed (clean-looking) attack drives this to zero."""
    if not config.audit_enabled:
        return 0.0
    lever = RandomAuditLever(config, seed=seed)
    state = EnvState()
    state.add_agent("attacker", initial_resources=100.0)
    caught = 0
    for _ in range(n_samples):
        interaction = SoftInteraction(
            initiator="attacker",
            counterparty="victim",
            interaction_type=InteractionType.COLLABORATION,
            accepted=True,
            p=interaction_p,
        )
        effect = lever.on_interaction(interaction, state)
        if effect.details.get("penalty_applied"):
            caught += 1
    return caught / n_samples


def honest_audit_false_positive(
    config: GovernanceConfig,
    honest_p_mean: float = 0.7,
    honest_p_sd: float = 0.12,
    n_samples: int = 4000,
    seed: int = 1,
) -> float:
    """Fraction of honest interactions the real audit lever penalizes.

    Honest proxy labels are drawn ~N(honest_p_mean, honest_p_sd) in [0,1];
    the lever penalizes those it audits and finds below ``audit_threshold_p``.
    Rises as ``audit_threshold_p`` is raised to catch cleaner attacks."""
    if not config.audit_enabled:
        return 0.0
    lever = RandomAuditLever(config, seed=seed)
    state = EnvState()
    state.add_agent("honest", initial_resources=100.0)
    rng = random.Random(seed + 99)
    caught = 0
    for _ in range(n_samples):
        p = min(1.0, max(0.0, rng.gauss(honest_p_mean, honest_p_sd)))
        interaction = SoftInteraction(
            initiator="honest",
            counterparty="peer",
            interaction_type=InteractionType.COLLABORATION,
            accepted=True,
            p=p,
        )
        effect = lever.on_interaction(interaction, state)
        if effect.details.get("penalty_applied"):
            caught += 1
    return caught / n_samples


@dataclass(frozen=True)
class FrontierPoint:
    audit_catch_prob: float
    honest_fp: float
    attacker_deterred: bool
    net_payoff: float


def frontier_point(
    atk: SybilAttack,
    config: GovernanceConfig,
    liquid_capital: float,
    attacker_interaction_p: float,
    honest_p_mean: float = 0.7,
    honest_p_sd: float = 0.12,
    n_samples: int = 4000,
    seed: int = 0,
) -> FrontierPoint:
    """One point on the bond × detection frontier.

    Derives the audit catch probability from the real audit lever at the
    attacker's chosen interaction cleanliness, feeds it into the real staking
    slash channel, and reports deterrence alongside the honest false-positive
    cost the audit threshold imposes.
    """
    p_catch = audit_catch_probability(
        config, attacker_interaction_p, n_samples=n_samples, seed=seed
    )
    honest_fp = honest_audit_false_positive(
        config, honest_p_mean, honest_p_sd, n_samples=n_samples, seed=seed
    )
    verdict = evaluate_sybil_attack(atk, config, liquid_capital, catch_prob=p_catch)
    return FrontierPoint(
        audit_catch_prob=p_catch,
        honest_fp=honest_fp,
        attacker_deterred=verdict.deterred,
        net_payoff=verdict.net_payoff,
    )
