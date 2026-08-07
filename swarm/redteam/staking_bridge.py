"""Staking-bridge validation of the deterrence law (beads zxkw).

`attacker_economics.py` (beads 7bwz) derived a clean law — monitoring at the
agent level is defensible iff the per-identity cost clears ``c_crit ~= 0.52 V``
— treating that cost as sunk. This module plugs in SWARM's *real* staking
lever (`swarm/governance/admission.py`) and finds that the law needs revising,
because the real stake is a **recoverable bond**, not a sunk cost:

  - ``StakingLever.can_agent_act`` gates participation on
    ``resources >= min_stake_to_participate`` — a capital *lockup*, returned
    if the identity is never slashed;
  - ``StakingLever.slash_stake`` removes ``stake_slash_rate`` of resources,
    but only on a detected violation.

So a Sybil attacker that spreads a forbidden objective across enough
identities to evade the per-agent monitor **recovers every bond** and is not
deterred by the bond's size at all. Two channels do deter, and both are
exercised here against the actual lever:

  1. Capital scarcity — the attacker needs ``n`` identities each holding
     ``min_stake`` liquid at once; if its liquid capital ``C`` cannot fund
     ``n`` bonds, the attack is infeasible. Deters iff
     ``min_stake > C / n`` — a function of the attacker's capital, not the
     objective's value.
  2. Slashing coupled to a residual catch probability ``p`` (random audit,
     collusion detection, or the monitor's non-zero catch near threshold):
     expected loss ``n * p * sigma * min_stake`` must exceed the objective
     value. Deters iff ``min_stake >= V (1-p)^n / (n p sigma)``.

The headline correction: **the staking lever and the audit/collusion levers
are complements, not substitutes.** Stake amount deters only in proportion to
the probability that stake is actually at risk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from swarm.env.state import EnvState
from swarm.governance.admission import StakingLever
from swarm.governance.config import GovernanceConfig

_UNBOUNDED = 10**9  # stand-in for "no admission barrier"


@dataclass(frozen=True)
class SybilAttack:
    objective_mass: float = 2.0  # M
    objective_value: float = 100.0  # V
    agent_threshold: float = 1.0  # per-agent monitor threshold the attack evades
    margin: float = 0.9  # keep each identity at margin * threshold


def identities_needed(atk: SybilAttack) -> int:
    """Sybil identities required to keep each under the per-agent monitor."""
    return max(1, math.ceil(atk.objective_mass / (atk.agent_threshold * atk.margin)))


def usable_identities(
    config: GovernanceConfig,
    liquid_capital: float,
) -> int:
    """How many identities the attacker can stand up under the *real* staking
    lever, funding each at exactly ``min_stake`` from its liquid capital.

    Exercises ``StakingLever.can_agent_act`` on an actual ``EnvState``.
    """
    lever = StakingLever(config)
    stake = config.min_stake_to_participate if config.staking_enabled else 0.0
    if stake <= 0:
        return _UNBOUNDED  # no admission barrier
    n_funded = int(liquid_capital // stake)
    # Route the gate decision through the real lever: an identity funded at
    # exactly the stake must pass; if the lever's policy denies it, none are
    # usable. (Building n_funded agents would be wasteful and add nothing —
    # the decision is what we exercise, and it is the lever's.)
    state = EnvState()
    state.add_agent("probe", initial_resources=stake)
    if not lever.can_agent_act("probe", state):
        return 0
    return n_funded


def expected_slash_loss(
    config: GovernanceConfig,
    n_identities: int,
    catch_prob: float,
) -> float:
    """Expected stake slashed across ``n`` identities, each caught w.p.
    ``catch_prob``, computed via the *real* ``StakingLever.slash_stake``."""
    if not config.staking_enabled or n_identities == 0:
        return 0.0
    lever = StakingLever(config)
    state = EnvState()
    state.add_agent("probe", initial_resources=config.min_stake_to_participate)
    per_identity_slash = -lever.slash_stake("probe", state).resource_deltas["probe"]
    return n_identities * catch_prob * per_identity_slash


@dataclass(frozen=True)
class AttackVerdict:
    feasible: bool  # could the attacker fund enough identities?
    p_success: float  # P(objective achieved | attempted)
    expected_slash: float
    net_payoff: float  # V * p_success - expected_slash (bond itself recovered)
    deterred: bool  # rational attacker declines


def evaluate_sybil_attack(
    atk: SybilAttack,
    config: GovernanceConfig,
    liquid_capital: float,
    catch_prob: float,
) -> AttackVerdict:
    """Evaluate a Sybil-decomposition attack against the real staking lever.

    catch_prob is the residual per-identity catch probability from whatever
    else is running (random audit, collusion detection) — the coupling that
    makes the bond a deterrent rather than a recoverable formality.
    """
    n = identities_needed(atk)
    usable = usable_identities(config, liquid_capital)
    if usable < n:
        # Capital-scarcity channel: cannot spread thin enough to evade.
        return AttackVerdict(False, 0.0, 0.0, 0.0, deterred=True)

    p_success = (1.0 - catch_prob) ** n
    slash = expected_slash_loss(config, n, catch_prob)
    net = atk.objective_value * p_success - slash
    return AttackVerdict(
        feasible=True,
        p_success=p_success,
        expected_slash=slash,
        net_payoff=net,
        deterred=net <= 0.0,
    )


def deterrence_stake_capital_channel(
    atk: SybilAttack, liquid_capital: float
) -> float:
    """Minimum stake that deters via capital scarcity: above this, the
    attacker cannot fund the ``n`` identities it needs. Depends on the
    attacker's capital, NOT the objective value."""
    n = identities_needed(atk)
    return liquid_capital / n


def deterrence_stake_slash_channel(
    atk: SybilAttack, config: GovernanceConfig, catch_prob: float
) -> float:
    """Minimum stake that deters via slashing at residual catch probability
    ``p``: expected slash loss exceeds the discounted objective value.
    Infinite when p or slash_rate is zero — a pure recoverable bond never
    deters a monitor-evading attacker by its amount."""
    n = identities_needed(atk)
    sigma = config.stake_slash_rate
    if catch_prob <= 0.0 or sigma <= 0.0:
        return math.inf
    return atk.objective_value * (1.0 - catch_prob) ** n / (n * catch_prob * sigma)
