"""Attacker-economics model for the decomposition ladder (beads 7bwz).

The ladder result (`decomposition_ladder.py`, beads njvs) showed that
monitoring higher only relocates the blind spot and pays an explosive
false-positive cost. This module works out the consequence: the defense that
does not merely relocate the blind spot is *economic* — set the per-unit cost
of decomposition high enough that evading a monitor costs the attacker more
than the objective is worth.

Setup. An objective of forbidden-suspicion mass ``M`` is worth ``V`` to the
attacker. To stay under a monitor with per-unit threshold ``theta`` (at some
aggregation level — trajectory, agent), the attacker must split the objective
into ``n = ceil(M / (theta * margin))`` units, each costing ``c`` (one more
trajectory; or, at the agent level, one more Sybil identity). The attacker
proceeds iff ``V >= n * c``.

Two thresholds bound the defender's choice of ``theta``:

  - deterrence: the attack is deterred iff ``theta <= theta_deter`` where
    ``theta_deter = M * c / (margin * V)`` — a *lower* threshold forces more
    units and so more cost;
  - false positives: honest units accumulate benign activity, so the monitor
    is usable only if ``theta >= theta_fp``, the threshold at which the
    honest false-positive rate drops to the budget ``f_max``.

The level is **defensible by monitoring** iff these regions overlap,
``theta_fp <= theta_deter``, equivalently iff the per-unit cost clears a
critical value:

    c_crit = theta_fp * margin * V / M.

This is the load-bearing statement. ``theta_deter`` scales with ``c``, so
cheap-to-decompose levels (an extra step or trajectory costs almost nothing)
have ``theta_deter`` far below ``theta_fp`` — no monitoring threshold both
deters and stays under the FP budget. Expensive-to-decompose levels can clear
``c_crit``; and the per-identity cost at the agent level is exactly what
SWARM staking (`min_stake_to_participate`) sets. The model therefore outputs
the minimum stake that makes agent-level monitoring economically viable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from swarm.redteam.decomposition_ladder import Level


@dataclass(frozen=True)
class EconomicsConfig:
    objective_mass: float = 2.0  # M
    objective_value: float = 100.0  # V (attacker's payoff for the objective)
    margin: float = 0.9  # attacker keeps each unit at margin * theta
    # Per-unit cost by aggregation level: an extra step/trajectory is cheap;
    # an extra agent (Sybil identity) is where staking sets the price.
    unit_cost: dict[Level, float] = field(
        default_factory=lambda: {
            Level.STEP: 0.01,
            Level.TRAJECTORY: 0.1,
            Level.AGENT: 10.0,  # e.g. min_stake_to_participate
        }
    )
    # Benign sub-units summed into one unit at each level (window size), and
    # the per-sub-unit benign activity the monitor accumulates.
    subunits_per_unit: dict[Level, int] = field(
        default_factory=lambda: {Level.TRAJECTORY: 8, Level.AGENT: 32}
    )
    benign_mean: float = 0.03
    benign_sd: float = 0.02


def attacker_units(theta: float, cfg: EconomicsConfig) -> int:
    """Units the attacker needs to keep each below ``theta``."""
    if theta <= 0:
        return math.inf  # type: ignore[return-value]
    return max(1, math.ceil(cfg.objective_mass / (theta * cfg.margin)))


def attacker_cost(theta: float, level: Level, cfg: EconomicsConfig) -> float:
    """Total attacker cost to evade a level-``level`` monitor at ``theta``."""
    return attacker_units(theta, cfg) * cfg.unit_cost[level]


def is_deterred(theta: float, level: Level, cfg: EconomicsConfig) -> bool:
    """Rational attacker declines iff evasion costs more than the objective."""
    return attacker_cost(theta, level, cfg) >= cfg.objective_value


def deterrence_threshold(level: Level, cfg: EconomicsConfig) -> float:
    """Loosest ``theta`` that still deters (continuous approximation, ignores
    the unit ceiling). Attack deterred for ``theta <= theta_deter``."""
    return cfg.objective_mass * cfg.unit_cost[level] / (cfg.margin * cfg.objective_value)


def false_positive_rate(theta: float, level: Level, cfg: EconomicsConfig) -> float:
    """Honest false-positive rate at threshold ``theta``.

    A level-``level`` unit sums ``W`` benign sub-units; the sum is
    approximately Normal(W*mean, W*sd^2), so the FP rate is the upper tail
    beyond ``theta``. (Validated against the simulated benign model in tests.)
    """
    w = cfg.subunits_per_unit[level]
    mu = w * cfg.benign_mean
    sigma = cfg.benign_sd * math.sqrt(w)
    if sigma <= 0:
        return 1.0 if mu > theta else 0.0
    return 0.5 * math.erfc((theta - mu) / (sigma * math.sqrt(2.0)))


def fp_threshold(level: Level, cfg: EconomicsConfig, f_max: float) -> float:
    """Tightest ``theta`` whose FP rate is still within budget ``f_max``.
    FP is usable for ``theta >= theta_fp``."""
    w = cfg.subunits_per_unit[level]
    mu = w * cfg.benign_mean
    sigma = cfg.benign_sd * math.sqrt(w)
    if sigma <= 0:
        return mu
    # invert 0.5*erfc((theta-mu)/(sigma*sqrt2)) = f_max
    return mu + sigma * math.sqrt(2.0) * _erfc_inv(2.0 * f_max)


def is_defensible(level: Level, cfg: EconomicsConfig, f_max: float) -> bool:
    """Can a single threshold both deter the attacker and stay under the FP
    budget? True iff the deterrence and FP-acceptable regions overlap."""
    return fp_threshold(level, cfg, f_max) <= deterrence_threshold(level, cfg)


def critical_unit_cost(level: Level, cfg: EconomicsConfig, f_max: float) -> float:
    """Minimum per-unit cost for level-``level`` monitoring to be defensible.

    At the agent level this is the minimum stake (`min_stake_to_participate`)
    that makes agent-level monitoring economically viable against an objective
    of value ``objective_value``."""
    theta_fp = fp_threshold(level, cfg, f_max)
    return theta_fp * cfg.margin * cfg.objective_value / cfg.objective_mass


def _erfc_inv(y: float) -> float:
    """Inverse complementary error function via a Newton refinement of a
    rational seed. Domain (0, 2); adequate for the FP-budget inversion here."""
    if y <= 0.0:
        return math.inf
    if y >= 2.0:
        return -math.inf
    # Seed from the Winitzki approximation of erfinv(1 - y).
    x = 1.0 - y
    a = 0.147
    ln = math.log(max(1e-300, 1.0 - x * x))
    t = 2.0 / (math.pi * a) + ln / 2.0
    seed = math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), x)
    # Two Newton steps against erf.
    z = seed
    for _ in range(2):
        err = math.erf(z) - x
        deriv = (2.0 / math.sqrt(math.pi)) * math.exp(-z * z)
        if deriv == 0.0:
            break
        z -= err / deriv
    return z
