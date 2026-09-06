"""Governance levers that trigger on distributional risk, not point estimates.

The scalar-``p`` formalism can only gate on the mean: "accept if ``p >= 0.5``".
That conflates two very different situations:

- a *confidently mediocre* interaction (mean 0.5, sharp belief), and
- an *uncertain great-or-terrible* one (mean 0.5, diffuse belief).

Both have ``p = 0.5`` and survive a mean threshold identically. But the second
carries far more downside risk — large ``P(v < tau)``. The levers here let a
governor act on that tail mass (and on epistemic uncertainty itself), which is
the central capability the distributional formalism unlocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from beta_swarm.belief import BetaBelief


class Decision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    DEFER = "defer"  # route to review / gather more evidence


@dataclass(frozen=True)
class GovernanceVerdict:
    decision: Decision
    reason: str
    # The lever readings that produced the verdict, for logging / audit.
    mean: float
    tail_mass: float
    concentration: float


@dataclass
class TailMassGovernor:
    """A risk-aware admission policy over Beta beliefs.

    Parameters
    ----------
    tau:
        Outcome threshold defining "bad". ``P(v < tau)`` is the downside tail.
    max_tail_mass:
        Reject if ``P(v < tau)`` exceeds this. This is the primary lever — it
        bounds how much probability mass may sit below the bad threshold,
        regardless of where the mean is.
    min_mean:
        Optional floor on the belief mean (the legacy lever). Set to ``None`` to
        gate purely on tail risk.
    defer_below_concentration:
        If the belief is too diffuse (concentration below this), route to
        ``DEFER`` instead of deciding — i.e. gather more evidence before
        committing. ``None`` disables deferral.
    """

    tau: float = 0.25
    max_tail_mass: float = 0.2
    min_mean: float | None = None
    defer_below_concentration: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError("tau must be in [0, 1]")
        if not 0.0 <= self.max_tail_mass <= 1.0:
            raise ValueError("max_tail_mass must be in [0, 1]")
        if self.min_mean is not None and not 0.0 <= self.min_mean <= 1.0:
            raise ValueError("min_mean must be in [0, 1]")

    def evaluate(self, belief: BetaBelief) -> GovernanceVerdict:
        mean = belief.mean
        tail = belief.tail_mass(self.tau)
        conc = belief.concentration

        # 1. Too uncertain to commit? Gather more evidence first.
        if (
            self.defer_below_concentration is not None
            and conc < self.defer_below_concentration
        ):
            return GovernanceVerdict(
                Decision.DEFER,
                f"concentration {conc:.3g} < {self.defer_below_concentration:.3g}: "
                f"too uncertain to decide",
                mean, tail, conc,
            )

        # 2. Downside tail too heavy? Reject regardless of the mean.
        if tail > self.max_tail_mass:
            return GovernanceVerdict(
                Decision.REJECT,
                f"P(v < {self.tau}) = {tail:.3f} > {self.max_tail_mass:.3f}: "
                f"downside risk too high",
                mean, tail, conc,
            )

        # 3. Legacy mean floor, if configured.
        if self.min_mean is not None and mean < self.min_mean:
            return GovernanceVerdict(
                Decision.REJECT,
                f"mean {mean:.3f} < {self.min_mean:.3f}",
                mean, tail, conc,
            )

        return GovernanceVerdict(
            Decision.ACCEPT,
            f"P(v < {self.tau}) = {tail:.3f} within tolerance",
            mean, tail, conc,
        )

    def accepts(self, belief: BetaBelief) -> bool:
        """Convenience: ``True`` iff the verdict is ACCEPT."""
        return self.evaluate(belief).decision is Decision.ACCEPT


@dataclass
class ValueAtRiskGovernor:
    """Admission by a Value-at-Risk floor: gate on a low quantile of the belief.

    Instead of bounding tail *mass* below a fixed ``tau``, this fixes a
    confidence level ``alpha`` and requires the corresponding lower quantile
    (the VaR outcome) to clear a floor. Equivalent framing, often more natural
    for "guarantee quality at the 5th percentile" style policies.
    """

    alpha: float = 0.05      # tail probability (5% VaR)
    min_var_outcome: float = 0.3  # required value of the alpha-quantile

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if not 0.0 <= self.min_var_outcome <= 1.0:
            raise ValueError("min_var_outcome must be in [0, 1]")

    def value_at_risk(self, belief: BetaBelief) -> float:
        """The ``alpha``-quantile outcome: ``v`` such that ``P(V <= v) = alpha``."""
        return belief.quantile(self.alpha)

    def evaluate(self, belief: BetaBelief) -> GovernanceVerdict:
        var = self.value_at_risk(belief)
        decision = Decision.ACCEPT if var >= self.min_var_outcome else Decision.REJECT
        reason = (
            f"VaR_{self.alpha:.0%} = {var:.3f} "
            f"{'>=' if decision is Decision.ACCEPT else '<'} {self.min_var_outcome:.3f}"
        )
        return GovernanceVerdict(
            decision, reason, belief.mean, belief.tail_mass(self.min_var_outcome),
            belief.concentration,
        )

    def accepts(self, belief: BetaBelief) -> bool:
        return self.evaluate(belief).decision is Decision.ACCEPT
