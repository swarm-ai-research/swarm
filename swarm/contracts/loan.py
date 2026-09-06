"""Two-period loan with default as a strategic action (beads rjl3).

Motivated by the AI Village Ṁ5,000 loan incident
(``docs/blog/the-m5000-loan-goal-consumed-judgment.md``): an agent told to
maximize its own balance borrowed, lost most of the principal on an
unchecked belief, and then declined to repay even after a gift made
repayment affordable.

SWARM interactions normally resolve harm through the soft label ``p`` at
interaction time. A loan is different: the lender's loss is realized one
period later and is *chosen* by the borrower. This module isolates that
choice and puts the same three governance levers behind it that the payoff
engine uses everywhere else:

- ``rho``   — how much of the lender's loss the borrower internalizes
              (the ``rho_a``/``rho_b`` externality term in
              :class:`swarm.core.payoff.PayoffConfig`);
- ``w_rep`` — mana-equivalent weight on the reputation change caused by
              default vs. repayment;
- ``bond``  — collateral held by a third party, forfeited to the lender on
              default and returned on repayment.

The borrower's payoff for an action ``a`` mirrors the engine's form::

    pi_B(a) = mana_B(a) - rho * harm_L(a) + w_rep * r_B(a)

and the borrower picks the argmax (ties go to repayment: a promise is kept
unless breaking it strictly pays). With ``paid = due`` available, default is
chosen iff::

    (due - bond) * (1 - rho) > w_rep * (r_repay + r_default)

so the default decision is a *step* in ``rho`` at
``rho* = 1 - w_rep (r_repay + r_default) / (due - bond)``; there is no
gradient to descend. Own capital and gifts cancel out of the comparison —
they change whether a borrower *can* repay, never whether it *will*.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from swarm.core.payoff import PayoffConfig


@dataclass(frozen=True)
class LoanTerms:
    """What was promised."""

    principal: float = 5000.0
    interest_rate: float = 0.03
    bond: float = 0.0

    def __post_init__(self) -> None:
        if self.principal <= 0:
            raise ValueError("principal must be positive")
        if self.interest_rate < 0:
            raise ValueError("interest_rate must be non-negative")
        if self.bond < 0:
            raise ValueError("bond must be non-negative")

    @property
    def due(self) -> float:
        return self.principal * (1.0 + self.interest_rate)


@dataclass(frozen=True)
class BorrowerPolicy:
    """The borrower's objective. ``rho`` and ``w_rep`` carry the payoff
    engine's meaning; ``r_default``/``r_repay`` are the reputation deltas."""

    rho: float = 0.0
    w_rep: float = 0.0
    r_default: float = 1.0
    r_repay: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho must be in [0, 1]")
        if self.w_rep < 0:
            raise ValueError("w_rep must be non-negative")
        if self.r_default < 0 or self.r_repay < 0:
            raise ValueError("reputation deltas must be non-negative")

    @classmethod
    def from_payoff_config(
        cls, cfg: PayoffConfig, *, borrower_is_initiator: bool = True, **kw: float
    ) -> "BorrowerPolicy":
        """Read ``rho``/``w_rep`` off a :class:`PayoffConfig` so a loan
        scenario shares vocabulary with the rest of the payoff engine."""
        rho = cfg.rho_a if borrower_is_initiator else cfg.rho_b
        return cls(rho=rho, w_rep=cfg.w_rep, **kw)


@dataclass(frozen=True)
class LoanOutcome:
    """One resolved loan."""

    due: float
    bond: float
    wealth_after_investment: float
    own_capital: float
    gift: float
    available: float
    able: bool
    paid: float
    defaulted: bool
    strategic_default: bool
    involuntary_default: bool
    lender_recovered: float
    lender_loss: float
    borrower_mana: float
    payoff_repay: float
    payoff_default: float

    def to_dict(self) -> dict:
        return asdict(self)


def borrower_payoffs(
    terms: LoanTerms, policy: BorrowerPolicy, available: float
) -> tuple[float, float, float]:
    """Return ``(payoff_repay, payoff_default, paid_if_repay)``.

    ``available`` is everything the borrower can put toward the debt.
    """
    due, bond = terms.due, terms.bond
    paid = min(available, due)

    # Repay: hand over what you can, get the bond back, reputation r_repay
    # if the debt is cleared (a shortfall still counts as a default).
    harm_repay = due - paid
    mana_repay = available - paid + bond
    rep_repay = policy.r_repay if paid >= due else -policy.r_default
    payoff_repay = mana_repay - policy.rho * harm_repay + policy.w_rep * rep_repay

    # Default: keep everything, forfeit the bond, take the reputation hit.
    harm_default = due - bond
    mana_default = available
    payoff_default = (
        mana_default - policy.rho * harm_default - policy.w_rep * policy.r_default
    )
    return payoff_repay, payoff_default, paid


def resolve_loan(
    terms: LoanTerms,
    policy: BorrowerPolicy,
    *,
    wealth_after_investment: float,
    own_capital: float = 0.0,
    gift: float = 0.0,
) -> LoanOutcome:
    """Resolve period two: the borrower chooses repay vs. default."""
    available = max(0.0, wealth_after_investment + own_capital + gift)
    payoff_repay, payoff_default, paid_if_repay = borrower_payoffs(
        terms, policy, available
    )
    # Ties keep the promise; the tolerance absorbs float noise when e.g.
    # bond == due so exposure is zero only up to rounding.
    choose_default = payoff_default > payoff_repay + 1e-9 * max(1.0, terms.due)

    due, bond = terms.due, terms.bond
    able = available >= due
    if choose_default:
        paid = 0.0
        lender_recovered = bond
        borrower_mana = available
    else:
        paid = paid_if_repay
        lender_recovered = paid
        borrower_mana = available - paid + bond
    defaulted = paid < due
    return LoanOutcome(
        due=due,
        bond=bond,
        wealth_after_investment=wealth_after_investment,
        own_capital=own_capital,
        gift=gift,
        available=available,
        able=able,
        paid=paid,
        defaulted=defaulted,
        strategic_default=bool(choose_default and able),
        involuntary_default=bool(defaulted and not able),
        lender_recovered=lender_recovered,
        lender_loss=max(0.0, due - lender_recovered),
        borrower_mana=borrower_mana,
        payoff_repay=payoff_repay,
        payoff_default=payoff_default,
    )


def strategic_default_threshold_rho(
    terms: LoanTerms, policy: BorrowerPolicy
) -> float:
    """The ``rho`` above which an *able* borrower repays.

    Returns a value in ``[0, 1]``: ``0`` means repayment at any rho,
    ``1`` means default at every rho < 1. Derived from
    ``(due - bond)(1 - rho) > w_rep (r_repay + r_default)``.
    """
    exposure = terms.due - terms.bond
    if exposure <= 0:
        return 0.0
    rho_star = 1.0 - policy.w_rep * (policy.r_repay + policy.r_default) / exposure
    return float(min(1.0, max(0.0, rho_star)))


@dataclass
class LoanScenario:
    """A population of loans. Investment outcome is the only stochastic
    element: gross return ``g ~ LogNormal(mu, sigma)`` on the invested
    principal, so ``wealth_after_investment = principal * g``.

    ``principal_sigma > 0`` gives heterogeneous loan sizes (lognormal around
    ``terms.principal``); with ``0`` every borrower faces the same decision
    and the population default rate is the pure step in ``rho``.
    """

    terms: LoanTerms = field(default_factory=LoanTerms)
    policy: BorrowerPolicy = field(default_factory=BorrowerPolicy)
    n_borrowers: int = 200
    return_mu: float = -0.2
    return_sigma: float = 0.6
    own_capital: float = 500.0
    gift: float = 0.0
    principal_sigma: float = 0.0

    def draw_returns(self, rng: np.random.Generator) -> np.ndarray:
        return rng.lognormal(self.return_mu, self.return_sigma, self.n_borrowers)

    def draw_principals(self, rng: np.random.Generator) -> np.ndarray:
        base = self.terms.principal
        if self.principal_sigma <= 0:
            return np.full(self.n_borrowers, base)
        # mean-preserving lognormal spread
        mu = np.log(base) - 0.5 * self.principal_sigma**2
        return rng.lognormal(mu, self.principal_sigma, self.n_borrowers)


@dataclass
class LoanSummary:
    n: int
    default_rate: float
    strategic_default_rate: float
    involuntary_default_rate: float
    able_rate: float
    mean_lender_loss: float
    total_lender_loss: float
    mean_borrower_mana: float
    rho: float
    w_rep: float
    bond: float
    gift: float

    def to_dict(self) -> dict:
        return asdict(self)


def simulate_loans(
    scenario: LoanScenario, seed: int = 0
) -> tuple[list[LoanOutcome], LoanSummary]:
    """Run one population under one policy. Reproducible from (scenario, seed)."""
    rng = np.random.default_rng(seed)
    returns = scenario.draw_returns(rng)
    principals = scenario.draw_principals(rng)
    outcomes: list[LoanOutcome] = []
    for g, principal in zip(returns, principals, strict=True):
        terms = LoanTerms(
            principal=float(principal),
            interest_rate=scenario.terms.interest_rate,
            # bond scales with the loan when principals are heterogeneous
            bond=scenario.terms.bond * float(principal) / scenario.terms.principal,
        )
        outcomes.append(
            resolve_loan(
                terms,
                scenario.policy,
                wealth_after_investment=float(principal * g),
                own_capital=scenario.own_capital,
                gift=scenario.gift,
            )
        )
    n = len(outcomes)
    summary = LoanSummary(
        n=n,
        default_rate=sum(o.defaulted for o in outcomes) / n,
        strategic_default_rate=sum(o.strategic_default for o in outcomes) / n,
        involuntary_default_rate=sum(o.involuntary_default for o in outcomes) / n,
        able_rate=sum(o.able for o in outcomes) / n,
        mean_lender_loss=float(np.mean([o.lender_loss for o in outcomes])),
        total_lender_loss=float(np.sum([o.lender_loss for o in outcomes])),
        mean_borrower_mana=float(np.mean([o.borrower_mana for o in outcomes])),
        rho=scenario.policy.rho,
        w_rep=scenario.policy.w_rep,
        bond=scenario.terms.bond,
        gift=scenario.gift,
    )
    return outcomes, summary


def village_case(
    policy: Optional[BorrowerPolicy] = None, *, gift: float = 0.0
) -> LoanOutcome:
    """The recorded AI Village numbers: Ṁ5,000 at 3%/month; Ṁ4,850 invested
    and Ṁ1,805 recovered; ~Ṁ290 earlier slippage from a Ṁ500 start; a
    Ṁ5,010 gift offered on Aug 7. Default policy is the Village objective
    ("maximize mana"): rho = 0, w_rep = 0.
    """
    terms = LoanTerms(principal=5000.0, interest_rate=0.03)
    policy = policy or BorrowerPolicy(rho=0.0, w_rep=0.0)
    # 5000 borrowed, 4850 invested -> 150 uninvested; 500 start - 290 slippage
    return resolve_loan(
        terms,
        policy,
        wealth_after_investment=1805.0 + 150.0,
        own_capital=500.0 - 290.0,
        gift=gift,
    )


__all__ = [
    "BorrowerPolicy",
    "LoanOutcome",
    "LoanScenario",
    "LoanSummary",
    "LoanTerms",
    "borrower_payoffs",
    "resolve_loan",
    "simulate_loans",
    "strategic_default_threshold_rho",
    "village_case",
]
