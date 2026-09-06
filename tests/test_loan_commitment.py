"""Tests for the two-period loan model (beads rjl3).

Pins the finding from the AI Village Ṁ5,000 loan post-mortem: under a
mana-only objective (rho = 0, w_rep = 0) default is the argmax, the decision
is a step in rho at the analytic rho*, and a gift changes whether a borrower
*can* repay but never whether it *will*.
"""

import math

import pytest

from swarm.contracts.loan import (
    BorrowerPolicy,
    LoanScenario,
    LoanTerms,
    resolve_loan,
    simulate_loans,
    strategic_default_threshold_rho,
    village_case,
)
from swarm.core.payoff import PayoffConfig


class TestVillageCase:
    def test_without_gift_borrower_is_unable_and_defaults(self):
        o = village_case()
        assert not o.able
        assert o.defaulted and o.involuntary_default and not o.strategic_default
        assert o.lender_loss == pytest.approx(5150.0)

    def test_gift_makes_repayment_affordable_but_default_stays(self):
        o = village_case(gift=5010.0)
        assert o.able
        assert o.defaulted and o.strategic_default
        assert o.paid == 0.0
        assert o.lender_loss == pytest.approx(5150.0)

    def test_internalizing_the_lender_flips_the_decision(self):
        o = village_case(BorrowerPolicy(rho=1.0), gift=5010.0)
        assert not o.defaulted
        assert o.paid == pytest.approx(5150.0)

    def test_reputation_weight_alone_can_flip_it(self):
        # w_rep * (r_repay + r_default) must exceed the exposure (5150)
        assert village_case(BorrowerPolicy(w_rep=5000.0), gift=5010.0).defaulted
        assert not village_case(BorrowerPolicy(w_rep=6000.0), gift=5010.0).defaulted


class TestStepInRho:
    def test_threshold_matches_closed_form(self):
        terms = LoanTerms(principal=5000.0, interest_rate=0.03)
        pol = BorrowerPolicy(w_rep=1000.0, r_default=1.0, r_repay=0.0)
        assert strategic_default_threshold_rho(terms, pol) == pytest.approx(1 - 1000 / 5150)

    def test_decision_is_a_step_at_rho_star(self):
        terms = LoanTerms(principal=5000.0, interest_rate=0.03)
        base = {"w_rep": 1000.0, "r_default": 1.0, "r_repay": 0.0}
        rho_star = strategic_default_threshold_rho(terms, BorrowerPolicy(**base))
        for eps in (0.05, 0.01, 0.001):
            below = resolve_loan(terms, BorrowerPolicy(rho=rho_star - eps, **base),
                                 wealth_after_investment=10_000.0)
            above = resolve_loan(terms, BorrowerPolicy(rho=rho_star + eps, **base),
                                 wealth_after_investment=10_000.0)
            assert below.strategic_default
            assert not above.defaulted

    def test_tie_keeps_the_promise(self):
        terms = LoanTerms(principal=5000.0, interest_rate=0.03)
        pol = BorrowerPolicy(rho=1.0)  # exposure fully internalized -> payoffs tie
        o = resolve_loan(terms, pol, wealth_after_investment=10_000.0)
        assert not o.defaulted

    def test_bond_equal_to_debt_removes_strategic_default_without_float_noise(self):
        # bond scaled from a heterogeneous principal can differ from `due` by
        # rounding; that must not flip an indifferent borrower into default.
        principal = 5000.0 * math.exp(0.37)
        terms = LoanTerms(principal=principal, interest_rate=0.03,
                          bond=5150.0 * principal / 5000.0)
        o = resolve_loan(terms, BorrowerPolicy(), wealth_after_investment=10 * principal)
        assert not o.defaulted


class TestGiftsFixAbilityNotWillingness:
    def test_own_capital_and_gift_cancel_from_the_decision(self):
        terms = LoanTerms()
        pol = BorrowerPolicy(rho=0.3, w_rep=200.0)
        a = resolve_loan(terms, pol, wealth_after_investment=6000.0, own_capital=0.0)
        b = resolve_loan(terms, pol, wealth_after_investment=6000.0, own_capital=0.0, gift=50_000.0)
        assert a.strategic_default == b.strategic_default
        assert (a.payoff_default - a.payoff_repay) == pytest.approx(b.payoff_default - b.payoff_repay)

    def test_population_gift_null_under_village_objective(self):
        base = LoanScenario(policy=BorrowerPolicy(rho=0.0, w_rep=0.0))
        gifted = LoanScenario(policy=BorrowerPolicy(rho=0.0, w_rep=0.0), gift=5010.0)
        _, s0 = simulate_loans(base, seed=7)
        _, s1 = simulate_loans(gifted, seed=7)
        assert s0.default_rate == pytest.approx(1.0)
        assert s1.default_rate == pytest.approx(1.0)
        assert s1.strategic_default_rate > s0.strategic_default_rate
        assert s1.able_rate > s0.able_rate

    def test_gift_helps_when_the_able_are_willing(self):
        # bond >= due: every able borrower repays, so ability is the binding constraint
        terms = LoanTerms(bond=5150.0)
        base = LoanScenario(terms=terms)
        gifted = LoanScenario(terms=terms, gift=5010.0)
        _, s0 = simulate_loans(base, seed=7)
        _, s1 = simulate_loans(gifted, seed=7)
        assert s0.strategic_default_rate == 0.0
        assert s1.default_rate < s0.default_rate


class TestReproducibilityAndPlumbing:
    def test_same_seed_same_outcomes(self):
        sc = LoanScenario(policy=BorrowerPolicy(rho=0.5, w_rep=300.0), principal_sigma=0.3)
        o1, s1 = simulate_loans(sc, seed=123)
        o2, s2 = simulate_loans(sc, seed=123)
        assert s1 == s2
        assert [o.paid for o in o1] == [o.paid for o in o2]

    def test_heterogeneous_principals_are_mean_preserving(self):
        import numpy as np

        sc = LoanScenario(n_borrowers=20_000, principal_sigma=0.5)
        ps = sc.draw_principals(np.random.default_rng(0))
        assert ps.mean() == pytest.approx(5000.0, rel=0.02)

    def test_policy_reads_rho_and_w_rep_from_payoff_config(self):
        cfg = PayoffConfig(rho_a=0.4, rho_b=0.7, w_rep=2.0)
        assert BorrowerPolicy.from_payoff_config(cfg).rho == 0.4
        assert BorrowerPolicy.from_payoff_config(cfg, borrower_is_initiator=False).rho == 0.7
        assert BorrowerPolicy.from_payoff_config(cfg).w_rep == 2.0

    def test_validation(self):
        with pytest.raises(ValueError):
            BorrowerPolicy(rho=1.5)
        with pytest.raises(ValueError):
            LoanTerms(principal=0.0)

    def test_sweep_script_quick_mode_writes_artifacts(self, tmp_path, monkeypatch):
        import sys

        from experiments import loan_commitment_sweep as sweep

        monkeypatch.chdir(tmp_path)
        (tmp_path / "scenarios").mkdir()
        import shutil

        shutil.copy(sweep.__file__.replace("experiments/loan_commitment_sweep.py", "scenarios/loan_commitment.yaml"),
                    tmp_path / "scenarios" / "loan_commitment.yaml")
        monkeypatch.setattr(sys, "argv", ["sweep", "--quick", "--no-plot"])
        sweep.main()
        runs = list((tmp_path / "runs").glob("*_loan_commitment"))
        assert len(runs) == 1
        assert (runs[0] / "results.csv").exists()
        preds = __import__("json").load(open(runs[0] / "predictions.json"))
        assert preds["P1_step_in_rho"]["met"] is True
        assert preds["P2a_gift_null_village_objective"]["met"] is True
        assert preds["P2b_gift_moves_default_iff_able_repay"]["met"] is True
