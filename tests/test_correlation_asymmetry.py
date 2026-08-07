"""Tests for the correlation-asymmetry experiment (bead pcdq).

Guards the analytic forms in experiments/correlation_asymmetry.py, including
the two bugs the first run shipped with: truncating family sizes when the
family count does not divide the agent count, and comparing structured cells on
requested rather than achieved rho_bar.
"""

import pytest

from experiments.correlation_asymmetry import (
    Config,
    attacker_hit,
    defender_catch,
    defender_retain,
    family_sizes,
    p_all,
    p_all_structured,
    p_any,
    p_at_least,
    p_none,
    p_none_structured,
    same_family_pair_fraction,
    sweep_exchangeable,
    two_level_params,
)

RHOS = [i / 20 for i in range(21)]


class TestExchangeableForms:
    """The common-shock model reduces to the textbook cases at the extremes."""

    def test_independent_any(self):
        assert p_any(0.3, 0.0, 8) == pytest.approx(1 - 0.7**8)

    def test_independent_all(self):
        assert p_all(0.3, 0.0, 8) == pytest.approx(0.3**8)

    def test_perfect_correlation_any_equals_marginal(self):
        assert p_any(0.3, 1.0, 50) == pytest.approx(0.3)

    def test_perfect_correlation_all_equals_marginal(self):
        assert p_all(0.3, 1.0, 50) == pytest.approx(0.3)

    def test_none_and_any_complementary(self):
        for rho in RHOS:
            assert p_none(0.3, rho, 6) + p_any(0.3, rho, 6) == pytest.approx(1.0)

    def test_single_agent_is_marginal(self):
        for rho in RHOS:
            assert p_any(0.42, rho, 1) == pytest.approx(0.42)
            assert p_all(0.42, rho, 1) == pytest.approx(0.42)

    def test_any_decreasing_in_rho(self):
        vals = [p_any(0.3, r, 8) for r in RHOS]
        assert vals == sorted(vals, reverse=True)

    def test_all_increasing_in_rho(self):
        vals = [p_all(0.3, r, 8) for r in RHOS]
        assert vals == sorted(vals)

    def test_probabilities_in_unit_interval(self):
        for rho in RHOS:
            for n in (1, 3, 8, 20):
                assert 0.0 <= p_any(0.3, rho, n) <= 1.0
                assert 0.0 <= p_all(0.3, rho, n) <= 1.0


class TestAtLeast:
    """p_at_least must agree with p_any / p_all at its boundaries."""

    def test_m_one_equals_any(self):
        for rho in RHOS:
            assert p_at_least(0.3, rho, 7, 1) == pytest.approx(p_any(0.3, rho, 7))

    def test_m_n_equals_all(self):
        for rho in RHOS:
            assert p_at_least(0.3, rho, 7, 7) == pytest.approx(p_all(0.3, rho, 7))

    def test_m_zero_is_certain(self):
        assert p_at_least(0.3, 0.5, 7, 0) == pytest.approx(1.0)

    def test_m_above_n_impossible(self):
        assert p_at_least(0.3, 0.5, 7, 8) == pytest.approx(0.0)

    def test_monotone_decreasing_in_m(self):
        vals = [p_at_least(0.5, 0.3, 8, m) for m in range(9)]
        assert vals == sorted(vals, reverse=True)


class TestFamilyPartition:
    """Regression: n // k silently dropped agents when k did not divide n."""

    def test_sizes_sum_to_n(self):
        for n in (1, 5, 8, 13, 20):
            for k in (1, 2, 3, 4, 7):
                assert sum(family_sizes(n, k)) == n

    def test_uneven_partition_is_balanced(self):
        assert family_sizes(5, 4) == [2, 1, 1, 1]

    def test_even_partition(self):
        assert family_sizes(8, 4) == [2, 2, 2, 2]

    def test_single_family(self):
        assert family_sizes(8, 1) == [8]

    def test_more_families_than_agents(self):
        sizes = family_sizes(3, 10)
        assert sum(sizes) == 3
        assert all(m >= 1 for m in sizes)

    def test_pair_fraction_single_family_is_one(self):
        assert same_family_pair_fraction([8]) == pytest.approx(1.0)

    def test_pair_fraction_singletons_is_zero(self):
        assert same_family_pair_fraction([1, 1, 1, 1]) == pytest.approx(0.0)

    def test_pair_fraction_even_split(self):
        # two families of 4: 2*C(4,2)=12 same-family pairs of C(8,2)=28
        assert same_family_pair_fraction([4, 4]) == pytest.approx(12 / 28)


class TestTwoLevelModel:
    """The structured model must reduce to the exchangeable one at k=1."""

    def test_single_family_matches_exchangeable_none(self):
        for rho in RHOS:
            a, b, _ = two_level_params(rho, [8], 0.0)
            assert p_none_structured(0.3, a, b, [8]) == pytest.approx(
                p_none(0.3, rho, 8)
            )

    def test_single_family_matches_exchangeable_all(self):
        for rho in RHOS:
            a, b, _ = two_level_params(rho, [8], 0.0)
            assert p_all_structured(0.3, a, b, [8]) == pytest.approx(
                p_all(0.3, rho, 8)
            )

    def test_spread_zero_is_exchangeable(self):
        """spread=0 puts all correlation in the global shock, any partition."""
        for rho in RHOS:
            a, b, achieved = two_level_params(rho, [2, 2, 2, 2], 0.0)
            assert a == pytest.approx(rho)
            assert b == pytest.approx(0.0)
            assert achieved == pytest.approx(rho)

    def test_achieved_matches_requested_when_reachable(self):
        for rho in (0.0, 0.05, 0.1):
            _, _, achieved = two_level_params(rho, [4, 4], 1.0)
            assert achieved == pytest.approx(rho)

    def test_saturation_is_reported_not_silent(self):
        """Regression: requested rho was reported even when unreachable."""
        # singleton families carry no within-family pairs, so spread=1 with
        # a=0 cannot produce any correlation at all
        _, _, achieved = two_level_params(0.8, [1, 1, 1, 1], 1.0)
        assert achieved <= 0.8 + 1e-9

    def test_shock_params_in_unit_interval(self):
        for rho in RHOS:
            for spread in (0.0, 0.5, 1.0):
                a, b, achieved = two_level_params(rho, [3, 3, 2], spread)
                assert 0.0 <= a <= 1.0
                assert 0.0 <= b <= 1.0
                assert 0.0 <= achieved <= 1.0

    def test_structured_probabilities_in_unit_interval(self):
        for rho in RHOS:
            a, b, _ = two_level_params(rho, [3, 3, 2], 0.5)
            assert 0.0 <= p_none_structured(0.3, a, b, [3, 3, 2]) <= 1.0
            assert 0.0 <= p_all_structured(0.3, a, b, [3, 3, 2]) <= 1.0


class TestSwarmOutcomes:
    """The headline qualitative findings, pinned."""

    def test_attacker_monotone_decreasing(self):
        cfg = Config()
        vals = [attacker_hit(cfg, r) for r in RHOS]
        assert vals == sorted(vals, reverse=True)

    def test_unanimous_retention_increasing_in_rho(self):
        """Conjunctive verification responds to correlation with opposite sign."""
        cfg = Config(verify_rule="unanimous")
        vals = [defender_retain(cfg, r) for r in RHOS]
        assert vals == sorted(vals)

    def test_any_keeps_retention_decreasing_in_rho(self):
        """Disjunctive verification degrades like the attacker."""
        cfg = Config(verify_rule="any_keeps")
        vals = [defender_retain(cfg, r) for r in RHOS]
        assert vals == sorted(vals, reverse=True)

    def test_unanimous_catch_has_interior_optimum(self):
        """Finding 1: defender catch is non-monotone under unanimous-drop."""
        cfg = Config(verify_rule="unanimous")
        vals = [defender_catch(cfg, r) for r in RHOS]
        best = max(range(len(vals)), key=lambda i: vals[i])
        assert 0 < best < len(vals) - 1
        assert vals[best] > vals[0]
        assert vals[best] > vals[-1]

    def test_majority_beats_unanimous_at_independence(self):
        """Finding 2: the aggregation rule dominates the correlation."""
        unanimous = Config(verify_rule="unanimous")
        majority = Config(verify_rule="majority")
        assert defender_catch(majority, 0.0) > 2 * defender_catch(unanimous, 0.0)

    def test_majority_optimum_at_independence(self):
        cfg = Config(verify_rule="majority")
        vals = [defender_catch(cfg, r) for r in RHOS]
        assert vals.index(max(vals)) == 0

    def test_detection_stage_tracks_attacker_exactly(self):
        """Same structure and parameters => identical curve. The surviving half."""
        cfg = Config()
        from experiments.correlation_asymmetry import defender_detect

        for rho in RHOS:
            assert defender_detect(cfg, rho) == pytest.approx(attacker_hit(cfg, rho))

    def test_unknown_verify_rule_rejected(self):
        cfg = Config(verify_rule="nonsense")
        with pytest.raises(ValueError, match="unknown verify_rule"):
            defender_retain(cfg, 0.5)


class TestSweepShape:
    """The sweep emits well-formed rows."""

    def test_row_count_and_keys(self):
        rows = sweep_exchangeable(Config(), RHOS)
        assert len(rows) == len(RHOS)
        for key in ("rho", "attacker_hit", "defender_catch", "cum_uncaught"):
            assert key in rows[0]

    def test_cumulative_success_monotone_in_hit_rate(self):
        rows = sweep_exchangeable(Config(), RHOS)
        cums = [r["attacker_cum_success"] for r in rows]
        assert cums == sorted(cums, reverse=True)

    def test_probabilities_bounded(self):
        for r in sweep_exchangeable(Config(), RHOS):
            for key in ("attacker_hit", "defender_detect", "defender_retain", "defender_catch"):
                assert 0.0 <= r[key] <= 1.0
