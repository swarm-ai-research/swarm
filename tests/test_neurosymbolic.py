"""Tests for the neurosymbolic behaviour-classification stack."""

from __future__ import annotations

import pytest

from swarm.neurosymbolic import (
    Atom,
    BehaviorConfig,
    KinematicPerceiver,
    MaxTimesProvenance,
    Program,
    Trajectory,
    Var,
    classify_behaviors,
    noisy_or,
    to_scallop_program,
)
from swarm.neurosymbolic.provenance import clamp01


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
class TestProvenance:
    def test_max_times_conjunction_is_product(self):
        prov = MaxTimesProvenance()
        assert prov.times(0.5, 0.4) == pytest.approx(0.2)

    def test_max_times_disjunction_is_idempotent(self):
        prov = MaxTimesProvenance()
        assert prov.plus(0.7, 0.7) == 0.7
        assert prov.plus(0.3, 0.9) == 0.9

    def test_noisy_or_independent(self):
        assert noisy_or([0.5, 0.5]) == pytest.approx(0.75)
        assert noisy_or([]) == 0.0
        assert noisy_or([1.0, 0.2]) == pytest.approx(1.0)

    def test_clamp01_keeps_invariant(self):
        assert clamp01(-0.5) == 0.0
        assert clamp01(1.5) == 1.0
        assert clamp01(0.3) == 0.3


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class TestEngine:
    def test_single_rule_propagates_probability(self):
        prog = Program()
        prog.fact("near", "a", "t", p=0.8)
        prog.rule(
            Atom("close", Var("X"), Var("Y")),
            Atom("near", Var("X"), Var("Y")),
        )
        db = prog.run()
        assert db.prob("close", "a", "t") == pytest.approx(0.8)

    def test_conjunction_multiplies(self):
        prog = Program()
        prog.fact("p", "x", p=0.5)
        prog.fact("q", "x", p=0.4)
        prog.rule(
            Atom("r", Var("X")),
            Atom("p", Var("X")),
            Atom("q", Var("X")),
        )
        db = prog.run()
        assert db.prob("r", "x") == pytest.approx(0.2)

    def test_alternative_derivations_take_max(self):
        prog = Program()
        prog.fact("edge", "a", "b", p=0.9)
        prog.fact("edge", "a", "b", p=0.3)  # combined via plus=max at load
        prog.rule(
            Atom("reach", Var("X"), Var("Y")),
            Atom("edge", Var("X"), Var("Y")),
        )
        db = prog.run()
        assert db.prob("reach", "a", "b") == pytest.approx(0.9)

    def test_transitive_closure_recursion_converges(self):
        prog = Program()
        prog.fact("edge", "a", "b", p=0.9)
        prog.fact("edge", "b", "c", p=0.8)
        prog.rule(
            Atom("path", Var("X"), Var("Y")),
            Atom("edge", Var("X"), Var("Y")),
        )
        prog.rule(
            Atom("path", Var("X"), Var("Z")),
            Atom("path", Var("X"), Var("Y")),
            Atom("edge", Var("Y"), Var("Z")),
        )
        db = prog.run()
        assert db.prob("path", "a", "b") == pytest.approx(0.9)
        assert db.prob("path", "a", "c") == pytest.approx(0.72)  # 0.9 * 0.8

    def test_wildcard_is_existential(self):
        prog = Program()
        prog.fact("knows", "a", "x", p=0.6)
        prog.fact("knows", "a", "y", p=0.9)
        prog.rule(
            Atom("social", Var("A")),
            Atom("knows", Var("A"), Var("_")),
        )
        db = prog.run()
        # max over the two existential witnesses
        assert db.prob("social", "a") == pytest.approx(0.9)

    def test_unsafe_rule_rejected(self):
        prog = Program()
        with pytest.raises(ValueError, match="Unsafe rule"):
            prog.rule(
                Atom("bad", Var("X"), Var("Y")),
                Atom("only", Var("X")),
            )

    def test_probabilities_stay_in_unit_interval(self):
        prog = Program()
        prog.fact("a", "x", p=1.0)
        prog.fact("b", "x", p=1.0)
        prog.rule(Atom("c", Var("X")), Atom("a", Var("X")), Atom("b", Var("X")))
        db = prog.run()
        for _, _, p in db.items():
            assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Perceiver (neural layer)
# ---------------------------------------------------------------------------
class TestPerceiver:
    def test_emits_soft_facts_in_unit_interval(self):
        traj = Trajectory(
            agent="a",
            positions=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            targets={"goal": [(5.0, 0.0)]},
        )
        prog = KinematicPerceiver().perceive(traj)
        facts = list(prog.edb.items())
        assert facts, "perceiver should emit facts"
        for _, _, p in facts:
            assert 0.0 <= p <= 1.0

    def test_moving_toward_high_when_aligned(self):
        # Agent walks straight at the target.
        traj = Trajectory(
            agent="a",
            positions=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            targets={"goal": [(10.0, 0.0)]},
        )
        prog = KinematicPerceiver().perceive(traj)
        assert prog.edb.prob("moving_toward", "a", "goal", 1) > 0.9
        assert prog.edb.prob("closing", "a", "goal", 1) > 0.9

    def test_increasing_distance_when_fleeing(self):
        traj = Trajectory(
            agent="a",
            positions=[(0.0, 0.0), (-1.0, 0.0), (-2.0, 0.0)],
            targets={"threat": [(5.0, 0.0)]},
        )
        prog = KinematicPerceiver().perceive(traj)
        assert prog.edb.prob("increasing_distance", "a", "threat", 1) > 0.9

    def test_requires_two_timesteps(self):
        with pytest.raises(ValueError):
            Trajectory(agent="a", positions=[(0.0, 0.0)], targets={})


# ---------------------------------------------------------------------------
# Behaviour classification (end-to-end)
# ---------------------------------------------------------------------------
class TestBehaviorClassification:
    def test_pursuit_dominates_for_chaser(self):
        # Straight-line chase, closing distance every step.
        traj = Trajectory(
            agent="hunter",
            positions=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
            targets={"prey": [(10.0, 0.0)]},
        )
        scores = classify_behaviors(traj)
        top, prob = scores.top()
        assert top == "pursuing"
        assert prob > 0.5
        assert scores.episodes["pursuing"], "should record a supporting run"

    def test_evasion_dominates_for_fleer(self):
        # Threat is close (detected), agent runs directly away from it.
        positions = [(0.0, 0.0)]
        for i in range(1, 6):
            positions.append((-float(i), 0.0))
        traj = Trajectory(
            agent="rabbit",
            positions=positions,
            targets={"fox": [(1.0, 0.0)]},
        )
        scores = classify_behaviors(traj)
        assert scores.scores["evading"] > 0.5
        assert scores.scores["evading"] >= scores.scores["pursuing"]

    def test_scores_are_probabilities(self):
        traj = Trajectory(
            agent="a",
            positions=[(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0)],
            targets={"goal": [(5.0, 0.0)]},
        )
        scores = classify_behaviors(traj)
        assert set(scores.scores) == {"pursuing", "evading", "foraging"}
        for p in scores.scores.values():
            assert 0.0 <= p <= 1.0

    def test_min_run_length_gates_pursuit(self):
        traj = Trajectory(
            agent="a",
            positions=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            targets={"g": [(10.0, 0.0)]},
        )
        # Only 2 closing steps available; requiring 4 should yield no run.
        strict = classify_behaviors(traj, config=BehaviorConfig(min_run_length=4))
        assert strict.scores["pursuing"] == 0.0

    def test_longer_pursuit_run_scores_below_shorter_window(self):
        # Each step probability < 1, so the product over more steps is smaller:
        # confirms probabilities genuinely compound through recursion.
        traj = Trajectory(
            agent="a",
            positions=[(0.0, 0.0), (1.0, 0.3), (2.0, 0.0), (3.0, 0.3), (4.0, 0.0)],
            targets={"g": [(10.0, 0.0)]},
        )
        short = classify_behaviors(traj, config=BehaviorConfig(min_run_length=2))
        long = classify_behaviors(traj, config=BehaviorConfig(min_run_length=4))
        if long.scores["pursuing"] > 0 and short.scores["pursuing"] > 0:
            assert long.scores["pursuing"] <= short.scores["pursuing"]

    def test_reported_run_is_highest_probability_not_longest(self):
        # A confident close-up chase that gains one extra, lower-confidence step
        # should not have its score dragged down by the longer window: the
        # readout keeps the most-confident qualifying run, not the longest.
        from swarm.neurosymbolic import KinematicPerceiver, add_behavior_rules

        traj = Trajectory(
            agent="a",
            positions=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 1.5)],
            targets={"g": [(10.0, 0.0)]},
        )
        min_len = 3

        # Ground truth: the max probability over *all* qualifying runs the engine
        # derived. Keeping the longest-per-start run could report less than this.
        prog = KinematicPerceiver().perceive(traj)
        add_behavior_rules(prog)
        db = prog.run()
        engine_max = max(
            (p for gt, p in db.get("pursuit_run").items()
             if gt[0] == "a" and int(gt[3]) - int(gt[2]) + 1 >= min_len),
            default=0.0,
        )
        assert engine_max > 0.0

        scores = classify_behaviors(traj, config=BehaviorConfig(min_run_length=min_len))
        assert scores.scores["pursuing"] == pytest.approx(engine_max)


# ---------------------------------------------------------------------------
# Scallop bridge
# ---------------------------------------------------------------------------
class TestScallopBridge:
    def test_program_text_mentions_all_behaviours(self):
        scl = to_scallop_program()
        for rel in ("pursuit_run", "evade_run", "forage_cycle", "alerted"):
            assert rel in scl

    def test_rule_text_matches_engine_relations(self):
        # The .scl text should reference the same input relations the perceiver emits.
        scl = to_scallop_program()
        for rel in ("moving_toward", "closing", "increasing_distance", "succ"):
            assert rel in scl
