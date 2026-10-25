"""Tests for scripts/layer_lag.py (docs/research/pace-layers-swarm-lag.md).

Pace layers as a check: a change to a file a prereg pins stays OPEN until a
doc commit names the pin. The naa8 case drives the fixtures: the commit that
moved the rubric default also touched the prereg, so "any doc edit" must not
count as ratification.
"""

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from layer_lag import (  # noqa: E402
    Commit,
    change_refs,
    churn_ratio,
    classify,
    extract_pins,
    fast_events,
    fill_weeks,
    incident_test,
    ratification_lags,
    ratifies,
    weekly_churn,
)

DOC = "docs/research/calibration-prereg.md"
PIN = "swarm/judges/rubric_v1.md"
T0 = datetime(2026, 6, 1, tzinfo=timezone.utc)


def at(days: float, sha: str, *files: str) -> Commit:
    return Commit(sha, T0 + timedelta(days=days), files)


class TestClassify:
    def test_layers(self):
        assert classify(".beads/issues.jsonl") == "fast"
        assert classify("swarm/core/proxy.py") == "middle"
        assert classify("docs/research/calibration-prereg.md") == "slow"
        assert classify("CLAUDE.md") == "slow"

    def test_unlayered(self):
        assert classify("docs/research/theory-notes.md") is None
        assert classify("README.md") is None


class TestPins:
    def test_known_paths_only(self):
        text = "locked at `swarm/judges/rubric_v1.md`, see `p` and `nope/x.py`"
        assert extract_pins(text, {PIN}) == [PIN]

    def test_line_and_anchor_suffixes(self):
        text = "`swarm/judges/rubric_v1.md:12` and `swarm/judges/rubric_v1.md#v1`"
        assert extract_pins(text, {PIN}) == [PIN]

    def test_ratifies_needs_the_pin_name(self):
        assert ratifies(["Arm B stays on `swarm/judges/rubric_v1.md`"], PIN)
        assert ratifies(["rubric_v1.md is still the lock"], PIN)
        assert not ratifies(["Pilot re-run (rubric v3): see findings"], PIN)
        assert not ratifies(["Arm A deviation analysis"], PIN)

    def test_touch_mode_ratifies_anything(self):
        assert ratifies([], None)

    def test_refs_from_subject(self):
        known = {"naa8", "5cdk"}
        assert change_refs("feat: metrics (beads-5cdk) (#501)", known) == {"beads-5cdk", "#501"}
        assert change_refs("fix: pin Arm B rubric (naa8)", known) == {"naa8", "beads-naa8"}
        assert change_refs("fix(blog): thing (deps)", known) == set()

    def test_ratifies_by_bead_ref(self):
        line = ["## Operating-point metrics (2026-06-05, beads-5cdk)"]
        assert ratifies(line, PIN, {"beads-5cdk"})
        assert not ratifies(line, PIN, {"#50"})
        assert not ratifies(["see #5010"], PIN, {"#501"})


class TestLag:
    def commits(self):
        return [
            at(0, "reg", DOC),
            at(0.5, "born", PIN),
            at(4, "drift", "swarm/judges/judge.py", DOC),  # naa8: default moves, doc gets a pointer
            at(10, "edit", PIN),
            at(20, "amend", DOC),
        ]

    def test_touch_scores_naa8_as_lag_zero(self):
        lags = ratification_lags(self.commits(), DOC, [PIN], "dir", None)
        drift = [c for c in lags if c.sha == "drift"]
        assert drift and drift[0].ratified == T0 + timedelta(days=4)

    def test_strict_keeps_naa8_open(self):
        edits = {"reg": ["lock `swarm/judges/rubric_v1.md`"], "drift": ["Pilot re-run (rubric v3)"],
                 "amend": ["rubric_v1.md amended"]}
        lags = {c.sha: c for c in ratification_lags(self.commits(), DOC, [PIN], "dir", edits)}
        assert lags["drift"].ratified == T0 + timedelta(days=20)
        assert lags["edit"].ratified == T0 + timedelta(days=20)

    def test_addendum_naming_the_bead_ratifies(self):
        commits = [at(0, "reg", DOC), at(1, "born", PIN),
                   Commit("fix", T0 + timedelta(days=2), (PIN, DOC), "feat: x (beads-5cdk)")]
        edits = {"reg": [], "fix": ["## Addendum (beads-5cdk)"]}
        (change,) = ratification_lags(commits, DOC, [PIN], "file", edits, {"5cdk"})
        assert change.ratified == T0 + timedelta(days=2)

    def test_file_scope_misses_sibling(self):
        shas = {c.sha for c in ratification_lags(self.commits(), DOC, [PIN], "file", None)}
        assert shas == {"edit"}

    def test_pin_creation_is_not_drift(self):
        shas = {c.sha for c in ratification_lags(self.commits(), DOC, [PIN], "file", None)}
        assert "born" not in shas

    def test_never_ratified_is_open(self):
        commits = [at(0, "reg", DOC), at(1, "born", PIN), at(2, "edit", PIN)]
        (change,) = ratification_lags(commits, DOC, [PIN], "file", {"reg": []})
        assert change.ratified is None
        assert change.lag_days(T0 + timedelta(days=5)) == 3

    def test_undocumented_doc_has_no_lags(self):
        assert ratification_lags([at(1, "x", PIN)], DOC, [PIN], "file", None) == []


class TestChurn:
    def test_counts_once_per_layer(self):
        churn = weekly_churn([at(0, "a", ".beads/a", ".beads/b", "swarm/x.py"), at(1, "b", DOC)])
        assert churn == {"2026-W23": {"fast": 1, "middle": 1, "slow": 1}}

    def test_ratio_is_smoothed(self):
        assert churn_ratio({"fast": 3, "middle": 0, "slow": 0}) == 4.0

    def test_incident_window(self):
        churn = {
            "2026-W20": {"fast": 9, "middle": 0, "slow": 0},
            "2026-W22": {"fast": 1, "middle": 0, "slow": 1},
            "2026-W30": {"fast": 0, "middle": 0, "slow": 0},
        }
        (row,) = incident_test(churn, [date(2026, 6, 5)], window=4)
        assert row["weeks_before"] == 2
        assert row["ratio_before"] == 5.5
        assert row["ratio_rest"] == 1.0


class TestFastEvents:
    def test_beads_and_messages_counted_per_week(self):
        beads = [{"created_at": "2026-06-01T10:00:00Z", "closed_at": "2026-06-02T10:00:00Z"},
                 {"created_at": "2026-06-09T10:00:00Z", "closed_at": None}]
        msgs = ["2026-06-03T00:00:00.000Z"]
        assert fast_events(beads, msgs) == {"2026-W23": 3, "2026-W24": 1}

    def test_fill_weeks_adds_quiet_weeks(self):
        churn = {"2026-W20": {"fast": 1, "middle": 0, "slow": 0},
                 "2026-W23": {"fast": 2, "middle": 0, "slow": 0}}
        filled = fill_weeks(churn)
        assert list(filled) == ["2026-W20", "2026-W21", "2026-W22", "2026-W23"]
        assert filled["2026-W21"] == {"fast": 0, "middle": 0, "slow": 0}
