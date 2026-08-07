"""Tests for scripts/dispatch_staleness.py (beads-y0cx).

The false-zombie bug: bd's updated_at doesn't bump on comments, and run
artifacts live outside bd, so staleness computed from updated_at alone
flagged actively-worked beads (k2yr: stale-46d despite a same-day run
and comments). These tests pin the effective-activity rule:
max(updated_at, last comment, newest referencing runs/ artifact).
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from dispatch_staleness import (  # noqa: E402
    build_report,
    effective_activity,
    latest_run_activity,
    parse_ts,
    short_id,
)

NOW = datetime(2026, 7, 27, 12, 0, 0, tzinfo=timezone.utc)


class TestParsing:
    def test_short_id(self):
        assert short_id("distributional-agi-safety-k2yr") == "k2yr"
        assert short_id("plain") == "plain"

    def test_parse_ts_offset_and_z(self):
        a = parse_ts("2026-07-18T22:42:00-04:00")
        b = parse_ts("2026-07-19T02:42:00Z")
        assert a == b

    def test_parse_ts_invalid(self):
        assert parse_ts("") is None
        assert parse_ts("not-a-date") is None


class TestEffectiveActivity:
    def test_max_of_signals_wins(self):
        upd = NOW - timedelta(days=46)
        comment = NOW - timedelta(days=1)
        run = NOW - timedelta(hours=2)
        ts, source = effective_activity(upd, comment, run)
        assert ts == run
        assert source == "run_artifact"

    def test_updated_at_alone(self):
        upd = NOW - timedelta(days=3)
        ts, source = effective_activity(upd, None, None)
        assert ts == upd
        assert source == "updated_at"

    def test_all_missing(self):
        ts, source = effective_activity(None, None, None)
        assert ts is None
        assert source == "none"


class TestRunScan:
    def _make_run(self, root, name, content="", age_days=0.0):
        d = root / name
        d.mkdir(parents=True)
        (d / "config.json").write_text(content or "{}")
        mtime = (NOW - timedelta(days=age_days)).timestamp()
        os.utime(d, (mtime, mtime))
        return d

    def test_reference_in_dir_name(self, tmp_path):
        self._make_run(tmp_path, "20260718_arm_a_k2yr_seed42", age_days=2)
        ts = latest_run_activity("distributional-agi-safety-k2yr", tmp_path)
        assert ts is not None
        assert abs((NOW - ts).total_seconds() - 2 * 86400) < 3600

    def test_reference_in_metadata_file(self, tmp_path):
        self._make_run(
            tmp_path, "20260718_calibration_seed42",
            content='{"bead": "distributional-agi-safety-k2yr"}', age_days=1)
        ts = latest_run_activity("distributional-agi-safety-k2yr", tmp_path)
        assert ts is not None

    def test_short_id_requires_word_boundary(self, tmp_path):
        # "k2yr" embedded inside a longer token must NOT match.
        self._make_run(tmp_path, "20260718_tok2yrx_seed0", age_days=1)
        assert latest_run_activity(
            "distributional-agi-safety-k2yr", tmp_path) is None

    def test_newest_of_multiple_references(self, tmp_path):
        self._make_run(tmp_path, "20260701_k2yr_old", age_days=20)
        self._make_run(tmp_path, "20260726_k2yr_new", age_days=1)
        ts = latest_run_activity("distributional-agi-safety-k2yr", tmp_path)
        assert abs((NOW - ts).total_seconds() - 1 * 86400) < 3600

    def test_missing_runs_dir(self, tmp_path):
        assert latest_run_activity(
            "distributional-agi-safety-k2yr", tmp_path / "nope") is None


class TestReport:
    def test_false_zombie_rescued_by_comment_and_run(self, tmp_path):
        """The k2yr scenario: updated_at 46d ago, same-day comment and
        run artifact -> effective staleness ~0, not 46."""
        runs = tmp_path / "runs"
        d = runs / "20260718_arm_a_k2yr_seed42"
        d.mkdir(parents=True)
        (d / "summary.csv").write_text("ok")
        mtime = (NOW - timedelta(hours=6)).timestamp()
        os.utime(d, (mtime, mtime))

        issues = [{
            "id": "distributional-agi-safety-k2yr",
            "title": "Calibration arm A",
            "updated_at": (NOW - timedelta(days=46)).isoformat(),
        }]
        comments = {"distributional-agi-safety-k2yr": [
            {"created_at": (NOW - timedelta(hours=12)).isoformat()},
        ]}
        rows = build_report(issues, comments, runs, NOW)
        row = rows[0]
        assert row["naive_stale_days"] == 46.0
        assert row["stale_days"] < 1.0
        assert row["activity_source"] == "run_artifact"

    def test_genuine_zombie_stays_stale(self, tmp_path):
        issues = [{
            "id": "distributional-agi-safety-dead",
            "title": "Actually abandoned",
            "updated_at": (NOW - timedelta(days=30)).isoformat(),
        }]
        rows = build_report(issues, {}, tmp_path / "runs", NOW)
        assert rows[0]["stale_days"] == 30.0
        assert rows[0]["activity_source"] == "updated_at"

    def test_sorted_stalest_first(self, tmp_path):
        issues = [
            {"id": "a-1", "title": "fresh",
             "updated_at": (NOW - timedelta(days=1)).isoformat()},
            {"id": "a-2", "title": "old",
             "updated_at": (NOW - timedelta(days=20)).isoformat()},
        ]
        rows = build_report(issues, {}, tmp_path / "runs", NOW)
        assert [r["id"] for r in rows] == ["a-2", "a-1"]
