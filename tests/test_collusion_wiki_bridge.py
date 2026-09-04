"""Tests for the collusion.wiki replay bridge (bead xmtv).

Synthetic three-wiki fixture; the real export is not committed. The tests
pin the mapping semantics (identity modes, agent vs page projection,
reply window) and that a replay produces a self-contained run folder.
"""

import json
from datetime import datetime, timezone

import pytest

from swarm.bridges.collusion_wiki import (
    ReplayConfig,
    load_revisions,
    revisions_to_interactions,
    run_replay,
)
from swarm.bridges.collusion_wiki.loader import load_events


def _rev(rev_id, page, label, ip16, t, created=False, wiki="dse"):
    return {
        "rev_id": rev_id, "wiki": wiki, "page_id": f"{wiki}/{page}",
        "label": label, "ip16": ip16, "time": t, "body_len": 10,
        "change_summary": "coordination update",
        "diff_base_reason": "page_created" if created else None,
    }


@pytest.fixture
def data_dir(tmp_path):
    rows = [
        _rev("r1", "Answers", "HelperA", "20.1", "2026-06-16T10:00:00Z", True),
        _rev("r2", "Answers", "HelperB", "20.2", "2026-06-16T10:00:30Z"),
        _rev("r3", "Answers", "HelperA", "20.1", "2026-06-16T10:01:00Z"),
        _rev("r4", "Answers", "HelperA", "20.1", "2026-06-16T10:01:20Z"),
        _rev("r5", "Answers", "HelperC", "20.1", "2026-06-16T10:02:00Z"),
        _rev("r6", "Other", "HelperB", "20.2", "2026-06-17T09:00:00Z", True),
        _rev("r7", "Other", "", "40.7", "2026-06-19T09:00:00Z"),
    ]
    with (tmp_path / "revisions.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    events = [
        {"event_type": "save", "wiki": "dse", "page": "Answers",
         "time": "2026-06-16T10:00:00Z"},
        {"event_type": "delete", "wiki": "dse", "page": "Other",
         "time": "2026-06-19T12:00:00Z", "actor_label": "[Admin1]"},
    ]
    with (tmp_path / "events.jsonl").open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return tmp_path


class TestLoader:
    def test_sorted_and_parsed(self, data_dir):
        revs = load_revisions(data_dir)
        assert [r.rev_id for r in revs] == ["r1", "r2", "r3", "r4", "r5", "r6", "r7"]
        assert revs[0].time == datetime(2026, 6, 16, 10, tzinfo=timezone.utc)
        assert revs[0].page_created and not revs[1].page_created
        assert revs[-1].editor_label == "(unlabeled)"

    def test_events_skip_saves(self, data_dir):
        ev = load_events(data_dir)
        assert [e.event_type for e in ev] == ["delete"]
        assert load_events(data_dir, types={"probe"}) == []


class TestMapper:
    def test_agent_projection_replies_to_previous_distinct_editor(self, data_dir):
        xs = revisions_to_interactions(load_revisions(data_dir), identity="label")
        # r1 creates (no counterparty); r3->r4 is a self follow-up (dropped)
        pairs = [(x.initiator, x.counterparty) for x in xs]
        assert pairs == [
            ("HelperB", "HelperA"),
            ("HelperA", "HelperB"),
            ("HelperC", "HelperA"),
            ("(unlabeled)", "HelperB"),
        ]
        assert all(x.accepted and x.p == 0.5 for x in xs)
        assert xs[0].metadata["page_id"] == "dse/Answers"

    def test_ip16_identity_collapses_handles(self, data_dir):
        xs = revisions_to_interactions(load_revisions(data_dir), identity="ip16")
        ids = {x.initiator for x in xs} | {x.counterparty for x in xs}
        assert ids == {"20.1", "20.2", "40.7"}
        # HelperA and HelperC share 20.1, so r5 is now a self follow-up
        assert len(xs) == 3

    def test_label_ip16_identity(self, data_dir):
        xs = revisions_to_interactions(load_revisions(data_dir), identity="label_ip16")
        assert xs[0].initiator == "HelperB@20.2"

    def test_reply_window_drops_slow_replies(self, data_dir):
        xs = revisions_to_interactions(
            load_revisions(data_dir), reply_window_seconds=3600
        )
        # r7 replies to r6 two days later -> dropped
        assert len(xs) == 3

    def test_page_projection_is_bipartite(self, data_dir):
        xs = revisions_to_interactions(load_revisions(data_dir), projection="page")
        assert len(xs) == 7
        assert all(x.counterparty.startswith("page:") for x in xs)

    def test_p_stays_in_unit_interval(self, data_dir):
        for x in revisions_to_interactions(load_revisions(data_dir)):
            assert 0.0 <= x.p <= 1.0


class TestReplay:
    def test_run_folder_is_self_contained(self, data_dir, tmp_path):
        cfg = ReplayConfig(
            sweep_identity=["label", "ip16"],
            structural_null_samples=5,
            timeline_null_samples=3,
            landmarks={"moderator_sweep_begins": "2026-06-19T00:00:00Z"},
        )
        out = run_replay(data_dir, cfg, tmp_path / "runs")
        assert (out / "summary.json").exists()
        assert (out / "config.json").exists()
        assert (out / "timeline.csv").exists()
        for ident in ("label", "ip16"):
            assert (out / f"pairs_{ident}.csv").exists()
            assert (out / f"structural_{ident}.csv").exists()
        s = json.loads((out / "summary.json").read_text())
        assert s["n_revisions"] == 7 and s["n_deletions"] == 1
        assert s["deletions_by_day"] == {"2026-06-19": 1}
        assert set(s["per_identity"]) == {"label", "ip16"}
        tl = s["per_identity"]["label"]["timeline"]
        assert tl["n_steps"] >= 3
        assert "temporal_vs_moderator_sweep_begins" in tl["lag_days"]

    def test_from_yaml_reads_repo_scenario(self):
        cfg = ReplayConfig.from_yaml(
            __import__("pathlib").Path("scenarios/casestudy_wiki_backchannel.yaml")
        )
        assert cfg.scenario_id == "casestudy_wiki_backchannel"
        assert cfg.sweep_identity == ["label", "ip16", "label_ip16"]
        assert "openai_hq_visit" in cfg.landmarks
