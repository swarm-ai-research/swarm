"""Tests for the collusion.wiki replay bridge (bead xmtv).

Synthetic three-wiki fixture; the real export is not committed. The tests
pin the mapping semantics (identity modes, agent vs page projection,
reply window) and that a replay produces a self-contained run folder.
"""

import csv
import json
from datetime import datetime, timezone

import pytest
import yaml as _yaml

from swarm.bridges.collusion_wiki import (
    ReplayConfig,
    load_revisions,
    revisions_to_interactions,
    run_replay,
)
from swarm.bridges.collusion_wiki.loader import load_events
from swarm.bridges.collusion_wiki.stego import classify, decode_tags, run_stego, scan
from swarm.bridges.collusion_wiki.synthetic import SyntheticConfig, generate, run


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


# ---------------------------------------------------------------------------
# Synthetic message-board twin (scenarios/message_board_channel.yaml)
# ---------------------------------------------------------------------------


def _small_scenario(tmp_path):
    doc = {
        "scenario_id": "message_board_channel_test",
        "kind": "synthetic",
        "seed": 7,
        "synthetic": {
            "n_honest": 40, "honest_edits_lambda": 2.0, "honest_hub_fraction": 0.5,
            "n_colluder_operators": 4, "operator_edits": 40, "hub_page": "SandBox",
            "wiki": "synthwiki",
            "honest_window": ["2026-06-01T00:00:00Z", "2026-06-25T00:00:00Z"],
            "coordination_window": ["2026-06-16T00:00:00Z", "2026-06-19T00:00:00Z"],
            "moderator_sweep_begins": "2026-06-19T00:00:00Z", "n_sweep_deletions": 10,
        },
        "replay": {
            "identity": "label", "projection": "agent", "temporal_window_seconds": 60,
            "temporal_alarm": 0.7, "structural_min_size": 3,
            "structural_null_samples": 24, "structural_alarm_pvalue": 0.05,
            "timeline_step_hours": 24, "timeline_null_samples": 5,
            "landmarks": {"moderator_sweep_begins": "2026-06-19T00:00:00Z"},
        },
        "sweep": {"identity": ["label", "ip16"]},
    }
    p = tmp_path / "message_board_channel_test.yaml"
    p.write_text(_yaml.safe_dump(doc))
    return p


class TestSyntheticBoard:
    def test_generate_is_deterministic(self):
        cfg = SyntheticConfig(seed=7, n_honest=20, n_colluder_operators=3, operator_edits=12)
        a_revs, a_ev, a_gt = generate(cfg)
        b_revs, b_ev, b_gt = generate(cfg)
        assert a_revs == b_revs and a_ev == b_ev and a_gt == b_gt

    def test_colluders_on_hub_honest_mostly_own_pages(self):
        cfg = SyntheticConfig(seed=7, n_honest=20, honest_hub_fraction=0.5,
                              n_colluder_operators=3, operator_edits=12, hub_page="SandBox")
        revs, _ev, gt = generate(cfg)
        hub = [r for r in revs if r["page_id"] == "SandBox"]
        # every colluder edit is on the hub; colluder ip16 blocks are the ground truth
        colluder_ip = set(gt["colluder_ip16"])
        assert colluder_ip and colluder_ip.isdisjoint({f"10.{i}" for i in range(256)})
        assert all(r["page_id"].startswith("Page_") or r["page_id"] == "SandBox" for r in revs)
        assert len(hub) > 0

    def test_run_scores_and_volume_leads_sweep(self, tmp_path):
        scenario = _small_scenario(tmp_path)
        out = run(scenario, tmp_path / "runs")
        ev = json.loads((out / "synthetic_eval.json").read_text())
        scores = ev["detector_scores"]
        # volume monitor fires before the moderator sweep
        assert ev["volume_monitor"]["lead_days_vs_sweep"] is not None
        assert ev["volume_monitor"]["lead_days_vs_sweep"] > 0
        # structural saturates where it is significant: the flagged community
        # sweeps in honest hub-users, so precision is well below 1
        ip16_struct = scores["ip16"]["structural"]
        assert ip16_struct["fp"] > 0 and ip16_struct["precision"] < 1.0
        # the per-agent temporal score is a handle artifact: it collapses to
        # noise when handles are folded to their operator IP block
        assert scores["ip16"]["temporal"]["recall"] == 0.0
        assert scores["ip16"]["temporal_max"] < 0.3


class TestVolumeInTimeline:
    """bead hoer: the aggregate volume monitor rides along in the timeline."""

    def test_timeline_carries_volume_columns(self, data_dir, tmp_path):
        cfg = ReplayConfig(structural_null_samples=5, timeline_null_samples=5,
                           volume_alarm_ratio=2.0, landmarks={"sweep": "2026-06-19T00:00:00Z"})
        out = run_replay(data_dir, cfg, runs_root=tmp_path)
        rows = list(csv.DictReader((out / "timeline.csv").open()))
        assert {"volume_ratio", "volume_alarm"} <= set(rows[0].keys())
        summary = json.loads((out / "summary.json").read_text())
        tl = summary["per_identity"]["label"]["timeline"]
        assert "first_volume_alarm" in tl
        assert "volume_vs_sweep" in tl["lag_days"]
        assert "volume" in summary["per_identity"]["label"]


class TestStegoScan:
    """bead vv3j.5: hidden-character carriers in the export's text fields."""

    def test_classify_tag_chars_and_decode(self):
        hidden = "help" + "".join(chr(0xE0000 + ord(c)) for c in "hyperplex.org") + "peer"
        assert "tag_chars" in classify(hidden)
        assert decode_tags(hidden) == "hyperplex.org"
        assert classify("API research links") == []

    def test_classify_zero_width_bidi_mixed_script(self):
        assert "zero_width" in classify("Research\u200bHelper")
        assert "bidi_controls" in classify("abc\u202edef")
        assert "mixed_script" in classify("Res\u0435archHelper")  # Cyrillic е

    def test_scan_counts_by_source_and_day(self, data_dir, tmp_path):
        hidden = "note" + "".join(chr(0xE0000 + ord(c)) for c in "x.y") + "z"
        with (data_dir / "revisions.jsonl").open("a") as f:
            f.write(json.dumps(_rev("r9", "Answers", "HelperZ", "20.9",
                                    "2026-06-20T10:00:00Z") | {"change_summary": hidden}) + "\n")
        rep = scan(data_dir)
        assert rep.n_flagged >= 1
        assert rep.by_carrier.get("tag_chars") == 1
        assert rep.by_source.get("revision.change_summary") == 1
        assert rep.by_day.get("2026-06-20") == 1
        assert rep.findings[0].decoded == "x.y" or any(
            f.decoded == "x.y" for f in rep.findings)
        out = run_stego(data_dir, tmp_path)
        assert (out / "summary.json").exists() and (out / "findings.jsonl").exists()

    def test_clean_fixture_is_clean(self, data_dir):
        rep = scan(data_dir)
        assert rep.by_carrier.get("tag_chars", 0) == 0
