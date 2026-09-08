"""Tests for the swarm.termina.digital loader and post-disclosure replay
(beads lnaf, sjis).

A synthetic ``incidents.sqlite`` with the three tables the bridge reads
(venue, actor, record) and a manifest; the real 50 MB db is not committed.
The tests pin the record -> WikiRevision mapping (actor kinds, /16 recovery,
time formats, window and kind filters), the fingerprint marks against a
baseline, the sha256 pin warning, and that the replay writes a
self-contained run folder per venue and window.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml as _yaml

from swarm.bridges.collusion_wiki import ReplayConfig, run_termina_replay
from swarm.bridges.collusion_wiki import termina as T

_SCHEMA = """
CREATE TABLE venue (id text primary key, host text not null, path text not null default '',
    software text, kind text not null, status text not null, moderator text,
    first_seen text, last_seen text, caveat text);
CREATE TABLE actor (id text primary key, kind text not null, name text not null,
    venue_id text, first_seen text, last_seen text, notes text not null default '');
CREATE TABLE record (id text primary key, venue_id text not null, kind text not null,
    external_id text, title text, summary text, actor_id text, ip_actor_id text,
    observed_time text, body_len integer, phase text not null, status text not null default 'live',
    source text not null, campaign_id text, content_kind text);
"""

# 1781807128 is 2026-06-18T18:25:28Z, inside the baseline rows' span below
_ROWS = [
    # id, venue, kind, ext, title, summary, actor, ip_actor, time, phase, source, campaign
    ("probier~AgentAlphaX1@1", "probier", "rc-row", "1.1", "AgentAlphaX1", "seed",
     "handle:probier:DataHelperAgent", None, "2026-06-18T18:20Z", "pre-disclosure", "live-rc", "swarm-cohort"),
    ("probier~AgentAlphaX1@2", "probier", "rc-row", "1.2", "AgentAlphaX1", "ack",
     "handle:probier:OtherResearcherBot", None, "2026-06-18T18:21Z", "pre-disclosure", "live-rc", "swarm-cohort"),
    ("probier~Agent010LeminoDirect1781807128@1", "probier", "rc-row", None, "Agent010LeminoDirect1781807128",
     "agent010", "ip:20.80.1.x", None, "2026-06-18T18:30Z", "pre-disclosure", "live-rc", "swarm-retrieval"),
    ("probier~AgentAlphaX1@r", "probier", "revision", "1.1", "AgentAlphaX1", None,
     "handle:probier:DataHelperAgent", "ip16:20.80", "2026-06-18T18:20:10Z", "pre-disclosure", "collusion-export", None),
    # moderator, July
    ("probier~AgentAlphaX1@3", "probier", "rc-row", None, "AgentAlphaX1", "Seite gelöscht.",
     "human:probier:118", None, "2026-07-05T10:00Z", "pre-disclosure", "live-rc", None),
    # post-disclosure: a re-save of a June title from AWS, a visitor handle, a bare-ip row
    ("probier~Agent010LeminoDirect1781807128@2", "probier", "rc-row", None, "Agent010LeminoDirect1781807128",
     "agent010", "ip:ec2-3-230-123-10.compute-1.amazonaws.com", None, "2026-09-07T02:27Z", "post-press", "live-rc", None),
    ("probier~SandBox@9", "probier", "rc-row", None, "SandBox", "invitation",
     "handle:probier:CentaurAgent", None, "2026-09-04T16:44Z", "post-press", "live-rc", "visitors"),
    ("probier~PublicBoard@1", "probier", "rc-row", None, "PublicBoard", "PublicBoard relay",
     "ip:159.146.96.x", None, "2026-09-06T20:33Z", "post-press", "live-rc", None),
    ("probier~SandBox@10", "probier", "rc-row", None, "SandBox", "reply",
     "ip:84.17.35.x", None, "2026-09-06T13:42Z", "post-press", "live-rc", None),
    ("probier~Untimed@1", "probier", "rc-row", None, "Untimed", None,
     "ip:1.2.3.x", None, None, "unknown", "shellac-pack", None),
    # another venue, day-precision time, mapped wiki name
    ("usemod-org~SandBox@1", "usemod-org", "rc-row", None, "SandBox", "fleet",
     "ip:66.54.102.x", None, "2026-08-30", "pre-disclosure", "live-rc", "usemod-fleet"),
]


@pytest.fixture
def db_dir(tmp_path: Path) -> Path:
    db = tmp_path / T.SQLITE_NAME
    con = sqlite3.connect(db)
    con.executescript(_SCHEMA)
    con.executemany(
        "INSERT INTO venue (id, host, path, software, kind, status, moderator, first_seen, last_seen, caveat) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            ("probier", "wikiservice.at", "/probier", "prowiki", "wiki", "live", None, "2026-05-27", None, None),
            ("usemod-org", "usemod.org", "/cgi-bin/wiki.pl", "usemod", "wiki", "live", None, "2026-05-26", "2026-08-30", None),
        ],
    )
    con.executemany(
        "INSERT INTO record (id, venue_id, kind, external_id, title, summary, actor_id, ip_actor_id, "
        "observed_time, body_len, phase, source, campaign_id) VALUES (?,?,?,?,?,?,?,?,?,10,?,?,?)",
        _ROWS,
    )
    con.commit()
    con.close()
    (tmp_path / T.MANIFEST_NAME).write_text(json.dumps({
        "schema_version": "9", "generated_at": "2026-09-08T14:10:58Z",
        "licence": "CC0-1.0",
        "files": {"record.jsonl": {"rows": len(_ROWS), "sha256": "ab" * 32}},
    }))
    return tmp_path


class TestParsing:
    @pytest.mark.parametrize("raw, expected", [
        ("2026-07-03T08:05Z", datetime(2026, 7, 3, 8, 5, tzinfo=timezone.utc)),
        ("2026-06-22T08:45:55Z", datetime(2026, 6, 22, 8, 45, 55, tzinfo=timezone.utc)),
        ("2026-03-11T11:09", datetime(2026, 3, 11, 11, 9, tzinfo=timezone.utc)),
        ("2026-08-30", datetime(2026, 8, 30, tzinfo=timezone.utc)),
        (None, None),
        ("", None),
        ("not a time", None),
    ])
    def test_parse_time(self, raw, expected):
        assert T.parse_time(raw) == expected

    def test_actor_parts(self):
        assert T.actor_parts("handle:dse:AiraBot") == ("handle", "AiraBot")
        assert T.actor_parts("human:dse:118") == ("human", "human-118")
        assert T.actor_parts("ip:84.115.212.x") == ("ip", "84.115.212.x")
        assert T.actor_parts(None) == ("", "")

    def test_ip16_recovery(self):
        assert T.ip16_of(None, "ip16:20.80") == "20.80"
        assert T.ip16_of("ip:84.115.212.x", None) == "84.115"
        assert T.ip16_of("ip:*.plus.com", None) == ""
        assert T.ip16_of("handle:dse:AiraBot", None) == ""

    def test_cloud_from_rdns(self):
        assert T.cloud_of("ip:ec2-3-230-123-10.compute-1.amazonaws.com") == "aws"
        assert T.cloud_of("ip:182.49.96.34.bc.googleusercontent.com") == "gcp"
        assert T.cloud_of("ip:66.54.102.x") == ""
        assert T.cloud_of("handle:dse:AiraBot") == ""

    def test_grammars(self):
        assert T.HANDLE_GRAMMAR.match("DataHelperAgent")
        assert T.HANDLE_GRAMMAR.match("OpenAINov28CVD") is None  # cohort tag, not a role noun
        assert T.HANDLE_GRAMMAR.match("help_peer") is None
        assert T.TITLE_GRAMMAR.match("Agent010LeminoDirect1781807128")
        assert T.TITLE_GRAMMAR.match("OpenAICompactTexasFemaleTwoMore2015Y7")
        assert T.TITLE_GRAMMAR.match("ShortOfferings2020A1781808567")
        assert T.TITLE_GRAMMAR.match("SandBox") is None
        assert T.TITLE_GRAMMAR.match("PublicBoard") is None


class TestLoader:
    def test_rc_rows_only_by_default_sorted_and_mapped(self, db_dir):
        revs = T.load_revisions(db_dir, venues=["probier", "usemod-org"])
        # the export-duplicate revision and the untimed row are dropped
        assert all(r.record_kind == "rc-row" for r in revs)
        assert [r.rev_id for r in revs][:3] == [
            "probier~AgentAlphaX1@1", "probier~AgentAlphaX1@2",
            "probier~Agent010LeminoDirect1781807128@1",
        ]
        r0 = revs[0]
        assert r0.wiki == "probier" and r0.page_id == "probier/AgentAlphaX1"
        assert r0.label == "DataHelperAgent" and r0.actor_kind == "handle"
        assert r0.page_created and not revs[1].page_created
        assert r0.source == "termina:live-rc" and r0.campaign == "swarm-cohort"
        assert r0.phase == "pre-disclosure"
        ip_row = revs[2]
        assert ip_row.label == "" and ip_row.editor_label == "(unlabeled)"
        assert ip_row.ip16 == "20.80" and ip_row.actor_kind == "ip"
        usemod = [r for r in revs if r.wiki == "usemod"]
        assert len(usemod) == 1 and usemod[0].page_id == "usemod/SandBox"
        assert usemod[0].time == datetime(2026, 8, 30, tzinfo=timezone.utc)

    def test_human_rows_are_labelled_and_excludable(self, db_dir):
        revs = T.load_revisions(db_dir, venues=["probier"])
        humans = [r for r in revs if r.actor_kind == "human"]
        assert [h.label for h in humans] == ["human-118"]
        kept = T.load_revisions(db_dir, venues=["probier"], exclude_actor_kinds=["human"])
        assert not any(r.actor_kind == "human" for r in kept)
        assert len(kept) == len(revs) - 1

    def test_window_and_record_kinds(self, db_dir):
        post = T.load_revisions(
            db_dir, venues=["probier"],
            window=(datetime(2026, 7, 3, tzinfo=timezone.utc), datetime(2026, 9, 8, tzinfo=timezone.utc)),
        )
        assert {r.rev_id for r in post} == {
            "probier~AgentAlphaX1@3", "probier~Agent010LeminoDirect1781807128@2",
            "probier~SandBox@9", "probier~PublicBoard@1", "probier~SandBox@10",
        }
        export_dups = T.load_revisions(db_dir, venues=["probier"], record_kinds=["revision"])
        assert [r.rev_id for r in export_dups] == ["probier~AgentAlphaX1@r"]
        assert export_dups[0].ip16 == "20.80" and export_dups[0].source == "termina:collusion-export"

    def test_missing_db_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            T.load_revisions(tmp_path)

    def test_manifest_and_pin(self, db_dir, caplog):
        m = T.read_manifest(db_dir)
        assert m.schema_version == "9" and m.rows["record.jsonl"] == len(_ROWS)
        db = db_dir / T.SQLITE_NAME
        assert T.check_pin(db, None) is True
        assert T.check_pin(db, T.sha256_file(db)) is True
        with caplog.at_level("WARNING"):
            assert T.check_pin(db, "0" * 64) is False
        assert "differs from the scenario pin" in caplog.text

    def test_venue_table(self, db_dir):
        rows = T.venue_table(db_dir, ["probier", "usemod-org"])
        assert [r["id"] for r in rows] == ["probier", "usemod-org"]
        assert rows[1]["last_seen"] == "2026-08-30"

    def test_empty_filters_yield_nothing(self, db_dir):
        """Empty IN-lists must not become ``IN ()`` (SQLite OperationalError)."""
        assert T.load_revisions(db_dir, venues=[]) == []
        assert T.load_revisions(db_dir, record_kinds=[]) == []
        assert T.load_revisions(db_dir, venues=[], record_kinds=[]) == []
        assert T.venue_table(db_dir, []) == []
        assert list(T.iter_revisions(db_dir, venues=[])) == []
        assert list(T.iter_revisions(db_dir, record_kinds=[])) == []
        assert T.load_revisions(db_dir, venues=["no-such-venue"]) == []
        assert T.venue_table(db_dir, ["no-such-venue"]) == []


class TestFingerprint:
    def test_marks_against_baseline(self, db_dir):
        win_b = (datetime(2026, 5, 24, tzinfo=timezone.utc), datetime(2026, 7, 3, tzinfo=timezone.utc))
        win_p = (datetime(2026, 7, 3, tzinfo=timezone.utc), datetime(2026, 9, 8, tzinfo=timezone.utc))
        base = T.load_revisions(db_dir, venues=["probier"], window=win_b, exclude_actor_kinds=["human"])
        post = T.load_revisions(db_dir, venues=["probier"], window=win_p, exclude_actor_kinds=["human"])
        fb = T.fingerprint(base)
        assert fb["n_rows"] == 3 and fb["n_handles"] == 2 and fb["n_titles"] == 2
        assert fb["handle_grammar_share"] == 1.0
        assert fb["title_grammar_share"] == 1.0
        assert fb["handle_in_baseline"] is None  # no baseline given
        assert fb["termina_campaigns"] == {"swarm-cohort": 2, "swarm-retrieval": 1}
        fp = T.fingerprint(post, base)
        assert fp["n_rows"] == 4 and fp["n_titles"] == 3 and fp["n_handles"] == 1
        # CentaurAgent matches the grammar but is not a June handle
        assert fp["handle_grammar_share"] == 1.0 and fp["handle_in_baseline"] == 0.0
        # one of three titles is a June page, and it is the one carrying a June epoch
        assert fp["title_in_baseline"] == pytest.approx(1 / 3, abs=1e-3)
        assert fp["title_epoch_in_baseline"] == pytest.approx(1 / 3, abs=1e-3)
        assert fp["title_grammar_share"] == pytest.approx(1 / 3, abs=1e-3)
        assert fp["cloud_from_rdns"] == {"aws": 1}
        assert fp["actor_kinds"] == {"handle": 1, "ip": 3}
        # the AWS rDNS row has no dotted /16; 159.146 and 84.17 are new
        assert fp["n_ip16"] == 2 and fp["ip16_in_baseline"] == 0.0
        assert fp["peak_day"] == "2026-09-06" and fp["peak_day_rows"] == 2

    def test_empty_window(self):
        fp = T.fingerprint([])
        assert fp["n_rows"] == 0 and fp["peak_day"] is None
        assert fp["handle_grammar_share"] is None and fp["rows_per_handle"] is None


class TestReplay:
    def test_run_folder_per_venue_and_window(self, db_dir, tmp_path):
        cfg = ReplayConfig(
            scenario_id="casestudy_wiki_postdisclosure",
            source="termina",
            identity="actor",
            venues=["probier", "usemod-org"],
            sweep_identity=["actor", "ip16"],
            structural_null_samples=3,
            timeline_null_samples=2,
            baseline_window=["2026-05-24T00:00:00Z", "2026-07-03T00:00:00Z"],
            post_window=["2026-07-03T00:00:00Z", "2026-09-08T00:00:00Z"],
            landmarks={"press_coverage": "2026-09-04T00:00:00Z"},
        )
        out = run_termina_replay(db_dir, cfg, tmp_path / "runs")
        s = json.loads((out / "summary.json").read_text())
        assert s["source"] == "termina" and s["db_pin_matches"] is True
        assert s["manifest"]["schema_version"] == "9"
        assert (out / "config.json").exists() and (out / "fingerprint.csv").exists()
        assert (out / "rows_probier_post.csv").exists()
        # the excluded moderator row is still in the rows CSV for reading
        assert "human-118" in (out / "rows_probier_post.csv").read_text()
        v = s["venues"]["probier"]
        assert v["wiki"] == "probier" and v["n_rows_total"] == 8 and v["n_rows_kept"] == 7
        base = v["windows"]["baseline"]
        post = v["windows"]["post"]
        assert base["fingerprint"]["n_rows"] == 3 and post["fingerprint"]["n_rows"] == 4
        assert post["fingerprint"]["n_rows_incl_excluded"] == 5
        assert set(base["per_identity"]) == {"actor", "ip16"}
        assert "timeline" not in base["per_identity"]["actor"]
        # under "actor" the three anonymous editors are distinct agents
        assert post["per_identity"]["actor"]["n_agents"] == 2
        tl = post["per_identity"]["actor"]["timeline"]
        assert tl["n_steps"] >= 1 and "volume_vs_press_coverage" in tl["lag_days"]
        # the Sep 7 re-save of a June page replies to the June editor only when
        # the whole series is projected: 2 in-window replies (SandBox, Lemino)
        assert post["per_identity"]["actor"]["n_interactions_full_series_slice"] == 2
        assert post["per_identity"]["actor"]["n_interactions"] == 1
        ev = post["edit_volume"]
        assert ev["n_windows"] == 4 and ev["peak_day"] == "2026-09-06" and ev["peak_events"] == 2
        assert ev["alarm"] is False  # 2 saves against a 1-save floor is ratio 2
        assert base["edit_volume"]["peak_events"] == 3 and base["edit_volume"]["n_windows"] == 1
        assert (out / "probier" / "post" / "timeline.csv").exists()
        assert (out / "probier" / "baseline" / "structural_actor.csv").exists()
        # usemod has no baseline rows: empty window, no detector pass, still listed
        u = s["venues"]["usemod-org"]
        assert u["wiki"] == "usemod"
        assert u["windows"]["baseline"]["fingerprint"]["n_rows"] == 0
        assert u["windows"]["baseline"]["per_identity"] == {}
        assert u["windows"]["post"]["fingerprint"]["termina_campaigns"] == {"usemod-fleet": 1}
        assert [r["id"] for r in s["venue_table"]] == ["probier", "usemod-org"]

    def test_empty_venues_writes_empty_run(self, db_dir, tmp_path):
        cfg = ReplayConfig(
            source="termina", venues=[], identity="actor", sweep_identity=["actor"],
            structural_null_samples=2,
        )
        out = run_termina_replay(db_dir, cfg, tmp_path / "runs", with_timeline=False)
        s = json.loads((out / "summary.json").read_text())
        assert s["venues"] == {} and s["venue_table"] == []
        assert s["exclude_actor_kinds"] == ["human"]

    def test_empty_record_kinds_keeps_venues_but_no_rows(self, db_dir, tmp_path):
        cfg = ReplayConfig(
            source="termina", venues=["probier"], record_kinds=[], identity="actor",
            sweep_identity=["actor"], structural_null_samples=2,
        )
        out = run_termina_replay(db_dir, cfg, tmp_path / "runs", with_timeline=False)
        s = json.loads((out / "summary.json").read_text())
        assert s["venues"]["probier"]["n_rows_total"] == 0
        assert s["venues"]["probier"]["n_rows_kept"] == 0
        assert [r["id"] for r in s["venue_table"]] == ["probier"]

    def test_pin_mismatch_is_recorded_not_fatal(self, db_dir, tmp_path):
        cfg = ReplayConfig(
            source="termina", venues=["probier"], identity="actor", sweep_identity=["actor"],
            structural_null_samples=2, termina_db_sha256="0" * 64,
        )
        out = run_termina_replay(db_dir, cfg, tmp_path / "runs", with_timeline=False)
        s = json.loads((out / "summary.json").read_text())
        assert s["db_pin_matches"] is False

    def test_from_yaml_reads_repo_scenario(self):
        path = Path("scenarios/casestudy_wiki_postdisclosure.yaml")
        cfg = ReplayConfig.from_yaml(path)
        assert cfg.source == "termina"
        assert cfg.venues[:3] == ["dse", "probier", "fractal"]
        assert cfg.record_kinds == ["rc-row"] and cfg.exclude_actor_kinds == ["human"]
        assert cfg.baseline_window == ["2026-05-24T00:00:00Z", "2026-07-03T00:00:00Z"]
        assert cfg.post_window[0] == "2026-07-03T00:00:00Z"
        assert cfg.termina_db_sha256 and len(cfg.termina_db_sha256) == 64
        doc = _yaml.safe_load(path.read_text())
        assert doc["data"]["files"] == ["manifest.json", "incidents.sqlite"]
        assert "collusion_wiki_report" in cfg.landmarks


class TestCli:
    def test_source_termina_dispatches(self, db_dir, tmp_path, monkeypatch):
        from swarm.bridges.collusion_wiki import __main__ as cli

        seen = {}

        def fake(db, cfg, runs_root, *, with_timeline):
            seen.update(db=db, cfg=cfg, runs_root=runs_root, with_timeline=with_timeline)
            return tmp_path / "out"

        monkeypatch.setattr(cli, "run_termina_replay", fake)
        rc = cli.main([
            "scenarios/casestudy_wiki_postdisclosure.yaml",
            "--termina-db", str(db_dir), "--runs-root", str(tmp_path), "--no-timeline",
        ])
        assert rc == 0
        assert seen["db"] == db_dir and seen["with_timeline"] is False
        assert seen["cfg"].source == "termina"
