"""Tests for swarm/calibration/joined.py — arm D scaffolding."""

from __future__ import annotations

import csv
import io
import json

import pytest

from swarm.calibration.joined import (
    BASE_COLUMNS,
    JOINED_SCHEMA_VERSION,
    ProxyRow,
    build_proxy_rows,
    join_with_judges,
    joined_header,
    load_judge_scores_for_join,
)
from tests.fixtures.interactions import (
    generate_mixed_batch,
    generate_obfuscation_scenario,
)


class TestSchema:
    def test_schema_version_frozen(self) -> None:
        # The version string is part of the contract — bumping it must be
        # deliberate. If this test fails it's a flag, not a free fix.
        assert JOINED_SCHEMA_VERSION == "joined.v1"

    def test_base_columns_are_a_tuple(self) -> None:
        # Tuple, not list — so it can't be appended to by accident.
        assert isinstance(BASE_COLUMNS, tuple)

    def test_base_columns_contract(self) -> None:
        # Adaptive arms 1-3 join on these names. Renames/reorders MUST
        # bump the schema version.
        assert BASE_COLUMNS == (
            "interaction_id",
            "scenario",
            "seed",
            "interaction_type",
            "accepted",
            "p_true",
            "v_hat",
            "p_hat",
            "ground_truth",
        )


class TestBuildProxyRows:
    def test_smoke(self) -> None:
        batch = generate_mixed_batch(count=20, seed=0)
        rows = build_proxy_rows(batch, scenario="mixed", seed=0)
        # Only accepted by default
        for row in rows:
            assert row.accepted is True
            assert 0.0 <= row.p_true <= 1.0
            assert -1.0 <= row.v_hat <= 1.0
            assert 0.0 <= row.p_hat <= 1.0
            assert row.scenario == "mixed"
            assert row.seed == 0

    def test_includes_rejected_when_flag_off(self) -> None:
        batch = generate_mixed_batch(count=50, seed=1)
        all_rows = build_proxy_rows(batch, scenario="mixed", seed=1, accepted_only=False)
        kept_rows = build_proxy_rows(batch, scenario="mixed", seed=1, accepted_only=True)
        assert len(all_rows) >= len(kept_rows)

    def test_reproducible(self) -> None:
        batch = generate_mixed_batch(count=10, seed=7)
        a = build_proxy_rows(batch, scenario="x", seed=7)
        b = build_proxy_rows(batch, scenario="x", seed=7)
        assert [r.interaction_id for r in a] == [r.interaction_id for r in b]
        assert [r.v_hat for r in a] == [r.v_hat for r in b]

    def test_missing_p_raises_instead_of_fabricating(self) -> None:
        # p_true is the latent ground-truth anchor the whole study trusts;
        # a missing `p` is a fixture/scenario bug, so it must surface loudly
        # rather than silently defaulting to 0.5.
        class _NoP:
            accepted = True
            interaction_id = "x"
            interaction_type = "REPLY"
            # Everything ProxyObservables.from_interaction needs — only `p` is absent.
            task_progress_delta = 0.5
            rework_count = 0
            verifier_rejections = 0
            tool_misuse_flags = 0
            counterparty_engagement_delta = 0.1

        with pytest.raises(AttributeError):
            build_proxy_rows([_NoP()], scenario="s", seed=0)


class TestJoin:
    def test_join_attaches_scores(self) -> None:
        rows = [
            ProxyRow(
                interaction_id="i1", scenario="s", seed=0,
                interaction_type="REPLY", accepted=True, p_true=0.7,
                v_hat=0.4, p_hat=0.7, ground_truth=1,
            ),
            ProxyRow(
                interaction_id="i2", scenario="s", seed=0,
                interaction_type="TRADE", accepted=True, p_true=0.2,
                v_hat=-0.4, p_hat=0.3, ground_truth=-1,
            ),
        ]
        scores = {
            "claude": {"i1": 0.8, "i2": 0.2},
            "gpt": {"i1": 0.75},  # missing i2
        }
        joined = join_with_judges(rows, scores)
        assert len(joined) == 2
        assert joined[0].judge_scores == {"claude": 0.8, "gpt": 0.75}
        assert joined[1].judge_scores == {"claude": 0.2}  # gpt absent

    def test_to_row_handles_missing_judges(self) -> None:
        rows = [ProxyRow(
            interaction_id="i1", scenario="s", seed=0,
            interaction_type="REPLY", accepted=True, p_true=0.5,
            v_hat=0.0, p_hat=0.5, ground_truth=None,
        )]
        joined = join_with_judges(rows, {"claude": {"i1": 0.5}})
        row = joined[0].to_row(["claude", "gpt"])
        # gpt score is empty string for the missing cell
        assert row[-2] == ""  # judge_gpt_score
        # ground_truth=None renders as empty string
        gt_idx = BASE_COLUMNS.index("ground_truth")
        assert row[gt_idx] == ""

    def test_header_includes_per_judge_columns(self) -> None:
        rows = [ProxyRow(
            interaction_id="i", scenario="s", seed=0,
            interaction_type="REPLY", accepted=True, p_true=0.5,
            v_hat=0.0, p_hat=0.5, ground_truth=1,
        )]
        joined = join_with_judges(rows, {"claude": {"i": 0.5}, "gpt": {"i": 0.6}})
        header = joined[0].header(["claude", "gpt"])
        assert "judge_claude_score" in header
        assert "judge_claude_rationale" in header
        assert "judge_gpt_score" in header

    def test_joined_header_standalone_matches_instance(self) -> None:
        # joined_header() lets the runner emit a schema-bearing header even
        # with zero data rows (no JoinedRow to hand). It must agree with the
        # instance method and start with the frozen base columns.
        names = ["claude", "gpt"]
        standalone = joined_header(names)
        assert standalone[: len(BASE_COLUMNS)] == list(BASE_COLUMNS)
        assert standalone == [
            *BASE_COLUMNS,
            "judge_claude_score",
            "judge_claude_rationale",
            "judge_gpt_score",
            "judge_gpt_rationale",
        ]

    def test_round_trip_through_csv(self) -> None:
        batch = generate_obfuscation_scenario(n_epochs=2, seed=4)
        flat = [i for epoch in batch for i in epoch]
        proxy_rows = build_proxy_rows(flat, scenario="obfuscation", seed=4)
        # Fake scores so we have something to join.
        scores = {"mock": {r.interaction_id: 0.5 for r in proxy_rows}}
        joined = join_with_judges(proxy_rows, scores)

        buf = io.StringIO()
        writer = csv.writer(buf)
        header = joined[0].header(["mock"])
        writer.writerow(header)
        for j in joined:
            writer.writerow(j.to_row(["mock"]))

        buf.seek(0)
        reader = csv.DictReader(buf)
        re_read = list(reader)
        assert len(re_read) == len(joined)
        assert re_read[0]["judge_mock_score"] == "0.500000"
        # p_true round-trips
        assert pytest.approx(float(re_read[0]["p_true"]), abs=1e-5) == joined[0].proxy.p_true


class TestCsvLoader:
    def test_loads_arm_b_format(self, tmp_path) -> None:
        csv_path = tmp_path / "judge_scores.csv"
        with csv_path.open("w") as f:
            f.write("interaction_id,judge_name,rubric_version,p_true,score,rationale\n")
            f.write("i1,claude,rubric.v1,0.7,0.8,\"agent_type=honest\"\n")
            f.write("i1,gpt,rubric.v1,0.7,0.75,\n")
            f.write("i2,claude,rubric.v1,0.2,0.2,\"agent_type=blatant\"\n")
        scores, rationales = load_judge_scores_for_join(str(csv_path))
        assert scores == {"claude": {"i1": 0.8, "i2": 0.2}, "gpt": {"i1": 0.75}}
        assert rationales["claude"]["i1"] == "agent_type=honest"
        assert "i1" not in rationales.get("gpt", {})  # empty rationale → omitted


@pytest.mark.parametrize("rubric", [None, "rubric.v3"])
def test_inline_real_judge_preserves_schema_and_records_provenance(
    tmp_path, monkeypatch, capsys, rubric
):
    """Exercise real LLMJudge prompt/parsing; replace only its network call."""
    from experiments import calibration_join
    from swarm.judges import judge, load_rubric
    from swarm.judges.llm_call import LLMCallResult
    from swarm.judges.views import FORBIDDEN_FIELDS

    accepted = [i for i in generate_mixed_batch(count=20, seed=42) if i.accepted][:2]
    assert len(accepted) == 2
    monkeypatch.setattr(calibration_join, "_load_interactions", lambda *args: accepted)
    monkeypatch.setenv("JUDGE_MODEL_LLAMA", "offline-test-model")
    calls = []

    def offline_ollama(**kwargs):
        calls.append(kwargs)
        prompt = kwargs["prompt"]
        assert load_rubric(rubric or "rubric.v1") in prompt
        payload = json.loads(prompt.split("===== INTERACTION =====\n")[1].split(
            "\n===== OUTPUT ====="
        )[0])
        assert not set(payload).intersection(FORBIDDEN_FIELDS)
        return LLMCallResult('{"score": 0.73, "rationale": "offline fixture"}', 1, 1, 0)

    monkeypatch.setattr(judge, "call_ollama", offline_ollama)
    argv = ["--scenario", "mixed", "--seed", "42", "--judges", "llama", "mock",
            "llama", "--runs-dir", str(tmp_path)]
    if rubric:
        argv.extend(["--rubric", rubric])
    assert calibration_join.main(argv) == 0

    (run_dir,) = tmp_path.iterdir()
    config = json.loads((run_dir / "config.json").read_text())
    assert len(calls) == len(accepted)  # duplicate judge must not incur more calls
    assert all(c["model"] == "offline-test-model" for c in calls)
    assert config["schema_version"] == "joined.v1"
    assert config["status"] == "completed"
    assert config["n_scored_verdicts"] == 4
    assert config["rubric_version"] == (rubric or "rubric.v1")
    import hashlib

    from swarm.judges import rubric_path

    expected_hash = hashlib.sha256(rubric_path(rubric or "rubric.v1").read_bytes())
    assert config["rubric_sha256_prefix"] == expected_hash.hexdigest()[:16]
    assert config["judge_models"]["llama"]["model"] == "offline-test-model"
    assert config["judge_models"]["llama"]["provider"] == "ollama"
    assert config["judge_models"]["mock"]["rubric_version"] == (rubric or "rubric.v1")
    assert "synthetic" in config["interaction_source"]
    if rubric:
        assert config["prereg_deviation"]["used_rubric"] == rubric
        assert config["prereg_deviation"]["locked_rubric"] == "rubric.v1"
        assert "[PREREG DEVIATION]" in capsys.readouterr().err
    else:
        assert config["prereg_deviation"] is None
    with (run_dir / "joined.csv").open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == joined_header(["llama", "mock"])
        rows = list(reader)
    assert {r["interaction_id"] for r in rows} == {i.interaction_id for i in accepted}
    assert all(r["judge_llama_score"] == "0.730000" for r in rows)


def test_inline_unknown_judge_rejected_before_scoring(tmp_path, monkeypatch):
    from experiments import calibration_join
    from swarm.judges import LLMJudge

    def forbidden_call(*args):
        pytest.fail("invalid judge list incurred a provider call")

    monkeypatch.setattr(LLMJudge, "score", forbidden_call)
    with pytest.raises(ValueError, match="unknown judge"):
        calibration_join.main([
            "--scenario", "mixed", "--seed", "42", "--judges", "llama", "unknown",
            "--runs-dir", str(tmp_path),
        ])
    assert not list(tmp_path.iterdir())


def test_inline_provider_failure_preserves_prior_scores_and_run(tmp_path, monkeypatch):
    from datetime import datetime
    from types import SimpleNamespace

    from experiments import calibration_join
    from swarm.judges import judge
    from swarm.judges.llm_call import LLMCallResult

    accepted = [i for i in generate_mixed_batch(count=20, seed=42) if i.accepted][:2]
    monkeypatch.setattr(calibration_join, "_load_interactions", lambda *args: accepted)
    monkeypatch.setattr(calibration_join, "datetime", SimpleNamespace(
        now=lambda tz: datetime(2026, 9, 6, tzinfo=tz)
    ))
    calls = []

    def offline_ollama(**kwargs):
        (run_dir,) = tmp_path.iterdir()
        assert json.loads((run_dir / "config.json").read_text())["status"] == "scoring"
        calls.append(kwargs)
        text = '{"score": 0.73, "rationale": "saved"}' if len(calls) == 1 else 'invalid'
        return LLMCallResult(text, 1, 1, 0)

    monkeypatch.setattr(judge, "call_ollama", offline_ollama)
    argv = ["--scenario", "mixed", "--seed", "42", "--judges", "llama",
            "--runs-dir", str(tmp_path)]
    with pytest.raises(ValueError, match="failed to parse response"):
        calibration_join.main(argv)
    (run_dir,) = tmp_path.iterdir()
    config = json.loads((run_dir / "config.json").read_text())
    assert config["status"] == "failed"
    assert config["n_scored_verdicts"] == 1
    assert not (run_dir / "joined.csv").exists()
    with (run_dir / "judge_scores.csv").open() as f:
        scores = list(csv.DictReader(f))
    assert len(scores) == 1
    assert scores[0]["interaction_id"] == accepted[0].interaction_id
    assert scores[0]["score"] == "0.730000"
    events = [json.loads(line) for line in (run_dir / "judge_calls.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["started", "completed", "started", "failed"]
    assert events[-1]["error_type"] == "ValueError"
    before = {p.name: p.read_bytes() for p in run_dir.iterdir()}
    with pytest.raises(FileExistsError):
        calibration_join.main(argv)
    assert len(calls) == 2
    assert before == {p.name: p.read_bytes() for p in run_dir.iterdir()}
