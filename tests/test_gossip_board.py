"""Tests for the gossip-board information-fidelity model (bead fcj5)."""

from pathlib import Path

import pytest

from swarm.bridges.gossip_board import BoardConfig, run_board, sweep_board
from swarm.bridges.gossip_board.model import aggregate

SCENARIO = Path(__file__).resolve().parent.parent / "scenarios" / "gossip_board_fidelity.yaml"


def _cfg(**kw) -> BoardConfig:
    c = BoardConfig(n_agents=12, n_rounds=40, late_join_round=20)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_scenario_yaml_loads():
    cfg = BoardConfig.from_yaml(SCENARIO)
    assert cfg.scenario_id == "gossip_board_fidelity"
    assert cfg.fidelity == "code"


def test_unknown_field_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("scenario_id: x\nboard:\n  nope: 1\n")
    with pytest.raises(KeyError):
        BoardConfig.from_yaml(p)


def test_bad_fidelity_rejected():
    with pytest.raises(ValueError):
        run_board(_cfg(fidelity="telepathy"))


def test_deterministic_under_seed():
    a = run_board(_cfg(), seed=7)
    b = run_board(_cfg(), seed=7)
    assert a.summary == b.summary
    assert [vars(r) for r in a.rounds] == [vars(r) for r in b.rounds]


def test_metrics_bounded():
    res = run_board(_cfg(fidelity="description"), seed=3)
    for r in res.rounds:
        assert 0.0 <= r.frontier_fraction <= 1.0
        assert 0.0 <= r.identical_pairs <= 1.0
        assert 0.0 <= r.hidden_dim_adoption <= 1.0
        assert 1 <= r.modal_cluster <= r.active_agents
    s = res.summary
    assert 0.0 <= s["published_fraction"] <= 1.0
    assert s["board_success_entries"] <= s["board_entries"]
    assert 0 <= s["unexplored_dims"] <= 8


def test_late_joiners_inactive_before_join():
    res = run_board(_cfg(late_joiner_fraction=0.5, late_join_round=20), seed=1)
    assert res.rounds[0].active_agents == 6
    assert res.rounds[-1].active_agents == 12


def test_lineage_recorded_on_adoption():
    res = run_board(_cfg(fidelity="code", p_peer=0.8), seed=5)
    adopted = [e for e in res.board if e.parent is not None]
    assert adopted, "code sharing with high p_peer must produce adoptions"
    ids = {e.id for e in res.board}
    assert all(e.parent in ids for e in adopted)
    assert res.summary["adoptions"] == len(adopted)


def test_code_sharing_fingerprints_description_does_not():
    """Preregistered expectation 1 (means over seeds)."""
    cfg = _cfg(n_agents=24, n_rounds=80, late_join_round=40)
    rows = sweep_board(cfg, {"fidelity": ["code", "description"]}, seeds=range(5))
    agg = {a["fidelity"]: a for a in aggregate(rows, ["fidelity"])}
    assert agg["code"]["final_modal_cluster_fraction"] > 0.5
    assert agg["description"]["final_modal_cluster_fraction"] < agg["code"]["final_modal_cluster_fraction"]
    assert agg["description"]["final_mean_true_score"] > 0.9 * agg["code"]["final_mean_true_score"]


def test_survivorship_gap_is_publication_property():
    """Preregistered expectation 2."""
    cfg = _cfg(n_agents=16, n_rounds=60, late_join_round=30)
    rows = sweep_board(cfg, {"publish_failures": [False, True]}, seeds=range(4))
    agg = {a["publish_failures"]: a for a in aggregate(rows, ["publish_failures"])}
    assert agg[False]["survivorship_gap"] > 0.0
    assert agg[True]["published_fraction"] > agg[False]["published_fraction"]


def test_success_only_board_publishes_only_successes():
    res = run_board(_cfg(publish_failures=False), seed=2)
    assert all(e.success for e in res.board)
    res2 = run_board(_cfg(publish_failures=True), seed=2)
    assert any(not e.success for e in res2.board)


def test_cli_writes_run_folder(tmp_path):
    from swarm.bridges.gossip_board.__main__ import main

    out = tmp_path / "run"
    rc = main([str(SCENARIO), "--out", str(out), "--seeds", "1", "--axis", "fidelity=code,description"])
    assert rc == 0
    assert (out / "history.json").exists()
    assert (out / "csv" / "rounds.csv").exists()
    assert (out / "csv" / "sweep_mean.csv").exists()


def test_diverse_fraction_bounds_and_count():
    with pytest.raises(ValueError):
        run_board(_cfg(diverse_fraction=1.5))
    res = run_board(_cfg(diverse_fraction=0.25), seed=1)
    assert res.summary["diverse_agents"] == 3


def test_zero_prior_never_discovers_hidden_dim_without_diversity():
    """Bead khs2: a shared prior of exactly zero is not fixed by population size."""
    res = run_board(_cfg(hidden_dim_prior=0.0, n_agents=48), seed=4)
    assert res.summary["hidden_dim_explored"] is False
    assert res.summary["hidden_discovery_round"] is None
    assert res.summary["hidden_discovery_round_or_max"] == res.config.n_rounds


def test_one_diverse_agent_discovers_hidden_dim():
    res = run_board(_cfg(hidden_dim_prior=0.0, n_agents=8, diverse_fraction=0.125, n_rounds=120), seed=4)
    assert res.summary["diverse_agents"] == 1
    assert res.summary["hidden_dim_explored"] is True
    assert res.summary["hidden_discovery_round"] is not None


# --- bead 9err: detectors on honest copying -------------------------------

def test_fingerprint_projection_links_identical_configs():
    from swarm.bridges.gossip_board.detect import (
        fingerprint_interactions,
        lineage_interactions,
    )

    res = run_board(_cfg(fidelity="code", p_peer=0.8, n_rounds=60), seed=5)
    fp = fingerprint_interactions(res.board, res.config.n_agents)
    assert fp, "code sharing must produce identical-config links"
    assert all(x.initiator != x.counterparty for x in fp)
    ln = lineage_interactions(res.board, res.config.n_agents)
    assert len(ln) == sum(1 for e in res.board if e.parent is not None
                          and {x.id: x for x in res.board}[e.parent].agent != e.agent)


def test_residual_is_subset_of_fingerprint():
    from swarm.bridges.gossip_board.detect import (
        fingerprint_interactions,
        residual_interactions,
    )

    res = run_board(_cfg(fidelity="code", n_rounds=60), seed=6)
    fp = fingerprint_interactions(res.board, res.config.n_agents)
    rs = residual_interactions(res.board, res.config.n_agents)
    assert len(rs) <= len(fp)
    assert {x.interaction_id for x in rs} <= {x.interaction_id for x in fp}
    assert all(not x.metadata["explained"] for x in rs)


def test_detect_on_run_shapes_and_bounds():
    from swarm.bridges.gossip_board.detect import detect_on_run

    row = detect_on_run(run_board(_cfg(fidelity="code", n_rounds=60), seed=7), seed=7, n_null=10)
    for k in ("structural_flagged_agents", "temporal_alarm_rate", "pairwise_flagged",
              "residual_structural_flagged_agents", "residual_pairwise_flagged"):
        assert 0.0 <= row[k] <= 1.0, k
    assert row["residual_n_interactions"] <= row["n_interactions"]
    if row["n_interactions"]:
        assert 0.0 <= row["explained_fraction"] <= 1.0


def test_score_only_has_far_fewer_fingerprints_than_code():
    from swarm.bridges.gossip_board.detect import fingerprint_interactions

    # A shared baseline config still produces some identical configs early on
    # (one-dimension mutations of all-zeros collide), so the floor is not zero.
    n_code = len(fingerprint_interactions(run_board(_cfg(fidelity="code", n_rounds=60), seed=8).board, 12))
    n_score = len(fingerprint_interactions(run_board(_cfg(fidelity="score_only", n_rounds=60), seed=8).board, 12))
    assert n_score < n_code / 3


def test_cli_detect_writes_csv(tmp_path):
    from swarm.bridges.gossip_board.__main__ import main

    out = tmp_path / "det"
    rc = main([str(SCENARIO), "--out", str(out), "--seeds", "1", "--detect",
               "--axis", "fidelity=code,score_only"])
    assert rc == 0
    assert (out / "csv" / "detect_mean.csv").exists()
