"""Tests for the router-intermediary experiment (bead ggiw)."""

import json
import math
import random
import sys
from pathlib import Path

import pytest
import yaml

import experiments.router_intermediary as router_intermediary
from experiments.router_intermediary import (
    Population,
    auc,
    ban_cell,
    benign_threshold,
    detection_cell,
    partition,
    population_from,
    z_score,
)

POP = Population(
    n_operators=100,
    adversary_share=0.05,
    interactions_per_epoch=10,
    benign_mean=0.8,
    benign_conc=8.0,
    adversary_mean=0.6,
    adversary_conc=8.0,
)


class TestPieces:
    def test_auc_extremes_and_ties(self):
        assert auc([2.0, 3.0], [0.0, 1.0]) == 1.0
        assert auc([0.0, 1.0], [2.0, 3.0]) == 0.0
        assert auc([1.0], [1.0]) == 0.5

    def test_auc_empty_is_nan(self):
        assert auc([], [1.0]) != auc([], [1.0])

    def test_partition_covers_every_operator_once(self):
        parts = partition(range(23), 5, random.Random(0))
        assert sorted(o for p in parts for o in p) == list(range(23))
        assert [len(p) for p in parts] == [5, 5, 5, 5, 3]

    @pytest.mark.parametrize("m", [1, 20])
    def test_benign_z_is_standardized(self, m):
        # the benign-only threshold at 5% FPR sits near the normal 1.645
        thr = benign_threshold(random.Random(1), POP, m, 0.05, samples=4000)
        assert 1.3 < thr < 2.0

    def test_z_score_zero_at_benign_mean(self):
        assert z_score(POP, 1.0 - POP.benign_mean, 7) == pytest.approx(0.0)

    def test_at_least_one_adversary(self):
        assert Population(10, 0.0, 1, 0.8, 8, 0.6, 8).n_adversaries == 1


class TestCells:
    def test_direct_accounts_detect_a_clear_adversary(self):
        r = detection_cell(POP, 1, 0.0, 0.05, reps=20, seed=3)
        assert r["auc"] > 0.95
        assert r["clean_account_flag_rate"] == pytest.approx(0.05, abs=0.03)

    def test_pooling_lowers_auc(self):
        direct = detection_cell(POP, 1, 0.0, 0.05, reps=30, seed=3)["auc"]
        pooled = detection_cell(POP, 10, 0.0, 0.05, reps=30, seed=3)["auc"]
        assert pooled < direct

    def test_full_kyc_restores_direct_detection(self):
        direct = detection_cell(POP, 1, 0.0, 0.05, reps=20, seed=3)["auc"]
        kyc = detection_cell(POP, 10, 1.0, 0.05, reps=20, seed=3)["auc"]
        assert kyc == pytest.approx(direct, abs=0.02)

    def test_fully_contaminated_pooling_reports_nan_not_crash(self):
        all_adversary = Population(
            n_operators=20,
            adversary_share=1.0,
            interactions_per_epoch=2,
            benign_mean=0.8,
            benign_conc=8.0,
            adversary_mean=0.6,
            adversary_conc=8.0,
        )
        r = detection_cell(all_adversary, 5, 0.0, 0.05, reps=3, seed=3)
        assert r["clean_account_share"] == 0.0
        assert math.isnan(r["auc"])
        assert math.isnan(r["clean_account_flag_rate"])

    def test_remainder_account_size_is_calibrated(self):
        pop = Population(23, 0.1, 2, 0.8, 8, 0.6, 8)
        r = detection_cell(pop, 5, 0.0, 0.05, reps=2, seed=3)
        assert 0.0 <= r["adversary_account_flag_rate"] <= 1.0

    def test_no_ban_keeps_every_adversary_online(self):
        r = ban_cell(POP, 5, "none", 0.05, horizon=5, reroute_adv=1, reroute_ben=4, reps=2, seed=0)
        assert r["adversary_persistence"] == 1.0
        assert r["benign_epochs_lost_per_rep"] == 0.0

    def test_bans_cost_more_benign_epochs_when_pooled(self):
        kw = {"horizon": 10, "reroute_adv": 1, "reroute_ben": 4, "reps": 5, "seed": 0}
        direct = ban_cell(POP, 1, "ban_account", 0.05, **kw)
        pooled = ban_cell(POP, 10, "ban_account", 0.05, **kw)
        assert pooled["benign_epochs_lost_per_rep"] > direct["benign_epochs_lost_per_rep"]

    def test_unknown_ban_policy_is_rejected(self):
        with pytest.raises(ValueError, match="unknown ban policy"):
            ban_cell(POP, 5, "ban_acount", 0.05, 5, 1, 4, 2, 0)

    def test_scenario_config_parses(self):
        cfg = yaml.safe_load(Path("scenarios/router_intermediary.yaml").read_text())
        pop = population_from(cfg)
        assert pop.n_adversaries == 20
        assert cfg["sweep"]["ban_policy"] == ["none", "ban_account"]

    def test_quick_main_writes_reproducible_artifacts(self, tmp_path, monkeypatch):
        cfg = yaml.safe_load(Path("scenarios/router_intermediary.yaml").read_text())
        cfg["population"]["n_operators"] = 20
        cfg["population"]["interactions_per_epoch"] = 2
        cfg["reps"] = 2
        cfg["bans"]["horizon_epochs"] = 2
        cfg["sweep"]["kyc_passthrough"] = [0.0, 1.0]
        cfg["sensitivity"]["adversary_mean_p"] = [0.6]
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg))

        original_threshold = router_intermediary.benign_threshold
        monkeypatch.setattr(
            router_intermediary,
            "benign_threshold",
            lambda rng, pop, m, fpr, samples=4000: original_threshold(
                rng, pop, m, fpr, samples=30
            ),
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "router_intermediary",
                "--quick",
                "--config",
                str(config_path),
                "--output",
                str(tmp_path),
            ],
        )
        assert router_intermediary.main() == 0

        [run_dir] = [p for p in tmp_path.iterdir() if p.is_dir()]
        for name in ("detection.csv", "bans.csv", "kyc.csv", "sensitivity.csv"):
            assert (run_dir / name).is_file()
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["config"] == str(config_path)
        assert summary["seed"] == cfg["seed"]
        assert summary["reps"] == 20
        assert summary["quick"] is True
