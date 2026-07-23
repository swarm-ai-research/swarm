"""Same-seed determinism smoke tests for calibration arms A–D (beads 69lz).

Each arm's CLI is run twice (three times for the judge arm) at seed=42 with
tiny sizes into separate tmp dirs; the content-bearing outputs must match
byte-for-byte and config.json must match after dropping volatile fields.
Catches seed regressions on the live calibration track (cf. the jkwl
seed=0→42 bug) for ~seconds of CI time.

Marked ``smoke``; modeled on ``test_adaptive.py::test_deterministic_under_seed``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments import (
    calibration_agreement,
    calibration_fidelity,
    calibration_join,
    calibration_judge,
)

pytestmark = pytest.mark.smoke

SEED = "42"
# Fields legitimately allowed to differ between two identical-seed runs.
VOLATILE_CONFIG_KEYS = {"ts_utc", "git_rev"}


def _run(main, tmp_path: Path, tag: str, argv: list[str]) -> Path:
    runs_dir = tmp_path / tag
    main([*argv, "--runs-dir", str(runs_dir)])
    (run_dir,) = list(runs_dir.iterdir())
    return run_dir


def _assert_runs_identical(a: Path, b: Path) -> None:
    """Every output file matches: CSVs byte-for-byte, configs modulo volatiles."""

    names_a = sorted(p.name for p in a.iterdir())
    names_b = sorted(p.name for p in b.iterdir())
    assert names_a == names_b
    for name in names_a:
        if name == "config.json":
            cfg_a = json.loads((a / name).read_text())
            cfg_b = json.loads((b / name).read_text())
            for key in VOLATILE_CONFIG_KEYS:
                cfg_a.pop(key, None)
                cfg_b.pop(key, None)
            assert cfg_a == cfg_b, "config.json differs beyond volatile fields"
        else:
            assert (a / name).read_bytes() == (b / name).read_bytes(), (
                f"{name} not deterministic under fixed seed"
            )


# ---------------------------------------------------------------------------
# Arm A — proxy fidelity (k2yr)
# ---------------------------------------------------------------------------
def test_arm_a_fidelity_deterministic(tmp_path):
    argv = [
        "--p-grid", "0.2,0.8",
        "--k-values", "1.0,2.0",
        "--per-bin", "20",
        "--seed", SEED,
    ]
    first = _run(calibration_fidelity.main, tmp_path, "a1", argv)
    second = _run(calibration_fidelity.main, tmp_path, "a2", argv)
    _assert_runs_identical(first, second)
    assert (first / "summary.csv").exists()


# ---------------------------------------------------------------------------
# Arm B — external judge anchor (5bpg)
# ---------------------------------------------------------------------------
def test_arm_b_judge_three_runs_deterministic(tmp_path):
    argv = ["--judges", "mock", "--per-bin", "2", "--seed", SEED]
    runs = [
        _run(calibration_judge.main, tmp_path, f"b{i}", argv) for i in range(3)
    ]
    _assert_runs_identical(runs[0], runs[1])
    _assert_runs_identical(runs[1], runs[2])
    # And the run stays on the preregistered rubric (naa8).
    config = json.loads((runs[0] / "config.json").read_text())
    assert config["rubric_version"] == "rubric.v1"


# ---------------------------------------------------------------------------
# Arm C — inter-rater agreement (p6bz)
# ---------------------------------------------------------------------------
def test_arm_c_agreement_invariant_across_reruns(tmp_path):
    # End-to-end offline: arm B with two deterministic raters (mock +
    # jittered mock_b, added in 55bd8b0b / beads gjc0) feeds arm C.
    scores_run = _run(
        calibration_judge.main,
        tmp_path,
        "scores",
        ["--judges", "mock", "mock_b", "--per-bin", "3", "--seed", SEED],
    )
    argv = ["--scores", str(scores_run / "judge_scores.csv")]
    first = _run(calibration_agreement.main, tmp_path, "c1", argv)
    second = _run(calibration_agreement.main, tmp_path, "c2", argv)
    _assert_runs_identical(first, second)
    assert (first / "summary.csv").exists()


# ---------------------------------------------------------------------------
# Arm D — frozen joined schema (h5bo)
# ---------------------------------------------------------------------------
def test_arm_d_join_deterministic_and_schema_complete(tmp_path):
    argv = ["--scenario", "mixed", "--seed", SEED, "--judges", "mock"]
    first = _run(calibration_join.main, tmp_path, "d1", argv)
    second = _run(calibration_join.main, tmp_path, "d2", argv)
    _assert_runs_identical(first, second)

    config = json.loads((first / "config.json").read_text())
    assert config["schema_version"]

    with (first / "joined.csv").open() as f:
        header = next(csv.reader(f))
    # The joined.v1 contract: exactly the frozen header for this judge set.
    from swarm.calibration.joined import joined_header

    assert header == joined_header(["mock"])


def test_injection_generator_pins_v_hat_exactly():
    """c89o: closed-form observables land compute_v_hat on each target."""
    import random

    from swarm.core.observable_generator import (
        CalibrationInjectionObservableGenerator,
    )
    from swarm.core.proxy import ProxyComputer

    targets = {f"c{i}": t for i, t in enumerate([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])}
    gen = CalibrationInjectionObservableGenerator(
        targets, ProxyComputer(), rng=random.Random(0)
    )
    # -1.0 needs the asymptotic penalty floor; everything else is exact.
    assert gen.max_target_error() < 3e-4
    proxy = ProxyComputer()
    for t in (-0.6, -0.2, 0.2, 0.6, 1.0):
        assert abs(proxy.compute_v_hat(gen._solve_observables(t)) - t) < 1e-12


def test_injection_ground_truth_rate_matches_latent_p():
    """gt draws follow p_latent = compute_p(target), not the realized labels."""
    import random

    from swarm.core.observable_generator import (
        CalibrationInjectionObservableGenerator,
    )
    from swarm.core.proxy import ProxyComputer

    proxy = ProxyComputer()
    gen = CalibrationInjectionObservableGenerator(
        {"a": 0.6}, proxy, rng=random.Random(123)
    )
    n = 5000
    rate = sum(1 for _ in range(n) if gen.draw_ground_truth("a") == 1) / n
    assert abs(rate - proxy.compute_p(0.6)) < 0.02
    assert gen.draw_ground_truth("unknown") is None


def test_injection_scenario_end_to_end(tmp_path):
    """Scenario run wires targets -> pinned v_hat + ground_truth in export."""
    import json as _json
    import subprocess
    import sys as _sys

    export = tmp_path / "export.json"
    res = subprocess.run(
        [
            _sys.executable, "-m", "swarm", "run",
            "scenarios/calibration_proxy_fidelity.yaml",
            "--seed", "5", "--epochs", "1", "--steps", "30",
            "--export-json", str(export), "--quiet",
        ],
        capture_output=True, text=True, timeout=300,
    )
    assert res.returncode == 0, res.stderr[-500:]
    events = _json.loads(export.read_text())["events"]
    cal = [e for e in events if e.get("ground_truth") is not None]
    assert cal, "no calibration interactions captured"
    allowed = {-1.0, -0.6, -0.2, 0.2, 0.6, 1.0}
    for e in cal:
        assert e["ground_truth"] in (1, -1)
        assert any(abs(e["v_hat"] - t) < 3e-4 for t in allowed), e["v_hat"]
