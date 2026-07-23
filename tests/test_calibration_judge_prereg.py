"""Regression for beads naa8: Arm B's collector must honor the prereg lock.

calibration-prereg.md version-locks Arm B's judge rubric at rubric_v1 before
any data collection. The library-wide DEFAULT_RUBRIC_VERSION has since moved
to v3 — the collector must NOT follow it silently.
"""

import json

from experiments import calibration_judge as cj


def test_collector_defaults_to_preregistered_rubric():
    parser_default = None
    # Parse with no --rubric: the default must be the preregistered v1,
    # regardless of where the library default has moved.

    for action in _build_parser()._actions:
        if "--rubric" in getattr(action, "option_strings", []):
            parser_default = action.default
    assert parser_default == cj.PREREG_RUBRIC_VERSION == "rubric.v1"


def _build_parser():
    # main() builds its parser inline; reconstruct via a dry parse_args probe.
    import argparse
    import unittest.mock as mock

    captured = {}

    def _capture(self, argv=None):
        captured["parser"] = self
        raise SystemExit(0)  # stop main() before any work

    with mock.patch.object(argparse.ArgumentParser, "parse_args", _capture):
        try:
            cj.main(["--help-probe"])
        except SystemExit:
            pass
    return captured["parser"]


def test_run_stamps_v1_and_no_deviation(tmp_path):
    cj.main([
        "--judges", "mock",
        "--per-bin", "2",
        "--runs-dir", str(tmp_path),
    ])
    run_dir = next(tmp_path.iterdir())
    config = json.loads((run_dir / "config.json").read_text())
    assert config["rubric_version"] == "rubric.v1"
    assert config["rubric_sha256_prefix"]
    assert config["prereg_deviation"] is None


def test_explicit_v3_is_recorded_as_deviation(tmp_path, capsys):
    cj.main([
        "--judges", "mock",
        "--per-bin", "2",
        "--rubric", "rubric.v3",
        "--runs-dir", str(tmp_path),
    ])
    assert "[PREREG DEVIATION]" in capsys.readouterr().err
    run_dir = next(tmp_path.iterdir())
    config = json.loads((run_dir / "config.json").read_text())
    deviation = config["prereg_deviation"]
    assert deviation["locked_rubric"] == "rubric.v1"
    assert deviation["used_rubric"] == "rubric.v3"
