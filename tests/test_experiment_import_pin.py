"""Experiment scripts must import `swarm` from their own checkout (bead hjyp).

`python experiments/x.py` puts experiments/ on sys.path[0], not the repo root,
so a bare `import swarm` resolves through the editable install -- which points
at whichever checkout ran `pip install -e .` last. On 2026-09-21 that was a
third session's worktree running an older module, and the numbers for bead
19n0 were computed against it. They happened to be identical when re-run
correctly; four of the functions involved differed, so that was luck.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"
IMPORTS_SWARM = re.compile(r"^(from swarm|import swarm)", re.M)
PIN = "__import__(\"sys\").path.insert("


def _scripts_importing_swarm() -> list[Path]:
    return sorted(p for p in EXPERIMENTS.glob("*.py")
                  if IMPORTS_SWARM.search(p.read_text()))


def test_there_are_scripts_to_check():
    """Positive control: a vacuous pass here would hide a broken glob."""
    assert _scripts_importing_swarm(), "no experiment scripts import swarm?"


@pytest.mark.parametrize(
    "script", _scripts_importing_swarm(), ids=lambda p: p.name)
def test_script_pins_swarm_to_this_checkout(script: Path):
    text = script.read_text()
    pin_at = text.find(PIN)
    assert pin_at != -1, (
        f"{script.name} imports swarm without pinning sys.path to the repo "
        f"root. Add the hjyp guard above its first swarm import."
    )
    first_import = IMPORTS_SWARM.search(text)
    assert first_import is not None
    assert pin_at < first_import.start(), (
        f"{script.name} pins sys.path AFTER importing swarm, which is too late."
    )


def test_guard_resolves_locally_when_run_from_elsewhere(tmp_path):
    """The behaviour the static check is a proxy for."""
    script = EXPERIMENTS / "ai_village_detector_fp.py"
    if not script.exists():                      # pragma: no cover
        pytest.skip("reference script not present")
    probe = (
        "import runpy, sys, pathlib;"
        f"runpy.run_path({str(script)!r}, run_name='notmain');"
        "p = pathlib.Path(sys.modules['swarm'].__file__).resolve();"
        f"print(int(pathlib.Path({str(REPO_ROOT)!r}) in p.parents))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], cwd=tmp_path,
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("1"), (
        "experiment script imported swarm from outside this checkout:\n"
        + out.stdout + out.stderr[-2000:]
    )
