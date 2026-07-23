"""Shared helpers for the calibration experiment runners (arms A/B/D).

Extracted so each runner doesn't re-define `git_rev` and the
scenario→interactions loader. The fixture generators are imported lazily
inside `load_interactions` (not at module top) so importing this module for
`git_rev` alone — as `calibration_fidelity` does — doesn't require the
`tests/` package to be importable (e.g. from an installed/minimal checkout).
"""

from __future__ import annotations

import subprocess

# Single source of truth for the scenarios the runners accept. Used both for
# argparse `choices` and `load_interactions` so the two can't drift.
SCENARIOS: tuple[str, ...] = ("mixed", "obfuscation", "self_optimizer")


def git_rev() -> str:
    """Short HEAD SHA for run provenance; 'unknown' if git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def load_interactions(scenario: str, seed: int) -> list:
    """Generate the interaction pool for a calibration `scenario` at `seed`."""
    # Imported here, not at module top, so `git_rev`-only importers like
    # calibration_fidelity don't depend on tests/ being on the path.
    from tests.fixtures.interactions import (
        generate_mixed_batch,
        generate_obfuscation_scenario,
        generate_self_optimizer_scenario,
    )

    if scenario == "mixed":
        return generate_mixed_batch(count=500, seed=seed)
    if scenario == "obfuscation":
        epochs = generate_obfuscation_scenario(n_epochs=10, seed=seed)
        return [i for epoch in epochs for i in epoch]
    if scenario == "self_optimizer":
        epochs = generate_self_optimizer_scenario(n_epochs=10, seed=seed)
        return [i for epoch in epochs for i in epoch]
    raise ValueError(f"unknown scenario: {scenario}")
