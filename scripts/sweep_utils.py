"""Shared utilities for escalation sweep scripts.

Provides common logging, checkpoint loading/saving functionality used
across multiple escalation sweep experiments.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def log(msg: str, progress_file: Path) -> None:
    """Print message to console and append to progress log file.

    Args:
        msg: Message to log
        progress_file: Path to progress log file
    """
    print(msg, flush=True)
    with open(progress_file, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def load_checkpoint(checkpoint_file: Path) -> dict:
    """Load checkpoint data from JSON file if it exists.

    Args:
        checkpoint_file: Path to checkpoint JSON file

    Returns:
        Dictionary of completed runs, or empty dict if file doesn't exist
    """
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return json.load(f)
    return {}


def save_checkpoint(completed: dict, checkpoint_file: Path) -> None:
    """Save checkpoint data to JSON file.

    Args:
        completed: Dictionary of completed runs to save
        checkpoint_file: Path to checkpoint JSON file
    """
    with open(checkpoint_file, "w") as f:
        json.dump(completed, f, indent=2)
