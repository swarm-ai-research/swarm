#!/usr/bin/env python3
"""Compare two SWARM run JSON artifacts.

This thin wrapper keeps the issue-requested ``run_diff.py`` script while the
implementation lives in ``swarm.replay.run_diff`` for reuse and tests.
"""

from swarm.replay.run_diff import main

if __name__ == "__main__":
    raise SystemExit(main())
