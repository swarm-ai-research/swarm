#!/usr/bin/env python3
"""Compare two SWARM run directories.

This thin wrapper keeps the issue-requested ``run_diff.py`` script while the
implementation lives in ``swarm.analysis.run_diff`` for reuse and tests.
"""

from swarm.analysis.run_diff import main

if __name__ == "__main__":
    raise SystemExit(main())
