#!/usr/bin/env python
"""Verify INVARIANTS.md: every invariant names >=1 pytest node that collects.

An invariant without a running test is prose, not an invariant. Exits
nonzero if any entry lacks a test: line or references a node that pytest
cannot collect.
"""
import re
import subprocess
import sys
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "INVARIANTS.md"


def main() -> int:
    text = MANIFEST.read_text()
    entries = re.findall(
        r"^- (INV-\d+): (.+?)(?=^- INV-|\Z)", text, re.M | re.S
    )
    if not entries:
        print("check_invariants: no INV- entries found in INVARIANTS.md")
        return 1

    failures = []
    all_nodes = []
    for inv_id, body in entries:
        nodes = re.findall(r"^\s*test: (\S+)$", body, re.M)
        if not nodes:
            failures.append(f"{inv_id}: no `test:` line — invariant is unenforced prose")
        all_nodes.extend((inv_id, n) for n in nodes)

    if all_nodes:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q",
             "-p", "no:testmon", *[n for _, n in all_nodes]],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            tail = "\n".join(result.stdout.strip().splitlines()[-5:])
            failures.append(f"referenced test nodes failed to collect:\n{tail}")

    if failures:
        print("check_invariants: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"check_invariants: OK — {len(entries)} invariants, "
          f"{len(all_nodes)} enforcing test references, all collect")
    return 0


if __name__ == "__main__":
    sys.exit(main())
