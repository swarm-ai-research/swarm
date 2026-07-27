# Invariants

Machine-checked manifest of the repo's load-bearing invariants. Every entry
MUST name at least one enforcing pytest node (`test:` line). CI runs
`python scripts/check_invariants.py`, which fails if an invariant has no
test or a referenced test no longer collects — an invariant without a
running test is prose, not an invariant.

Add new invariants here when you add the test that enforces them. The
statements mirror CLAUDE.md's "Safety / invariants" section; this file is
the enforcement map.

- INV-1: `p` must remain in [0, 1] everywhere it is surfaced or logged.
  test: tests/test_invariants.py::TestPBounds
- INV-2: Event logs (`*.jsonl`) are append-only and replayable — replay
  returns events in order with fields preserved.
  test: tests/test_event_log.py::TestReplay
  test: tests/test_event_log.py::TestToInteractions
- INV-3: Runs are reproducible from scenario YAML + seed.
  test: tests/test_governance_arena.py::TestDemoSimulation::test_seed_reproducibility
- INV-4: The per-interaction payoff hot path stays within its latency
  budget (deliberately inlined formulas — see payoffs_both).
  test: tests/test_performance.py::TestPerformance::test_payoffs_both_within_budget
- INV-5: The diversity gate only ever discounts evidence — effective
  sample size never exceeds the raw vote count, at any correlation.
  A safety gate that can inflate confidence is not a gate.
  test: tests/test_diversity.py::TestEffectiveN::test_never_exceeds_n
