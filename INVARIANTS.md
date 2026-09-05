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
- INV-6: Composed wards never widen the parent — a dispatched child's
  effective bounds are narrower-or-equal to its parent's in every
  dimension (numeric min, boolean or, permission intersection, negative
  spec accumulation), and an over-scoped declaration is rejected, never
  clamped. An authority bound a child can loosen is not a bound.
  test: tests/test_agentgit_wards.py::TestNeverWidens::test_composed_never_widens_parent
  test: tests/test_agentgit_wards.py::TestNeverWidens::test_chain_never_widens_root
- INV-7: A delegation link bound to a context verifies only in that
  context — lifting a valid signed grant from one task, scenario, or
  session into another is refused, and a nonce reused under a different
  payload is refused. A credential valid everywhere is a bearer token,
  not a delegation.
  test: tests/test_agentgit_identity.py::test_bound_link_replayed_into_foreign_context_is_refused
  test: tests/test_agentgit_identity.py::test_bundle_bound_to_task_verifies_and_replay_into_other_task_fails
  test: tests/test_agentgit_identity.py::test_nonce_reuse_under_different_payload_is_refused
