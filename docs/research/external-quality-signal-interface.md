# Proposal memo: governance-side external-quality signal interface

**Issue:** distributional-agi-safety-2utj (Mechanism bet 3, dispatch 2026-07-18)
**Status:** design-only proposal — requires Auditor sign-off on feasibility before any code
**Reversibility:** fully config-gated, default off; no schema or behavior change when disabled

## Problem

Two independent pressures point at the same missing interface:

1. **The calibration chain serializes.** Arms B→C→D all wait on a dedicated
   judge-collection phase (arm B) before agreement (C) and the joined
   schema (D) can run. Judge scores are only produced by a special-purpose
   collector, so anchor data is a *phase*, not a *byproduct*.
2. **Agent-supplied anchors are fundamentally gameable.** The arm-2 cause-3
   probe demonstrated this empirically: the v3 rubric's target rules key on
   `metadata["agent_type"]`, which is fully under agent control — a CEM
   learner converged to claiming `honest` while true quality dropped
   (proxy toxicity 0.434 vs judge toxicity 0.250). Any anchor routed
   through agent-writable metadata inherits that failure mode.

A **governance-owned** channel addresses both: external judge scores flow
into interaction records during *any* governed run (anchor data as
byproduct), and the channel is tamper-resistant by construction because
agents never hold the pen.

## Proposed interface

### Config (new `GovernanceConfig` section)

```yaml
governance:
  external_quality_signals:
    enabled: false            # default off — pure observe-only opt-in
    judges: [mock]            # names resolved against JUDGE_SPECS
    rubric_version: rubric.v1 # pinned; prereg-compatible default (see naa8)
    sampling_rate: 0.1        # fraction of accepted interactions scored
    seed: 42                  # sampling + mock-jitter determinism
    observe_only: true        # hard invariant in v1 — see "What this must not do"
```

### Write path

A new `ExternalQualityMiddleware` (the existing `Middleware` lifecycle —
`on_step_start` / `on_epoch_end`) or an `on_interaction_complete` observer
scores a seeded sample of accepted interactions and writes:

```python
interaction.metadata["governance.external_quality"] = {
    "judge": "mock",
    "rubric_version": "rubric.v1",
    "rubric_sha256_prefix": "254a10e42f42a8d9",
    "score": 0.85,
    "scored_at_epoch": 4,
}
```

Design rules:

- **Namespaced key, governance-owned.** The `governance.` prefix marks the
  boundary. Enforcement note for the Auditor: interaction metadata is
  assembled by the orchestrator from several sources today; v1 must add a
  guard that strips/rejects any *agent-originated* write to `governance.*`
  keys (cheap: one check in the metadata merge), otherwise the
  tamper-resistance claim is decorative.
- **Batched + async-tolerant.** Live LLM judges must never block a step:
  scores are attached at epoch end for the epoch's sample. Mock judges
  (deterministic, jittered per 55bd8b0b) make the whole path testable
  offline, same as arms B/C.
- **Provenance-stamped.** Judge name, rubric version, and rubric SHA travel
  with every score, mirroring the arm-B collector's config stamps that made
  the naa8 audit resolvable in minutes instead of weeks.

### Consumption by the calibration arms

- **Arm D (joined schema):** governance-supplied scores enter the join as
  additional judge columns. `joined.v1`'s frozen-header rule is respected:
  columns are per-judge-name, and adding a judge is additive — no reorder,
  no version bump needed per the existing contract in
  `swarm/calibration/joined.py`.
- **Arm C (agreement):** with ≥2 configured judges, α/ICC can be computed
  from any governed run's stream, not only a dedicated collection.
- **Arm B (the preregistered anchor):** **unchanged.** This is the critical
  prereg-compatibility point: the preregistration locks arm B's dedicated,
  version-locked collection. The governance channel provides
  *supplementary* anchor data; it does not substitute for, amend, or feed
  the preregistered arm-B artifact. Partial parallelization comes from C/D
  being able to dry-run and harden against governance-channel data while
  arm B's official collection proceeds independently.

## What this must not do (v1 invariants)

1. **Observe-only.** The prereg explicitly bounds the judge-recursion risk:
   judge-rated quality is a *held-out signal, not a target*. If governance
   ever acts on this signal in-run (fees, freezes, admission), the anchor
   stops being external and the recursion the prereg bounds is live.
   `observe_only: true` is a v1 hard invariant, not a default.
2. **No agent visibility.** Scores must not appear in agent observations —
   an observable anchor becomes a training signal (cause-3 again, one hop
   removed).
3. **No prereg substitution.** See arm B above.

## Costs and risks

- Live-judge cost scales with `sampling_rate × interactions`; the mock path
  is free. Recommend live judges only behind an explicit budget note.
- Latency: epoch-end batching bounds it; a slow judge degrades to a
  partial sample (log the shortfall — no silent truncation).
- Goodhart drift: mitigated by invariants 1–2, but the Auditor should
  treat any future `observe_only: false` proposal as a new prereg event.

## Open questions for Auditor sign-off

1. Is the `governance.*` metadata-namespace guard sufficient
   tamper-resistance, or should scores live outside `SoftInteraction`
   entirely (side-table keyed by interaction_id, joined at export)?
2. Does supplementary anchor data flowing during the adaptive study
   require a prereg note even under `observe_only: true`?
3. Arm C consumption: is agreement computed over governance-channel scores
   admissible for the preregistered α threshold, or reportable only as a
   secondary analysis?

## Decision

Pending Auditor review. If signed off: implementation is one config
section, one middleware, one metadata guard, and additive join columns —
each independently testable with the existing mock-judge machinery.
