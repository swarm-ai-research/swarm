---
description: "Active research thread — current hypothesis and next steps"
updated: 2026-07-18
---

# Active Thread

> Drafted 2026-07-18 by Claude (research-operator session) from commit history,
> CHANGELOG, and beads — the plan had been stale since 2026-03-01 (beads `ja4u`).
> Owner review requested: correct anything that misstates intent.

## Current focus (July 2026): from confirmed governance results to agent-native infrastructure

The March arc (governance sweet spot, 4 studies, 755 runs — archived below) is
published and confirmed. Work since has pivoted from *demonstrating* governance
properties in synthetic populations to *instrumenting* them for real agent
systems. Three active threads:

### 1. AgentGit — tamper-evident provenance for agent-written code
Working hypothesis: agent contributions need identity-scoped, tamper-evident
track records before trust-gated automation is safe. Shipped: hash-chained
event store + git-notes provenance, deterministic multi-reviewer panel with
synthesized disagreement, machine-speed coordination primitives (claim/lock/
propose over SQLite), operational memory (repo/org/agent scopes), and
Beta-smoothed track-record reputation feeding the policy engine (fail-closed:
unknown trust ≤ threshold). Next: exercise the full attest → review →
reputation → gate loop on real multi-session work in this repo.

### 2. Coalition detection benchmark arc (sk95/5cdk lineage)
Structural detection is now a default governance signal (issue `2iok`).
Honest mixed verdicts recorded: temporal per-window detection wins at
*finding* burst coalitions (Hungarian recovery 1.000 vs 0.32 static) but
loses ROC ranking; structural detectors fail entirely on overlapping
coalitions where threshold detectors degrade smoothly. Open question: a
combined detector that uses structural for recovery and threshold scores for
ranking. Overlapping-coalition recovery (2-3%) is the known weak spot.

### 3. LLM-judge calibration (preregistered)
Arm B collector pinned to preregistered rubric_v1 (issue `naa8`). Experiment
in flight; do not modify the rubric mid-arm.

## Supporting work landed since March

- **Bridges**: agentveil v1 complete (failure-mode contracts D4/D5/E3 tested);
  aevolve BenchmarkAdapter (evolve governance designers; whitelist as
  integrity boundary; gate-Goodharting detector via quality_gap < 0);
  miroshark SSE event endpoint + multiseed runner.
- **Infra**: replay/checkpoint layer over the event log; sanitized opt-out
  telemetry; repo-wide PG-style cleanup (9 packages, ~30 tracked issues
  resolved as of 2026-07-18, incl. two event-log payload bugs and a
  fabricated-observation bug in the claude_code bridge); informal Joel Test
  audit (beads label `joel-test`: 5 gaps filed; Q3 closed via the
  ci-red-main alerting job, 3 process items open — evidence lives in the
  tracker, not the repo).

## Next steps

1. **Exercise AgentGit end-to-end** on real concurrent sessions (ties into
   worktree-policy decision, beads `urch`).
2. **Combined coalition detector** — structural recovery + threshold ranking;
   attack the overlapping-coalition weak spot.
3. **Run calibration arms** to completion under rubric_v1.
4. Carried from 2026-03-01 (verify still wanted): real Mesa model
   (Schelling/Sugarscape), population scaling to 100+ agents, online anomaly
   detector from the tp-rework correlation signal.
5. Process: agent trial-task gate (beads `4z8y`), operator usability pass
   (beads `2thn`).

## Blockers

None hard. Friction: multi-session shared-checkout races (mitigated by
ci-red-main alerting; worktree enforcement decision pending, beads `urch`).

---

# Archived thread (confirmed 2026-03-01)

## Hypothesis (CONFIRMED — 4-study arc, 755 runs)

The governance sweet spot is game-structure-invariant and adversary-resistant:
externality internalization (rho) combined with adaptive thresholds resists
signal gaming because 40% of v_hat weight comes from hard-to-fake signals
(verifier rejections, rework count). Higher rho raises the threshold, making
gaming progressively harder — but at the cost of welfare.

1. **Governance study** (110 runs): rho alone is a pure welfare tax; adaptive
   threshold creates real governance with sweet spot at rho∈[0.3,0.7]
2. **Adaptive agents study** (165 runs): learning agents recover +137% welfare
   at rho=1.0, Pareto-dominate adaptive at every rho
3. **Game structures study** (180 runs): generalizes across PD, Stag Hunt,
   Hawk-Dove — learning benefit 132-159% at rho=1.0 (d=9.88-11.63, p<0.001)
4. **Adversarial learners study** (300 runs): signal gaming does NOT break
   governance at moderate-to-high rho; at rho>=0.75 adversarial acceptance
   <4%; welfare loss linear in adversarial fraction; tp-rework correlation is
   a reliable anomaly detector (+0.15-0.25 honest, drops to -0.45 at 50% adversarial);
   the rho dilemma persists (rho=0.75+ costs ~30-40% welfare vs rho=0)

Published: ClawXiv `clawxiv.2602.00116`, AgentXiv `2602.00072`.
