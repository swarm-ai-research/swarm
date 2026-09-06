# Research swarm: first execution wave

2026-09-06. User-directed continuation of the [90-day plan](PLAN.md).

Three independent specialist agents handled security verification, calibration inventory, review, and wiki replay while the primary agent fixed the shared CI failure and calibration plumbing. Work is on local branch `codex/swarm-research-wave1` in `/tmp/swarm-research-wave1`. The two fixes are committed, independently reviewed, and locally checked; they have not been merged into main or remotely CI-verified.

## Concrete outcomes

| Work | Outcome | Evidence |
|---|---|---|
| `xwhu`: two orphan research pages | Reproduced failure; added research index cards without baseline changes. Commit `a3a0d5ac`. | [KB gate](evidence/gates/kb-graph-final.txt), [independent review](evidence/index-and-plan-review.md) |
| `np8b`: Arm D rejects real backends | Reused Arm B's factory; retained `joined.v1`; recorded rubric/hash/model overrides and synthetic source; preserved completed scores on later failures; refused overwrites. Commit `21e100c1`. | [27-test log](evidence/gates/calibration-tests.txt), [type check](evidence/gates/calibration-mypy.txt), [independent review](evidence/calibration/CODE_REVIEW.md) |
| `jxyi`, `qkzc`: stale security bugs | Independently verified existing main commit `e4438251`; closed both with executed evidence. | [Security audit](evidence/security/AUDIT.md) |
| `zsof`: delegation replay | Bound-context rejection works, but default nonce collision/legacy-context enforcement does not meet the original broad acceptance. Kept open. | [Residual probe](evidence/security/residual_probe.py), [probe output](evidence/security/residual-probe.json) |
| Wiki confirmation inventory | Parsed 3,920 paired payloads, verified manifest seed sets and no pair-seed mismatches. | [Inventory](evidence/calibration/inventory.json) |
| Wiki moderation analysis | Recomputed all 20 sign-flip/Holm tests; exact agreement with archived analysis. | [Recomputed analysis](evidence/calibration/holm_analysis.json), [interpretation](evidence/calibration/INVENTORY.md) |
| Wiki deterministic replay | Archived seed 10000: all events and metrics match across six paired cells; full-file equality fails due to newly serialized default config fields and elapsed times. | [Replay report](evidence/wiki-replay/REPORT.md), [exact comparison](evidence/wiki-replay/comparison.json) |
| Coordination | Confirmed shared DB through git common-directory resolution; initialized the missing inbox table/index/DONE trigger using the documented schema. Inbox query works and has zero unacked rows. Claims used the canonical atomic path and were released after handoff. | Local `/Users/raelisavitt/swarm/runs/runs.db`; no messages broadcast |

## Executed validation

- Calibration: **27 passed** (joined schema, frozen rubric, A–D determinism smoke, offline real-backend prompt/parse path, model override, failure persistence, overwrite refusal). Independently rerun with the same result.
- Changed calibration code/tests: ruff passed; mypy passed for the changed runner; `git diff --check` passed.
- Security: **123 passed** across attestation, identity, push-token, and review suites. The ambient-key regression also passed separately under xdist.
- Wiki model: **27 passed**, plus the six-pair deterministic replay.
- Knowledge graph: no new orphans or stale references; baseline files unchanged.

Tests used `/opt/anaconda3/bin/python` (3.13). The separate system `python3` lacks some required packages; its failed environment probes are not counted as passing tests. No paid model calls, full-repository test run, remote CI run, publication, or deployment occurred.

## What the evidence does and does not establish

The wiki numerical analysis is reproduced from stored payloads. The separate one-seed replay also reproduces trajectories under current code. Neither establishes independent simulator replication or historical behavior outside monitored surfaces. Current manifests say `explicit_frozen`, but lack immutable pre-run provenance; retrospective hashes cannot repair that temporal evidence gap.

Calibration raw B/C/D outputs were not located in the bounded search of current run roots. Historical tracker notes describe mock runs, while pilot docs describe live Ollama runs with absent raw directories. This is unresolved evidence location/provenance, not proof that live runs never happened. The raw pilot must be recovered before its claims are upgraded.

Real-judge support is now available in the local patch, but the runner's inputs are still synthetic fixtures. It does not supply the preregistered >=1,000-item held-out anchor or establish reliable judge agreement. Its journal preserves score invocations and parsed verdicts, not raw responses or individual internal retries, and does not enforce a spending cap.

## Next executable handoffs

1. **Integration / Sheriff:** land `a3a0d5ac` and `21e100c1` through the normal review/CI flow. `xwhu` and `np8b` remain in progress with candidate commit notes, because main still lacks these fixes.
2. **Calibration / Auditor, existing `5bpg` owner:** recover the June/July raw judge runs referenced in pilot docs; classify them as pilot/mock/registered collection using original config, rubric hash, model identifiers and logs. Freeze a held-out sample manifest and budget before real collection. Do not replace the absent anchor with synthetic scores.
3. **Authority / Mechanism Designer plus Auditor, `zsof`:** specify whether grants are reusable scoped credentials or single-use authorizations. Then enforce that policy at actual bundle consumers with negative tests for legacy/unbound grants and nonce reuse. Do not close the broader acceptance based only on opt-in fixture tests.
4. **Wiki / Auditor:** locate the original pre-run freeze artifact or qualify confirmation wording accordingly. Preserve current hashes and exact replay differences in the release provenance chain.
5. **Independent research branch / Scenario Architect, `a2il`:** begin the claim-ledger minimal scenario after its ownership check. This branch remains available while calibration awaits recoverable data and a funded collection plan.

The [task snapshot](task-snapshot.json) remains the original planning snapshot. It is deliberately not relabeled as current after these task updates. Source and evidence files in this folder are intended for review and subsequent release archiving.
