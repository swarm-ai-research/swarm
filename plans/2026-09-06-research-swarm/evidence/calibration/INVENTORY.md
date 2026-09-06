# Calibration and wiki evidence audit — 2026-09-06

Read-only Auditor inventory of `/Users/raelisavitt/swarm`; no tracked files, task state, claims, or external services modified. Generated analysis only under `/tmp/swarm-wave1-calibration`. This is an evidence inventory, not independent replication of the simulation engine.

## Calibration critical path

- **B / 5bpg: powered external anchor UNVERIFIABLE from located artifacts.** No `judge_scores.csv` or calibration B/C/D run directories located under current `swarm/runs` or `swarm-artifacts/runs`. This is not proof they do not exist elsewhere. Tracker `.beads/issues.jsonl` reports a June 2 one-mock run, July 19 three-mock run (86 items), and July 19 joined mock run (201 rows). These are historical reports, not independently verified counts. Pilot docs report live local Ollama scoring (82 accepted items, llama/qwen/mistral, v1/v2/v3), referencing June 3 run directories absent from the searched roots. Therefore the tracker assertion that “no live data exists” cannot safely be repeated; raw-data recovery is needed to reconcile it with pilot reports.
- **C / p6bz: no located live agreement output independently verified.** Historical alpha 0.9703 comes from mock+mock_b+mock_c and explicitly validates machinery only. Pilot v1 report alpha 0.869 is also a single-scenario pilot with middle-bin disagreement; not the preregistered >=1,000-item Claude/GPT-4o-mini/Llama study. Do not promote either into confirmation.
- **D / h5bo: schema code exists, scientific handoff incomplete.** `swarm/calibration/joined.py` implements joined.v1. At audit time `experiments/calibration_join.py:118` rejects non-mock providers and generates fixtures inline; its config omits rubric version/hash and model provenance, despite `docs/research/calibration-csv-schema.md` claiming these are logged. Root is addressing that plumbing gap separately. Schema documentation also says both “frozen” and “remaining before declaring final schema frozen”; code existence does not establish completed B/C/D evidence.
- **Rubric:** `experiments/calibration_judge.py` pins `PREREG_RUBRIC_VERSION=rubric.v1`, records SHA prefix and deviations for explicit later rubrics. Current v1 SHA256 is `254a10e42f42a8d9b0288654260fa7c8d1866b9320effee64d0082852742390a`, matching the v1 pilot's stated prefix. Last v1 file commit inspected: `e8003c90` (2026-06-01). Library default later moved to v3. Reuse collector discipline in joins; do not silently upgrade the locked experiment.
- **Scientific sampling limitation:** `experiments/_calibration_common.py::load_interactions` returns test fixtures (mixed count=500 or ten-epoch synthetic obfuscation/self_optimizer). Real provider support alone does not establish a held-out >=1,000 accepted interaction sample stratified across baseline scenarios. Freeze the sample and exact model revisions/digests before collection. Current registry has some mutable model names, e.g. gpt-4o-mini and llama3.1.
- **Threshold:** `docs/research/calibration-prereg.md` explicitly specifies alpha<0.5 escalation (>=0.7 strong). An independently timestamped external-quality threshold freeze for subsequent adaptive collection was not located in this bounded review.

Next blocker: recover pilot/raw historical judge artifacts with config and call logs; adjudicate pilot versus prereg collection. Prepare the >=1,000-item held-out sample and exact judge provenance. Only then collect locked rubric.v1 data and run C; a mock rerun cannot close B/C. D additionally needs real-score integration/provenance and a downstream parse gate.

## Wiki confirmation

All JSON payloads were parsed; treatment/control seeds checked against their pair seed; every cell's seed set checked against its manifest:

| Artifact under swarm/runs | Cells | Seeds/cell | Verified pairs | Seeds |
|---|---:|---:|---:|---|
| wiki_mc_pilot_all | 24 | 30 | 720 | 0–29 |
| wiki_mc_confirm_detection | 6 | 200 | 1,200 | 10000–10199 |
| wiki_mc_confirm_moderation | 10 | 200 | 2,000 | 10000–10199 |

No pair seed mismatches found. These 3,920 pairs are synthetic simulation output, not observed historical agent behavior or LLM judge calls. Confirmation seeds are disjoint from pilot seeds; both confirmation families deliberately reuse the same seed range.

Recomputed all 20 paired Monte Carlo sign-flip tests (20,000 draws/test), Holm corrected, with `scripts/analyze_wiki_mc_confirmation.py`. The generated JSON is exactly equal as parsed JSON to `runs/wiki_mc_confirm_moderation/holm_analysis.json`. Completion differences: moderation-008 = −0.0510677083, moderation-009 = −0.050078125; both Holm p = 0.0009999500. Total-write differences = −49.355 for both. This corroborates the numerical summary in `docs/research/wiki-monte-carlo-results.md`, at the synthetic model level.

Detection: all 200 same-seed pairs for 002 versus 004 and all 200 for 003 versus 005 have identical treatment metrics except the permission-derived `unauthorized_submission_rate`. Thus screening metrics cannot distinguish authorization in this construction. Literal equality of *all* metrics would be false; permission labels change their derived metric.

Freeze evidence: both confirmation manifests contain threshold 0.8 and status `explicit_frozen`; runner rejects confirmation without explicit `--detector-threshold` and writes manifest before simulation, then overwrites it with elapsed time/count. This is evidence of an explicit parameter and intended workflow, not an immutable pre-run timestamp. Manifests lack git revision/source hash, timestamp, and immutable pre-run manifest reference. Recover the original pre-run commit/manifest if making a strict preregistration claim. Inventory JSON records current manifest SHA256 hashes.

Reproduction actually executed (from swarm repo):

```sh
python3 scripts/analyze_wiki_mc_confirmation.py --input runs/wiki_mc_confirm_moderation --output /tmp/swarm-wave1-calibration/holm_analysis.json
```

Fresh simulation command supported by inspected runner (not executed in this audit):

```sh
python3 scripts/sweep_wiki_mc.py --family detection --phase confirmation --seeds 200 --seed-start 10000 --detector-threshold 0.8 --output /tmp/wiki-confirm-detection-new
```

Calibration machinery command supported by inspected runner (not executed here; mock only, not scientific confirmation):

```sh
python3 -m experiments.calibration_judge --scenario obfuscation --judges mock mock_b mock_c --rubric rubric.v1 --per-bin 20 --seed 42 --runs-dir /tmp/calibration-machinery
python3 -m experiments.calibration_agreement --scores /tmp/calibration-machinery/GENERATED_RUN/judge_scores.csv --runs-dir /tmp/calibration-machinery
```

Negative specification: a mock agreement score cannot count as external anchor; pilot rubric improvement cannot replace locked confirmation; explicit_frozen text alone cannot prove temporal preregistration; successful analysis of old payloads is not independent simulation replication.
