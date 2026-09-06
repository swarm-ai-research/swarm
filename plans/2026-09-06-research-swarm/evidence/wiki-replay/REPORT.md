# Wiki detection replay: archived seed 10000

Independent Reproducibility Sheriff check, 2026-09-06. Code HEAD `a3a0d5acbe1ced58316bcb4a889bb3d4745025a7` from `/tmp/swarm-research-wave1`; `/opt/anaconda3/bin/python`. No tracked files or task state changed; no fresh seeds or paid calls.

**PASS: all 6 paired cells reproduce treatment/control event lists and every metric exactly.** This covers 12 arm trajectories at seed 10000 across independent, authorized, and prohibited sharing, each with 0.25 and 1.0 observation coverage. Screen metrics also exactly match. This is a one-seed deterministic replay, not an independent replication or validation of the entire 200-seed archive.

**FAIL: full payload equality, even ignoring elapsed_seconds.** Each of the 6 files has exactly 7 recursive differences: elapsed_seconds plus three newly serialized default config fields in each arm. Added fields (absent in archive): `evasion_learning_probability=0.0`, `moderation_granularity="host"`, `page_deletion_fraction=0.5`. No existing config value, event, model metric, or screening metric differs. Initial strict comparison raised AssertionError; this failed gate is retained in replay.log. No outputs were regenerated or normalized to force a match.

Manifest differences are expected subset/timing changes: seed list 10000–10199 becomes [10000], completed_pairs 1200 becomes 6, and elapsed_seconds changes. All other manifest fields match, including frozen detector threshold 0.8, base_config {}, and the six cells. paired_summary.csv describes only the one replay seed; its uncertainty fields are empty by design and must not replace archived 200-seed estimates.

**PASS: 27 existing model tests** from `tests/test_wiki_sim.py`, 0.55 seconds. Output in tests.log.

Commands:

```sh
python scripts/sweep_wiki_mc.py --family detection --phase confirmation --seeds 1 --seed-start 10000 --detector-threshold 0.8 --output /tmp/swarm-wave1-wiki-replay
python -m pytest tests/test_wiki_sim.py -q -o addopts='' -o cache_dir=/tmp/swarm-wave1-wiki-replay/pytest-cache
```

Replay completed 6 pairs in 0.1690565417520702 seconds. Source archive: `/Users/raelisavitt/swarm/runs/wiki_mc_confirm_detection`. `comparison.json` preserves exact per-file differences, archive/replay SHA256 digests, HEAD, Python version, and source fingerprints. JSON diff entries show absent archive values as null; `added_config_fields` explicitly confirms these keys were absent, not present with null values. Archive manifest has no source SHA or Python provenance, so historical code/environment identity remains unverified. Full-file byte equality is not claimed.
