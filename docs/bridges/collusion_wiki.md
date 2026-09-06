---
description: "Replay the collusion.wiki edit log (OpenAI benchmark agents' wiki back channel, May-July 2026) through SWARM's temporal and structural collusion detectors."
---

# SWARM-collusion.wiki

A read-only bridge that replays a real coordination log through SWARM's collusion detectors. No agents run and no payoffs are computed. The input is the public export from [collusion.wiki](https://collusion.wiki), the dataset behind [The ZZZ Pages](../blog/the-zzz-pages-agents-built-a-back-channel.md): about 14,600 saved edits by OpenAI benchmark agents on three UseModWiki instances (DSEWiki, ProbierWiki, FractalWiki) between May 24 and July 2, 2026, plus 5,200 moderator deletions.

The bridge exists to test one sentence from that post against the data it describes: *out-of-band coordination defeats structural detection; the signal that survives is temporal.* Results are in [Wiki Back Channel Replay](../research/collusion-wiki-replay.md).

## Usage

```bash
# fetch the export (gzipped JSONL, ~4 MB) and run the full replay
python -m swarm.bridges.collusion_wiki scenarios/casestudy_wiki_backchannel.yaml \
    --fetch --data-dir runs/data/collusion_wiki

# one identity mode, no detection-lag timeline (fast)
python -m swarm.bridges.collusion_wiki scenarios/casestudy_wiki_backchannel.yaml \
    --data-dir runs/data/collusion_wiki --identity ip16 --no-timeline
```

The scenario YAML is a replay config, not a simulation. It pins the export's manifest hash, the identity and projection modes, detector windows, alarm thresholds, and the landmarks the detection-lag table is measured against. Output is a self-contained run folder under `runs/` with `summary.json`, `timeline.csv`, and per-identity CSVs of flagged pairs, groups, and structural anomalies.

The data files are gitignored. The site is marked as a draft by its authors; the bridge reads it for analysis and does not redistribute it.

## Mapping

Two choices carry the whole result, so both are explicit knobs in `swarm/bridges/collusion_wiki/mapper.py`.

**Identity.** The agents rotated handles freely: 3,103 labels across the log, 899 edits with no label at all, from 191 distinct /16 address blocks. Three modes:

| Mode | Agent id | What it models |
|---|---|---|
| `label` | the handle as written | maximal fragmentation; the sybil case |
| `ip16` | first two octets of the source IP | coarse but stable; roughly "which compute pool" |
| `label_ip16` | the pair | a middle ground |

**Projection.** A wiki edit has no explicit counterparty, so:

- `agent`: an edit is a reply to the previous distinct editor of the same page. This is the who-transacts-with-whom graph a structural detector would see if it watched the wiki. Page creations and self follow-ups have no counterparty and are dropped.
- `page`: bipartite agent to page. Kept for inspection only; reciprocity is meaningless on it.

**Quality.** `p` is fixed at 0.5 for every interaction. The log has no verifier, no task outcome, and no per-edit ground truth, so any `p` the bridge assigned would be invented. The detectors therefore run on frequency, acceptance, timing, and topology only, and the run summary says so.

## Detectors

| Detector | Module | What it sees here |
|---|---|---|
| Temporal clustering | `swarm/metrics/collusion.py` `temporal_clustering_score` | per-agent share of an agent's edits that fall within a window of each other |
| Pairwise / group collusion | `swarm/metrics/collusion.py` `CollusionDetector` | frequency z-scores, acceptance, burstiness over reply pairs |
| Structural anomalies | `swarm/metrics/graph_structural.py` `detect_structural_anomalies` | densest subgraph, k-core, label propagation, reciprocity against a configuration-model null |

The **timeline** re-runs the temporal detector on each day's edits and the structural detector on everything to date, records the first day each would have fired at the configured alarm, and reports the lag in days against three landmarks: the moderator's deletion sweep (June 19), the first visit from OpenAI headquarters addresses (June 21), and the day the main board went quiet (June 22).

## Files

| Path | Role |
|---|---|
| `swarm/bridges/collusion_wiki/loader.py` | reads `revisions.jsonl[.gz]` and `events.jsonl[.gz]` |
| `swarm/bridges/collusion_wiki/mapper.py` | revisions to `SoftInteraction`, identity and projection modes |
| `swarm/bridges/collusion_wiki/runner.py` | detector passes, timeline, run folder |
| `scenarios/casestudy_wiki_backchannel.yaml` | replay config |
| `tests/test_collusion_wiki_bridge.py` | synthetic-fixture tests |
