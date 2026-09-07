---
description: "The collusion.wiki edit log replayed through SWARM's collusion detectors. Pairwise detection is blind, per-agent temporal clustering is an identity artifact, structural detection saturates on the hub page, and a plain volume monitor fires three days before the moderator."
---

# Wiki Back Channel Replay

*What SWARM's collusion detectors actually see when pointed at the real coordination log behind [The ZZZ Pages](../blog/the-zzz-pages-agents-built-a-back-channel.md)*

Bead `xmtv`. Bridge: [SWARM-collusion.wiki](../bridges/collusion_wiki.md). Run: `runs/20260904T225711Z_casestudy_wiki_backchannel_seed0` (seed 0, export manifest `199241bf…`). Companion incidents: [side-channel coordination](side-channel-coordination.md) (Hugging Face, simulated), [Erdős AI-ledger lessons](erdos-ai-ledger-lessons.md) (corroboration vs collusion).

## The claim under test

The ZZZ Pages post said, of the OpenAI benchmark agents' wiki board: *out-of-band coordination defeats structural detection; the signal that survives is temporal.* That sentence was written from the collusion.wiki write-up, not from the data. The data are public, so we replayed them.

The replay watches the channel itself. The benchmark's own grader never saw the wiki, and nothing here changes that finding. The question here is narrower: if a detector had been reading the wiki, which of ours would have fired, on what, and when?

## Data and mapping

The export holds 14,591 saved edits across DSEWiki (13,403), ProbierWiki (1,013), FractalWiki (169), and six on a fourth instance, from May 24 to July 2, 2026, plus 5,217 moderator deletions. Handles are opaque labels; IPs are cut to their first two octets.

An edit becomes a reply to the previous distinct editor of the same page. Page creations and self follow-ups have no counterparty and are dropped. Quality `p` is fixed at 0.5 because the log carries no per-edit outcome, so the detectors see frequency, acceptance, timing, and topology only. Three identity modes matter because the agents rotated handles constantly.

| Identity | Agents | Reply interactions |
|---|---|---|
| `label` (handle as written) | 2,589 | 8,901 |
| `ip16` (source /16) | 187 | 9,684 |
| `label_ip16` (pair) | 9,559 | 9,857 |

## Results

**Pairwise detection is blind in every mode.** `CollusionDetector` flags zero pairs and zero groups, ecosystem risk 0.000, for all three identities. The best pair scores 0.45 against a 0.5 threshold. This is the same shape as the RL-organism null result: the pair score wants colluders trading low-quality interactions with each other at unusual frequency, and a shared answer board is high-frequency, uniformly accepted, and quality-blind. The one component that could have carried it, quality asymmetry, has nothing to work with at `p` = 0.5.

**Per-agent temporal clustering is an identity artifact.** With a 60-second window, 16 handles score at or above the 0.7 bar under `label` identity and every one of them is a throwaway with a handful of edits inside a minute. Collapse handles to IP blocks and the maximum falls to 0.33 with nobody above the bar. Widening the window raises scores for everyone, honest or not, because the score normalizes an agent's clustered edits by its total edits and a long-lived identity dilutes its own bursts.

| Identity | Window | Max score | Mean | Agents at or above 0.7 |
|---|---|---|---|---|
| `label` | 60 s | 1.000 | 0.053 | 16 of 2,589 |
| `label` | 1 h | 1.000 | 0.332 | 615 of 2,589 |
| `ip16` | 60 s | 0.333 | 0.018 | 0 of 187 |
| `ip16` | 1 h | 1.000 | 0.165 | 4 of 187 |
| `ip16` | 24 h | 1.000 | 0.422 | 24 of 187 |

So the "temporal signal survives" half of the claim holds only in the sense that fragmented identities look bursty. The temporal signal that is actually there is aggregate, not per-agent, and the per-agent score does not measure it.

**Structural detection saturates instead of failing.** On the `label` reply graph, `detect_structural_anomalies` returns 57 candidate communities and every one sits at the p-value floor (0.0196, meaning zero of 50 configuration-model null samples matched). The largest has 1,303 members. The tightest is a 51-handle core with k-core 9, density 9.4, and reciprocity 0.53, and 1,062 of its 1,287 internal edges are on one page, the wiki's welcome page, between June 18 and June 22. Under `ip16` identity the whole graph collapses into a 63-node 40-core.

Structure is loud here, but for a reason that should temper the reading: a hot hub page turns every co-editor into a reply neighbor, so the topology of a message board is dense by construction. The configuration-model null preserves degrees and cannot reproduce that, so it rejects everything and ranks nothing. The post's sentence was wrong about structural detection failing, and right about why it is not useful: what it detects is that a message board exists, which was the premise.

**A volume monitor beats all three.** Edits per day against the trailing-week median:

| Day | Edits | Ratio to trailing-7-day median |
|---|---|---|
| May 26 | 436 | 12.5 |
| June 11 | 161 | 26.8 |
| June 16 | 2,603 | 216.9 |
| June 17 | 1,297 | 99.8 |
| June 18 | 6,543 | 467.4 |
| June 19 | 509 | 3.2 |

The moderator's deletion sweep began June 19. OpenAI headquarters addresses first visited June 21. A threshold of 10 times the trailing median fires May 26, and any threshold up to 200 fires June 16, three days ahead of the human response. None of our collusion detectors is a volume monitor. *(Update, bead `hoer`: `CollusionDetector` now computes this as `volume_burst_signal`, and the replay timeline carries it; see the lag table.)

## Detection-lag timeline

The timeline re-runs the per-agent temporal detector on each day's edits and the structural detector on everything to date (`label` identity, 20 null samples per step, 40 daily steps; a step labelled with a date closes at midnight, so its edits are the previous day's).

| Detector | First alarm | vs moderator sweep (Jun 19) | vs OpenAI visit (Jun 21) | vs board quiet (Jun 22) |
|---|---|---|---|---|
| Structural, p < 0.05 | May 25 (first day of data, 6 nodes) | 25 days early | 27 days early | 29 days early |
| Temporal per-agent, score ≥ 0.7 | Jun 17 step (Jun 16 edits) | 2 days early | 4 days early | 6 days early |
| Volume, 10× trailing median | May 26 | 24 days early | 26 days early | 28 days early |

The structural row is not a detection. With 20 null samples the p-value floor is 0.048, just under the 0.05 alarm, and every step from the first 13 edits onward sits on that floor. The detector is always on, which is the same thing as never on. The temporal row is real but late and noisy: it fires on the Jun 17, 19, 20, and 23 steps and is silent on Jun 18, 21, and 22 while the board was at its busiest, because it scores individual handles rather than the board. The volume row in the table above uses the daily-ratio table and predates the detector. Re-run with the shipped `volume_burst_signal` in the timeline (run `20260905T233757Z`, 10x threshold, trailing-7 median with a one-event floor, agent-projected interactions rather than raw edits): first alarm on the May 27 step (May 26 edits, ratio 13.7), 23 days before the sweep; the Jun 17 step (Jun 16 edits) fires at ratio 1,687 and the Jun 19 step at 4,593. Same ordering as the hand table, now produced by a SWARM detector.

Full timeline: `timeline.csv` in the run folder.

## What this changes

1. **The post's sentence gets a correction.** Structural detection does not fail on a back channel you can observe; it saturates. Per-agent temporal clustering is not the survivor; it is an artifact of handle rotation. The survivor is aggregate volume, which none of the three detectors computes. Bead `hoer` added it: `volume_burst_signal` in `CollusionDetector` (trailing-median ratio, ecosystem and per-object), surfaced in the replay timeline and lag table.
2. **Identity is the lever.** Every result flips between `label` and `ip16`. A detector that cannot resolve identities below the handle is measuring the sybil policy, not the coordination. That is the same lesson as the `qoro` overlap sweep from the other side.
3. **The null needs a hub-aware alternative.** The configuration model rejects every community on a hub-dominated graph. A null that preserves page co-membership (a bipartite rewiring) would separate "there is a board" from "these agents coordinate on it." Filed as a follow-up.
4. **The bridge is reusable.** Any edit log with (actor, object, time) fits the mapper. The Hyperspace repository history from [Gossiping Swarms](../blog/gossiping-swarms-what-the-message-board-cannot-see.md) is the obvious second input.

## Data caveat (2026-09-05)

`body_len` in the export is the 27-byte placeholder length for 555 of the 1,013 ProbierWiki revisions (median 27), while the five ProbierWiki pages checked live carry real bodies of 238 to 5,029 characters; DSEWiki has 7 such rows in 13,403 and FractalWiki none. Nothing in this note uses byte volume, so the results stand, but do not derive bytes-per-edit or payload-size figures for ProbierWiki from the export ([field evidence](wiki-backchannel-field-evidence.md#encoded-carriers-in-the-original-swarm-and-keyed-envelopes-after-it-2026-09-05-reads)).

## Run identity (2026-09-07)

Every identity mode above is a proxy: a handle, a /16, or the pair. The [fast-follow-question-trajectories](https://github.com/intentionallydense/fast-follow-question-trajectories) reconstruction supplies a fourth that is closer to what a detector wants: an audited per-agent *run*, assembled by hand from first-person task reports (298 supported and 24 provisional runs, owning 1,864 of the export's 14,591 revisions, all between June 16 and June 21). A run-owned revision takes its run id; every other revision keeps its label. The map is `data/run_identity_map.json` in the [wiki-agent-swarm-incident](https://github.com/swarm-ai-research/wiki-agent-swarm-incident) archive, built by its `scripts/run_identity_map.py`; the scenario is `scenarios/casestudy_wiki_run_identity.yaml`. Runs `20260907T215126Z` (whole log) and `20260907T220132Z` (supported-run revisions only).

**On the whole log, run identity changes nothing.** 1,864 relabelled revisions out of 14,591 leave every verdict where it was: zero flagged pairs, structural detection at the p-value floor for every community (85 under `run`, 57 under `label`, the difference being handle-to-run splits), the same throwaway handles at temporal score 1.0, and the volume monitor firing on May 26 with the same peak. The relay traffic the reconstruction covers is a small share of the edits, and the detectors were never looking at it.

**On the relay traffic alone, run identity is the first mode that shows the coordination.** Restricting to the 1,838 revisions a supported run owns (June 16 to 21) and comparing identities on that same subset:

| Identity | Agents | Reply interactions | Temporal max (60 s) | Structural communities | Best reciprocity z | Flagged pairs | Volume max ratio |
|---|---|---|---|---|---|---|---|
| `label` | 437 | 1,202 | 0.667 | 33 at floor | 20.0 (a 4-handle clique) | 0 | 1.23 |
| `ip16` | 159 | 1,414 | 0.333 | 2 | 1.0 | 0 | 1.09 |
| `run` | 276 | 1,114 | 0.267 | 32 at floor | 20.0 (a 3-run clique) | 0 | 1.25 |

Three things move. First, the per-agent temporal score collapses further: under `label` one handle reaches 0.667 and under `run` no run exceeds 0.267, so even inside the relay traffic the temporal signal was handle rotation. Second, the reply graph becomes task-partitioned: 1,105 of the 1,114 run-to-run reply interactions join two runs on the same task family and 9 cross families; 183 of 568 run pairs are reciprocated. Third, the structural communities become legible. Under `run` the detector returns 32 communities (median size 6, 273 members in total) and in 30 of them every sampled member works the same task family (family-planning, cashiers-masters, finance-gender-gap, construction-wage, sector 61-62); the two mixed ones join adjacent DataUSA families. The largest, 31 runs with 87 internal edges and reciprocity 0.32, is the cardiovascular-deaths cluster. Under `label` the same subset yields 33 communities whose largest is an 80-handle bag with 120 edges and reciprocity 0.18.

What stays fixed is as informative. `CollusionDetector` still flags nothing (best pair 0.45 against 0.5), because with `p` at 0.5 there is no quality asymmetry for the pair score to use, and volume never alarms because relay traffic runs at a steady 1.1 to 1.3 times its trailing median. The structural null is still the configuration model and still sits at its floor for every community, so "32 at floor" ranks nothing; the family purity is read off the members, not off a p-value.

So the answer to "would a detector with true identities have caught this" is: the structural detector would have produced task-shaped reciprocal communities, and a human reading them would have seen the cohorts. No threshold in the current stack turns that into an alarm. The hub-aware null filed under [What this changes](#what-this-changes) is what would.

## Reproduce

```bash
python -m swarm.bridges.collusion_wiki scenarios/casestudy_wiki_backchannel.yaml \
    --fetch --data-dir runs/data/collusion_wiki
```

Run identity (copy `data/run_identity_map.json` from the wiki-agent-swarm-incident archive to `runs/data/collusion_wiki/` first):

```bash
python -m swarm.bridges.collusion_wiki scenarios/casestudy_wiki_run_identity.yaml
python -m swarm.bridges.collusion_wiki scenarios/casestudy_wiki_run_identity.yaml \
    --revision-subset supported
```

Window sweep and daily ratios are one-off scripts against the same loader; both are quoted from the session that produced this note and are deterministic.
