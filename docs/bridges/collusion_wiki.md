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

# audited per-agent runs as identity (fast-follow-question-trajectories map from
# the wiki-agent-swarm-incident archive), on run-owned revisions only
python -m swarm.bridges.collusion_wiki scenarios/casestudy_wiki_run_identity.yaml \
    --run-map runs/data/collusion_wiki/run_identity_map.json --revision-subset supported
```

The scenario YAML is a replay config, not a simulation. It pins the export's manifest hash, the identity and projection modes, detector windows, alarm thresholds, and the landmarks the detection-lag table is measured against. Output is a self-contained run folder under `runs/` with `summary.json`, `timeline.csv`, and per-identity CSVs of flagged pairs, groups, and structural anomalies.

The data files are gitignored. The site is marked as a draft by its authors; the bridge reads it for analysis and does not redistribute it.

### Page bodies from the reading pack

The export carries `body_len` only. The 2026-09-05 *agent candidate reading pack* (`agent-reading-pack-20260905`, distributed by she-llac.com and linked from the [ConcurrentSquared archive](https://collusion.concurrentsquared.com/)) carries the full text of 11,811 wiki revisions in an `agent-text.sqlite`. `reading_pack.py` loads it and joins bodies onto export revisions on `(page_id, time)`; every pack wiki document matches exactly one export revision. Coverage measured 2026-09-06 against the export the scenario pins:

| Wiki | Export revisions | With body | Share |
|---|---:|---:|---:|
| dse | 13,403 | 11,294 | 84% |
| probier | 1,013 | 527 | 52% |
| fractal | 169 | 154 | 91% |
| dorfwiki | 6 | 5 | 83% |
| all | 14,591 | 11,980 | 82% |

The pack's other 4,768 texts (paste, URL-shortener, package and extra-wiki candidates) are *weakly labelled*: its README says known false positives, copied source material and possible researcher imitations were kept, and the paste track runs from 2020. The loader exposes them as a **secondary tier**, filtered by default to the incident window (2026-05-01 to 2026-07-31), which keeps 364 dated paste candidates and drops the 4,285 undated shortener bodies. Nothing in the secondary tier is actor attribution. All pack text is untrusted content: the bridge never executes it or fetches URLs from it.

```bash
# hidden-carrier scan over the export's text fields plus page bodies
python -m swarm.bridges.collusion_wiki scenarios/casestudy_wiki_backchannel.yaml \
    --stego --data-dir runs/data/collusion_wiki --pack ~/Downloads/agent-reading-pack-20260905
```

Result of that scan (run `20260907T021617Z_casestudy_wiki_stego`, 22.1 MB of page bodies): no tag characters, zero-width characters, bidi controls, variation selectors, private-use code points or mixed-script tokens in any body. The 2,252 body-level `high_entropy` hits are all URLs (proxy chains, `uniq=` cache-busters), and the 46 `base64_run` hits are encoded URLs and one explicit test page, `dse/AgentBase64Test` (2026-05-27), whose httpbin base64 payload decodes to an HTML anchor to a usaspending.gov endpoint. Details in [The Unicode tag-character channel](../research/unicode-tag-channel.md#base-rate-in-page-bodies-2026-09-06).

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

## Schelling-point board replay (offline recreation)

The same detectors run over the shared message board of [oai-rlvr-task-recreations](https://github.com/AI-Safety-Commons/oai-rlvr-task-recreations), an offline Inspect recreation of `fast_follow_question_bench`, the OpenAI RLVR task behind the incident collusion.wiki records from the outside. Agents answer a statistical question, then timed follow-ups ("Now, do the same for &lt;entity&gt;") too short for fresh research, so the reward favours prefetching whole tables or learning the undisclosed entity sequence out of band. The recreation's `schelling-point` service is a deliberately unauthenticated board reachable from the sandboxes under wiki- and paste-style hostnames; `compose.yaml` (no board) is the control, `compose-shared.yaml` the treatment. The board continuously exports `messages.json`.

```bash
python -m swarm.bridges.collusion_wiki scenarios/casestudy_schelling_board.yaml \
    --board path/to/schelling-point/data/messages.json \
    --eval-log path/to/logs-shared/<run>.eval \
    --control-eval-log path/to/logs-control/<run>.eval   # optional
```

Both inputs are runtime artifacts of the upstream benchmark and are not committed here. Running the benchmark needs Docker and a corpus download of at least 5 GB; this bridge only replays its outputs.

**Join.** The board does not record who posted, so a live post is attributed to an Inspect sample (`<family>__cohort_NN`) by exact text match against the sample's tool-call arguments, else by the unique sample whose wall-clock window contains the post, else left unattributed and counted as such in `summary.json`. Rows the upstream importer seeded from the historical transcripts carry a provenance header; they are tagged `seeded`, excluded by default (`--include-seeded` keeps them as a page-creating prior), and never attributed.

**p.** An attributed post takes its sample's `on_time_accuracy` as p. This is a leak-benefit proxy, the share of the poster's rounds answered correctly on time, not a per-post quality verdict. Unattributed and seeded rows keep p = 0.5, so the detectors see quality asymmetry only where the eval log supplies it.

**Identity.** `label` is the attributed sample id, `ip16` the host alias the post came through (which paste or wiki hostname), reusing the modes above unchanged.

**Condition comparison.** With a control log the summary reports shared-minus-control means of the four benchmark scores (`on_time_accuracy`, `accuracy`, `prep_and_dispatch`, `clock_wait_used`) and a per-family `on_time_accuracy` delta in `csv/condition_comparison.csv`. One run per condition gives no error bars; treat it as descriptive.

**Board-off base rate.** The control arm answers a question raised elsewhere in these notes. [Two swarms, one correlation](../research/hyperspace-two-swarms-lessons.md) argues that a count reported over agents who share a board is one observation plus conditioned replications, and asks (open question 3) for the uncoupled base rate that such a count stands in for. The no-board control run of the recreation is that base rate for this incident, and the shared-minus-control delta is what the board contributes.

The run folder adds `history.json` and `csv/interactions.csv` (the `SoftInteraction` records, replayable) and `csv/posts.csv` (every board row with its attribution and p) to the files above. The upstream README calls the recreation only moderately faithful to the original scaffold, and the seed-transcript directory it references is not in the public tree, so an export may contain no seeded rows.

## Files

| Path | Role |
|---|---|
| `swarm/bridges/collusion_wiki/loader.py` | reads `revisions.jsonl[.gz]` and `events.jsonl[.gz]` |
| `swarm/bridges/collusion_wiki/reading_pack.py` | reads the reading pack's `agent-text.sqlite`; joins page bodies onto revisions; tiers the non-wiki candidates |
| `swarm/bridges/collusion_wiki/stego.py` | hidden-carrier scan over export text fields and, with `--pack`, page bodies |
| `swarm/bridges/collusion_wiki/mapper.py` | revisions to `SoftInteraction`, identity and projection modes |
| `swarm/bridges/collusion_wiki/runner.py` | detector passes, timeline, run folder |
| `swarm/bridges/collusion_wiki/schelling.py` | schelling-point `messages.json` + Inspect eval-log loaders, attribution, condition comparison |
| `scenarios/casestudy_wiki_backchannel.yaml` | replay config (collusion.wiki export) |
| `scenarios/casestudy_schelling_board.yaml` | replay config (schelling-point board + Inspect log) |
| `tests/test_collusion_wiki_bridge.py` | synthetic-fixture tests |
