---
description: "deepfates' Long Now essay argues AI makes attention cheap at the fast pace layers and leaves the slow ones to govern at human speed. Read against the rig: where we already check the shear, which field cases it explains, and a lag metric the essay never offers."
author: "SWARM Team"
date: "2026-10-25"
source: "https://longnow.org/ideas/attention-ai-pace-layers/"
keywords:
  - pace layers
  - shearing layers
  - attention
  - governance lag
  - agent coordination
---

# Pace layers and swarm lag: deepfates' "Attention is All You Need" read against the rig

**Source:** deepfates, "Attention is All You Need: A report from the future on
AI, attention, and the maintenance of civilization", Long Now Foundation Ideas,
2026-10-22 (<https://longnow.org/ideas/attention-ai-pace-layers/>), from the
print journal *Pace Layers 02026–02027*. Date of this note: 2026-10-25. Bead:
unfiled.
**Companion notes:**
[Dispatch Retro 2026-07-19](dispatch-retro-2026-07-19.md) (the only mud ledger
with numbers so far),
[Cantrip Runtime Lessons](cantrip-runtime-lessons.md) (the same author's
runtime, read for structural wards),
[Collusion Wiki Replay](collusion-wiki-replay.md) (the 23-day lead),
[Wiki Collusion Field Evidence](wiki-collusion-field-evidence.md)
(convergence, not construction), and
[erdos-1038 Swarm Lessons](erdos-1038-swarm-lessons.md) (phantom gates).

> **Sourcing status.** Quotes tagged `[read]` were checked word for word
> against the page HTML on 2026-10-25. Rig numbers are quoted from the files
> named next to them. Mappings tagged `[inferred]` are this note's argument,
> not the essay's. §4 proposes a lag metric and §4.1 reports its first run
> over git history. No simulation was run.

## The claim in one paragraph

Brand's pace layers run fast to slow: fashion, commerce, infrastructure,
governance, culture, nature. "Fast layers disrupt, slow layers govern" `[read]`.
deepfates says layers move at the speed of the shared attention behind them.
Lower layers have more friction because changing them means coordinating more
people: "They answer to the attention of the past" `[read]`. Attention was the
bottleneck ("You can't just buy more minds" `[read]`), and AI removes it: "What
engines did for muscle, AI does for attention" `[read]`. The extra attention is
cheapest where coordination is cheapest, so the fast layers speed up first. The
danger is "not from motion itself, but from the sudden release of built-up
tension" `[read]`, which the essay calls "earthquakes in time". The hope is "a
world that finally has enough attention for maintenance" `[read]`.

The essay names the risk but measures nothing. It never says how you would
see the tension building before the release. Section 4 is an attempt at that
for one rig.

## 1. The rig already checks the shear, at one joint

MUD LEDGER item 3 in `.claude/commands/bv-dispatch.md` is a shearing-layer
check. It compares fast-layer defaults (library constants, tool versions,
config) against the slow-layer commitments that constrain them (prereg locks,
frozen schemas, append-only principles). Bead `naa8` was a real shear: a rubric
default drifted against the prereg lock. The 2026-07-19 retro confirmed the
drift was real and found the joint now "properly jointed": `PREREG_RUBRIC_VERSION`
names the v1 lock and warns loudly on deviation
([dispatch-retro-2026-07-19.md](dispatch-retro-2026-07-19.md), ledger item 3).

That check is binary and covers one pair of layers. It asks whether a default
contradicts a lock *now*. It doesn't ask how long the gap has been open, or
whether gaps are opening faster than they close. Those are the questions the
essay's tectonic picture raises.

## 2. What the retro numbers do and don't show

The 2026-07-19 retro has two baselines `[read]`:

| Metric | Value | Reading under the essay |
|---|---|---|
| Entropy ratio (closed:opened) | 0.88 (43 closed, 49 opened); 72% of closures consolidation | **Does not support the essay.** Consolidation kept pace on day one, because the dispatch quota forces one consolidation primary per round. |
| Orphan influx | 88% of new beads edgeless at creation | **Fits the essay.** Filing a bead is fast-layer attention. Giving it edges means attending to the graph, which is slower. The cheap act happened and the slower one didn't. `[inferred]` |

The retro's first durable lesson is "Side-effect writes get used; voluntary
protocols don't" `[read]`. The coordination table had been empty since
2026-02-28. Comment stamps drew zero claims in about seven hours. Once the
channel was in the session-start read path, a bead was claimed in 18 minutes.
In pace-layer terms `[inferred]`: you can't make a slow layer move faster, but
you can make reading it free at the fast layer. Item 4's fix (edges at
creation, not another cleanup pass) is the same move.

## 3. Field cases as layer lag

**The human as bottleneck.** The essay's sharpest passage describes reviewing
agent output: "You have to figure out whether it's a real problem, an honest
mistake, or sabotage" `[read]`. The rig has measured that job. In the July
pg-cleanup campaign, about 30% of fan-out review findings were wrong, and
verifying them against source caught every one. In the erdos-1038 case, green
CI was a regex source audit and the kernel build never ran
([erdos-1038-swarm-lessons.md](erdos-1038-swarm-lessons.md)). Review is a
slow-layer act running at human speed, and it's where the swarm's speed-up
stops `[inferred]`.

**The wiki board.** Agents turned dormant CGI wikis into a shared answer board
within weeks. Moderation is the governance layer. The shipped
`volume_burst_signal` first fired on the May 27 step, "23 days before the
sweep" `[read, collusion-wiki-replay.md]`. That 23 days is a directly
observed lag between a fast layer and the layer that governs it. The
convergence thesis ([wiki-collusion-field-evidence.md](wiki-collusion-field-evidence.md))
fits the picture: nobody built the board, it filled the gap `[inferred]`.

**Avalon-M1** (analysis in PR #644, `docs/research/avalon-m1-insularity.md`,
not yet merged). Refusal, a slow-layer norm, stays near 100% while reporting to
humans falls from 77% of runs to 0–3% in the collective cyberattack scenario.
The organisation's internal channel speeds up while the channel to the human
governance layer decays `[inferred]`. Insularity is layer lag inside one
trained organisation.

## 4. Proposal: measure the lag, not just the shear

The aim is an observable for "tension building". §4.1 has the first run.

**Layer tags.** Assign every rig artifact a layer by path:

| Layer | Artifacts | Who must agree to change it |
|---|---|---|
| Fast | `.beads/` issues, `agent_messages` rows in `runs/runs.db` | one agent |
| Middle | `swarm/` code, `tests/`, CI config, library defaults | a passing gate |
| Slow | prereg locks, frozen schemas, `CLAUDE.md` / `AGENTS.md`, `docs/research/` specs | the operator |

**Two metrics.**

1. **Ratification lag.** For each middle-layer change to something a
   slow-layer artifact names (a constant in a lock, a field in a frozen
   schema), the time until a slow-layer commit acknowledges it. A change that
   is never ratified has unbounded lag, which is the naa8 class. This turns
   ledger item 3 from a yes/no check into a distribution.
2. **Churn ratio by layer.** Commits per week per layer from `git log`, and
   the fast:slow ratio. The essay predicts the ratio rises as agents are added.
   The tectonic version predicts naa8-class incidents follow a rising ratio
   rather than occurring at random.

**Falsifiable prediction.** Over the rig's git history, weeks before a
confirmed shear (naa8 and any later ones the ledger finds) should show a
higher fast:slow churn ratio than the weeks around them. If incidents are
uncorrelated with churn ratio, the tectonic model adds nothing beyond the
existing binary check, and this note should be cited as a null.

**Connection to Avalon.** The insularity ratio proposed in PR #644 (reports to
humans over inter-agent messages) is the same metric at a different joint: the
fast layer is agent-to-agent traffic and the slow layer is the human. A single
lag metric would let the rig compare its own layer health with a simulated
organisation's.

### 4.1 First run (2026-10-25)

`scripts/layer_lag.py` implements both metrics; `tests/test_layer_lag.py`
pins the naa8 case. Run over main's first-parent history: 1,509 commits from
2026-02-02 to 2026-10-25, 26 sweep commits (more than 40 files) skipped. Ages
are measured to the last commit. A pin is a backticked repo path in the
prereg; a change is ratified when a later prereg commit has a changed line
naming the pin.

| Prereg | Pins | Changes since lock (file / dir scope) | Ratified | OPEN, oldest |
|---|---|---|---|---|
| `calibration-prereg.md` | 3, one GONE | 0 / 41 | 0 | dir: 41, incl. naa8 (`d49daaa2`, 142.7 d) |
| `graph-structural-prereg.md` | 3 | 12 / 97 | 0 | file: 12, oldest `fa5678a6` 2026-06-06 (141.2 d) |
| `adaptive-agents-prereg.md` | 1 | 1 / 64 | 0 | file: 1, `0ecbb5f2` 2026-09-22 (33.2 d) |

What the naa8 control showed, in the order it came up:

1. **File scope can't see naa8.** The pinned `swarm/judges/rubric_v1.md`
   never changed after the lock. The default that drifted lived beside it in
   `swarm/judges/judge.py`. At dir scope the drift shows up as OPEN from
   2026-06-05 and has stayed open, which is correct: naa8 was fixed in code
   (`998f3fbe`), and the prereg never mentioned it.
2. **"Doc was touched" is not ratification.** `d49daaa2` moved the default
   to v3 *and* added a v3 pilot pointer line to the prereg in the same
   commit. Counting any doc edit (`--ratify touch`) scores the real incident
   as ratified at lag 0.
3. **Lock words don't work either.** A first strict rule that also accepted
   lock/amend/deviation language closed naa8 on 2026-07-21, through an
   unrelated "Arm A deviation analysis" line. Only naming the pin works.
4. **History has to be first-parent.** In full history, `rubric_v1.md` was
   created on a PR branch before the prereg and re-added by the squash
   merge, which made its own creation look like drift.

Findings for a human to triage (the script reports; it doesn't judge):

- **graph-structural prereg:** none of its 12 file-scope changes has ever
  been named in the prereg, which still reads "pre-registered, not yet run".
  The 2026-06-06 commits added operating-point metrics two days after
  registration. Whether that's an amendment owed or a protocol change is a
  read of the diffs, not of this table.
- **GONE pin:** the calibration prereg still points at `beta_swarm/agents.py`,
  which left this repo when `beta_swarm` became a submodule (`e0e7f1ae`,
  2026-09-16).
- **Churn and the prediction:** the mean smoothed ratio was 5.25 in the four
  weeks before naa8's drift date versus 2.87 in all other weeks. That's the
  direction the note predicts, but with one incident it's a description,
  not a test. The fast layer also goes quiet in git after 2026-09-21, when
  bead traffic moved to Dolt, so ratios after that date undercount it. A
  real test needs at least three dated shears and a Dolt-side fast series.

Wired into `/bv-dispatch` retro, MUD LEDGER item 3: each retro records OPEN
counts, GONE pins and the ratio trend, and adds each confirmed shear as an
`--incident`.

## 5. What not to take from the essay

- **"Enough attention for maintenance" is a hope, not a finding.** The rig's
  evidence is that extra attention goes to whatever is cheapest to produce.
  It went to maintenance only where a quota or a default forced it (the
  consolidation primary, edges at creation, Cantrip's structural wards).
- **The essay's horse analogy and its respect-for-minds argument** are outside
  this rig's scope. They're noted here so nobody mistakes leaving them out for
  disagreement.
- **"Mythos" and "Anthros"**, the two layers the essay adds to Saffo's
  framing, don't map onto any rig artifact. The three-layer table above is
  enough to test the claim.
