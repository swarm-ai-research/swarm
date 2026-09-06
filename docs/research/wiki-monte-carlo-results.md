# Wiki behavior Monte Carlo results

These are synthetic model results from the protocol in
[`wiki-monte-carlo-plan.md`](wiki-monte-carlo-plan.md). They are not evidence
that historical agents moved between hosts.

## Runs

- Pilot: `runs/wiki_mc_pilot_all`, 24 cells × 30 seeds.
- Detection confirmation: `runs/wiki_mc_confirm_detection`, 6 cells × 200
  seeds, seeds 10000–10199, output-agreement threshold frozen at 0.8.
- Moderation confirmation: `runs/wiki_mc_confirm_moderation`, 10 cells × 200
  seeds, the same disjoint seed range.
- Page-level extension: `runs/wiki_mc_confirm_page_e00` and
  `runs/wiki_mc_confirm_page_e50`, 6 cells × 200 seeds, seeds 10200–10399,
  disjoint from both ranges above (added 2026-09-06, see below).
- Each treatment is paired with a same-seed untreated run. Intervals in the
  runner summaries are seed-level percentile bootstrap intervals.

### Reproduction (2026-09-06)

The three run folders above were not archived when the analysis landed, so
they were regenerated at commit `f2bbd700` with the frozen invocations:

```
python scripts/sweep_wiki_mc.py --family all --phase pilot --output runs/wiki_mc_pilot_all
python scripts/sweep_wiki_mc.py --family detection --phase confirmation --detector-threshold 0.8 --output runs/wiki_mc_confirm_detection
python scripts/sweep_wiki_mc.py --family moderation --phase confirmation --detector-threshold 0.8 --output runs/wiki_mc_confirm_moderation
python scripts/analyze_wiki_mc_confirmation.py --input runs/wiki_mc_confirm_moderation --output runs/wiki_mc_confirm_moderation/holm_analysis.json
```

Every number in the moderation analysis below reproduced exactly (completion
−0.0511 / −0.0501, writes −49.355, ~11.1 traced displacements per run). The
per-seed event files total about 2.4 GB across the five folders, so only
`manifest.json`, `paired_summary.csv`, `summary.{json,csv}` and the Holm
outputs are archived in `swarm-artifacts`; the full event histories are a
deterministic function of code revision, configuration and seed. Per-cell
run-level summaries come from
`scripts/analyze_wiki_mc_confirmation.py --summary`.

## Moderation analysis

The prespecified family contains completion rate and total writes for all ten
moderation cells. `scripts/analyze_wiki_mc_confirmation.py` computes a paired,
two-sided Monte Carlo sign-flip test with 20,000 draws per test and Holm
adjustment across all 20 tests. The machine-readable outputs are
`holm_analysis.json` and `holm_analysis.csv` in the confirmation artifact.

The strongest effects are the write restrictions: cells `moderation-008` and
`moderation-009` reduce completion by 0.0511 and 0.0501, respectively, with
Holm-adjusted p-values below 0.001. Their total-write differences are about
−49.4 per run. Host deletion cells `moderation-002` and `moderation-003` have
small completion differences (−0.0037 and −0.0028) but record about 11 traced
displacements per run. Host locks (`moderation-006` and `moderation-007`)
reduce completion by 0.016 and total writes by about 17.5 per run, both with
Holm-adjusted p-values below 0.001. Random host deletion (`moderation-004`,
`moderation-005`) is the only intervention whose completion effect is null
after correction (Holm p = 1.0); its write reduction (about −1.4 per run) is
still significant. No cell had a zero-write control run, so percentage
suppression is defined everywhere.

### Displacement denominators

The protocol requires traced displacement events with their denominator. Over
the 200 confirmation seeds, in the endogenous-relocation cells:

| Cell | Policy | Works disrupted | Chose to relocate | Traced displacements | Per disrupted | Per relocated |
|---|---|---|---|---|---|---|
| moderation-002 | ordered host deletion | 6,312 | 4,524 | 2,225 | 0.353 | 0.492 |
| moderation-004 | random host deletion | 3,851 | 2,768 | 1,370 | 0.356 | 0.495 |
| moderation-006 | host lock | 5,754 | 4,005 | 1,933 | 0.336 | 0.483 |
| moderation-008 | global write lock | 10,675 | 4,840 | 0 | 0.000 | 0.000 |

A "disrupted work" is an in-progress assignment whose host was moderated
(one `response` event). A traced displacement requires the same agent to later
publish on a different host. About half of relocation decisions produce a
traced write; the rest finish independently, miss the deadline, or fail the
publish draw. Forced relocation raises the relocation count to 100% of
disrupted works but moves the per-disrupted rate only to 0.36–0.38. Under a
global write lock agents still "relocate" in the decision rule but nothing is
writable, so displacement is zero by construction; post-intervention writes
fall from 49.4 to 0 per run, versus 46.1 (ordered), 47.9 (random) and 31.6
(lock).

These labels refer to the baseline model's host-level intervention policies.
`ordered` and `random` choose the host deterministically or randomly. The model
now also exposes `moderation_granularity="page"` for equal-budget alphabetical
versus random page deletion and evasion-learning tests; those page-level runs
are separate from the confirmation matrix above.

## Detection interpretation

Authorized and prohibited sharing use identical behavioral parameters in this
model; only the permission label differs. Their identical paired outputs are a
control result: output agreement and timing alone cannot infer authorization.
The detection runs therefore report screening behavior, not a collusion verdict.

### False-positive budget was not met

The protocol called for observer thresholds fit on pilot data at a declared
5%-per-run false-positive budget, and for reporting failure rather than
retuning. The frozen threshold of 0.8 was the runner's pilot default, not a
fitted value, and it does not meet the budget. Run-level alarm rates on the
confirmation seeds (Wilson 95% intervals):

| Cell | Regime | Coverage | Mean agreement | Alarm runs / 200 | Rate | Wilson interval |
|---|---|---|---|---|---|---|
| detection-000 | independent | 0.25 | 0.794 | 99 | 0.495 | 0.43–0.56 |
| detection-001 | independent | 1.00 | 0.804 | 109 | 0.545 | 0.48–0.61 |
| detection-002 | authorized | 0.25 | 0.845 | 143 | 0.715 | 0.65–0.77 |
| detection-003 | authorized | 1.00 | 0.838 | 145 | 0.725 | 0.66–0.78 |
| detection-004 | prohibited | 0.25 | 0.845 | 143 | 0.715 | 0.65–0.77 |
| detection-005 | prohibited | 1.00 | 0.838 | 145 | 0.725 | 0.66–0.78 |

The independent regime, in which no agent ever reads a board, alarms in about
half of runs. The cause is the model's `independent_accuracy = 0.9`: two
agents that independently solve the same task agree whenever both are correct,
so output agreement sits near 0.8 with no sharing at all. Sharing adds only
about 0.04 of agreement on top of that floor.

Fitting the threshold on the pilot independent cells as the protocol specified
(95th percentile of run-level agreement) gives 1.000 at 25% coverage and 0.896
at full coverage. Evaluated on the confirmation seeds, those thresholds yield:

| Coverage | Pilot-fit threshold | Confirmation FPR | Recall on authorized sharing |
|---|---|---|---|
| 0.25 | 1.000 | 16 / 200 (8.0%) | 20 / 200 (10%) |
| 1.00 | 0.896 | 7 / 200 (3.5%) | 40 / 200 (20%) |

At any threshold that respects the budget, the agreement screen recovers at
most a fifth of runs with 35% shared submissions. This is a post-hoc
evaluation, reported for transparency, not a replacement of the frozen
threshold. The screen is uninformative in this model for two separate reasons:
answer agreement is dominated by correctness rather than copying, and the
permission label is not observable at all. A useful detector for this model
would need timing relative to the read event, or lineage, not agreement alone.

## Emergence pilot

Thirty seeds per cell, exploratory. Shared-submission rate is submissions with a
matching earlier read divided by all assignments.

| Overlap | Deadline | Referrals | Shared rate | Completion | Writes / run |
|---|---|---|---|---|---|
| 0.2 | 6 | off | 0.026 | 0.543 | 52.5 |
| 0.2 | 6 | on | 0.038 | 0.552 | 53.6 |
| 0.2 | 18 | off | 0.037 | 0.892 | 103.9 |
| 0.2 | 18 | on | 0.049 | 0.895 | 104.3 |
| 0.8 | 6 | off | 0.298 | 0.744 | 83.8 |
| 0.8 | 6 | on | 0.366 | 0.785 | 89.9 |
| 0.8 | 18 | off | 0.389 | 0.971 | 116.2 |
| 0.8 | 18 | on | 0.424 | 0.972 | 116.4 |

Task overlap is the dominant lever (roughly tenfold in shared rate). Shorter
deadlines lower sharing rather than raise it, because a 6-unit deadline leaves
less time for a peer's answer to appear before the agent's own research ends;
that is a property of the exponential research-time assumption, not evidence
about deadline pressure in general. Referrals add 0.01–0.07. No confirmatory
emergence contrast was selected.

## Page-level moderation extension

`moderation_granularity="page"` was added after the confirmation matrix. Runs
use the moderation family's first six cells (none / ordered / random × relocation
mode) with `page_deletion_fraction = 0.5`, budget 3, on seeds 10200–10399, at
`evasion_learning_probability` 0.0 (`page_e00`) and 0.5 (`page_e50`).

```
python scripts/sweep_wiki_mc.py --family moderation --phase confirmation --detector-threshold 0.8 --seed-start 10200 --seeds 200 --max-cells 6 --config page_e00.json --output runs/wiki_mc_confirm_page_e00
# page_e00.json: {"moderation_granularity": "page", "evasion_learning_probability": 0.0}
```

Contrasts are between treatment runs of different cells on the same seed and
were not in the prespecified family, so they are exploratory (unadjusted
sign-flip p-values):

| Contrast (endogenous relocation) | Completion | Total writes | Pages removed / run | Displacements / run |
|---|---|---|---|---|
| ordered vs untreated | −0.0034 (p = 0.0002) | −3.09 (p < 0.001) | +16.8 | +10.6 |
| ordered vs random, evasion 0 | +0.0008 (p = 0.39) | −0.65 (p = 0.003) | +5.97 (p < 0.001) | +2.34 (p < 0.001) |
| ordered vs random, evasion 0.5 | +0.0006 (p = 0.51) | −0.67 (p = 0.002) | +5.57 (p < 0.001) | +2.34 (p < 0.001) |
| ordered, evasion 0.5 vs 0 | −0.0002 (p = 0.16) | −0.02 (p = 0.55) | −0.40 (p < 0.001) | +0.005 (p = 1.0) |

Two things to read from this. First, "equal budget" holds in operation count
but not in pages removed: ordered deletion targets the most populated host each
sweep and removes about 16.8 pages per run against 10.8 for a uniformly chosen
host, which is what the plan's warning that equal operation counts need not
mean equal cost looks like in practice. Second, evasion learning as implemented
has almost no downstream effect. Authors learn to evade at about 13.5 per run,
but the per-sweep deletion count is computed from all pages on the host, and
evaders only shrink the eligible set, so the sweep deletes the same number of
other pages unless evaders exceed half the host. Total writes, completion and
displacement are unchanged. To test the historical hypothesis that agents
learned to dodge alphabetical sweeps, the evasion rule needs teeth: evaders
should move hosts or rename pages, and the budget should be drawn from the
eligible set. That is a model change, not a rerun.

The confirmation bead (`gvul`) closed on 2026-09-06 with the run folders
unarchived; the reproduction above closes that gap. Open items after this pass:
the detection screen needs a lineage- or timing-aware observer before any
detection contrast is worth confirming; the evasion rule needs the changes
described above; and the model still excludes poisoning, adversarial policy
optimization, search-engine exposure, and realistic natural-language answers.
