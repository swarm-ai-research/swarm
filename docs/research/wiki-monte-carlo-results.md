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

The change is implemented as an opt-in (`evasion_style="sort_last"`,
`page_budget_basis="eligible"`). Defaults keep the published `exclude`+`host`
cells. The follow-up protocol, smoke, and negative spec are in
[`wiki-monte-carlo-lessons.md`](wiki-monte-carlo-lessons.md). A 10-seed smoke
shows later ordered deletions falling after prefixing (12.3 → 10.7 pages)
with unchanged completion at that sample; that is not confirmation.

The confirmation bead (`gvul`) closed on 2026-09-06 with the run folders
unarchived; the reproduction above closes that gap. Open items after this pass:
the detection screen needs a lineage- or timing-aware observer before any
detection contrast is worth confirming; the ZZZ-aware page rule still needs a
frozen 200-seed confirmation on seeds 10400–10599 before any claim about
effect size (a third style, `evasion_style="move"`, landed 2026-09-07:
learners relocate their live pages to another writable host at learning time;
an exploratory rerun is in the next section); and the model still excludes
poisoning, adversarial policy
optimization, search-engine exposure, and realistic natural-language answers.

### Evasion-with-teeth rerun (2026-09-07, exploratory)

Treatment-run contrasts against `page_e50` (published `exclude`+`host`), same
200 seeds (10200–10399), six moderation cells, `page_deletion_fraction` 0.5,
budget 3. Only the two ordered cells differ; none/random cells are identical
by construction (evasion is learned only under ordered sweeps). Mean
difference and unadjusted sign-flip p (5000 draws); exploratory, not the
frozen confirmation called for above.

Provenance: these numbers were produced on the PR branch before it was
reconciled with the `evasion_style` / `page_budget_basis` knobs that landed
in #586, under a parallel implementation whose modes map as follows.
`shrink` is `exclude`+`host` and `shrink_elig` is `exclude`+`eligible`; both
have the same semantics on `main`, so those rows carry over. `move` and
`move_elig` are `move`+`host` and `move`+`eligible`; the `move` style was
ported unchanged, so those rows carry over too (they coincide because every
page on the host is eligible under `move`). `rename` kept evaders' pages
eligible and sorted them last with the host-wide budget; the landed
`sort_last` instead removes prefixed pages from the eligible set, so the
`rename` row is only an analogue. The `sort_last` and `sort_last_elig` rows
(`sort_last`+`host`, `sort_last`+`eligible`) were rerun on the merged branch
against the same baseline files; their none/random cells are identical to the
baseline, as expected.

| cell | contrast | completion_rate | task_success_rate | total_writes | displacements | removed_pages |
|---|---|---|---|---|---|---|
| moderation-002 (ordered/endogenous) | shrink mean | 0.911 | 0.821 | 103.645 | 10.605 | 16.410 |
| moderation-002 | e50_shrink_elig minus shrink (n=200) | +0.000 (p=0.001) | +0.001 (p=0.000) | +0.065 (p=0.020) | -0.050 (p=0.182) | -2.710 (p=0.000) |
| moderation-002 | e50_rename minus shrink (n=200) | -0.000 (p=1.000) | +0.000 (p=1.000) | -0.010 (p=1.000) | -0.005 (p=1.000) | +0.400 (p=0.000) |
| moderation-002 | e50_move minus shrink (n=200) | -0.001 (p=0.086) | -0.001 (p=0.176) | -0.075 (p=0.577) | -0.800 (p=0.000) | +0.475 (p=0.000) |
| moderation-002 | e50_move_elig minus shrink (n=200) | -0.001 (p=0.086) | -0.001 (p=0.176) | -0.075 (p=0.577) | -0.800 (p=0.000) | +0.475 (p=0.000) |
| moderation-002 | e50_sort_last minus shrink (n=200) | +0.000 (p=0.383) | +0.000 (p=0.059) | +0.010 (p=0.629) | +0.000 (p=1.000) | +0.180 (p=0.000) |
| moderation-002 | e50_sort_last_elig minus shrink (n=200) | +0.000 (p=0.001) | +0.001 (p=0.000) | +0.065 (p=0.006) | -0.025 (p=0.121) | -1.490 (p=0.000) |
| moderation-003 (ordered/forced) | shrink mean | 0.911 | 0.821 | 103.680 | 10.945 | 16.445 |
| moderation-003 | e50_shrink_elig minus shrink (n=200) | +0.000 (p=0.003) | +0.001 (p=0.001) | +0.050 (p=0.064) | -0.045 (p=0.344) | -2.730 (p=0.000) |
| moderation-003 | e50_rename minus shrink (n=200) | -0.000 (p=1.000) | +0.000 (p=1.000) | -0.005 (p=1.000) | +0.005 (p=1.000) | +0.395 (p=0.000) |
| moderation-003 | e50_move minus shrink (n=200) | -0.001 (p=0.269) | -0.001 (p=0.431) | -0.020 (p=0.915) | -0.680 (p=0.000) | +0.490 (p=0.000) |
| moderation-003 | e50_move_elig minus shrink (n=200) | -0.001 (p=0.269) | -0.001 (p=0.431) | -0.020 (p=0.915) | -0.680 (p=0.000) | +0.490 (p=0.000) |
| moderation-003 | e50_sort_last minus shrink (n=200) | +0.000 (p=1.000) | +0.000 (p=0.253) | +0.005 (p=1.000) | +0.000 (p=1.000) | +0.185 (p=0.000) |
| moderation-003 | e50_sort_last_elig minus shrink (n=200) | +0.000 (p=0.003) | +0.001 (p=0.000) | +0.050 (p=0.018) | -0.030 (p=0.245) | -1.515 (p=0.000) |

Reading: giving evaders teeth does not rescue the historical hypothesis inside this
model. `rename` (the sort-last ZZZ trick) and `move` change pages removed by under half a
page per run and leave completion, success and writes untouched; `move` lowers
displacements by about 0.7–0.8 because the evader's page has already left the host
before the sweep disrupts it (a voluntary relocation the displacement metric does
not count). Drawing the budget from the eligible set is the only knob with a
visible cost effect: about 2.7 fewer pages removed per run under `exclude`
and about 1.5 fewer under `sort_last`, in both cases with no completion change.
The landed `sort_last` behaves like `rename` where it matters: with the
host-wide budget it adds under 0.2 pages removed per run and moves nothing
else; with the eligible budget it removes fewer pages than `exclude`+`eligible`
does, because prefixed pages stay on the host and keep feeding the host count
while the eligible set shrinks (about 6.6 renames and 13.6 prefixed pages per
run at learning rate 0.5). The constraint is structural: a host
holds at most one page per task (`boards[host][task]`), so a sweep touches a
handful of pages and evasion can only shuffle those. A test of "agents learned to
dodge alphabetical sweeps" needs a page-population model (many pages per task,
name-keyed) before deletion order can matter. Runs: `runs/wiki_mc_confirm_page_e50_{shrink_elig,rename,move,move_elig,sort_last,sort_last_elig}`. The two `sort_last` folders are archived as summaries plus `contrast_vs_page_e50.json` in [swarm-artifacts](https://github.com/swarm-ai-research/swarm-artifacts) (355d312); the four pre-merge folders are not archived. To regenerate on `main`, use `page_e50.json` plus `{"page_budget_basis": "eligible"}`, `{"evasion_style": "sort_last"}`, `{"evasion_style": "sort_last", "page_budget_basis": "eligible"}`, `{"evasion_style": "move"}`, and `{"evasion_style": "move", "page_budget_basis": "eligible"}` respectively; the pre-merge `rename` row has no `main` equivalent.

### What the field evidence says about the ZZZ premise (2026-09-08)

Everything above tests the ZZZ trick *inside the model*. The premise it reaches
for — that DSEWiki agents prefixed pages with `ZZZ` to sort last against an
alphabetical moderator sweep — has since been measured against the export in
[wiki-collusion-field-evidence.md](wiki-collusion-field-evidence.md) §6.4. It
does not survive, on four counts:

- **The June sweep was not alphabetical.** Across the full sequence of 5,217
  hand deletions (2026-06-04 → 07-14) the rank correlation between deletion
  position and title alphabetical position is **0.009**; the June deletions run
  in RecentChanges order, newest first.
- **The sustained alphabetical passes came after the agents had stopped.** Of
  41 ascending runs of ≥10 titles, the five longest all begin on or after
  2026-07-12 (89 titles from `2026-07-13T19:40`, then 78, 63, 58, 49) — three
  weeks after the swarm had gone. June's largest run is 35.
- **The `ZZZ` pages predate that phase by about seven weeks**, and are not
  backups: zero records anywhere in the bundle carry the `backup` content kind,
  and the classified `ZZZ` rows are answer-share (26), probe-test (11),
  coordination (9), bridge (8), data-cache (4), with unique-test-page titles.
- **It bought nothing anyway.** 3,898 of 3,908 distinct dse titles were deleted
  — 0.26% survival. All 77 `Z`-prefixed and all 16 `ZZZ`-containing titles were
  deleted.

None of this touches the `sort_last` / `move` knobs or any number in the table
above; those are model behaviour and stand. It constrains only how they may be
cited. The sweep results here — and the `pi02` results they extend — are
**simulation predictions with no field instance behind them**. The paragraph
above is right that a real test needs a page-population model; it is now also
true that even with one, the historical episode it was reaching for did not
happen as described.
