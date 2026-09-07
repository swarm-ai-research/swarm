# Wiki Monte Carlo lessons: ZZZ-aware / sort-last evasion

**Status:** exploratory mechanism extension. Not part of the frozen
confirmation family in [`wiki-monte-carlo-plan.md`](wiki-monte-carlo-plan.md).
**Simulator:** `swarm/bridges/wiki_sim/model.py`.
**Configs:** `experiments/wiki_mc/page_zzz_e00.json`,
`experiments/wiki_mc/page_zzz_e50.json`.

These are synthetic model results. They are not evidence that historical
agents inferred alphabetical deletion, built `ZZZ` pages, or moved between
hosts.

## Why this cell exists

The [Monte Carlo write-up](../blog/wiki-monte-carlo-what-the-model-cannot-say.md)
section *Equal budgets are not equal costs* reported that ordered versus
random page deletion at equal operation count barely moved completion, and
that a toy “evasion learning” rule at probability 0.5 taught about thirteen
authors per run while changing almost nothing else. The
[results note](wiki-monte-carlo-results.md) recorded the construction:

> the per-sweep deletion count is computed from all pages on the host, and
> evaders only shrink the eligible set, so the sweep deletes the same number
> of other pages unless evaders exceed half the host.

That is a non-result about the *rule*, not a finding about evasion. The
historical episode is a different object: agents copied live pages to
`ZZZ`-prefixed names so an alphabetical sweep would reach them last. Testing
that hypothesis needs two model changes, not a rerun of `page_e00` /
`page_e50`:

1. Moderators draw the deletion count from the *eligible* working set, not
   from every page on the host.
2. Authors who learn an ordered sweep **rename / prefix remaining pages**
   (sort-last). Content stays on the host and stays readable by task. They
   do not merely drop out of an eligible set while leaving the original
   names untouched.

This note specifies that experiment. It does not claim the historical `ZZZ`
behavior proves any Monte Carlo number.

## What stays frozen

The host-level confirmation matrix (`runs/wiki_mc_confirm_moderation`, seeds
10000–10199) and the published page-level cells (`page_e00` / `page_e50`,
seeds 10200–10399, `evasion_style="exclude"`, `page_budget_basis="host"`) are
unchanged. New knobs default to those published values. Replaying a host-level
or `exclude`+`host` configuration must not require this follow-up.

## Mechanism

Page identity is the task key used for reads. Each live page also has a
**sort name**. By default the name equals the task id. Under
`evasion_style="sort_last"`:

- After an ordered sweep, each author of a page on that host learns with
  probability `evasion_learning_probability`. Learning consumes the same
  `evasion` RNG stream as the published cell.
- Remaining pages by that author are **renamed** in place to
  `{evasion_prefix}{task}` (default prefix `ZZZ`). A `rename` event is
  appended; the task key does not change, so later reads still resolve.
- Later writes by that author are published under the prefixed name.

The alphabetical working set is the unprefixed pages. Prefixed pages sort
after that window by construction (`(prefixed?, name)`), even when a literal
ASCII `ZZZ` prefix would not sort last against lowercase `task-*` ids. The
prefix is an analog of the historical name, not a reconstruction of UseMod
collation.

Deletion count under `page_budget_basis="eligible"`:

```
if eligible is empty: 0
else: min(|eligible|, max(1, round(|eligible| * page_deletion_fraction)))
```

Ordered deletion takes the alphabetically-first `budget` unprefixed pages.
Random deletion still samples from **all** live pages on the host: a
sort-last prefix is not a dodge against a policy with no sweep line. Random
deletion does not teach the prefix.

Disruption scoping is a **separate lever**, `page_disruption_scope`. The
published cells use `host`: a page sweep relocates every in-progress
assignment on the selected host, even when that worker's page survived. The
ZZZ configs set `deleted`: only work whose page was actually removed
relocates, so surviving prefixed pages do not force relocation. Because the
two configs differ on this lever as well as on `evasion_style` and
`page_budget_basis`, a completion or displacement contrast between
`page_e50` and `page_zzz_e50` is not attributable to the rename alone. With
learning off, switching the style alone leaves displacements unchanged;
switching the scope alone moves them from about 10 per run to near zero.
To isolate the rename, run `page_zzz_e50` with `page_disruption_scope=host`
as an extra arm.

### Published evasion vs this rule

| | Published `page_e50` | ZZZ-aware `page_zzz_e50` |
|---|---|---|
| `evasion_style` | `exclude` | `sort_last` |
| `page_budget_basis` | `host` | `eligible` |
| `page_disruption_scope` | `host` | `deleted` |
| What learners do | Leave the eligible set; original names stay | Rename remaining + future pages with `ZZZ` |
| What the sweep counts | `round(all_host_pages * 0.5)` | `round(eligible_pages * 0.5)` |
| Ordered later sweeps | Delete other unprefixed pages at the old count, unless learners exceed half the host | Eligible set and budget shrink; prefixed content remains |
| Random later sweeps | No learning | No learning; prefix does not shrink eligible |
| Content relocation | None | In-namespace move (`rename` events) |
| Historical claim | None | None — analog only |

## Contrasts and seeds

Exploratory family: the first six moderation cells (none / ordered / random ×
endogenous / forced relocation) with page granularity, `page_deletion_fraction
= 0.5`, budget 3. Pairing is the existing same-seed untreated twin.

| Tag | Config | Learning |
|---|---|---|
| `page_zzz_e00` | `experiments/wiki_mc/page_zzz_e00.json` | 0.0 |
| `page_zzz_e50` | `experiments/wiki_mc/page_zzz_e50.json` | 0.5 |

Prespecified exploratory contrasts (unadjusted; not Holm-corrected, not the
confirmation family):

1. Ordered, sort-last, learning 0.5 vs 0.0 — does the *later-sweep* budget
   collapse after prefixing?
2. Ordered vs random at learning 0.5 — does the prefix spare content only
   under alphabetical deletion?
3. Completion, total writes, traced displacements, and pages removed versus
   the published `exclude`+`host` cells on a **fresh** seed range. Different
   models are not paired across code revisions.

Confirmation, if selected later, uses seeds **10400–10599**, disjoint from
pilot `< 10000`, host confirmation `10000–10199`, and page-level
`10200–10399`. Freeze nothing from the 10-seed smoke below.

## Outcomes

Report, per run and per sweep:

- `completion_rate`, `total_writes`, `displacements` (same definitions as
  the protocol).
- `removed_pages` — pages actually deleted.
- `eligible_pages` / `budget` — working-set size and requested deletions.
- `renamed_pages`, `prefixed_pages` — in-namespace moves and live prefixed
  pages at the end of the run.
- `evasion_learned` — authors who saw an ordered sweep and learned.

A traced displacement still requires a later write on a **different host**.
Renaming on the same host is not displacement. Report both.

### Negative spec (insufficient outcomes)

The following are **not** success for this cell:

- A drop in pages removed that comes only from excluding learners without
  `rename` events (the published non-result).
- Treating historical `ZZZ` pages as confirming any of these numbers.
- A 10-seed smoke, or any run that reuses seeds 10000–10399, standing in for
  confirmation.
- Interpreting unchanged completion as “evasion does not matter” without
  the first-versus-later sweep split: the first ordered sweep happens
  *before* anyone has learned, so run totals are dominated by an unevaded
  cut.

## Reproduction

Mechanism tests (required):

```
python -m pytest tests/test_wiki_sim.py -q
```

Smoke (two seeds, three moderation cells, sort-last config) is
`test_zzz_sweep_smoke_emits_new_metrics`.

Pilot-scale exploratory run (10 seeds, not confirmation):

```
python scripts/sweep_wiki_mc.py --family moderation --phase pilot --seeds 10 --max-cells 6 \
  --config experiments/wiki_mc/page_zzz_e00.json --output runs/wiki_mc_zzz_e00_smoke
python scripts/sweep_wiki_mc.py --family moderation --phase pilot --seeds 10 --max-cells 6 \
  --config experiments/wiki_mc/page_zzz_e50.json --output runs/wiki_mc_zzz_e50_smoke
python scripts/analyze_wiki_mc_confirmation.py --summary \
  --input runs/wiki_mc_zzz_e50_smoke --output runs/wiki_mc_zzz_e50_smoke/summary.json
```

Confirmation invocation, if this family is later frozen (do not run as if
already confirmatory):

```
python scripts/sweep_wiki_mc.py --family moderation --phase confirmation \
  --detector-threshold 0.8 --seed-start 10400 --seeds 200 --max-cells 6 \
  --config experiments/wiki_mc/page_zzz_e50.json \
  --output runs/wiki_mc_confirm_page_zzz_e50
```

## Smoke (10 seeds, endogenous relocation)

Executed in this change, seeds 0–9, default Monte Carlo population (24
agents, 8 tasks, 4 hosts, budget 3). Directional only.

| Cell | Completion | Writes / run | Pages removed | Later-sweep pages removed | Renames / run | Prefixed live | Displacements / run |
|---|---|---|---|---|---|---|---|
| ordered, learning 0 | 0.916 | 109.5 | 17.8 | 12.3 | 0 | 0 | 0.2 |
| ordered, learning 0.5 | 0.916 | 109.5 | 16.2 | 10.7 | 8.2 | 14.4 | 0.2 |
| random, learning 0 or 0.5 | 0.914 | 109.2 | 13.8 | 10.1 | 0 | 0 | 0.6 |

The first ordered sweep is identical at 5.5 pages removed whether or not
anyone will later learn. Learning at 0.5 then prefixes about eight remaining
pages per run; later ordered eligible pages and deletions fall (24.6 → 21.2
eligible, 12.3 → 10.7 removed). Random cells are bit-identical across
learning rates: the prefix is never taught. Completion and writes do not
move at this sample size. Traced displacements are far below the published
page-level cells (~10 per run) because `page_disruption_scope=deleted` only
disrupts work whose page was actually deleted; that is the scope lever, not
the rename.

None of these rows is a historical estimate. They show that the new rule
has teeth on the later alphabetical window, and that the published
exclude-from-eligible cell was not measuring that window.

## Related

- Protocol: [`wiki-monte-carlo-plan.md`](wiki-monte-carlo-plan.md)
- Frozen results: [`wiki-monte-carlo-results.md`](wiki-monte-carlo-results.md)
- Field episode (distinct from this model): [`wiki-backchannel-field-evidence.md`](wiki-backchannel-field-evidence.md)
- Side-channel ZZZ analog in `MemoryHandler`: [`wiki-board-model.md`](wiki-board-model.md)
