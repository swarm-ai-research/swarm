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
- Each treatment is paired with a same-seed untreated run. Intervals in the
  runner summaries are seed-level percentile bootstrap intervals.

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
displacements per run. Other deletion/lock contrasts include null completion
effects after correction.

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

The confirmation bead remains open for a final review of selected contrasts and
the frozen manifest. The model still excludes poisoning, adversarial policy
optimization, search-engine exposure, and realistic natural-language answers.
