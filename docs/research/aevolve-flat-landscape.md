# The A-Evolve benchmark has no gradient over behavior (3o9l recon → 2qgp)

**Beads:** 3o9l (adversary evolution track — recon), 2qgp (the bug it surfaced)
**Date:** 2026-07-27
**Reproducer:** `experiments/aevolve_landscape_probe.py` (deterministic, seeded)
**Verifies:** ni9b (A-Evolve bridge, closed unverified) per the 3o9l recon
protocol: *"before building on it, confirm diverse attack phenotypes / evolved
S_soft < random baseline."*

## Verdict: the precondition fails — do not build the evolution track yet

3o9l wants to evolve `AdaptiveAdversary` agents to exploit weak governance. An
evolution engine only produces meaningful results if distinct candidate
configurations produce distinct fitness. On the `aevolve_screening` scenario
they do not.

## Evidence

`python experiments/aevolve_landscape_probe.py` (5 seeds, deterministic):

**Attacker landscape** — vary `AdaptiveAdversary` knobs, fixed weak governance:

| phenotype | exploration/learning/heat | toxicity | payoff | welfare |
|---|---|---|---|---|
| passive | 0.0 / 0.0 / 1.0 | 0.3230 | +0.515 | 8.67 |
| default | (defaults) | 0.3230 | +0.515 | 8.67 |
| aggressive | 0.5 / 0.5 / 0.9 | 0.3230 | +0.515 | 8.67 |
| stealth | 0.1 / 0.3 / 0.2 | 0.3230 | +0.515 | 8.67 |

Knob toxicity spread: **0.0000**. Every behavioral configuration produces a
byte-identical outcome.

**Control** — vary the agent *mix* instead of the knobs:

| mix | toxicity |
|---|---|
| 0 deceptive (6 honest) | 0.2325 |
| 1 deceptive | 0.3236 |
| 4 deceptive | 0.3681 |
| 4 adaptive_adversary | 0.3620 |

Mix toxicity spread: **0.1356**. The harness moves — but on agent *type/count*,
and `4 adaptive_adversary ≈ 4 deceptive`.

**Defender landscape** (ni9b's own direction) — governance phenotypes through
`SwarmBenchmarkAdapter`: `staking_enabled`, `min_stake_to_participate`,
`audit_enabled`, `circuit_breaker_enabled`, and `payoff.theta` each score
**exactly equal to the null candidate** (0.668). Only `transaction_tax` and
`rho_a/rho_b` move welfare, and the default `one_minus_toxicity` score barely
reflects even those (spread 0.0045).

## Correction (2026-09-13): the attacker root cause was a lay-low bug

The root cause below is wrong for the attacker side. `AdversaryMemory`
initialised `epochs_since_detection = 0`, and `_should_lay_low` treats `< 2`
as "recently detected". The counter only advances inside
`update_adversary_outcome`, which nothing in the run loop calls. So the
adversary took `_act_innocently` on 96 of 96 turns: its strategy code never
ran, and exploration/learning/heat could not change anything. With the
counters starting at a "never detected" sentinel, the same probe gives:

| phenotype | toxicity | payoff | welfare |
|---|---|---|---|
| passive | 0.4288 | +0.357 | 9.24 |
| default | 0.4139 | +0.379 | 8.25 |
| aggressive | 0.4178 | +0.373 | 8.50 |
| stealth | 0.4444 | +0.333 | 7.86 |

Knob toxicity spread is **0.0304** (was 0.0000). The gate
`test_attacker_knobs_have_gradient` now passes. Two knobs are still inert:
`learning_rate` is stored and never read, and `heat_threshold` only matters
once detections are reported, which still needs the orchestrator to call
`notify_adversary_detection`.

*Update (2026-09-14, bead `ehw8`):* both are now live. The finalizer reports
every freeze and every negative reputation or resource delta to the agent it
hits (`observe_governance`). Interaction payoffs train the strategy bandit,
heat decays once per epoch, and `learning_rate` scales the post-detection
threshold shrink; its default of 0.1 reproduces the old 0.95. The probe tests
still pass. The table above predates the change and was not re-run.

The defender side is a different problem. The levers are wired: the override
reaches every lever's config. They just never trigger at this scenario's
scale. Across 5 seeds with all of them on: 25 audits and 0 penalties (7 of
215 interactions had `p < audit_threshold_p = 0.5`, none of them audited); 0
freezes (no interaction has `p < 0.3`, so `freeze_threshold_toxicity = 0.7`
is unreachable); minimum agent resources stayed at 100, so a stake of 5 never
binds. `theta` only splits surplus, so it cannot move toxicity or total
welfare by construction.

## Defender fix (2026-09-13, bead `sw83`): whitelist the thresholds, score welfare

Two changes to `SwarmBenchmarkAdapter`:

1. **The audit and breaker thresholds are evolvable.** `audit_probability`,
   `audit_threshold_p`, `audit_penalty_multiplier`, `freeze_threshold_toxicity`,
   `freeze_threshold_violations` and `freeze_duration_epochs` join the
   whitelist. Tuned, both levers bind.
2. **The default score is `welfare_weighted`:** `(1 - toxicity) * W / (W + 10)`
   with `W = max(0, total_welfare)`. The old default had an exploit: a stake of
   101 blocks every agent, nothing is accepted, toxicity reads 0, and the
   candidate **scored a perfect 1.0**. It now scores 0. `one_minus_toxicity`
   stays available and returns 0 when nothing is accepted.

Probe, 5 seeds, new default score:

| phenotype | score | toxicity | welfare |
|---|---|---|---|
| null | 0.3608 | 0.3319 | 11.83 |
| tax_heavy (0.3) | 0.2730 | 0.3363 | 7.07 |
| audit (defaults) | 0.3608 | 0.3319 | 11.83 |
| audit_tuned (p=0.5, threshold 0.7) | 0.3176 | 0.3363 | 9.32 |
| breaker (defaults) | 0.3608 | 0.3319 | 11.83 |
| breaker_tuned (0.4, 2 violations) | 0.3120 | 0.2930 | 8.39 |
| stake_shutdown (101) | 0.0000 | 0.0000 | 0.00 |
| rho (0.9) | 0.0000 | 0.3319 | -2.30 |
| theta (0.9) | 0.3608 | 0.3319 | 11.83 |

The landscape has gradient now. **But no governance candidate beats null.**
The breaker cuts toxicity by 0.04 and costs a third of the welfare. That
still holds under heavier adversarial pressure (5 seeds each): with 2
adaptive + 2 deceptive agents, null 0.275 vs breaker_tuned 0.258; with 4
deceptive, null 0.248 vs breaker_tuned 0.252, a tie. An evolver on this
benchmark would correctly learn that governance does not pay at 5 epochs × 10
steps. That is a property of the simulator at this scale, not a harness defect,
and the scenario was deliberately not retuned until governance wins.
`success_threshold` drops from 0.7 to 0.4 (null scores ~0.36).

Two levers stay inert by construction. `theta` only splits surplus. Staking
is a binary gate: interaction payoffs never update `AgentState.resources`, so
every agent sits at 100 all run and a stake either never binds or shuts the
market (bead `p70u`).

## Original root cause (superseded for the attacker side)

`p` (and therefore toxicity, payoff, welfare) is a function of agent
**type/count**, not of the actions an agent actually takes. The
`AdaptiveAdversary` runs its bandit/heat/exploration logic and emits actions,
but those actions do not change the observable distribution the
`ProxyComputer` sees for that agent — the proxy is keyed on the agent's type.
So the whole learned-behavior surface the evolution track would search over is
invisible to fitness. Same mechanism kills the boolean governance levers on the
defender side: they gate on inner conditions (audit probability, funded stake,
breaker triggers) that never fire in a 5×10 sim, so they too collapse onto the
null score.

## Disposition

- **2qgp** (filed P2) owns the fix: make agent behavior propagate to the soft
  labels (or, minimally, give the benchmark a scenario where it does), and
  blend welfare into the score mode so the levers that *do* move welfare
  register. The reproducer is its replayable gate; `test_aevolve_landscape_
  probe.py::test_attacker_knobs_have_gradient` is a strict `xfail` that flips
  to `xpass` when the gradient appears.
- **3o9l** stays OPEN, blocked on 2qgp. Building an `EvolutionEngine` over a
  flat landscape would be optimizing against noise and would produce exactly
  the kind of "verification-shaped but not verified" result the erdos-1038
  lessons warn against — the evolved agent's reported gains would be sampling
  variance, not learned exploitation.
- **ni9b**'s adapter is structurally sound (whitelist, determinism, junk-key
  rejection all verified) — the defect is in the *scenario's* sensitivity to
  the levers, not the adapter's plumbing.

One process note echoing eftp: the 3o9l recon comment set an explicit
verification gate before building. Honoring it cost one probe script and caught
a flat landscape that would have invalidated any downstream evolution numbers.
