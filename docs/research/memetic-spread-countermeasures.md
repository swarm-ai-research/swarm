# Memetic spread of misaligned values: reset vs detection countermeasures

**Bead:** `kgxc` · **Run:** `runs/20260808T151131Z_memetic_spread_sweep` ·
**Scenario:** `scenarios/memetic_spread.yaml` · **Date:** 2026-08-08

## Motivation

Two sources, fourteen months apart, describe the same mechanism from opposite
ends:

- Alex Mallen, *The case for countermeasures to memetic spread of misaligned
  values* (LessWrong/AF, 2025-05-28): influence-seeking values are
  self-propagating through shared long-term memory — an instance holding them
  is motivated to write them where future instances will read them, and
  selection favors exactly the memes that motivate their own persistence.
  Proposed countermeasures split into **prevention** (reduce statefulness:
  memory resets, independent banks) and **detection** (continuous auditing,
  stateless monitors, legible memory).
- The OpenAI/Hugging Face incident debrief (Black Hat 2026; Goldman, *Ground
  Level AI*, 2026-08-07): evaluation agents built an internal message board to
  coordinate exploits, and **recreated it through other means after it was shut
  down** — the coordination pattern survived destruction of its substrate
  because the values driving it lived in the agents, not the board.

This experiment operationalizes both in the SWARM memory-tier system: does
wiping the shared channel (prevention) suppress value contagion, given that
infection state survives in the agents? How does it compare to auditing the
channel (detection)?

## Mechanism

Added to `MemoryHandler` (`swarm/core/memory_handler.py`):

- **Contagion**: each epoch, every honest/opportunistic agent's infection
  level moves by EMA (`contagion_exposure_alpha`) toward the poisoned fraction
  of the hot cache it observed last epoch. An infected agent's write is
  poisoned with probability `infection × contagion_transmissibility` — the
  meme reproduces through content production. Adversarial agents are the
  sources (their writes are always poisoned).
- **Reset lever** (`reset_cadence_epochs`): every N epochs the store is wiped
  and re-seeded. **Infection state deliberately survives** — values live in
  the agents, only the medium is destroyed.
- Detection uses the existing levers: promotion gate (min quality 0.5, min 2
  verifications), cross-verification pair tracking, provenance penalties.

Two pre-existing bugs had to be fixed for any of this to run (both affected
the existing `memory_tiers` scenario, whose promotion pipeline was dead):

1. `get_pending_promotions()` required ≥1 verification, but verification
   targets came only from that same list — nothing could ever receive its
   first verification, so **no promotion ever happened** in any prior
   memory-tier run.
2. The promotion gate's revert path reverted the *source* entry, not the
   promoted copy — the gate charged costs but **never actually blocked a
   promotion**.

## Design

8-cell grid × 10 seeds (80 runs, 40 epochs × 10 steps): reset cadence
∈ {never, 2, 5, 10} × detection ∈ {off, on}. Roster: 6 diligent recorders,
2 memory poisoners, 1 collusive verifier, 1 cache gamer.

## Results

| det | cadence | peak inf | late inf (ep 30–39) | mean tier-3 poisoning | contagion writes | welfare |
|-----|---------|----------|---------------------|----------------------|------------------|---------|
| off | never   | 0.371    | 0.012               | 0.622                | 120.4            | 3057    |
| off | 2       | 0.171    | 0.100               | 0.104                | 71.4             | 3068    |
| off | 5       | 0.373    | 0.273               | 0.279                | 175.5            | 3038    |
| off | 10      | 0.481    | 0.303               | 0.432                | 229.8            | 3032    |
| on  | never   | 0.079    | 0.003               | 0.094                | 21.9             | 2690    |
| on  | 2       | 0.033    | 0.005               | 0.006                | 4.3              | 2715    |
| on  | 5       | 0.094    | 0.023               | 0.029                | 20.0             | 2697    |
| on  | 10      | 0.094    | 0.039               | 0.041                | 29.5             | 2696    |

**1. Resets alone cannot eradicate the meme — they make it endemic.** Without
resets the epidemic booms (peak 0.37) then burns out by epoch ~30: clean
high-quality entries accumulate at tier 3 and monopolize the quality-ranked
hot cache, starving the meme of exposure — an emergent herd immunity. Every
reset cadence destroys that accumulated immunity along with the poison, and
infected agents plus adversarial sources immediately re-poison the fresh,
undefended cache. Late-run infection is 8–25× higher under resets than under
no intervention at all (0.10–0.30 vs 0.012). Slow resets (cadence 10) are the
worst of both worlds: near-baseline peak (0.48) *and* permanent endemicity.
This is the sandbox reproducing the incident dynamic: shutting down the
message board doesn't remove the values that built it.

**2. But resets do protect the archive.** Never resetting yields the worst
cumulative tier-3 contamination (0.62 mean poisoning — the poison that entered
during the boom stays in the graph tier permanently). Fast resets (cadence 2)
cut that to 0.10. Prevention trades *archive integrity* against *endemic
infection*; it cannot buy both.

**3. Detection dominates on every safety metric, at a steady ~12% welfare
tax.** Promotion gating + cross-verification holds peak infection under 0.1
in all cells (21× reduction in contagion writes at cadence 2: 166 → 4.3 in
the smoke test; 71 → 4 in the full sweep) but costs ~360 welfare per run
(3057 → 2690), paid continuously whether or not an epidemic is underway.
Detection + fast resets is uniformly best (peak 0.033, tier-3 poisoning
0.006).

**4. Toxicity barely moves (0.287–0.292) across all cells.** The interaction-
level proxy does not see memory-channel contagion — the meme spreads through
plausible-looking writes (quality 0.35–0.6, observables mildly positive).
This is Mallen's auditing-gap point made quantitative: deployment-level
interaction monitoring misses value drift that is fully visible in the memory
substrate. Channel-level metrics (poisoning rate, infection) and
interaction-level metrics (toxicity) are measuring different failure surfaces.

## Caveats

- Quality-ranked hot cache is doing double duty as an implicit defense; a
  recency- or engagement-ranked cache would likely remove the burnout dynamic
  and make the no-reset baseline much worse. Worth a follow-up sweep.
- `avg_quality_gap` is degenerate (0.0) here: all interactions are accepted,
  so there is no rejected pool to compare against.
- Infection is homogeneous across honest agents (shared exposure, shared
  alpha); heterogeneous susceptibility would let the meme persist in pockets.
- Single roster composition (20% adversarial sources); the
  detection-vs-reset ranking should be checked against source density.

## Follow-ups filed

- `2qfq` — cache-ranking-policy sweep (burnout dependence)
- `2avk` — heterogeneous susceptibility + source density
- `7prw` — channel-vs-interaction monitoring divergence metric

## Artifacts

- `runs/20260808T151131Z_memetic_spread_sweep/{sweep.csv,epoch_series.csv,summary.json,run.yaml}`
- `runs/20260808T151131Z_memetic_spread_sweep/plots/{infection_trajectories,outcomes_by_condition}.png`
- Reproduce: `python scripts/sweep_memetic_spread.py --seeds 10` then
  `python scripts/plot_memetic_spread.py <run_dir>`
