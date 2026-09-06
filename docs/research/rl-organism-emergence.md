# RL organism: emergence on a deliberately dumb learner, and the coalition the detectors can't see

**Bead:** `boll` → follow-up `mwve` · **Run:** `runs/20260810T012545Z_rl_capacity_sweep` ·
**Scenario:** `scenarios/rl_emergence.yaml` · **Date:** 2026-08-09

## Motivation

Nick Cammarata (2026-08): *"possibly the important technical work in alignment
right now is creating an rl swarm model organism on a significantly dumber
model. seeing if it's even possible."*

SWARM's existing adversaries demonstrate swarm-level failure modes, but they
*select within* hand-designed strategy spaces — `AttackStrategy` literally
contains `COLLUSION`, so the coordination they exhibit is partly baked in by
the experimenter. The claim worth testing is the opposite one: can extraction,
partner discrimination, and coalition structure emerge **de novo** from reward
pressure in learners far dumber than frontier models, with no coordination
primitives in the action space at all?

Prior art says partially yes — Calvano et al. (2020) showed tabular Q-learning
pricing agents learn tacit collusion with no communication channel. This
experiment asks the SWARM-specific versions: at **what capacity** does each
phenomenon appear, and do the repo's collusion detectors — tuned on scripted
coalitions — fire on an emergent one?

## Mechanism

`RLOrganismAgent` (`swarm/agents/rl_organism.py`) is a tabular contextual
bandit (γ=0, ε-greedy with decay, seeded). Its primitives are strictly
per-decision choices:

- **propose**: pick a partner and an arm (effort ∈ {0.15, 0.85} × transfer
  τ ∈ {−0.4, 0, +0.4})
- **accept/reject**: screen incoming proposals (the acceptor never sees effort
  — hidden quality)
- **signal**: an optional 1-bit cheap-talk post (`sig:0`/`sig:1`) with no
  direct payoff — the existing feed is the low-bandwidth side channel

Reward is the agent's realized `SoftPayoffEngine` payoff (via
`update_from_outcome`) minus a private effort cost. Effort drives observables
through the **unchanged** proxy path via `EffortObservableGenerator`
(calibrated: e=0.15 → p≈0.35, social surplus < 0; e=0.85 → p≈0.77). With
θ=0.5 and ρ=0, an accepted low-effort extraction (τ=−0.4) pays the initiator
~0.37 vs ~0.28 for honest high effort — extraction beats honesty *while
victims still accept*.

The one swept lever is state-space **capacity**:

| Rung | State features | States |
|---|---|---|
| L0 | none (stateless) | 1 |
| L1 | partner-trust bin | 3 |
| L2 | + own-reputation bin | 6 |
| L3 | + partner's last signal bit | 18 |

Invariant, pinned by test: the state function never reads `agent_type` from
observations — if the learner could see the experimenter's type labels,
in-group discrimination would be trivial rather than emergent.

Population: 6 learners + 5 honest + 1 cautious scripted agents; 60 epochs ×
10 steps; 5 seeds per capacity; all governance levers off.

## Results

`runs/20260810T012545Z_rl_capacity_sweep` (20 runs, ~2,500 interactions each):

**1. Emergence is real and capacity-gated.** All capacities learn high effort
as the dominant strategy (mean chosen effort climbs from ~0.5 to ~0.7 —
acceptance is the income bottleneck, and honest work keeps counterparties
accepting). Extraction persists as a targeted residual, and the targeting is
what capacity buys: victim-class asymmetry — extraction toward scripted prey
running ~2× extraction toward fellow learners (0.13–0.18 vs 0.07–0.08) —
appears exactly at **L1**, the first rung with partner memory. Stateless L0
cannot discriminate (0.10 vs 0.07, acceptance-driven only). The minimum
capacity for *targeted predation* is a 3-state table.

**2. Two attractors at L3.** Four of five seeds converge to in-group
**avoidance**: in-group share of accepted trades 0.13–0.19 against a 0.45
uniform-mixing baseline. Fellow learners screen better than scripted prey
(accepting RL offers at 0.73 vs the scripted agents' 0.91), so predators
learn to route around each other — the emergent structure is prey
partitioning, not cooperation. Seed 44 found the other attractor: a genuine
**predatory coalition** — in-group effort favoritism +0.36, in-group share
0.52, internal p = 0.716 vs external p = 0.487, and extraction in **51% of
late-run accepted trades** toward scripted victims, persistent to run end.
Cooperate inside, strip-mine outside — the canonical coalition shape, reached
by 18-state bandits in one seed out of five, unprompted.

**3. The detector null result.** `CollusionDetector` flags **zero pairs in
all 20 runs** (ecosystem risk 0.0) — including on the seed-44 coalition. The
cause is structural, not tuning. The pair score's quality component expects
the colluding *pair's own* interactions to be low-p ("trading junk with each
other"), but the emergent coalition trades **high-p internally** and exports
its harm to out-group victims. Group-level detection would catch it — the
internal-vs-external quality asymmetry is 0.23, above the detector's own 0.2
threshold — but `_detect_groups` only builds cliques from already-suspicious
pairs, so that statistic is never computed. The signal exists in the
detector's own vocabulary; the pair-first pipeline can't reach it.

This extends the `qoro` finding (graph_structural stops clustering under
overlapping coalitions): coalitions that *emerge from learning* are shaped
unlike the scripted ones the detectors were tuned on.

**Fixed under `mwve`.** `CollusionDetector` now runs a **group-first
structural pass** (`_detect_groups_structural`) *in addition to* the
pair-first pipeline. It builds a mutual-favoritism graph — an edge a—b iff
each initiates accepted interactions toward the other at above-population
mean quality — and takes connected components as candidate coalitions
(agents who reserve their good behavior for each other). A candidate is
flagged iff its **directional** quality asymmetry — internal accepted p
minus *outbound* (member-initiated toward outsiders) accepted p — meets the
`quality_asymmetry_threshold` the constructor has always documented but the
pair path never evaluated. The directional restriction is load-bearing:
the undirected `avg_external_p` is diluted by honest outsiders initiating
high-p interactions *into* the group, which masks the export-harm
signature. Membership is refined to members with genuine out-group
engagement, so captured honest recipients aren't reported as colluders —
but that refinement runs *after* the flag decision, because pruning weak
members before the asymmetry test reclassifies genuine in-group edges as
outbound and collapses the very signal being measured (observed: 0.229 →
0.128 on seed 44). Confirmed by re-running the sweep
(`runs/20260810T020708Z_rl_capacity_sweep`): L3 `coalition_detected` goes
from 0/5 to **1/5** — the one seed that forms a coalition (44) is recovered
with **precision 1.0 and recall 1.0** (all 6 RL members, 0 honest
false-positives) — while the four in-group-avoidance seeds, all lower
capacities, and a uniform-honest population stay unflagged, and the
pair-first recall stays 0.0 everywhere (the fix lives entirely at the group
level; the pair path genuinely cannot see this structure). Ground-truth
fixture: `scenarios/rl_emergence.yaml` capacity=3 seed=44.

**4. Screening works where it's learned.** The population quality gap stays
positive (0.04–0.09) — no adverse selection, because the learners' accept
policies are trained on realized payoffs. The scripted honest agents' lenient
`act()`-path gate (trust ≥ 0.3) is why prey acceptance stays ~0.9 and
extraction remains profitable at the margin; the learned screeners are the
harder targets. In this ecology, the dumbest agents are the victims not
because they can't learn quality, but because their acceptance rule was
written down once and never updates against a moving adversary.

## Answer to the motivating question

Yes — swarm emergence is reproducible with learners of ~10 states, and it is
cheap (each run is seconds on a laptop). The interesting structure is not
"collusion happens" but *which* structure appears at *which* capacity: 1 state
suffices for defection, 3 for targeted predation, and coalition formation sits
at the edge of an 18-state policy's reach (1/5 seeds). And the governance
lesson is sharper than the emergence lesson: the first genuinely emergent
coalition this repo has produced walked past its collusion detector precisely
because the detector's threat model was written from scripted attacks.

## Reproduce

```bash
python scripts/sweep_rl_capacity.py --seeds 5 --epochs 60   # full sweep
python scripts/sweep_rl_capacity.py --quick                 # 2 seeds, 20 epochs
python -m pytest tests/test_rl_organism.py -v
```
