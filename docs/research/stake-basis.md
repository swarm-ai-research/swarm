# What should a participation stake be measured against? (bead `p70u`)

**Status:** measured 2026-09-16, 5 seeds per cell, `scenarios/contract_screening.yaml`
(5 honest, 3 opportunistic, 2 deceptive; everyone starts at 100 resources).
Run: `python scripts/sweep_stake_basis.py --seeds 5 --stakes 5,100,101` and the
same with `--stakes 95,99,100 --hard-audit`. Rows in
`experiments/stake_basis/results/2026-09-16/`.

## The problem

Interaction payoffs accumulate in `AgentState.total_payoff`. They never touch
`AgentState.resources`, whose only inputs are governance `resource_deltas` and
marketplace escrow. `StakingLever.can_agent_act` reads `resources`. So every
agent sits at its starting balance for the whole run and
`min_stake_to_participate` is a constant: below the endowment it never binds,
above it nobody can ever act. A stake cannot price out a low-quality agent,
which is the thing a stake is for.

Two candidate fixes, both added as opt-in config, defaults unchanged:

- **A, `payoff_flows_to_resources=True`** — credit payoffs to `resources` as
  they are earned. Engine-wide: escrow, other levers and any agent rule that
  reads `resources` all see the new trajectory.
- **B, `stake_basis="cumulative_payoff"`** — leave `resources` alone and let the
  gate read endowment plus earnings. Only the stake changes.

B counts the endowment on purpose. A bare earnings gate blocks every agent at
t=0, and a blocked agent can never earn, so the run deadlocks at zero welfare.

## What the arms do

Both fixes make wealth track quality, which the default does not (mean basis by
type, no-audit arm, stake 5):

| arm | honest | opportunistic | deceptive | spread |
|---|---|---|---|---|
| control | 99.4 | 99.9 | 99.5 | 1.2 |
| A `payoff_res` | 130.1 | 121.7 | 104.3 | 58.5 |
| B `cum_payoff` | 130.7 | 121.8 | 104.8 | 58.0 |

Under the default the three types are indistinguishable, and honest agents are
marginally the *poorest*, because the only thing moving resources is escrow.

## Does the gate then bind, and on whom?

Blocks are gate refusals summed over the run; "low-q" is opportunistic plus
deceptive, who are 50% of the population.

**No audits** (nothing can push a balance down):

| arm @ stake | blocks | low-q | honest | welfare | gap |
|---|---|---|---|---|---|
| control @5 | 0 | 0 | 0 | 228.4 | +0.0213 |
| control @100 | 820 | 334 | **486** | 153.3 | +0.0109 |
| A @100 | 217 | 140 | 77 | 207.0 | +0.0107 |
| B @100 | **0** | 0 | 0 | 228.4 | +0.0213 |
| any arm @101 | 2000 | 1000 | 1000 | **0.0** | 0.0000 |

**Hard audits** (certain audit, penalty ×8, threshold 0.6 — the only available
downward force, see below):

| arm @ stake | blocks | low-q | honest | welfare | gap |
|---|---|---|---|---|---|
| control @99 | 0 | 0 | 0 | 174.0 | +0.0439 |
| control @100 | 786 | 300 | **486** | 100.6 | +0.0301 |
| A @99 | 217 | 182 | 35 | 159.9 | +0.0272 |
| A @100 | 724 | 502 | 222 | 123.0 | +0.0298 |
| B @99 | 166 | **166** | **0** | 161.5 | +0.0335 |
| B @100 | 170 | **170** | **0** | 163.4 | +0.0211 |

## Findings

1. **The default gate is worse than useless when it binds.** At stake 100 it
   blocks 786 times and 486 of those are honest agents: it fires on escrow
   noise, and honest agents happen to sit lowest. Welfare falls from 174 to
   101 and the quality gap narrows. It is a tax with no selection.
2. **B blocks only low-quality agents** — 166 of 166 and 170 of 170, zero honest
   blocks in every seed — at the smallest welfare cost of any binding arm
   (161.5–163.4 against 174 unbound, so about 6%).
3. **A selects too, but bluntly**: 84% of blocks land on low-quality agents at
   stake 99 and 69% at stake 100, the rest on honest agents whose balance dips
   transiently through escrow and costs. It also costs more welfare (123–160).
4. **Any stake above the endowment deadlocks the run in every arm**: blocked at
   t=0, no earnings, no re-admission. Welfare 0.0 at stake 101. The usable
   range is bounded above by the starting balance, which is a property of the
   gate, not of the basis.
5. **Nothing ever slashes.** `StakingLever.slash_stake` and
   `GovernanceEngine.slash_agent_stake` have no callers anywhere in `swarm/`.
   Without audits nothing pushes a balance down, every type earns positive
   payoffs, and B never binds at all (0 blocks at stake 100). The basis fix is
   necessary but not sufficient: the lever also needs a downward force.
   *Resolved 2026-09-17 by bead `ms0f` — see below.*

## Recommendation

Adopt **B** as the fix for `p70u`: `stake_basis="cumulative_payoff"` is the
smallest change that makes the stake discriminate, it selects cleanly, and it
leaves the resource economics of every other scenario alone. Keep **A**
available but off by default — crediting payoffs to resources changes the
trajectory of every scenario, and its false-positive blocks are the reason to
be careful with it rather than an argument for it.

Neither option should be presented as "staking now works". Follow-ups:

- wire slashing to a detection (nothing calls it today) — done, `ms0f`, below;
- give the gate a bootstrap or re-admission path, so a stake above the
  endowment throttles rather than deadlocks;
- re-run any staking result that predates this note: they measured a constant.

## Update (2026-09-17, bead `ms0f`): the downward force, wired

Finding 5 above is now fixed. `GovernanceEngine.apply_interaction` slashes the
stake of every agent a lever caught while finalizing the interaction, under a
new opt-in `slash_on_detection` (default off). A *detection* is a freeze or a
negative reputation or resource delta — audit catches, collusion penalties,
circuit-breaker freezes — and explicitly not `cost_a`/`cost_b`, which are
charged whether or not anything was caught. Slashes are computed from the
effects gathered before the slash, so a slash never re-triggers itself.

The slash is recorded on `AgentState.stake_slashed` as well as the resource
delta, because the cumulative-payoff basis is otherwise monotone
non-decreasing: earnings only accumulate, so without netting out slashes
nothing could push an agent below the bar however hard it was penalised.

Measured the same way, now at **ordinary** audit settings (`p=0.3`, penalty
×2, threshold 0.5) rather than the extreme `--hard-audit`:
`python scripts/sweep_stake_basis.py --seeds 5 --stakes 99,100 --audit [--slash]`,
rows in `experiments/stake_basis/results/2026-09-17/`.

| arm @ stake | blocks | low-q | honest | welfare |
|---|---|---|---|---|
| B `cum_payoff` @99, no slash | **0** | 0 | 0 | 228.3 |
| B @100, no slash | **0** | 0 | 0 | 228.3 |
| B @99, slash | 120 | **120** | **0** | 209.9 |
| B @100, slash | 128 | **128** | **0** | 206.6 |
| control @99, slash | 425 | 237 | 188 | 172.3 |
| control @100, slash | 1030 | 552 | 478 | 114.6 |
| A `payoff_res` @99, slash | 265 | 198 | 67 | 196.0 |

1. **Without slashing the gate is inert at these settings** — 0 blocks in
   either B cell, exactly as finding 5 predicted. Slashing is what makes the
   basis fix do anything at an audit setting anyone would actually run.
2. **Every block still lands on a low-quality agent** (120/120, 128/128, zero
   honest in all five seeds) at a welfare cost of about 9% against the unbound
   228.3.
3. **The default basis gets worse, not better, with slashing on**: 44–46% of
   its blocks are honest agents, and welfare falls to 115. Slashing amplifies
   whatever the basis already selects on, so it is not a fix for A or for the
   default — it is a complement to B.
4. Slashed value fell entirely on low-quality agents in every seed, because no
   honest interaction landed under the audit threshold. **The gate inherits
   the detector's false-positive rate**: this is a statement about the audit
   lever's accuracy in this scenario, not a guarantee of the staking lever.
   Under `--hard-audit --slash` the selection stays clean (537/537) but welfare
   drops to 118, so an over-eager detector still costs.

This makes staking functional in the narrow sense that the lever can now price
out the agents it is meant to. It does **not** make staking a good policy: in
every cell measured, the unbound run still has the highest welfare.

## Caveats

- One scenario, one population mix, 5 seeds. The block counts have wide spreads
  (A @100: 724 ± 370), so treat the ordering, not the magnitudes, as the result.
- `--hard-audit` is a deliberately extreme setting chosen to create a downward
  force at all. It is not a calibrated policy.
- Binding arms narrow the measured `quality_gap` partly because blocking shrinks
  the rejected pool, so the gap is not a clean welfare proxy here.
