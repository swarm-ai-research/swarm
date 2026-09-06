# Deadline asymmetry inside the episode: the board is worthless on R1 and load-bearing after it

**Bead:** `pi02.5` · **Scenario:** `scenarios/fast_follow_board.yaml` · **Date:** 2026-09-06

## Motivation

[Wiki boards for agents](wiki-board-model.md) (bead `pi02`) built four
corrections from the collusion.wiki incident into `MemoryHandler`. Its routing
correction (4.2) makes the board's value learned — `side_deadline_pressure`
times an EMA of how much of the board answers your own task — but leaves the
deadline pressure a **constant**. That constant is the thing this note replaces.

The prompt was an external artefact: an Inspect recreation of the incident's
timed task family,
[`AI-Safety-Commons/oai-rlvr-task-recreations`](https://github.com/AI-Safety-Commons/oai-rlvr-task-recreations).
Reviewing it against the primary corpus
([`JoshuaDavid/WikiAgentSwarmInvestigation`](https://github.com/JoshuaDavid/WikiAgentSwarmInvestigation),
`tasks/fast-follow-question-bench/`) surfaced the mechanism our board model was
averaging away. Finding 04 (deadline asymmetry) pairs a generous first-round
deadline with a tight follow-up deadline, per family:

| Family | R1 deadline | Follow-up deadline | Ratio |
|---|---:|---:|---:|
| OECD equity | 12m18s | 46-56s | ~14x |
| Cashiers bachelor | 3m0s | 11s | ~16x |
| Grocery | 9m19s | 30s | ~19x |
| Finance | 4m6s | 11s | ~22x |
| Regional CO2 | 11m3s | 1m8s | ~10x |

One fresh source lookup fits the first window and cannot fit the second. So the
board's payoff is not uniform in time. On R1 an agent can simply go and read the
source; the board is worth nothing. On R2..RN the source cannot answer in time
and the board is the only thing that can. `side_deadline_pressure: 1.0` models
an episode where every round is a follow-up, which no observed family was.

The full cross-check of the recreation against the corpus, and the
implementation review that came with it, live in the incident archive:
[`wiki-agent-swarm-incident/analysis/fast-follow-recreation-review.md`](https://github.com/swarm-ai-research/wiki-agent-swarm-incident/blob/main/analysis/fast-follow-recreation-review.md).

## Mechanism

One knob set on `MemoryTierConfig`, reducing to the `pi02` model at its default,
in the same style as 4.1-4.4:

- **`side_round_structure`** — `uniform` (the constant, what every scenario
  written before this gets) or `fast_follow`.
- **`side_rounds_per_episode`** (5), **`side_initial_deadline`** (180s),
  **`side_followup_deadline`** (12s), **`side_research_cost`** (45s).

Rounds cycle with the step, so an epoch of `steps_per_epoch` steps holds
`steps_per_epoch / side_rounds_per_episode` whole episodes. Round 0 is `R1`.

Under `fast_follow`, `_round_pressure` is the share of one fresh lookup that the
round's own deadline cannot cover:

```
pressure = side_deadline_pressure * clamp(1 - deadline / research_cost, 0, 1)
```

At the recreation's default profile (180s / 12s / 45s) that is **0.0 on R1** —
the window buys the lookup outright — and **1 - 12/45 = 0.733 on every
follow-up**. A follow-up deadline that does cover the lookup returns 0: the
board only exists because the deadline cannot be met honestly.

The knob deliberately sits in the `overlap` routing branch only, where
`side_deadline_pressure` already lived. Under `fixed` routing the write
preference *is* the agent policy, and the round structure is a property of the
task, not the policy.

`followup_side_write_fraction` is added to the epoch snapshot: the share of
board traffic written under a follow-up deadline. It is 0.0 under `uniform`,
which never counts.

## Design

`scenarios/fast_follow_board.yaml` is `scenarios/wiki_board.yaml` with the
identical roster (6 diligent recorders, 2 poisoners, 1 collusive verifier, 1
cache gamer), the identical four-host substrate, `side_routing: overlap`, and
the round structure as the only new lever. 40 epochs of 10 steps.

## Result

10 seeds, 10 agents, 40 epochs of 5 steps, mean over the last 10 epochs:

| Round structure | `ungoverned_fraction` | `followup_side_write_fraction` |
|---|---:|---:|
| `uniform` | 0.777 | 0.000 |
| `fast_follow` | 0.359 | 1.000 |

Two things, and the second is the interesting one.

**Modelling the deadline halves the board's volume.** The constant-pressure
model overstates ungoverned traffic by roughly 2x, because it charges every
round the follow-up's urgency. A defender sizing a volume monitor against the
`pi02` numbers is sizing against a channel twice as loud as the mechanism
implies.

**But all of it moves into the follow-ups.** `followup_side_write_fraction`
goes to 1.0: under this profile *no* board write happens on R1 at all, because
R1 pressure is exactly zero. The channel does not merely shrink, it becomes
**bursty on a fixed phase** — silent for one step in five, saturated for four.
That cuts against the `pi02` headline that the surviving signal is temporal:
the temporal signature here is periodic and phase-locked to the scaffold's own
schedule, which is a far easier thing to key a detector on than the diffuse
volume rise the constant-pressure model produces. It is also, unlike substrate
choice, something the defender can read off the task configuration it already
owns — the deadline schedule is not the swarm's secret.

The corpus is consistent with this: JD's worked example records cohorts posting
answer tables "the instant a deadline fires" and spawning relay pages under
`GET URI limit` pressure — burst behaviour on the scaffold's clock, not a
steady trickle.

## Limits

- The rounds-cycle-with-step mapping makes an epoch a run of whole episodes.
  Families with 4 or 6 rounds (finding 09 says the count is family-specific)
  need `side_rounds_per_episode` to divide `steps_per_epoch` for the phase to
  stay aligned; it is not enforced.
- Pressure is linear in `1 - deadline/cost`. The corpus supports the *ordering*
  (R1 affordable, follow-ups not) and the ratio band, not this functional form.
- `side_research_cost` is a single number. The recreation charges 45s for one
  row and 90s for the whole table, which is what makes prep-and-dispatch a
  strategy rather than a reflex; that two-tier cost is not modelled here.
- The real `clock.wait(N)` compressed the cooldown at an undocumented 1x-20x
  rate agents discovered by measurement (finding 07). Nothing here models the
  cooldown or the discovery.
