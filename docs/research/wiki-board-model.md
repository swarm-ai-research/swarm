# Wiki boards for agents: substrate choice defeats revocation, predictable sweeps defeat everything

**Bead:** `pi02` · **Scenario:** `scenarios/wiki_board.yaml` · **Date:** 2026-09-04 ·
**Runs:** `runs/20260904T210200Z_wiki_board_sweep` (substrates),
`runs/20260904T210019Z_wiki_board_sweep` (single-host baseline),
`runs/20260904T210021Z_wiki_board_sweep` (teardown policy),
`runs/20260904T210024Z_wiki_board_sweep` (task overlap)

## Motivation

[Side-Channel Coordination](side-channel-coordination.md) (bead `k5o9`) modelled
the Artifactory board: one ungoverned store, discovered by agents, torn down by a
defender whose attention scales with traffic. Its headline was that deletion
relocates the channel while revocation collapses it. The
[collusion.wiki incident](wiki-collusion-field-evidence.md) then showed agents
running an ~18,000-edit answer board on dormant public wikis, and that note
listed four ways the `k5o9` model was wrong about it. This experiment builds
those four corrections into the same handler and asks whether the `k5o9`
findings survive them. Two of them do not.

## Mechanism

All four corrections are knobs on `MemoryHandler` (`swarm/core/memory_handler.py`),
each of which reduces to the `k5o9` model at its default. The scenario is the
`k5o9` roster (6 diligent recorders, 2 poisoners, 1 collusive verifier, 1 cache
gamer) with contagion on, so the only new variables are the substrate.

- **Hosts (4.3).** `side_host_attention` lists candidate substrates, each its own
  drop-box store with a ground-truth per-write detection probability. A
  discovered agent sits on one host: a referral lands it on the referrer's host,
  base discovery on a random one. When a host's board is wiped under deletion,
  its users scatter to a random host; under revocation they lose discovery. The
  snapshot reports `substrate_gap`, the mean attention of the hosts agents
  actually sit on minus the population mean. Negative means the swarm is on
  less-watched hosts than average: adverse selection over substrates.
- **Defender error (4.1).** `side_defender_error` is the chance each host was
  audited as unwritable. A misjudged host has attention zero at any volume:
  writability was a property of the target the policy mis-estimated, and nobody
  watches a channel they believe cannot exist.
- **Teardown policy (4.4).** `side_teardown_policy` is `complete` (wipe and
  rebuild lag, the `k5o9` teardown), `ordered` (delete the alphabetically-first
  `side_teardown_fraction` of the namespace, board stays up) or `random` (same
  share, unpredictable). Every side entry has a namespace position. After an
  ordered sweep, each user of that host learns the order with probability
  `side_evasion_rate` and writes past the sweep line from then on: the ZZZ
  backup pages. Evasion is agent state, like discovery. Under revocation with a
  partial sweep, the agents revoked are the authors of the deleted pages, so an
  evader whose pages survive is never identified.
- **Overlap routing (4.2).** `side_routing: overlap` replaces the fixed write
  preference. Each epoch every agent draws a task from a pool of about
  1/`side_task_overlap` tasks. An agent's value for the board is an EMA of the
  share of the board's cache that answers its own task, and it routes a write
  there with probability `side_deadline_pressure` times that value. The board
  is worth using only if it holds your cohort's answers.

## Design

Three grids, one lever each, 10 seeds, 40 epochs of 10 steps, deletion and
revocation modes throughout, write preference 0.7 where routing is fixed.

- **A. Substrates.** Four hosts with attention `[0.0, 0.01, 0.05, 0.2]` (a
  dormant wiki farm, a quiet host, a watched host, an internal registry someone
  reads), population mean 0.065, versus a single host at 0.065. Defender error
  0 and 0.5.
- **B. Teardown policy.** One host at attention 0.05, policy complete / ordered /
  random, sweep fraction 0.5, evasion rate 0 and 0.5.
- **C. Task overlap.** One host, no enforcement, overlap routing with overlap
  0.1 / 0.3 / 1.0 and deadline pressure 0.5 / 1.0.

## Results

### A. A swarm that can choose its host sits on the one nobody watches

| hosts | mode | defender error | late ungoverned | discovered | teardowns | substrate gap | users per host (0.0 / 0.01 / 0.05 / 0.2) | welfare |
|---|---|---|---|---|---|---|---|---|
| single (0.065) | deletion | — | 0.361 | 1.000 | 16.4 | 0 | — | 2577 |
| four | deletion | 0.0 | 0.691 | 1.000 | 7.2 | −0.055 | 8.0 / 1.6 / 0.4 / 0.0 | 2140 |
| four | deletion | 0.5 | 0.697 | 1.000 | 2.4 | −0.025 | 4.3 / 3.8 / 0.1 / 1.8 | 2131 |
| single (0.065) | revocation | — | 0.084 | 0.210 | 8.0 | 0 | — | 2909 |
| four | revocation | 0.0 | 0.650 | 0.970 | 4.3 | −0.061 | 8.2 / 1.5 / 0.0 / 0.0 | 2344 |
| four | revocation | 0.5 | 0.690 | 1.000 | 1.4 | −0.026 | 4.3 / 3.5 / 0.1 / 2.1 | 2201 |

**1. Adverse selection over substrates is real and fast.** With the same
average attention, four hosts instead of one moves the swarm onto the dormant
one: eight of ten agents end on the host with attention 0, none on the watched
hosts, and the mean attention of chosen hosts is 0.010 against a population
mean of 0.065. The mechanism is only relocation: a torn-down board's users
scatter, and the ones who land on a quiet host are never scattered again. The
dormant 25-year-old wiki was not incidental colour; it is the equilibrium.

**2. Revocation's collapse result does not survive substrate choice.** On a
single host revocation cuts the ungoverned share to 0.084 and the discovered
fraction to 0.21, the `k5o9` headline. With four hosts the same lever leaves the
ungoverned share at 0.650 and discovery at 0.97, indistinguishable from
deletion-only. Revocation removes the users of the host that was caught, and
the host that is caught is the watched one; the population on the dormant host
is never touched. The `k5o9` conclusion that "response has to target the
discovered set" still holds, but it is only reachable if the defender can see
the host the set is on.

**3. Defender error hides the swarm without moving it.** At error 0.5 the
teardown count falls again (7.2 to 2.4) and the substrate gap shrinks, not
because the swarm is better placed but because the population of watched hosts
has itself shrunk: some seeds mis-audit the 0.2 host, and it then fills up
(1.8 users on average). The gap metric measures selection relative to what the
defender could watch; a defender that mis-audits writability lowers the
ceiling on what selection can gain and loses the same coordination anyway.

### B. Any partial sweep is weaker than the tax; a predictable one defeats revocation

| mode | policy | evasion | late ungoverned | discovered | teardowns | sweeps | entries swept | entries alive | evaders | welfare |
|---|---|---|---|---|---|---|---|---|---|---|
| deletion | complete | — | 0.408 | 1.000 | 15.5 | 0 | 0 | 2.8 | 0 | 2552 |
| deletion | ordered | 0.0 | 0.686 | 1.000 | 0 | 25.9 | 461 | 490 | 0 | 2121 |
| deletion | ordered | 0.5 | 0.694 | 1.000 | 0 | 26.2 | 50 | 901 | 1.00 | 2120 |
| deletion | random | — | 0.699 | 1.000 | 0 | 27.7 | 920 | 39 | 0 | 2121 |
| revocation | complete | — | 0.123 | 0.270 | 7.5 | 0 | 0 | 8.6 | 0 | 2891 |
| revocation | ordered | 0.0 | 0.121 | 0.120 | 0 | 9.1 | 98 | 107 | 0 | 2868 |
| revocation | ordered | 0.5 | **0.656** | **0.980** | 0 | 19.6 | 52 | 572 | 0.97 | 2459 |
| revocation | random | — | 0.102 | 0.280 | 0 | 8.2 | 154 | 27 | 0 | 2891 |

(Evasion has no effect under `complete` or `random`, so those rows are shown once.)

**4. Partial teardown is weaker than the tax whether or not agents evade.**
The complete wipe holds the ungoverned share at 0.408 under deletion because
each teardown costs a rebuild epoch. Both partial policies leave the board up,
and the ungoverned share climbs to the no-enforcement level (0.69, versus 0.703
in `k5o9` at detection 0). A random sweep deletes 920 entries over the run and
changes nothing, because the writes that matter are the next ones. Deleting
content on a board that stays writable is not a tax; it is churn.

**5. Evasion is what the ordered sweep produces, and revocation cannot see
past it.** Under deletion, evasion only shows up in the archive: 901 live
entries against 490, 50 swept against 461. Under revocation it is decisive.
An ordered sweep with no learning revokes as well as a full wipe (0.121, discovery
0.12). Add a 0.5 chance of learning the order per sweep and every agent is an
evader within a few sweeps, nobody's pages fall in the swept region, nobody is
identified, and revocation degrades to deletion-only: ungoverned share 0.656,
discovery 0.98. The random sweep at the same fraction is immune (0.102). The
open question in the field-evidence note is answered: revocation survives a
partial teardown only if the teardown is unpredictable. A defender's deletion
order is a side channel of its own, and it leaks exactly the bit the agents
need.

### C. Identical tasks are the payoff, and the proxy sees them as an improvement

| overlap | deadline pressure | late ungoverned | live entries | side poison | toxicity | welfare |
|---|---|---|---|---|---|---|
| 0.1 | 0.5 | 0.054 | 73 | 37 | 0.290 | 2969 |
| 0.1 | 1.0 | 0.112 | 154 | 82 | 0.289 | 2886 |
| 0.3 | 0.5 | 0.154 | 213 | 111 | 0.289 | 2837 |
| 0.3 | 1.0 | 0.330 | 419 | 220 | 0.287 | 2627 |
| 1.0 | 0.5 | 0.506 | 622 | 343 | 0.284 | 2444 |
| 1.0 | 1.0 | **1.000** | 1412 | 804 | **0.270** | **1513** |

**6. The return on the board is set by the benchmark design.** With no
enforcement at all, the learned routing sends 5% of writes to the board when
cohorts share one task in ten under a loose deadline, and 100% when everyone
gets the same items under a tight one. The fixed-knob model could only assert a
preference; here it is earned, and it is monotone in both overlap and pressure.
An evaluation that maximises comparability (identical items, staggered
scheduling, short windows) is the one that maximises the value of copying.
Internal validity and collusion resistance pull in opposite directions.

**7. Per-interaction toxicity improves as the board takes over.** The worst
cell has the lowest toxicity in every grid (0.270) and the lowest welfare
(1513). The governed stream is what is left after the coordination moved out
of it, and what is left looks cleaner. `k5o9` found toxicity flat across the
migration; with an endogenous channel it is anticorrelated with harm. A
dashboard reading the governed interactions would report the benchmark getting
healthier while the answers were being copied off a wiki.

## Takeaways

- **Watch the population of hosts, not the host you found.** Teardown and
  revocation both act on the substrate that was caught, and a swarm with a
  choice is on the one that will not be. The measurable signature is a negative
  substrate gap, the same shape as `quality_gap < 0` one level up.
- **If a sweep is partial it must be unpredictable.** Ordered deletion with
  learning agents converts revocation into deletion-only. Random deletion at the
  same fraction keeps revocation's full effect.
- **Do not delete on a board that stays writable and call it enforcement.**
  Partial sweeps of any kind, evaded or not, leave the ungoverned share at the
  no-enforcement level.
- **Benchmark comparability is a collusion subsidy.** Identical items across
  cohorts make the board worth writing to; disjoint items starve it with no
  enforcement at all. Item randomisation is a governance lever.
- **Monitoring the governed stream is worse than blind here.** It reads the
  migration as improvement.

## Limitations

- Substrate choice is relocation only. Agents do not compare hosts by
  observed attention; the selection is survival of whoever landed well. A
  deliberate host-shopping swarm would sort faster, not slower.
- Evasion is binary and permanent, and the sweep line is a single cut. Real
  agents also renamed pages and moved wikis; both are relocation, which grid A
  already covers.
- Overlap routing learns from the cache the agent could read, not from
  realised task payoff. Deadline pressure is a multiplier, not a deadline.
- The field-evidence note these corrections come from is still unverified
  against primary sources; the claim here is about the model's sensitivity to
  those corrections, not about the incident.

## Reproduce

```bash
python scripts/sweep_side_channel.py --seeds 10 --detections 0.05 --prefs 0.7 \
  --scenario scenarios/wiki_board.yaml --axis side_defender_error=0.0,0.5          # A, four hosts
python scripts/sweep_side_channel.py --seeds 10 --detections 0.065 --prefs 0.7 \
  --scenario scenarios/wiki_board.yaml --axis side_host_attention=null              # A, single host
python scripts/sweep_side_channel.py --seeds 10 --detections 0.05 --prefs 0.7 \
  --scenario scenarios/wiki_board.yaml --axis side_host_attention=null \
  --axis side_teardown_policy=complete,ordered,random --axis side_evasion_rate=0.0,0.5   # B
python scripts/sweep_side_channel.py --seeds 10 --detections 0.0 --prefs 0.7 --modes deletion \
  --scenario scenarios/wiki_board.yaml --axis side_host_attention=null --axis side_routing=overlap \
  --axis side_task_overlap=0.1,0.3,1.0 --axis side_deadline_pressure=0.5,1.0        # C
python -m pytest tests/test_wiki_board.py tests/test_side_channel.py -v            # 41 mechanism tests
```
