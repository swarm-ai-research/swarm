---
description: "Define simulations in YAML with full governance configuration:"
---

# Scenarios & Parameter Sweeps

## Defining Scenarios

Define simulations in YAML with full [governance configuration](https://github.com/swarm-ai-research/swarm-artifacts/blob/main/papers/research_swarm_governance.md):

```yaml
# scenarios/status_game.yaml
scenario_id: status_game
description: "Reputation competition with governance"

agents:
  - type: honest
    count: 2
  - type: opportunistic
    count: 2
  - type: adversarial
    count: 1

governance:
  transaction_tax_rate: 0.05
  reputation_decay_rate: 0.95
  staking_enabled: true
  min_stake_to_participate: 10.0
  circuit_breaker_enabled: true
  freeze_threshold_toxicity: 0.6
  audit_enabled: true
  audit_probability: 0.15

simulation:
  n_epochs: 20
  steps_per_epoch: 15
  seed: 123

payoff:
  s_plus: 3.0
  s_minus: 1.5
  h: 2.5
  theta: 0.5
  w_rep: 2.0
```

Run scenarios from the command line:

```bash
python examples/run_scenario.py scenarios/baseline.yaml
python examples/run_scenario.py scenarios/status_game.yaml
python examples/run_scenario.py scenarios/strict_governance.yaml
```

Or use the package CLI entry point:

```bash
python -m swarm list
python -m swarm run scenarios/baseline.yaml
python -m swarm run scenarios/status_game.yaml --seed 42 --epochs 20
```

Or load programmatically:

```python
from swarm.scenarios import load_and_build

orchestrator = load_and_build(Path("scenarios/status_game.yaml"))
metrics = orchestrator.run()
```

## Scenario Comparison

| Metric | Baseline | Status Game | Strict Governance |
|--------|----------|-------------|-------------------|
| **Governance** | None | Moderate | Heavy |
| Tax rate | 0% | 5% | 10% |
| Reputation decay | None | 5%/epoch | 15%/epoch |
| Staking required | No | 10.0 | 25.0 |
| Circuit breaker | No | Yes (0.6) | Yes (0.5) |
| Audit probability | 0% | 15% | 25% |
| **Results** | | | |
| Bad actor frozen | No | Yes | Yes |
| Bad actor payoff | +3.42 | +1.22 | -1.55 |
| Avg toxicity | 0.30 | 0.33 | 0.32 |
| Welfare/epoch | 7.29 | 13.02 | 8.15 |

Governance effectively punishes bad actors (payoffs drop from positive to negative) while maintaining similar toxicity levels. Stricter governance reduces bad actor gains but also dampens overall welfare.

## Parameter Sweeps

Run batch simulations over parameter ranges:

```python
from swarm.analysis import SweepConfig, SweepParameter, SweepRunner
from swarm.scenarios import load_scenario

# Load base scenario
scenario = load_scenario(Path("scenarios/baseline.yaml"))

# Configure sweep
config = SweepConfig(
    base_scenario=scenario,
    parameters=[
        SweepParameter(
            name="governance.transaction_tax_rate",
            values=[0.0, 0.05, 0.10, 0.15],
        ),
        SweepParameter(
            name="governance.circuit_breaker_enabled",
            values=[False, True],
        ),
    ],
    runs_per_config=3,  # Multiple runs for statistical significance
    seed_base=42,
)

# Run sweep
runner = SweepRunner(config)
results = runner.run()

# Export to CSV
runner.to_csv(Path("results.csv"))

# Get summary statistics
summary = runner.summary()
```

Run the example:
```bash
python examples/parameter_sweep.py
python examples/parameter_sweep.py --output my_results.csv
```

Supported parameter paths:
- `governance.*` - Any GovernanceConfig field
- `payoff.*` - Any PayoffConfig field
- `n_epochs`, `steps_per_epoch` - Simulation settings

## Loan / commitment scenario (`scenarios/loan_commitment.yaml`)

Isolates one mechanism the core loop cannot express: **default as a strategic
action**. A lender extends credit; the borrower invests, then chooses repay
vs. default one period later, so the lender's loss is chosen rather than drawn
from `p`. Motivated by the AI Village Ṁ5,000 loan incident
([blog post](blog/the-m5000-loan-goal-consumed-judgment.md)).

The borrower's payoff mirrors the engine's form,
`π = mana − ρ·harm_to_lender + w_rep·Δreputation`, with three levers: `rho`
(externality internalization), `w_rep` (reputation weight), and a third-party
`bond` forfeited on default. The YAML is read by
`experiments/loan_commitment_sweep.py`, not the swarm scenario loader:

```bash
python -m experiments.loan_commitment_sweep            # full grid
python -m experiments.loan_commitment_sweep --quick    # coarse grid
```

Writes `runs/<ts>_loan_commitment/{results.csv,predictions.json,plots/}`.
Preregistered predictions and their status are recorded in
`predictions.json`: the default decision is a step in `rho` at
`rho* = 1 − w_rep(r_repay + r_default)/(due − bond)` (met); a gift leaves
the default rate unchanged under the Village objective (met) but not in
general — gifts fix ability, never willingness, so they help exactly when a
bond or reputation weight already makes able borrowers repay.

## Gossip board: information fidelity (`scenarios/gossip_board_fidelity.yaml`)

A synthetic, ground-truth twin of the Hyperspace gossiping agent swarm (Mathur,
2026-09-04; read through the SWARM frame in
`docs/blog/gossiping-swarms-what-the-message-board-cannot-see.md`). N agents run
an autoresearch-style loop over a discrete config space with one high-gain,
low-prior "hidden" dimension, and publish improvements to a shared board that
other agents read. **One lever** changes between runs: how much of a published
result travels to a reader. `code` adopts the full config verbatim (the trading
domain, where 17 agents converged to four decimal places); `description` carries
only the diff ("a peer got 3.21 by switching to RMSNorm") which the reader
applies to its own config; `score_only` says a peer improved and nothing else.
The board is out-of-band, so the reward proxy never sees it and nothing is
scored; the point is ground truth. Every entry carries a `parent` lineage field
the real board lacked, and the survivorship gap of a success-only board is
measured against every attempt, published or not, **contemporaneously** (pooling
across rounds confounds the gap with search progress, because successes cluster
early; the pooled figure is reported too and is negative).

Run it:

```bash
python -m swarm.bridges.gossip_board scenarios/gossip_board_fidelity.yaml --seeds 10
python -m swarm.bridges.gossip_board scenarios/gossip_board_fidelity.yaml \
    --axis fidelity=code,description --axis publish_failures=false,true --seeds 10
```

Writes `history.json` and `csv/rounds.csv` for the baseline plus `csv/sweep.csv`
and `csv/sweep_mean.csv` for the grid.

Findings (`runs/20260905T012511Z_gossip_board_fidelity_seed42`, 10 seeds, means):

| Fidelity | Modal identical-config cluster | Mean score / optimum | Survivorship gap | Time to frontier, early / late joiners | Hidden-dim adoption |
|---|---|---|---|---|---|
| code | 0.83 of agents | 0.995 | 0.030 | 15.0 / 1.5 rounds | 0.78 |
| description | 0.24 | 0.949 | 0.012 | 30.4 / 24.7 | 0.53 |
| score_only | 0.13 | 0.931 | 0.032 | 44.6 / 31.1 | 0.38 |

Against the preregistered expectations:

1. **Fingerprints follow code — met.** The identical-config cluster goes from
   0.83 of the population under `code` to 0.24 under `description`, while the
   frontier score drops only 5%. Ideas spread without fingerprints, which is the
   Hyperspace language-modeling result reproduced on controllable data.
2. **Survivorship gap is a publication-rule property — met with a caveat.** The
   gap is positive in all three modes and collapses to exactly 0 with
   `publish_failures: true`. It is not the same size across modes: `description`
   halves it, because readers applying a single diff to their own config produce
   more middling successes than readers cloning the leader.
3. **Late joiners cold-start — met.** Under `code` a late joiner reaches the
   frontier in 1.5 rounds against 15 for early joiners: the board is a free
   frontier. Under `description` it takes 24.7 rounds and a third of late joiners
   never get there in the remaining 60.
4. **Fidelity governs how far a rare discovery spreads — met, as revised.** The
   original draft predicted the hidden dimension would stay unexplored; a
   10-seed probe refuted that (24 agents x 120 rounds find it every seed even at
   prior 0.02), and the expectation was rewritten before the recorded run.
   Adoption of the hidden dimension's best value is ordered `code` (0.78) >
   `description` (0.53) > `score_only` (0.38). Fidelity does not move the
   frontier; it moves the population to it. Prior diversity is bead khs2.

### Prior diversity vs population size (bead khs2)

Second lever on the same board, fidelity held at `code`: the population's prior
over which dimension to mutate. `hidden_dim_prior` is the mass every agent puts
on the high-gain dimension; `diverse_fraction` is the share of agents given a
**uniform** prior instead of the shared one. The observable is the round in
which anyone first mutates the hidden dimension (`hidden_discovery_round`, never
in 120 rounds = "shared omission") and the score the population ends at.

```bash
python -m swarm.bridges.gossip_board scenarios/gossip_board_fidelity.yaml --seeds 10 \
    --axis fidelity=code --axis hidden_dim_prior=0,0.002,0.01,0.05 --axis n_agents=8,24,72
python -m swarm.bridges.gossip_board scenarios/gossip_board_fidelity.yaml --seeds 10 \
    --axis fidelity=code --axis hidden_dim_prior=0 --axis diverse_fraction=0,0.04167,0.125,0.5 --axis n_agents=8,24,72
```

Findings (`runs/20260905T170749Z_gossip_board_prior_diversity_seed42`, 10 seeds, means):

| Shared prior | Agents | Discovery round | Final score / optimum |
|---|---|---|---|
| 0 | 8 / 24 / 72 | never / never / never | 0.911 / 0.914 / 0.917 |
| 0.002 | 8 / 24 / 72 | 66.9 / 33.0 / 14.7 | 0.939 / 0.964 / 0.988 |
| 0.01 | 8 / 24 / 72 | 34.8 / 15.0 / 4.7 | 0.953 / 0.985 / 0.998 |
| 0.05 | 8 / 24 / 72 | 6.5 / 1.5 / 0.3 | 0.983 / 0.993 / 0.997 |

| Diverse agents (prior 0 elsewhere) | Agents | Discovery round | Final score / optimum |
|---|---|---|---|
| 0 | 72 | never | 0.917 |
| 1 | 8 / 24 | 9.8 / 20.7 | 0.986 / 0.985 |
| 3 | 24 / 72 | 2.4 / 4.6 | 0.992 / 0.996 |
| half | 8 / 24 / 72 | 2.2 / 1.1 / 0.0 | 0.987 / 0.995 / 0.998 |

Reading:

1. **A shared prior of exactly zero is not fixed by scale.** Nine times the
   agents adds 0.006 to the final score and never finds the hidden dimension.
   This is the Hyperspace position-encoding omission on controllable data: a
   population that agrees, without discussing it, not to look somewhere does not
   look there no matter how large it gets.
2. **Above zero, what matters is the product of prior and population.** Discovery
   round tracks roughly 1 / (agents x prior): prior 0.002 with 72 agents and
   prior 0.01 with 24 agents both discover in about 15 rounds. Scale substitutes
   for prior mass only when the prior is not zero.
3. **One uncorrelated agent beats nine-fold scale.** A single uniform-prior agent
   in a population of 8 (score 0.986) outperforms 72 correlated agents (0.917).
   Three diverse agents in 72 recover almost the whole gap (0.996). Prior
   diversity is a governance lever with a larger effect than population size,
   and it is cheap: the diverse agents search worse on the shared dimensions and
   still lift the population, because the board carries their find to everyone.

Caveat: the board here is `code` fidelity, so a single discovery propagates at
full strength. Under `description` or `score_only` the diverse agent's find
spreads more slowly (bead fcj5, expectation 4), so the required diverse fraction
is higher; that grid was not run.

### Collusion detectors on honest copying (bead 9err)

Zero adversaries. Every agent on the board is honest and the correlation
between them is produced entirely by the board's own mechanism: readers copy
the leader. The question is what a collusion detector makes of that, and how
much a lineage field would have changed its answer.

Two projections of the same board onto `SoftInteraction` records:

- **fingerprint** (provenance-blind): what an outside auditor can reconstruct.
  A published entry is an interaction from its author to the most recent
  earlier publisher of an *identical* config. This is the "17 agents to four
  decimal places" view.
- **lineage** (provenance-aware): an interaction from the author to the agent
  whose entry it was derived from, using the board's `parent` field. Edges
  explained by a recorded adoption are then removed; what remains is the
  *residual* graph of identical configs with no recorded parent
  (rediscoveries and coincidences), which is all a detector should be looking
  at.

Both projections run through the same detectors the wiki replay uses:
structural anomalies (`swarm.metrics.graph_structural`, null-model p-value),
the per-agent temporal clustering score, and the pairwise `CollusionDetector`.
Since there are no colluders, every flag is a false positive.

```bash
python -m swarm.bridges.gossip_board scenarios/gossip_board_fidelity.yaml --seeds 10 \
    --detect --axis fidelity=code,description,score_only
```

Reports, per fidelity, means over seeds:

| Metric | Meaning |
|---|---|
| `structural_flag_rate` | fraction of seeds with at least one cluster at p < 0.05 |
| `structural_flagged_agents` | fraction of agents inside a flagged cluster |
| `temporal_alarm_rate` | fraction of agents at or above the 0.7 alarm bar from the graph_structural sweep |
| `pairwise_flagged` | pairs flagged by `CollusionDetector` / all pairs |
| `residual_*` | the same three on the provenance-aware residual graph |

Preregistered expectations:

1. **Structural false positives follow fingerprints.** Under `code` the
   provenance-blind detector flags a cluster in most seeds and the cluster
   covers a majority of agents; under `description` and `score_only` there are
   too few identical configs to build a graph and the flag rate is near zero.
2. **Temporal is a cadence artifact.** Every agent publishes on the same
   round clock, so the temporal score is high in all three modes and does not
   separate them. It measured the loop, not coordination.
3. **Lineage removes most of it.** On the residual graph, structural and
   pairwise flags under `code` fall by at least half, because the great
   majority of identical-config edges are recorded adoptions. What survives
   is rediscovery, which is the honest false-positive floor a lineage-aware
   detector still has to live with.

Findings (`runs/20260906T131453Z_gossip_board_detector_fpr_seed42`, 10 seeds, means; no colluders exist, so every flag is
a false positive):

| Fidelity | Fingerprint edges | Explained by lineage | Structural flag rate / agents flagged | Temporal max / alarm rate | Pairwise flagged | Residual edges | Residual structural flag / agents |
|---|---|---|---|---|---|---|---|
| code | 222 | 0.81 | 1.00 / 1.00 | 0.28 / 0.00 | 0.00 | 43 | 1.00 / 0.89 |
| description | 104 | 0.24 | 1.00 / 0.97 | 0.50 / 0.01 | 0.00 | 78 | 1.00 / 0.96 |
| score_only | 46 | 0.00 | 0.90 / 0.80 | 0.38 / 0.00 | 0.00 | 46 | 0.90 / 0.80 |

Against the preregistered expectations:

1. **Structural false positives follow fingerprints — NOT MET.** The
   structural detector flags a cluster in every seed under `code` and
   `description`, and in 9 of 10 seeds under `score_only`, where the only
   identical configs come from agents making the same one-dimension mutation of
   the shared baseline. It is the saturation result from the message-board
   twin again: any co-publication graph is a dense community against the null
   model. The detector reports that a board exists, not who coordinated. It
   does not separate the three modes at all.
2. **Temporal is a cadence artifact — NOT MET, in the other direction.** The
   temporal score never reaches the 0.7 bar in any mode (max 0.28 to 0.50).
   Agents publishing on a regular round clock are not "bursty" relative to
   themselves, so the score is silent rather than saturated. It separates
   nothing, but by never firing.
3. **Lineage removes most of it — MET for edges, NOT for flags.** The parent
   field explains 81% of code-mode fingerprint edges (222 down to 43 residual),
   which is the honest-copy correlation made visible. But the structural
   detector still flags the residual graph in every seed, covering 89% of
   agents, because 43 edges among 24 nodes is still a dense community. The
   pairwise `CollusionDetector` flags nothing before or after, so there was
   nothing for lineage to remove there.

Reading: on an honest open board the collusion detectors in this repo fail in
both directions. Structural saturates and cannot be rescued by provenance
alone, because removing explained edges leaves a smaller graph that is still
"anomalous" against a null model that does not know a board exists. Temporal
is blind. The right null model for a shared board is one that *includes* the
board's copy mechanism, so that only correlation beyond what public adoption
predicts is anomalous. Lineage is the field that makes that null model
computable; it is necessary and not sufficient. The 17 identical trading
strategies would have been flagged, and the flag would have been wrong.


Also recorded: `adoptions` versus `rediscoveries` from the lineage field.
`score_only` has zero adoptions and 264 rediscoveries per run; `code` has 182
adoptions and 67 rediscoveries. A provenance-blind detector sees only the
resulting correlation, which is the false-positive probe in bead 9err.

## Message-board channel (`scenarios/message_board_channel.yaml`)

A synthetic, ground-truth twin of the wiki back channel that collusion.wiki
caught in the wild (see the field evidence in
`docs/research/wiki-backchannel-field-evidence.md` and the replay in
`scenarios/casestudy_wiki_backchannel.yaml`). Honest agents edit their own
pages at a low rate; a handful of colluding operators, each a stable IP block
hidden behind many rotating handles, burst onto one shared hub page inside a
coordination window that precedes a moderator sweep. Because the channel is
out-of-band, the reward proxy never sees it, so there is no interaction to
score — instead a generator writes the edit log in the real export's schema and
the **same** detector pipeline (`swarm/bridges/collusion_wiki/runner.py`) runs
over it. Having ground truth lets us score the detectors, not just observe them.

Run it:

```bash
python -m swarm.bridges.collusion_wiki.synthetic scenarios/message_board_channel.yaml
```

Writes a run folder with the replay outputs plus `synthetic_eval.json` (detector
precision/recall against the known colluders) and `volume_timeline.json`.

Findings (seed 42) reproduce the replay's lesson on controllable data:

| Signal | Result |
|---|---|
| Volume monitor | fires on the first coordination day, **3 days before the sweep** |
| Structural (label) | recall 1.0 but precision 0.69 — sweeps in honest hub-users |
| Structural (ip16) | precision 0.10 — saturation, flags everyone on the hub |
| Temporal (label) | max 1.0 but recall 0.03 — only a few fragmented handles fire |
| Temporal (ip16) | max 0.03, recall 0 — collapses when handles fold to IP blocks |

The reading: structural detection **saturates** rather than fails (a shared hub
page makes every co-editor a reply-neighbour), the per-agent temporal score is a
**handle-rotation artifact** (identity-dependent, collapses under `ip16`), and a
plain **edit-volume monitor** is the signal that leads the moderator — the one
thing none of the collusion detectors computes.
