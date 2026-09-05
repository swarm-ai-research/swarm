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
