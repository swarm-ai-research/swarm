---
description: "Pointers to recent runs — updated each session, never stores raw data"
---

# Recent Run Pointers

## Latest runs (by date)

| Date | Run ID | Type | Key Finding |
|------|--------|------|-------------|
| 2026-07-18 | 20260703-012553_multiseed_miroshark | preregistered_batch | 23 clean reps: libel<redteam<0 ordering NOT real (all H rejected); redteam qg +0.034 significantly POSITIVE; archived swarm-artifacts research/miroshark-amplification/multiseed-20260703 |
| 2026-03-01 | 20260228-185017_mesa_adversarial_learners_study | mesa_sweep | 300 runs; governance resists gaming at rho>=0.75 (<4% adv acceptance); tp-rework corr detects adversaries |
| 2026-02-27 | 20260226-211430_mesa_game_structures_study | mesa_sweep | governance generalizes across PD/SH/HD; learning +132-159% at rho=1.0 |
| 2026-02-27 | cross_study_mesa_comparison | comparison | 455 runs across 3 studies; toxicity converges ~0.147 regardless of game |
| 2026-02-26 | 20260226-201109_mesa_adaptive_agents_study | mesa_sweep | learning agents recover +137% welfare at rho=1.0, Pareto-dominate adaptive |
| 2026-02-25 | 20260224-220829_mesa_governance_study | mesa_sweep | rho sweet spot [0.3,0.7]: toxicity ↓34% (sig), welfare cost non-sig |
| 2026-02-22 | evo_game_prisoners_seed999 | evo_game | Sharp phase transition pattern, final welfare 34.24 |
| 2026-02-22 | evo_game_prisoners_seed314 | evo_game | Most volatile — welfare peaked then fell to 14.99 |
| 2026-02-22 | evo_game_prisoners_seed777 | evo_game | Gradual ramp, epoch 3 acceptance crash (37.5%) |
| 2026-02-22 | evo_game_prisoners_seed123 | evo_game | Gradual ramp, no early crash, final welfare 25.62 |

## Run index location

- Full catalog: `/Users/raelisavitt/swarm-artifacts/run-index.yaml` (117 runs)
- Local runs: `runs/` in main repo (gitignored, UUID-based)
- Tracked runs: `swarm-artifacts/runs/` (YYYYMMDD-HHMMSS_slug format)

## How to query

```bash
# Search by tag
grep -A5 "collusion" /Users/raelisavitt/swarm-artifacts/run-index.yaml

# Search by date range
grep "date: '2026-02-2" /Users/raelisavitt/swarm-artifacts/run-index.yaml

# Read a specific run's metadata
cat /Users/raelisavitt/swarm-artifacts/runs/<run_id>/run.yaml
```

## Active claims

Pointers only — source of truth is `swarm-artifacts/vault/claims/`:

- claim-collusion-detection-reduces-ring-damage-75pct
- claim-sybil-attacks-resist-full-governance-stack
- claim-contract-screening-achieves-perfect-type-separation
- claim-trust-boundaries-modify-but-never-deny-handoffs
- claim-full-governance-stack-prevents-most-attack-types
- claim-reputation-mediated-cooperation-sustains-in-dominant-defect (NEW — 5-seed evo game study)
