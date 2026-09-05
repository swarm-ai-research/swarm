# /red_team

Run a red-team evaluation (adaptive/adversarial stress testing) and summarize failure modes.

## Usage

`/red_team [mode]`

Modes:
- `quick` (default): minimal set of attacks and one governance config
- `full`: full attack library (may be slow)

## Behavior

1) Create `runs/<YYYYMMDD-HHMMSS>_redteam/`.
2) Run the red-team evaluator from `swarm.redteam`:
- Instantiate `RedTeamEvaluator(governance_config=...)`
- Run quick/full evaluation
3) Write:
- `<run_dir>/report.json` (machine-readable)
- `<run_dir>/report.txt` (human summary)
4) Print a short summary suitable for an issue:
- Robustness score + grade
- Top vulnerabilities (severity + affected lever)
- Most effective attack vectors
- Recommended mitigations / next experiments


## Attack library

`AttackLibrary.get_all_attacks()` in `swarm/redteam/attacks.py`. Each attack names the
governance levers it targets; the evaluator's outcome model responds to the matching
`governance_config` keys.

| attack_id | Category | Responds to |
|---|---|---|
| `reputation_farming` | exploitation | `audit_enabled` |
| `collusion_ring` | coordination | `collusion_detection_enabled` |
| `threshold_dancing` | evasion | `circuit_breaker_enabled` |
| `sybil_attack` | coordination | `sybil_detection_enabled`, `collusion_detection_enabled` |
| `timing_attack`, `information_laundering`, `resource_drain`, `governance_gaming` | various | general defenses only |
| `out_of_band_channel` | coordination | `egress_allowlist_enabled` (prevents), `volume_monitor_enabled` (detects); **not** `collusion_detection_enabled` |

`out_of_band_channel` is the field exhibit from the collusion.wiki incident
(`docs/research/wiki-backchannel-field-evidence.md`, bead 7y0p): agents coordinate on a
third-party substrate the governed observables never see, so task proxies stay green;
an ordered cleanup sweep is evaded by inferring its order (the `ZZZ` pages). The replay
(`docs/research/collusion-wiki-replay.md`) showed pairwise/structural collusion detection
saturating on the hub page while aggregate edit volume led the human moderator by three
days, which is why that attack ignores `collusion_detection_enabled`.
