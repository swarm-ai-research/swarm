# False-positive rate of collusion detectors on honest convergence (9err)

**Question.** The [wiki incident](wiki-backchannel-field-evidence.md) posed the
*convergence–collusion identifiability problem*: emergent convergence (independent
agents that discover the same board) produces the same observable signature as
designed collusion. So how often do our collusion detectors fire on a swarm that is
*honestly* converging, and does ground-truth provenance help? Bead `9err`.

**Setup.** The [`gossip_board`](../../swarm/bridges/gossip_board) model is an honest
code-sharing swarm with **no adversaries** — 24 agents searching a discrete config
space, publishing improvements others copy. Every agent is honest, so *any* pair or
group a detector flags is a false positive. Ground truth is the `Entry.parent`
lineage field the real wiki board lacked. Code fidelity, 30 rounds, 10 seeds;
`runs/…_probe_9err_fpr_convergence_seed0/`.

## Results

| Detector | FPR on honest convergence |
|---|---|
| Naive identical-output (share a config ⇒ suspect) | **54.4%** of all agent pairs |
| …same, but provenance-aware (lineage explains the match) | **17.1%** |
| Composite `CollusionDetector` (pair + group scores) | **0%** |
| Volume-burst (`hoer`) at the operational 10× threshold | **0/10 seeds** |

- **Honest convergence produces a huge identical-output cluster.** A naive detector
  that reads "same config ⇒ coordinated" flags 150 of 276 pairs — the "17 agents to
  four decimals" fingerprint at population scale. On the honest board this is 54%
  false positives.
- **Lineage is the fix, and a partial one.** Using `parent` to discount matches
  explained by copying drops the flagged pairs from 150 to 47 (54% → 17%). The 17%
  residual is *independent rediscovery* — agents that reached the same optimum by
  separate search, which no provenance field can explain away because there is no
  copy to point to. This motivates **lineage as a required board field**: it removes
  the copying-driven false positives, which are the bulk.
- **The composite detector is already robust.** `CollusionDetector` flags nothing
  (0%), because its score needs benefit-correlation and directional quality
  asymmetry, and honest convergence has neither — everyone benefits symmetrically and
  no one dumps low-`p` work on outsiders. The identifiability danger is specific to
  *co-occurrence / identical-output* detectors, not to the quality-asymmetry ones.
- **The volume-burst signal separates by magnitude, not kind.** Honest convergence
  does produce publish bursts, but mild ones (mean peak ratio **2.2×**); at the
  operational **10× threshold** it never fires, while the incident burst was
  **2603×** — roughly 1000× sharper. So the identifiability problem is real at the
  level of *kind* (both produce bursts), but here a well-chosen threshold discriminates
  by *magnitude*. The caveat is that this margin is regime-dependent: a slow-drip
  collusion or a virally-adopted honest tool could invert it.

## Takeaway

The identifiability worry is precise, not fatal. It bites exactly one detector class —
naive co-occurrence / identical-output — and lineage/provenance is the structural
answer for the copying-driven part (54% → 17%). The quality-asymmetry composite
detector and a well-thresholded volume-burst are already robust to honest convergence.
The general lesson: **do not screen on co-location alone; screen on co-location that
is not explained by shared provenance, and confirm with a quality-asymmetry signal.**

**Content-free follow-up.** The two content-free signals this note leaves open —
output correlation and re-derivation infeasibility — are measured in
[Content-free discriminators on an overlap-routing board](content-free-discriminators.md)
(bead `vv3j.2`).

**Companion probe.** The same bead was independently run against the graph-structural
and temporal detectors (`docs/scenarios.md`, "Collusion detectors on honest copying";
commit 2d028bda, `runs/20260906T131453Z_gossip_board_detector_fpr_seed42`). Structural
flags a cluster in every seed in all three fidelity modes and still flags the
lineage-residual graph; temporal never reaches its 0.7 bar. So the "screen on
co-location not explained by provenance" rule above holds for pair rules, but a
community detector needs a null model that includes the board's copy mechanism, not
just edge removal. The content-free detectors this motivates (output correlation, re-derivation infeasibility) and their precision/recall on the pi02 overlap board are written up in [content-free-discriminators.md](content-free-discriminators.md) (bead `vv3j.2`).
