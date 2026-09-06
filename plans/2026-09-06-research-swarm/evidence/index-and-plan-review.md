# Independent wave-one review

Reviewer: separate Reproducibility Sheriff instance. Date: 2026-09-06.
Scope: read-only review of `docs/research/index.md` in `/tmp/swarm-research-wave1`, HEAD `7dcb672b28872373ba991f0dd3680ce6a3ad5c56`, plus the local 90-day plan. No repository files, tracker records, protocols, datasets, or coordination messages changed. This report is the only written artifact.

## Verdict

PASS for the two research-index additions. No blocking finding in the plan's frozen-input or ownership instructions. This is a documentation/hygiene review, not independent reproduction of the linked experimental results or approval to claim completion of their open research tasks.

## Executed checks

- `python scripts/build_kb_graph.py --check`: exit 0; 432 nodes, 1481 edges, 149 orphans, 0 stale references; no new orphans or stale references versus baseline.
- `git diff --check`: exit 0, no output.
- Read both linked studies in full and checked their summaries against the exact diff.
- Read the full 90-day plan and cross-checked its frozen-data statements against the local wiki, calibration, and adaptive preregistrations.

## Summary fidelity

`Content-Free Collusion Discriminators` correctly frames the probe as synthetic copying detection. The linked note defines output agreement and re-derivation timing, reports ten-seed copying performance, and explicitly warns that these observables cannot establish collusion. The new summary preserves that limitation and introduces no numerical claim.

`Wiki Behavior Monte Carlo Results` correctly states that the runs are paired and synthetic, cover moderation effects and permission-blind detection, and cannot establish historical migration. The study's authorization control uses identical behavioral parameters with different permission labels. The summary makes no claim that the still-open manifest/contrast review is complete.

## Plan review

The plan explicitly preserves existing frozen protocols unless prospectively amended, rejects retuning on confirmation data, retains the registered five-seed adaptive comparison, and requires rubric provenance rather than silently promoting later rubrics. These match the reviewed prerequisites: the wiki protocol freezes contrasts/seeds/thresholds before confirmation; calibration locks Arm B rubric v1 and requires a frozen downstream schema; the adaptive protocol fixes five seeds and its external-quality threshold before runs.

Ownership is bounded: the plan prohibits takeover of in-progress tasks without explicit handoff, labels milestone owners as role slots rather than dispatched assignments, requires canonical atomic claims and isolated worktrees, and requires independent review. Capacity expansion and paid experiments are conditional; dates and positive research findings are not guaranteed. No dangerous or unsupported operational commitment was found within this narrow review.

Execution cautions, not blockers:

- The plan's heading currently says execution has not launched. Retain that as a dated planning-session statement or add a separate execution-status artifact as work proceeds; do not use it as current status after execution starts.
- Existing synthetic confirmation folders are not proof of pre-run freeze or successful manifest review. Those gates remain open until their provenance and selected contrasts are independently checked.
- Tracker ownership, current run inventory, coordination-store identity, and capacity must still be verified at claim time. The plan explicitly acknowledges these unresolved operational facts.

No remote CI was run. No experimental artifacts or historical claims were independently reproduced in this review.
