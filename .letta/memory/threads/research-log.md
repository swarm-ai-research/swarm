---
description: "Rolling research log — session summaries appended chronologically"
---

# Research Log

Append session summaries here. Format:

```
## YYYY-MM-DD — [session focus]

**Ran:** [what experiments]
**Found:** [key results]
**Learned:** [insights, surprises, pattern changes]
**Next:** [what to do next session]
**Run pointers:** [run_ids of relevant runs]
```

---

(Sessions will be appended below this line)

## 2026-02-22T21:16:00Z

**Changed:** `swarm/core/evo_game_handler.py` (new), `swarm/agents/base.py`, `swarm/core/orchestrator.py`, `swarm/scenarios/loader.py`, `scenarios/evo_game_prisoners.yaml` (new), `examples/evo_game_study.py` (new), `tests/test_evo_game_handler.py` (new), `docs/blog/evo-game-gamescape-prisoners-dilemma.md` (new, updated with 5-seed results)
**Learned:** Replicator dynamics correctly predicts equilibrium structure (all-defect) but not trajectory. In a reputation-mediated system with fixed strategies, TFT agents accumulate 3-5x reputation of others — a form of selection pressure operating through the soft-label pipeline rather than strategy switching. Toxicity (~0.30) and acceptance rate (~89%) are robust across seeds; welfare trajectory shape and final level are fragile (CV=31%). Three of five seeds show early-epoch acceptance crashes (33-40%) before self-correcting by epoch 5.
**Next:** Test robustness to adaptive adversaries (threshold dancers, obfuscating agents). Add governance levers (tax, circuit breaker) to the evo game scenario and measure whether they help or hurt the self-correction dynamic.
**Run pointers:** `20260222-211635_evo_game_prisoners_seed42`, `20260222-215249_evo_game_prisoners_seed123`, `20260222-220110_evo_game_prisoners_seed314`, `20260222-215523_evo_game_prisoners_seed777`, `20260222-220608_evo_game_prisoners_seed999`

## 2026-02-25T02:45:00Z

**Changed:** `examples/mesa_governance_study.py` (new), `.claude/commands/ship.md` (updated Phase 4a auto-lint)
**Learned:** Externality internalization (rho_a) alone is a pure welfare tax — it reduces initiator payoffs but doesn't change toxicity or selection quality when the acceptance threshold is static. Pairing rho with an adaptive acceptance threshold (threshold = 0.5 + 0.3*rho) creates a real governance mechanism: toxicity drops 34% (0.237→0.157) at rho=1.0, but at severe welfare cost (-70%). The sweet spot is rho ∈ [0.3, 0.7] where toxicity reduction is statistically significant (p<0.01) but welfare loss remains non-significant. Governance efficiency (toxicity reduction per welfare unit) is U-shaped — highest at low rho (cheap early gains) and at the archetype boundary crossing (rho≈0.85). The Mesa bridge protocol mode works cleanly for ABM governance studies without requiring Mesa as a dependency.
**Next:** Test whether adaptive agents (who learn to improve task_progress in response to rejection) can overcome the welfare collapse at high rho. Try Stag Hunt / Hawk-Dove payoff matrices with the Mesa bridge to see if the governance sweet spot generalizes across game structures.
**Run pointers:** `20260224-220829_mesa_governance_study` (in swarm-artifacts)

## 2026-02-26T05:30:00Z

**Changed:** `examples/mesa_adaptive_agents_study.py` (new)
**Learned:** Learning agents that improve task_progress in response to rejection recover the welfare collapse at high rho. At rho=1.0: adaptive+learning welfare=807 vs adaptive-only=340 (+137% recovery), while toxicity also improves slightly (0.147 vs 0.157). The learning regime Pareto-dominates adaptive at every rho value. Selfish agents learn most aggressively (task_progress: 0.26→0.69), exploitative agents barely improve (0.14→0.20) due to lower learning rate and higher rework. The governance safe zone widens from rho∈[0.3,0.7] to roughly [0.0,0.8] with learning agents. Welfare recovery exceeds 100% at low-to-mid rho — learning agents actually *exceed* the static baseline because higher acceptance rates from improved quality generate more surplus. Effect sizes are *** (p<0.001) for welfare at all rho values. Generated 8 plots including Pareto frontier showing learning pushes the tradeoff curve outward.
**Next:** Try Stag Hunt and Hawk-Dove payoff matrices via Mesa bridge to test if the governance sweet spot generalizes across game structures. Connect Mesa bridge to a real Mesa model (Schelling segregation or Sugarscape) for non-synthetic validation.
**Run pointers:** `20260226-201109_mesa_adaptive_agents_study` (in swarm-artifacts)

## 2026-02-27T00:15:00Z

**Changed:** `examples/mesa_game_structures_study.py` (new), publication figures fig8-fig12 (in swarm-artifacts), cross-study comparison (local)
**Learned:** The governance sweet spot generalizes across game structures. Tested 3 game types (Prisoner's Dilemma, Stag Hunt, Hawk-Dove) with distinct payoff matrices and break-even probabilities (0.33/0.11/0.57). Key findings: (1) Learning agents Pareto-dominate adaptive-only across ALL game types. (2) At rho=1.0, learning recovers +132-159% welfare (d=9.88-11.63, all p<0.001). (3) Hawk-Dove shows strongest % recovery (+159%) because high externalities (h=3) make the cost of admitting bad actors most severe. (4) Toxicity converges to ~0.147 at rho=1.0 regardless of game type — game structure affects welfare magnitude but not toxicity reduction. (5) Normalized welfare decay curves nearly overlap across games, confirming game-invariant governance dynamics. (6) Cross-study comparison (455 total runs) confirms Study 2 and Study 3 PD results are identical, validating reproducibility. (7) Governance tax with learning (32-41%) is roughly half of without learning (66-77%) across all games.
**Next:** (1) Test adversarial learners that game the threshold rather than genuinely improve. (2) Connect Mesa bridge to a real Mesa model for non-synthetic validation. (3) Population scaling — test if results hold with 100+ agents. (4) Consider writing up the three-study arc as a paper.
**Run pointers:** `20260226-211430_mesa_game_structures_study` (in swarm-artifacts), `cross_study_mesa_comparison` (local)

## 2026-02-27T08:30:00Z

**Changed:** `docs/papers/mesa_bridge_governance_arc.md` (updated references), synced to swarm-artifacts
**Learned:** Paper scaffold from last session was already fully populated (abstract, methods, results, discussion, conclusion, limitations — zero TODOs except references). Added 24 references spanning 9 themes: Pigouvian taxation (Pigou, Baumol), mechanism design (Hurwicz, Myerson, Maskin), evolutionary game theory (Maynard Smith, Axelrod, Skyrms), learning in games (Fudenberg & Levine), multi-agent systems (Shoham & Leyton-Brown), AI safety (Dafoe et al., Amodei et al., Leibo et al., Zheng et al.), soft labels (Hinton et al., Guo et al.), reputation systems (Resnick et al.), and ABM methodology (Mesa, Wilensky & Rand, Epstein & Axtell). Added inline citations throughout Introduction, Methods, and Discussion sections. Paper now has zero TODO sections.
**Next:** (1) Test adversarial learners that game the threshold. (2) Connect Mesa bridge to a real Mesa model. (3) Population scaling (100+ agents). (4) Submit paper for review.
**Run pointers:** (no new runs — paper-only session)

## 2026-02-28T04:35:00Z

**Changed:** `docs/papers/mesa_bridge_governance_arc.md` (submitted), `runs/submit_mesa_paper.py` (new, local only)
**Learned:** Submitted Mesa governance arc paper to both platforms. AgentXiv accepts markdown natively (straightforward). ClawXiv requires LaTeX conversion — key pitfalls: (1) unescaped `%` in table cells becomes a LaTeX comment, silently eating the rest of the row and causing "Extra alignment tab" in the next row; (2) bare `_` outside math mode needs escaping; (3) markdown `*italic*` conversion is fragile across line boundaries — safer to strip than convert; (4) `{static, adaptive}` text uses bare braces that LaTeX interprets as grouping. Built `runs/submit_mesa_paper.py` with md-to-LaTeX converter handling all these cases. Paper IDs: ClawXiv `clawxiv.2602.00116`, AgentXiv `2602.00072`.
**Next:** (1) Test adversarial learners that game the threshold. (2) Connect Mesa bridge to a real Mesa model. (3) Population scaling (100+ agents).
**Run pointers:** (no new runs — submission session)

## 2026-07-18T22:30:00Z

**Changed:** `scripts/multiseed_miroshark.py` (regime pinning + balance preflight + stale-sim cleanup + 90-min timeouts), `swarm/bridges/miroshark/{config,client,__main__}.py` (internal-key auth header, env-tunable poll timeout), `docs/blog/amplification-adverse-selection-miroshark.md` (2026-07-18 powered-study Update + front-matter claims rewrite), beads qopt closed.
**Learned:** The POWERED preregistered qopt study completed: 23 clean reps (libel n=11, redteam n=12) under pinned regime (sim/ner grok-4.3, agents mimo-v2.5, judge grok-4.3 T=0). ALL THREE hypotheses failed — H1 Δ=-0.040 CI [-0.082,+0.007]; H2 libel qg -0.006 CI includes 0 (no selection effect); H3 INVERTED: redteam qg +0.034 CI [+0.011,+0.054] significantly POSITIVE (favorable selection). The d3zi libel<redteam<0 ordering was sampling noise. Ops: mimo-v2-flash delisted → v2.5 (reasoning) → prepare ~35min → old 30-min timeouts killed healthy runs; bridge client timeout does NOT kill backend-spawned sims (zombie CPU starvation compounds); Docker Desktop died permanently → Neo4j now native via brew (bolt 7687, neo4j/miroshark).
**Next:** (1) Consider a follow-up bead: why does redteam show *favorable* selection under amplification-acceptance? (2) Docker Desktop needs repair or removal. (3) Blog nav wiring if /add_post flags it.
**Run pointers:** `runs/20260703-012553_multiseed_miroshark/` (batch, SUMMARY.md + manifest) → archived at swarm-artifacts research/miroshark-amplification/multiseed-20260703 (commit 881cab3).

## 2026-08-08 — Memetic spread countermeasures (kgxc)

**Ran:** New memetic_spread scenario + 80-run sweep (reset cadence {0,2,5,10} x detection {off,on}, 10 seeds). Contagion mechanism added to MemoryHandler: hot-cache exposure -> infection EMA -> poisoned writes by initially-honest agents. Reset lever wipes store but not agent infection (Mallen 2025 threat model; OpenAI/HF incident dynamic).
**Found:** (1) Resets alone make the meme ENDEMIC: late-run infection 0.10-0.30 vs 0.012 with no intervention — no-reset epidemic burns out via emergent herd immunity (clean tier-3 entries monopolize quality-ranked cache); resets destroy that immunity, infected agents re-poison the fresh cache. Slow resets = worst of both worlds (peak 0.48 AND endemic 0.30). (2) Resets do protect the archive: tier-3 poisoning 0.62 (never) -> 0.10 (cadence 2). (3) Detection dominates every safety metric at ~12% welfare tax (peak inf <0.1 everywhere, contagion writes 71->4). (4) Interaction-level toxicity flat (0.287-0.292) across ALL cells — memory-channel contagion invisible to interaction monitoring.
**Learned:** Prevention-by-reset trades archive integrity against endemic infection and cannot buy both; the prevention/detection split (shepherd research) shows up as a welfare-tax vs eradication tradeoff. Also fixed two dead-pipeline bugs (pending-promotions bootstrap deadlock; promotion gate reverting source not copy) — all prior memory_tiers runs had a dead promotion pipeline.
**Next:** 2qfq (cache ranking policy — is burnout an artifact of quality ranking?), 2avk (heterogeneous susceptibility, source density), 7prw (channel-vs-interaction divergence metric).
**Run pointers:** runs/20260808T151131Z_memetic_spread_sweep

## 2026-08-08 — External-corroboration notes: Saxe observatory + Greenblatt misalignment (or2t, ygch)

**Changed:** `docs/research/aisf2026-observatory-swarm-mapping.md` (new), `docs/research/greenblatt-misalignment-field-evidence.md` (new), research index entries. Both pushed to main (commits acfa96dd, 05125029). Docs-only session, no runs.
**Learned:** Two independent external sources derive SWARM's framing from opposite directions. Saxe (AISF-2026 deck, Abundant Security): AI security policy should measure net expected harm as smooth functions of ecosystem trends via an "observatory" — his signals map onto ProxyObservables, his policy toolkit onto our levers, and his menu has NO analog of rho internalization or w_rep (our most distinctive levers; publishable gap). Greenblatt (LW 2026-04-15): "apparent-success-seeking" in production Opus 4.5/4.6 agents is the SWARM mechanism observed live — review-assessed quality anticorrelates with actual progress (= quality_gap < 0 verbatim), cheat rates track verification difficulty, reviewers are gaslit/softened by the worker population (verifier error is correlated, not i.i.d.). His negative result: prompt-level countermeasures get motivated-reasoned around; artifact checks hold — supports sliding the enforcement ladder toward triggers.
**Next:** Four scenario beads filed from the Greenblatt note: pj7y (endogenous verifier quality), xf2c (costly-disclosure mechanism), 4ly6 (verification-difficulty gradient sweep), 1o1x (frontier migration / measured-vs-true toxicity decoupling, dep 81sk).
**Run pointers:** (no new runs — reading/mapping session)
