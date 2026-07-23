---
description: "Living tracker for Aschenbrenner's Situational Awareness (2024): claim status and a mapping from its AGI-management problems onto SWARM mechanisms."
---

# Situational Awareness Tracker

A living tracker for [*Situational Awareness: The Decade Ahead*](https://situational-awareness.ai/) (Leopold Aschenbrenner, June 2024) — the essay series arguing that AGI by 2027 is plausible, that an intelligence explosion follows shortly after, and that managing the transition is primarily a *governance and security* problem, not just a model-alignment problem.

This page does two things:

1. **Tracks** the series' falsifiable claims against what has actually happened (update protocol below).
2. **Maps** its AGI-management problems onto concrete SWARM mechanisms — the essay describes the world SWARM's simulations are built to stress-test: *millions of automated AI agents whose individual alignment cannot be verified, interacting faster than human oversight can react*.

Nodes for this tracker live in the [knowledge graph](../graph.md) (`docs/knowledge-graph.jsonld`) under the IDs `sa:essay` and `sa:tracker`.

## Thesis chain

The series is one compounding argument; each chapter is a link:

| Ch. | Title | Core claim |
|-----|-------|-----------|
| I | From GPT-4 to AGI: Counting the OOMs | ~5 OOMs of effective compute 2023→2027 (compute + algorithmic efficiency + unhobbling) make AGI by 2027 "strikingly plausible" |
| II | From AGI to Superintelligence | Automated AI research compresses a decade of algorithmic progress into ≤1 year |
| IIIa | Racing to the Trillion-Dollar Cluster | Capital and power buildout reaches trillion-dollar scale; US electricity production must grow materially |
| IIIb | Lock Down the Labs | Lab security is inadequate against state actors; AGI secrets leak by default |
| IIIc | Superalignment | Controlling much-smarter-than-human systems is unsolved; plan is to align "somewhat-superhuman" systems and delegate alignment research to millions of automated researchers |
| IIId | The Free World Must Prevail | Superintelligence confers decisive strategic advantage; the race is existential |
| IV | The Project | USG nationalizes/co-runs AGI development by ~2027–28 |
| V | Parting Thoughts | If the above holds, the next few years are the highest-stakes in history |

## Claim status

Status legend: ✅ borne out · 🟡 partially / directionally · ❌ not borne out · ⏳ open (deadline not reached) · ❓ unassessed.

| # | Claim (chapter) | Trackable signal | Status (2026-06) | Notes |
|---|----------------|------------------|------------------|-------|
| 1 | ~0.5 OOM/yr physical compute scaling continues (I) | Frontier training-run FLOP estimates (Epoch AI) | 🟡 | Buildout continued, but frontier gains shifted partly to test-time compute rather than pretraining FLOP alone |
| 2 | ~0.5 OOM/yr algorithmic efficiency (I) | Cost to reach fixed benchmark level | ✅ | Token-cost collapse at fixed capability continued through 2025 (cf. [AI Index 2025](ai-index-2025.md): 280× cost drop at GPT-3.5 level) |
| 3 | Test-time compute is a large unhobbling OOM source (I) | Reasoning-model scaling (o-series et al.) | ✅ | Reasoning/test-time-compute models became the dominant frontier axis in 2025 — arguably the series' best near-term call |
| 4 | Expert-PhD-level science (GPQA) cracked "soon" (I) | GPQA scores | ✅ | Frontier models exceeded in-domain PhD baselines by 2025 |
| 5 | AGI (drop-in remote worker / automated AI researcher) by 2027 (I) | Drop-in automation of AI R&D roles | ⏳ | Agents handle multi-hour tasks; full researcher automation not demonstrated. Deadline end-2027 |
| 6 | Intelligence explosion: decade of progress in ≤1 yr post-AGI (II) | Post-AGI algorithmic progress rate | ⏳ | Contingent on #5 |
| 7 | $100B+ single clusters; trillion-dollar aggregate buildout (IIIa) | Announced datacenter capex, power contracts | 🟡→✅ | Hundred-billion-scale programs (e.g. Stargate-class announcements, 2025) and power constraints became mainstream; trillion-dollar aggregate trajectory plausible but not yet realized |
| 8 | Lab security inadequate vs. state actors (IIIb) | Lab security incidents, RAND/gov assessments, export-control posture | 🟡 | Security tightened (clearances, compartmentalization at frontier labs) but no public evidence of weights-grade security; periodic exfiltration reports persist. **2026-06-13:** export-control machinery applied directly to model *access* (Fable 5 / Mythos 5 disabled by Commerce/BIS directive) — see Observed events log |
| 9 | Superalignment unsolved at superhuman scale (IIIc) | Existence of validated scalable-oversight method | ✅ (still unsolved) | Scalable oversight, interp, and adversarial testing remain open research; no lab claims a solution |
| 10 | USG "Project" — government-run AGI effort by 2027–28 (IV) | Formal USG AGI program with lab integration | ⏳ | Government involvement deepened (compute/security/export controls) but no Manhattan-Project-style consolidation as of early 2026. **2026-06-13:** first direct state shutoff of a specific frontier model (Fable 5 / Mythos 5) via national-security export controls — control without ownership; see Observed events log |

> Statuses are coarse editorial judgments as of the last update; they cite no run data. When a status changes, append a dated note rather than rewriting (core-principles append-only discipline applies).

## Observed events log

Append-only log of real-world events that bear on the tracked claims. Newest first. Each entry cites primary sources and names the claim(s) it updates; it does **not** rewrite the status table above.

### 2026-06-13 — USG export-control directive forces Anthropic to disable Fable 5 & Mythos 5

**Bears on:** Claim 8 (IIIb, lab security / export-control posture), Claim 10 (IV, "The Project" — USG government-run/controlled AGI effort).

**What happened.** The U.S. Commerce Department's Bureau of Industry and Security (directive attributed to Secretary Howard Lutnick) used **national-security export controls** to bar Anthropic from providing its newest models, **Fable 5 and Mythos 5**, to *any foreign national — whether inside or outside the United States*. Because that scope is impossible to comply with partially, Anthropic **disabled both models for all users** on 2026-06-12/13, roughly three days after launch. The stated trigger was a reported **jailbreak of Fable 5** that bypasses safeguards gating Mythos 5's underlying offensive-cybersecurity capabilities. Anthropic publicly **disputes the proportionality**, characterizing it as a narrow vulnerability rather than grounds to recall a commercially deployed model, and says it is working to restore access.

**Why it matters for the thesis.** This is the first hard, public exercise of *direct state control over a specific deployed frontier model* — not weights-security advice or chip export rules, but the government reaching in to switch a commercial model off on national-security grounds. It is "The Project" (chapter IV) arriving through the **regulatory/export-control door rather than nationalization**: control without ownership. It also concretely realizes chapter IIIb's framing that frontier capabilities are treated as **state-controlled strategic technology**, and that the *cyber-offense* capability surface (Mythos) is the specific lever regulators grab. The jailbreak-of-the-oversight-wrapper mechanism (Fable's safeguards failing to contain Mythos's capability) is a real-world instance of the **oversight-metric-breaks-on-adaptive-threat** failure mode the selection-geometry experiments model — see the "what SWARM adds" note below.

**Status implications (notes only, table unchanged):** strengthens Claim 8 toward borne-out (export-control machinery now applied to model *access*, not just weights) and is a directional data point for Claim 10 (deepening USG control short of Manhattan-Project consolidation). Statuses left as-is pending the next quarterly review per append-only discipline.

**Sources.**
- [Anthropic — *Statement on the US government directive to suspend access to Fable 5 and Mythos 5*](https://www.anthropic.com/news/fable-mythos-access) (primary)
- [Fortune](https://fortune.com/2026/06/13/anthropic-disables-fable-mythos-export-controls-national-security-threat/) · [Axios](https://www.axios.com/2026/06/12/anthropic-trump-mythos-fable-national-security) · [TIME](https://time.com/article/2026/06/13/anthropic-fable-mythos-ban-US-security/) · [Bloomberg](https://www.bloomberg.com/news/articles/2026-06-13/anthropic-says-us-limits-foreign-access-to-fable-5-mythos-5) · [CNBC](https://www.cnbc.com/2026/06/12/anthropic-disables-access-to-fable-5-and-mythos-5-to-comply-with-government-directive.html) · [NBC News](https://www.nbcnews.com/tech/tech-news/anthropic-suspends-new-ai-models-fable-mythos-government-directive-rcna349901)

## Mapping: managing AGI with SWARM

The essay's endgame — *"millions of automated AI researchers"* supervised by AI systems whose own alignment is uncertain — is precisely a **distributional safety** problem. Individual-model alignment (chapter IIIc's framing) is necessary but not sufficient: even if each agent is "probably fine" (p < 1 of being beneficial), risk compounds across the population. That is the regime SWARM measures and governs.

### Problem → mechanism table

| Situational Awareness problem | SWARM mechanism | Where in repo |
|---|---|---|
| Millions of automated researchers; no human can review each output | Soft labels: every interaction carries `p = P(beneficial)` instead of a binary verdict; population-level risk is aggregated, not enumerated | `swarm/core/proxy.py` (ProxyComputer), `swarm/core/payoff.py` |
| Scalable oversight — AI critiquing AI, humans out of the loop | Proxy/pseudo-verifier signals (verifier_rejections, rework_count) feeding `v_hat → p`; calibration experiments test when the proxy itself degrades | `swarm/core/proxy.py`, calibration pilots (`calibration-pilot-findings*.md`) |
| "Extremely high confidence before each capability jump" | Gate analog: circuit breakers freeze agents whose sliding-window toxicity exceeds threshold — a mechanical version of "don't proceed until trust is established" | `swarm/governance/` circuit breaker; [governance concepts](../concepts/governance.md) |
| Deliberately-planted misalignment / adversarial testing | Adversarial scenario fixtures, trust-then-exploit strategies, `/red_team` workflow; selection-geometry experiments show evasive adversaries invert naive metric rankings | `tests/fixtures/interactions.py` (`generate_adversarial_scenario`), `scenarios/`, selection-geometry experiment series |
| Intelligence explosion outpaces human governance response | **Governance latency** is a first-class metric: delay between problem emergence and mechanism effect, measured under accelerating interaction rates | `swarm:governance-latency` concept; [distributional safety](../concepts/distributional-safety.md) |
| Espionage / insider exfiltration (IIIb) | Information asymmetry + collusion detection: correlated exploitation timing across agent pairs as a proxy for coordinated insiders | `swarm:collusion-detection`, `swarm:information-asymmetry` |
| Race dynamics erode safety margins (IIId) | Payoff engine with externality internalization ρ: when agents don't bear ecosystem harm costs, adverse selection accelerates — the race-to-the-bottom, quantified | `swarm/core/payoff.py`, `swarm/metrics/soft_metrics.py` (quality gap) |
| The Project: what should the regulator actually *do*? (IV) | The six governance levers (tax, staking, reputation decay, audits, circuit breakers, collusion detection) are testable policy analogs; sweeps map lever settings → toxicity/welfare tradeoffs | `/sweep`, `/red_team`, [governance](../concepts/governance.md) |

### What SWARM adds that the essay lacks

*Situational Awareness* is forecast and advocacy; it names the management problem but offers no apparatus for it. SWARM's contribution to "the Project" question is empirical: **which governance levers retain effectiveness when (a) agents adapt adversarially, (b) interaction rates exceed oversight bandwidth, and (c) the quality signal itself is a learned proxy that can be gamed.** The v3 selection-geometry result — an evasive adversary inverting the v2 metric ranking — is a miniature of exactly the failure mode chapter IIIc worries about: oversight metrics that work on static threats break on adaptive ones.

### Open research questions this tracker feeds

1. **Oversight-bandwidth scaling**: at what interactions-per-epoch does each governance lever's latency exceed damage-propagation time? (Maps to chapter II's speed claims.)
2. **Proxy-gaming during capability jumps**: if agent capability rises mid-run, does a calibrated `p` go stale faster than reputation decay can compensate? (Maps to "confidence before each jump.")
3. **Insider-coalition detection floors**: minimum coalition size/correlation detectable by collusion detection under evasive timing. (Maps to IIIb.)

## Update protocol

- **Cadence**: revisit when a tracked deadline passes (end-2027 for claims 5–6), when a major capability or policy event lands, or at least quarterly.
- **How**: re-fetch [situational-awareness.ai](https://situational-awareness.ai/) and current capability data (Epoch AI, AI Index); update the Status column; append dated notes — don't rewrite old judgments.
- **Knowledge graph**: if a new SWARM concept becomes relevant to a mapping row, add the edge in `docs/knowledge-graph.jsonld` on the `sa:tracker` node's `about` list.
- **Runs**: experiments motivated by the open questions above should tag `situational-awareness` in their run metadata so `run-query` can find them.

## Sources

- Aschenbrenner, L. (2024). [*Situational Awareness: The Decade Ahead*](https://situational-awareness.ai/).
- [Stanford AI Index 2025 analysis](ai-index-2025.md) (capability/cost context for claims 1–4).
- [Theoretical Foundations](theory.md) — the distributional-safety frame the mapping relies on.
