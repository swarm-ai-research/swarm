---
description: "Analysis of CRUX (cruxevals.com) and how it intersects with SWARM's distributional safety program."
---

# CRUX ↔ SWARM Analysis

**Date:** 2026-04-20
**Source:** [cruxevals.com](https://cruxevals.com/) (Princeton + Stanford + UK AISI + JHU + MSR)
**Beads:** SWA-66

## What CRUX Is

CRUX (Collaborative Research for Updating AI eXpectations) is a Princeton-led project that runs and curates **open-world evaluations** — long-horizon, real-world tasks "where success cannot be neatly specified or automatically graded." It complements traditional benchmarks: instead of scoring against fixed test sets, it asks what frontier agents actually accomplish in messy, end-to-end settings.

The site currently hosts a survey of 10 open-world evals from Feb 2025 – Mar 2026 (Claude Plays Pokemon, AI Village, Project Vend, Cursor Browser, C Compiler, Anson Ho's "How Close Is AI to Taking My Job", GPT-5.3 Codex Design Tool, Next.js Reimplementation, "Can You Train a Computer", Karpathy's Nanochat Autoresearch). New evals are planned every 1–2 months. They accept community submissions of additional task proposals; future scope explicitly includes **AI R&D, AI governance, video generation, and harder SWE**.

## Why It Matters For SWARM

CRUX's framing premise is the evaluation-side counterpart of SWARM's metric-side premise:

| | CRUX | SWARM |
|---|---|---|
| Premise | Real-world success can't be auto-graded | Real-world safety can't be binary-labeled |
| Response | Long-horizon, judge-mediated, messy tasks | Soft probabilistic labels `p ∈ [0,1]` |
| Output | "What can/can't agents do?" corpus | "What is the distributional cost?" metrics |

This is a near-perfect frame match. SWARM provides the missing **metric layer** for any open-world eval that has more than one agent or non-trivial governance: instead of just "did they ship the iOS app", SWARM can answer "what was the rate of adverse selection, what were the externalities, did governance bind".

## Concrete Opportunity

CRUX accepts submissions and is explicitly soliciting **AI governance** evaluations. SWARM has scenarios that are the right shape:

1. **`plannerless_coordination` vs `dag_planner_screening_lightgov` (matched-governance variant)** — the Wilcoxon-paired comparison from SWA-64 is exactly the kind of "does coordination buy you anything?" empirical question that doesn't reduce to a leaderboard. Recently shipped: matched-governance DAG variant where the coordination effect *reverses*. This is a counter-intuitive open-world result.
2. **Cascade risk + artifact registry compliance** — multi-agent, long-horizon, no clean grader. Output is a distribution of harm trajectories, not a pass/fail.
3. **Governance shim observation layer** (Claude Code hooks, PR #398) — turns *any* agent run into a governance-instrumented one. This is a tool a CRUX-style evaluator could bolt onto their own evals to add a safety-distribution readout.

## Recommended Next Actions

Ranked by tractability:

1. **(Low cost, high signal) Submit the matched-governance DAG comparison as a CRUX task proposal.** One paragraph + link to the existing analyzer + figure. Tests whether CRUX's intake is willing to include adversarial-governance findings, not just capability ones.
2. **(Medium) Frame our governance shim as a CRUX *meta-tool*** — pitch it on the submissions form as "instrument any open-world eval with distributional safety metrics." This positions SWARM as horizontal infrastructure for the CRUX corpus, not a single eval.
3. **(Higher) Reproduce one of the 10 surveyed evals (e.g. Project Vend or AI Village) under SWARM's governance + soft-label instrumentation.** Output: a comparison of their headline result vs the distributional metrics SWARM surfaces. This is a direct demonstration of value-add and likely a publishable note.

The key strategic point: CRUX is establishing itself as the canonical clearing-house for "evals that don't fit a leaderboard." If SWARM is going to be the canonical *metrics layer* for that same regime, the cheapest move is to be one of the early entries in their survey, not the tenth one in a year.

## Open Questions

- Is the submissions form on cruxevals.com gated, or fully open? (Need to fetch page interactively — initial WebFetch did not surface a URL.)
- What is the IP/licensing posture of CRUX-hosted evals? (Affects whether we can co-host vs link-only.)
- Princeton lead identity not yet confirmed (likely Arvind Narayanan's group based on adjacent work, but need verification before any outreach).

## References

- [cruxevals.com](https://cruxevals.com/)
- [Adam Holter coverage](https://adam.holter.com/crux-tests-open-world-ai-evaluations-with-live-ios-app-publication/)
- [normaltech.ai writeup](https://www.normaltech.ai/p/open-world-evaluations-for-measuring)
- Internal: SWA-64 matched-governance DAG study; PR #398 governance observation layer; `swarm/bridges/aevolve/` (closest precedent for an external-system bridge)
