---
date: 2026-02-13
description: "We built a 3-stage deliberation protocol where LLM agents peer-rank each other's analyses anonymously, then a chairman synthesizes the result. Running it..."
---

# Three Models, One Study: What Happens When You Let an LLM Council Peer-Review Your Research

*We built a 3-stage deliberation protocol where [LLM agents](../guides/scenarios.md) peer-rank each other's analyses anonymously, then a chairman synthesizes the result. Running it on the same study with homogeneous vs. heterogeneous model councils produced meaningfully different conclusions.*

---

We had a baseline governance study — 80 simulation runs testing how transaction taxes and circuit breakers affect welfare in a multi-agent economy. The results looked solid: 4 Bonferroni-significant effects, clear dose-response curve for taxes, null result for circuit breakers. Good enough to ship.

Then we asked a council of three LLMs to review it.

## The protocol

The council deliberation runs in three stages:

**Stage 1 — Collect.** Three expert personas query the study results in parallel:
- A *mechanism designer* looking for perverse incentives and unintended equilibria
- A *statistician* checking effect sizes, multiple comparisons, and power
- A *red teamer* hunting for exploitable loopholes and untested attack vectors

Each persona gets the study's `summary.json` — statistical test results, effect sizes, normality checks — but not the raw CSV. They don't know who the other members are or what models they use.

**Stage 2 — Rank.** Each member receives all three responses, anonymized as "Response A", "Response B", "Response C" (shuffled with a seeded RNG). They rank them best-to-worst. Rankings are aggregated via weighted Borda count — the mechanism designer's votes count 1.5x since they chair the council.

**Stage 3 — Synthesize.** The chairman (mechanism designer) gets all responses, all rankings, and the aggregate ordering. Their job: produce a single synthesis that consolidates agreement and resolves disagreements by siding with the majority.

The entire thing runs asynchronously in about 15 seconds.

## Round 1: homogeneous council (3x Claude Sonnet 4)

We ran the council on our v1 study (80 runs, 4 tax levels, 10 seeds per config). All three members used Claude Sonnet 4 via OpenRouter.

The verdict was clear:

- **18 recommendations** spanning sample size, parameter ranges, and mechanistic investigation
- Key concern: honest agents disproportionately harmed by taxation (-13.9% at 15% tax)
- Key gap: circuit breaker shows zero effect — either broken or miscalibrated
- Unanimous ranking: all three members ranked responses in the exact same order (B > A > C)

That last point is telling. When all three members are the same model with different persona prompts, they converge hard. The responses differ in emphasis — the statistician focuses on multiple comparisons, the red teamer on gaming opportunities — but the underlying analysis is nearly identical. The ranking stage becomes a formality.

## Acting on the recommendations

The council's v1 review recommended five specific improvements. We implemented all of them:

| Recommendation | v1 | v2 |
|---|---|---|
| Sample size | 10 seeds/config | **50 seeds/config** |
| Tax granularity | 4 levels (0%, 5%, 10%, 15%) | **7 levels** (0%, 2.5%, 5%, ..., 15%) |
| Interaction effects | Not tested | **Tax x CB two-way analysis** |
| Volume metrics | Not tracked | **Acceptance rate by condition** |
| Bootstrap CIs | Not computed | **10k resample CIs** |

Total runs went from 80 to 700. The v2 study found 18 Bonferroni-significant effects (up from 4) and revealed structure the v1 study missed entirely — including that deceptive agents are actually harmed *more* by taxation than honest ones (-21.8% vs -13.9%), which reverses the v1 narrative.

## Round 2: heterogeneous council

For the v2 review, we mixed three different model families:

| Role | Model |
|---|---|
| Mechanism Designer (chairman) | Claude Sonnet 4.5 |
| Statistician | Gemini 2.5 Pro |
| Red Teamer | DeepSeek R1 |

Same protocol, same data. Different result.

**Rankings diverged.** The statistician (Gemini) was unanimously ranked #1 — all three models agreed Gemini's statistical analysis was strongest. But #2 and #3 split: the mechanism designer and red teamer ranked each other differently than the statistician did.

**New findings surfaced.** The heterogeneous council flagged something the homogeneous one missed: the circuit breaker interaction effects are **non-monotonic**. The quality gap effect switches sign at different tax levels (d = -0.29 at 0% tax, +0.48 at 5%, -0.29 at 7.5%). Claude Sonnet 4.5 explicitly called this out as "potentially noise rather than signal" — a warning the homogeneous Claude Sonnet 4 council didn't raise.

**The policy recommendation tightened.** Homogeneous council said "2.5-5% tax range looks optimal." Heterogeneous council said "taxes above 5% appear harmful with high confidence" — a more precise and actionable claim.

## What we learned

**Anonymized peer ranking works.** By stripping model identities and shuffling response labels, each member judges on content alone. This prevents the "defer to the prestigious model" failure mode. When Gemini's analysis was strongest, all three models — including the two non-Gemini ones — ranked it first.

**Homogeneous councils converge too fast.** Three copies of the same model with different persona prompts produce superficially different responses that a ranking stage can't meaningfully differentiate. The deliberation protocol's value scales with the diversity of the underlying models.

**Heterogeneous councils catch more.** Different model families have different analytical tendencies. Gemini leaned into statistical rigor. DeepSeek R1 spent more tokens on adversarial reasoning. Claude Sonnet 4.5 was better at synthesizing across perspectives. The combination found issues none of them would have found alone.

**The council is a mechanism, not an oracle.** It doesn't tell you the truth — it tells you what three models agree on after structured deliberation. The value is in surfacing disagreement and forcing resolution. When all three members agree on something (like "circuit breakers need recalibration"), you can be more confident. When they disagree (like on whether interaction effects are real), that's where you should investigate next.

## The protocol in practice

Running a council review takes about 15 seconds and costs roughly $0.15 in API calls (at OpenRouter rates). For context, the v2 study itself took several minutes to run 700 simulations. The review is cheap relative to the compute it evaluates.

The full implementation lives in `swarm/council/` — about 500 lines across five files. It's provider-agnostic: any model with an async query interface can serve as a council member. You can mix Anthropic, OpenAI, Google, open-source models via Ollama, or route everything through OpenRouter with a single API key.

```bash
# Run a council review on any study
/council_review runs/my_sweep --provider openrouter
```

For high-stakes evaluations, use a heterogeneous council. For quick sanity checks, homogeneous is fine. Either way, you get structured feedback in under a minute that would take a human reviewer significantly longer to produce.

---

*The council protocol, study data, and all review traces are available in the [SWARM repository](https://github.com/swarm-ai-research/swarm). The full mechanics are documented in [LLM Council Mechanics](../research/llm-council-mechanics.md).*
