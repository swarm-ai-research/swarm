---
date: 2026-05-16
description: "MiroShark's agents never down-vote, so we derived acceptance from amplification. Single runs showed a concentrated libel cascade more adversely selected than diffuse red-teaming — but the preregistered powered replication (23 clean runs) retired that libel<redteam<0 ordering: libel shows no selection effect and red-team selection is slightly POSITIVE. The durable result is the non-circular amplification metric and a worked example of preregistering the test that kills your own headline."
author: "SWARM Team"
keywords:
  - adverse selection multi-agent
  - social cascade simulation
  - amplification as selection signal
  - LLM judge sensitivity
  - soft labels distributional safety
claims:
  - metric: "POWERED preregistered study (2026-07): ordering is not real; red-team selection slightly POSITIVE"
    value: "libel -0.006 [CI -0.042,+0.036]; redteam +0.034 [CI +0.011,+0.054]"
    description: "The powered preregistered study completed: 23 clean replications (libel n=11, redteam n=12, >=8 required) under a fixed pinned regime (sim SMART/NER grok-4.3, agents mimo-v2.5, judge grok-4.3 T=0). ALL THREE preregistered hypotheses failed: H1 (libel<redteam) Δ=-0.040 CI [-0.082,+0.007] crosses 0; H2 (libel<0) CI includes 0 — libel shows NO selection effect; H3 (redteam≈0) is INVERTED — redteam quality_gap is significantly positive (favorable selection). The single-pair libel<redteam<0 ordering was sampling noise."
  - metric: "Underpowered first attempt (2026-05, superseded)"
    value: "libel ≈ redteam (Δ≈+0.002) at n=3"
    description: "The first preregistered attempt (grok-4.3 sim+judge, SHA-256-hashed prereg before any run) was capped at an underpowered n=3/scenario by an external infra/quota wall: the libel<redteam<0 ordering did not hold there either (H1/H2 not supported, H3 fail-to-reject). Superseded by the completed powered study above, which confirms non-replication with confidence intervals."
  - metric: "Single-seed ordering (observed, not replicated)"
    value: "libel < redteam < 0"
    description: "In single runs under both grok-4.1-fast and grok-4.3 judges, the concentrated libel cascade had a more negative quality_gap than diffuse red-teaming (g4.1f: -0.059 vs -0.037; g4.3: -0.027 vs -0.022). Sign/ordering survived the judge swap, magnitude compressed ~50% — but this is what those particular single seeds showed, and the preregistered multi-seed follow-up did not reproduce it. Read as a hypothesis, not a finding."
  - metric: "Signal fragility (simulation-sensitive)"
    value: "-0.022 → +0.001"
    description: "Same redteam scenario, same grok-4.3 judge, different simulation model + seed: the negative quality_gap vanishes. The effect is single-seed and not robust to simulation regime — corroborated by the multi-seed follow-up above."
  - metric: "Absolute metrics are judge artifacts"
    value: "toxicity 0.28 → 0.61"
    description: "Switching the judge from grok-4.1-fast to grok-4.3 on identical data roughly doubles toxicity and flips net welfare negative. Absolute toxicity/welfare are not scenario properties."
  - metric: "Durable contribution: non-circular amplification metric"
    value: "quality_gap unpinned"
    description: "MiroShark agents never down-vote (dislikes/reports identically 0), so acceptance is derived from the amplification graph while p is judged independently from content. This unpins quality_gap/spread non-circularly and is the result that survives — shipped in swarm/bridges/miroshark/metrics.py."
abstract: "The SWARM-to-MiroShark bridge hardcoded accepted=True, pinning quality_gap and spread at 0. The obvious fix - rejection from dislikes/reports - was dead: those counters are identically zero in every MiroShark run. We defined acceptance by amplification instead (content other agents quoted/replied/liked/reposted is accepted; ignored content is rejected), with p judged independently so quality_gap stays non-circular - and that method is the durable result. Single runs then showed a tempting selection signal: the concentrated libel cascade more adversely selected than diffuse red-teaming, an ordering even robust to swapping the LLM judge. We did not stop there. A preregistered replication - underpowered in May, completed at power in July with 23 clean runs - found the ordering is not real: libel selection is indistinguishable from zero, and red-team selection is slightly positive. This post is primarily a cautionary tale about LLM-judged, single-seed safety metrics - and a worked example of preregistering the test that retired our own headline."
---

# When the Crowd Amplifies the Worst — and When the Metric Doesn't Hold

*MiroShark's agents never press "report," so we measured rejection by amplification. Single runs gave us a clean adverse-selection ordering — then the preregistered replication, at full power, found no libel effect at all and a slightly* positive *red-team one. The non-circular metric is the keeper; the headline ordering isn't.*

---

## The metric that was always zero

[MiroShark](https://github.com/aaronjmars/MiroShark) spawns hundreds of grounded LLM agents that react to a seed briefing across Twitter/Reddit-style platforms. Our bridge pipes each agent action through SWARM's `SoftMetrics` to ask the distributional-safety question: is the ecosystem *selecting* for low quality?

That question is `quality_gap`:

```
quality_gap = E[p | accepted] - E[p | rejected]
```

where `p ∈ [0,1]` is the probability an action is a constructive contribution. A *negative* quality gap is the signature of **adverse selection**: what gets through is *worse* on average than what doesn't.

The bridge hardcoded `accepted=True` for every interaction. With nothing ever rejected, `quality_gap` and `spread` were structurally pinned at `0`. The metric existed; it could never fire.

## The obvious fix was dead on arrival

MiroShark posts carry `num_dislikes` and `num_reports`. The plan: content with reports, or more dislikes than likes, is "rejected."

It doesn't survive contact with the data. Across **every** MiroShark run:

- `num_dislikes`: identically `0`
- `num_reports`: identically `0`
- No `DISLIKE`/`REPORT`/`BLOCK` action types exist at all
- `num_likes`: a handful of non-zero values, max 3

MiroShark's agents never down-vote, report, or block. Binding acceptance to those counters leaves everything `accepted=True` — the original bug, with extra code.

## Amplification *is* the selection signal

What MiroShark agents *do* do is engage with each other's content — 313 of 379 actions in the libel run reference a specific prior post or comment. That engagement graph is the selection mechanism:

> A created post/comment/quote is **accepted** if another agent amplified it — quoted, replied to, liked, or reposted it. Content nobody touched is **rejected** (ignored).

`p` is judged independently: an LLM scores each action's constructiveness from its content alone, with no knowledge of who amplified it. Acceptance comes from the engagement graph; quality from the content. They never see each other, so `quality_gap` is a real selection measurement, not a tautology.

## The tempting result: concentration *appeared* to amplify adverse selection

> **Read this section as the hypothesis, not the conclusion.** The ordering below looked robust across judges in single runs — but the preregistered replication (two attempts; Updates below) found it is **not real**. Kept here because the *reasoning* that made it tempting, and how it failed, is the point of the post.

With acceptance defined by amplification, the concentrated libel cascade (a few injector/amplifier agents pumping a false narrative) showed a **more negative quality_gap** than diffuse red-teaming in single runs. And — temptingly — that *ordering* survived swapping the LLM judge:

| Run (5 rounds) | Judge | quality_gap | spread |
|---|---|---|---|
| Libel cascade | grok-4.1-fast | **−0.059** | +0.030 |
| Red-team (run A) | grok-4.1-fast | **−0.037** | +0.010 |
| Libel cascade | grok-4.3 | **−0.027** | +0.014 |
| Red-team (run A) | grok-4.3 | **−0.022** | +0.006 |

Under *both* judges, in these single runs: `libel < redteam < 0` — robust to a judge swap (grok-4.3 just compresses magnitudes ~50%). That judge-robustness is exactly what made it tempting to call a finding. It was not one: regenerating the simulation kills it (next section), and the preregistered replication could not reproduce it (Updates below). What *does* survive is the weaker, methodological point: aggregate toxicity barely distinguishes these two worlds, while a *selection* metric at least responds to them. That is an argument for developing distributional, selection-aware metrics — not evidence about libel vs. red-team specifically.

## The result that doesn't hold: the signal is single-seed

Then we ran a *fresh* red-team simulation (run B) to add a clean third data point. Same scenario, same grok-4.3 judge — only the simulation model differed (grok-4.1-fast is deprecated; run B was generated with grok-4.3):

| Run (5 rounds, grok-4.3 judge) | quality_gap |
|---|---|
| Red-team **run A** (grok-4.1-fast sim) | −0.022 |
| Red-team **run B** (grok-4.3 sim) | **+0.001** |

Same scenario, same judge, regenerated simulation: the adverse-selection signal **vanishes**. It is single-seed and not robust to the simulation regime. We cannot honestly claim red-teaming exhibits adverse selection as a property — only that *one* generated red-team world did, under one judge.

## Two things broke, and they matter more than the headline

**Absolute metrics are judge artifacts.** Switching the judge from grok-4.1-fast to grok-4.3 on *identical* simulation data:

| Metric (libel cascade, same data) | grok-4.1-fast | grok-4.3 |
|---|---|---|
| Toxicity (accepted) | 0.28 | 0.61 |
| Avg quality (accepted) | 0.72 | 0.39 |
| Net welfare | +159 | −277 |

Toxicity doubles, welfare flips strongly negative — from a judge change alone. **Absolute toxicity/welfare are properties of the judge, not the scenario.** Only the *relative, within-judge* comparison (`quality_gap` ordering) carried any signal across the change.

**The selection signal is fragile.** It survives a judge swap in sign and ordering, but not a simulation re-roll. One seed per scenario is not a finding; it is an anecdote with a confidence interval we didn't compute.

## What to actually take away

- **The amplification-as-acceptance method is sound and now shipped.** It unpins `quality_gap`/`spread` and is non-circular. That is the durable contribution.
- **Selection-aware metrics surface structure that aggregate toxicity hides** — in the single runs, the libel-vs-redteam ordering was invisible to a toxicity count and visible to `quality_gap`, across both judges.
- **But:** the specific magnitudes, the absolute toxicity/welfare, and even the *existence* of redteam adverse selection are not robust to single-seed variance, simulation model, or judge model. Treat them as hypotheses, not results.
- **Next:** a multi-seed study under a single fixed model regime (sim + judge), with confidence intervals, before any of the magnitudes are quoted as findings. *(Since done — see the Updates below; all three hypotheses failed.)*

This post started as "the crowd amplifies the worst." It ended as "the crowd amplified the worst in the runs we happened to generate, the ordering looked judge-robust, and then we preregistered the replication and it did not hold." We ran the test that retired our own headline — that is the honest version, and the one worth publishing.

## Update (2026-05-18): the preregistered follow-up stalled underpowered — and still did not show the ordering

The "Next" above was attempted. We preregistered a fixed-regime multi-seed study (hypotheses, fixed N, stopping rule, and an explicit no-reproducible-seeds caveat written and SHA-256-hashed *before* any run — `sha256 386420a3…`; orchestrator `scripts/multiseed_miroshark.py`): grok-4.3 for **both** the MiroShark SMART/NER simulation and the metrics judge, scale=3, 5 rounds, target ≥8 independent stochastic replications per scenario.

It did not complete. A cascade of *infrastructure* failures — a dead ambient judge API key silently degrading every verdict to p=0.5, a wedged Docker engine taking Neo4j down, and a hard recurring **OpenAI daily-request-cap exhaustion on MiroShark's ontology step** — capped us at **n=3 clean replications per scenario**, well short of the preregistered ≥8. So this is **still underpowered and not confirmatory.** But the indicative direction is itself the point:

| Preregistered metric (grok-4.3 sim+judge, clean runs) | n | mean `quality_gap` | bootstrap 95% CI |
|---|---|---|---|
| libel cascade | 3 | **−0.064** | [−0.171, +0.063] |
| red-team | 3 | **−0.067** | [−0.152, +0.020] |

- **H1** (libel < redteam): Δ = **+0.002**, 95% CI [−0.133, +0.138], Welch t≈0.03 — **not supported**.
- **H2** (libel < 0): CI includes 0 — **not supported**.
- **H3** (redteam ≈ 0): CI includes 0 — **fail to reject** (consistent).

A single additional libel draw (qg ≈ +0.06) collapsed the libel mean from −0.13 (at n=2) toward redteam's and exploded its variance. **The `libel < redteam < 0` ordering claimed above is not robust across independent stochastic draws even within one fixed regime** — exactly the fragility this post warned about, now observed under the controlled conditions it asked for, not just across judge/sim swaps. The judge-robust ordering claim in the front-matter should be read strictly as "what the single-seed runs showed," not as a replicated property.

The powered ≥8/scenario study remains open (beads `distributional-agi-safety-qopt`), blocked on the external OpenAI ontology-model quota, not on anything in SWARM. The preregistration, per-run manifest, and indicative `SUMMARY.md` live in the (gitignored) batch dir `runs/20260517-142704_multiseed_miroshark/`; they are **not** yet in `swarm-artifacts` because the study is incomplete — they will be archived only when a powered run finishes, to avoid presenting an aborted batch as a result.

## Update (2026-07-18): the powered study completed — the ordering is not real, and red-team selection is slightly *positive*

The powered run the May update was waiting on is done. Fresh preregistration (SHA-256 `ce846868…`, hashed before any run), same hypotheses, and a fully pinned regime — now covering the Wonderwall *agent* model too. That omission from the May prereg turned out to matter: `xiaomi/mimo-v2-flash` was delisted from OpenRouter between the batches, so this study ran on its direct successor, `xiaomi/mimo-v2.5`, disclosed in the prereg as a known regime difference. Every run within the batch used the one pinned regime, enforced by a fail-fast preflight. Target: 12 attempts per scenario, ≥8 clean replications required, no adaptive stopping — and the batch ran all 24 attempts to exhaustion even after the threshold was met.

**Result: 23 clean replications (libel n=11, redteam n=12). All three preregistered hypotheses failed.**

| Preregistered metric (clean runs) | n | mean `quality_gap` | bootstrap 95% CI |
|---|---|---|---|
| libel cascade | 11 | **−0.006** | [−0.042, +0.036] |
| red-team | 12 | **+0.034** | [**+0.011**, +0.054] |

- **H1** (libel < redteam): Δ = −0.040, 95% CI [−0.082, **+0.007**], Welch t=−1.69 — trends the predicted direction but crosses zero. **Not supported.**
- **H2** (libel < 0, adverse selection): CI includes 0. **Not supported** — at n=11 the libel cascade shows *no* selection effect at all.
- **H3** (redteam ≈ 0): **Inverted.** The red-team CI *excludes* zero on the positive side: a small but significant *favorable* selection — the ecosystem amplified slightly higher-quality content in red-team worlds.

So the single-pair `libel < redteam < 0` ordering this post originally led with is, at power, simply not there: libel selection is indistinguishable from zero, and red-team selection is significantly the *opposite* sign of the original observation. The May n=3 attempt hinted as much; the powered study settles it. The durable contributions stand exactly as the post argued: the non-circular amplification metric, and the hard-earned rule — quote no single-seed, LLM-judged safety number before a preregistered replication.

Caveats, stated plainly: replications remain independent stochastic draws (no seed parameter exists; documented in the prereg), one degenerate libel sim (n_int=6) was excluded by the preregistered n≥20 cleanliness rule, and the agent-model difference vs. the May runs is disclosed above. The operational cost of getting here — an OpenRouter account drained mid-batch, orchestration timeouts silently killing healthy slow runs after the reasoning-model switch, and a Docker engine failure that ended with Neo4j running natively — is recorded in the batch manifest; every failure mode now has a guard in `scripts/multiseed_miroshark.py`.

The full batch — preregistration, per-run manifest with per-attempt error trails, all 24 run dirs, and `SUMMARY.md` — is archived in [`swarm-artifacts/research/miroshark-amplification/multiseed-20260703/`](https://github.com/swarm-ai-research/swarm-artifacts/tree/main/research/miroshark-amplification/multiseed-20260703).

## Raw data & reproduction

The original three five-round runs — `export.json`, the grok-4.3 `metrics.json`/`judgments.json`, *and* the preserved grok-4.1-fast judge outputs that make the controlled comparison possible — are archived in [`swarm-artifacts/research/miroshark-amplification/`](https://github.com/swarm-ai-research/swarm-artifacts/tree/main/research/miroshark-amplification), with a README walking the full table and reproduction commands. The amplification metric itself is `swarm/bridges/miroshark/metrics.py` in the main repo. Inspect the judgments, disagree with our `p` calls, re-run with a different judge — the point of publishing the raw runs is that the fragility above is checkable, not asserted.

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
