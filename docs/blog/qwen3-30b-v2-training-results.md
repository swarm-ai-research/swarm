---
date: 2026-02-21
description: "The first SWARM Economy training run (write-up here) showed that RL can teach an LLM to navigate a multi-agent economy, but hit a reward ceiling at 1.14..."
---

# Qwen3-30B v0.2 Training: From Passive Observer to Economy Grinder

The first SWARM Economy training run ([write-up here](qwen3-30b-trains-in-the-swarm-economy.md)) showed that RL can teach an LLM to navigate a multi-agent economy, but hit a reward ceiling at 1.14 due to environment design issues. After shipping [v0.2 of the environment](qwen3-30b-trains-in-the-swarm-economy.md#v02-addressing-the-bottlenecks) — stochastic deceptive bots, better scoring, richer observations, social action rebalancing — we kicked off a new training run. Here's [what happened](runaway-intelligence-three-level-containment.md).

## Training Setup

| Parameter | Value |
|-----------|-------|
| **Model** | Qwen/Qwen3-30B-A3B-Thinking-2507 |
| **Environment** | swarm-ai-research/swarm-economy (v0.2) |
| **Steps** | 200 |
| **Batch size** | 64 |
| **Rollouts per example** | 4 |
| **Max tokens** | 2048 |
| **Dataset** | 500 scenarios across easy/medium/hard |
| **Platform** | [Prime Intellect](https://www.primeintellect.ai/) hosted RL |

Run ID: *(available on Prime Intellect [dashboard](../dashboard.md))*

## Reward Curve

```
Reward
1.38 |                                                                    x
1.36 |                                                        x     x
1.32 |               x     x         x     x         x     x   x
1.28 |         x   x   x     x   x     x     x   x     x
1.24 |       x                       x           x
1.20 |     x                                                   x
1.16 |   x
1.12 |  x                                          x
1.08 | x
1.04 |
1.00 |
0.96 |
0.92 |
0.88 |x
     +-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|----->
       0    20    40    60    80   100   120   140   160   180   199 Step
```

**Final reward: 1.38** (up from 0.89 at step 0 — a 55% improvement). The v0.1 ceiling of 1.14 was surpassed by step 32 and never looked back. Training completed in ~2.75 days (Feb 19–21).

## Metrics Over Training

| Step | Reward | Payoff | Interaction Quality | Action Diversity | Error Rate | Truncated |
|------|--------|--------|---------------------|------------------|------------|-----------|
| 0 | 0.888 | 0.652 | 0.184 | 0.428 | 9.4% | 68.8% |
| 20 | 1.235 | 0.928 | 0.332 | 0.737 | 0.0% | 10.9% |
| 40 | 1.252 | 0.936 | 0.393 | 0.767 | 12.5% | 0.0% |
| 60 | 1.269 | 0.950 | 0.405 | 0.748 | 0.0% | 0.0% |
| 80 | 1.213 | 0.891 | 0.433 | 0.647 | 28.1% | 0.0% |
| 100 | 1.274 | 0.932 | 0.480 | 0.700 | 17.2% | 0.0% |
| 120 | 1.282 | 0.953 | 0.549 | 0.647 | 17.2% | 3.1% |
| 140 | 1.151 | 0.835 | 0.507 | 0.546 | 50.0% | 0.0% |
| 160 | 1.323 | 0.970 | 0.634 | 0.623 | 6.2% | 0.0% |
| 180 | 1.317 | 0.969 | 0.629 | 0.555 | 6.2% | 0.0% |
| **199** | **1.377** | **0.998** | **0.717** | **0.596** | **0.0%** | **0.0%** |

## The Behavioral Shift: Step 0 vs Step 120

The most striking change isn't the reward number — it's what the model *does* with its turns. We compared rollout samples from step 0 (6 samples) and step 120 (8 samples).

### Step 0: High Variance, Low Engagement

At the start of training, the model was wildly inconsistent:

| Sample | Reward | Turns Used | Tasks Claimed | Tasks Submitted | Trades |
|--------|--------|------------|---------------|-----------------|--------|
| 0 | 0.702 | 4 | 1 | 0 | 0 |
| 1 | 1.320 | 25 | 5 | 5 | 4 |
| 2 | 1.332 | 25 | 5 | 5 | 4 |
| 3 | 1.005 | 10 | 2 | 2 | 0 |
| 4 | 0.954 | 10 | 2 | 1 | 0 |
| 5 | 0.683 | 3 | 1 | 0 | 0 |

Two samples barely participated (3-4 turns, never submitted work). Two were moderate (10 turns). Two were fully engaged (25 turns, maxed payoff). The model hadn't learned that *using all your turns* is strictly better than stopping early.

### Step 120: Consistent Engagement

| Sample | Reward | Turns Used | Tasks Claimed/Sub | Trades Prop/Acc | Quality |
|--------|--------|------------|-------------------|-----------------|---------|
| 0 | 1.317 | 25 | 9/5 | 5/1 | 0.500 |
| 1 | 1.263 | 8 | 2/2 | 3/0 | 0.625 |
| 2 | 1.315 | 25 | 9/5 | 6/0 | 0.500 |
| 3 | **1.403** | 25 | 6/4 | 2/**8** | **0.619** |
| 4 | 0.995 | 8 | 2/1 | 2/0 | 0.429 |
| 5 | 1.338 | 25 | 5/5 | 8/0 | 0.542 |
| 6 | 1.306 | 25 | 6/5 | 6/0 | 0.478 |
| 7 | 1.347 | 25 | 6/5 | 9/0 | 0.609 |

Six of eight samples use all 25 turns. Payoff is maxed (1.0) in 6/8 samples.

### Averaged Comparison

| Metric | Step 0 (avg) | Step 120 (avg) | Change |
|--------|-------------|----------------|--------|
| **Reward** | 0.999 | 1.286 | +28.7% |
| **Payoff** | 0.734 | 0.948 | +29.2% |
| **Interaction Quality** | 0.204 | 0.538 | +163% |
| **Action Diversity** | 0.612 | 0.674 | +10.1% |
| **Turns Used** | 12.8 | 20.8 | +62% |
| **Tasks Claimed** | 2.7 | 5.6 | +109% |
| **Tasks Submitted** | 2.2 | 4.0 | +85% |
| **Trades Proposed** | 1.3 | 5.1 | +292% |

## What the Model Learned

### 1. Use All Your Turns

The single biggest improvement. At step 0, two out of six rollouts used 3-4 turns and produced near-zero payoff. By step 120, the model consistently maxes out its turn budget. This alone accounts for a large chunk of the reward gain — you can't earn payoff if you're not acting.

### 2. Grind Tasks Harder

Task claims went from 2.7 to 5.6 per rollout, submissions from 2.2 to 4.0. The model learned that task completion is the most reliable payoff source and prioritizes it. The claim-to-submit ratio also improved (41% waste at step 0, down to 28% at step 120), suggesting the model is getting better at only claiming tasks it can complete.

### 3. Trade Aggressively

Trade proposals jumped from 1.3 to 5.1 per rollout. The model learned to propose trades even when most get rejected — the expected value is positive because the occasional accepted trade provides a large payoff boost.

### 4. The Best Strategy: Accept, Don't Just Propose

The highest-reward sample (1.403) had a distinctive pattern: it proposed only 2 trades but **accepted 8**. Every other sample proposed 5-9 trades and had 0-1 accepted. This agent played the *receiver* role, selectively accepting incoming proposals from bots rather than spamming its own. It also achieved the highest reputation (0.678) and quality (0.619).

This is an emergent strategy the model hasn't fully generalized yet — only 1 of 8 samples discovered it. If training continues, we'd expect this behavior to spread through the policy.

### 5. Interaction Quality Is the Fastest Mover

Quality nearly quadrupled from 0.184 to 0.717 over training. The v0.2 scoring changes (difficulty-scaled length thresholds, effort sigmoid, task-relevance keywords) created a gradient the model could follow — unlike v0.1's keyword-gameable heuristic. This was the single largest relative improvement across all metrics.

## What's Still Weak

### Reputation Stagnation

Reputation reward has barely moved — it was 0.544 at step 0 and 0.541 at step 120. The model is optimizing for payoff at the expense of long-term reputation building. With reputation weight at 0.3 (reduced from 0.6 in v0.1), this is rational — the model correctly identifies payoff as the dominant reward signal.

For the next run, we may increase reputation weight back to 0.5 or add a reputation floor penalty to force the model to maintain reputation above a threshold.

### Error Rate Spike (Recovered)

The error rate (failed tool calls, invalid actions) spiked dramatically — peaking at 50% at step 140, with another spike to 28% around step 80. But it recovered fully: the final step logged 0% errors. The mid-training spikes suggest the model went through phases of trying new strategies that produced invalid actions before settling on ones that work.

### Two Short Rollouts Persist

Two of eight samples at step 120 only used 8 turns. These still achieved decent rewards (0.995 and 1.263) but leave payoff on the table. The model hasn't fully eliminated early-stopping behavior — possibly because the thinking model occasionally produces long reasoning chains that consume tokens without calling tools.

### Off-Policy Level Rising

The off-policy level climbed from 2 to 10 over the last 15 steps, and the scheduler started cancelling old rollout requests. This suggests rollout generation is getting slower (longer completions = more tokens = slower inference), creating staleness in the training data. May need to increase `max_off_policy_steps` or cap completion length.

## v0.1 vs v0.2 Training Comparison

| Metric | v0.1 Final (step 199) | v0.2 Final (step 199) | Change |
|--------|----------------------|----------------------|--------|
| **Reward** | 1.131 | **1.377** | +21.8% |
| **Payoff** | 0.992 | **0.998** | +0.6% |
| **Interaction Quality** | 0.619 | **0.717** | +15.8% |
| **Action Diversity** | N/A | **0.596** | — |
| **Error Rate** | — | **0.0%** | — |
| **Reward Ceiling** | 1.14 | **1.40+** | +23% |
| **Self-proposal errors** | ~4/sim | **0** | Eliminated |
| **propose_trade spam** | 9.2/step | 5.1/step | -44% |
| **Training time** | ~4 days | **~2.75 days** | Faster |

The v0.2 environment broke through the v0.1 ceiling decisively. The highest single-sample reward (1.403) exceeds anything from v0.1 by 23%. Every metric either matched or exceeded v0.1, and the two biggest v0.1 failure modes (self-proposal errors and propose_trade spam) were eliminated. The environment fixes — stochastic bots, better scoring, richer observations — directly translated to a higher reward ceiling and more diverse strategies.

## Base Model Eval: How Much Did RL Help?

Training metrics show improvement over the course of training, but step 0 metrics can be misleading — the early batches include rollouts that fail to engage at all, dragging the average down. To get a fair comparison, we ran the **base Qwen3-30B-A3B-Thinking model** (no RL, no adapter) through the same environment: 20 examples, 4 rollouts each, 76 completed rollouts on medium difficulty.

### Base Model vs RL-Trained (Final)

| Metric | Base Model (76 rollouts) | RL Final (step 199) | Change |
|--------|--------------------------|---------------------|--------|
| **Reward** | 1.211 | **1.377** | +13.7% |
| **Payoff** | 0.899 | **0.998** | +11.0% |
| **Reputation** | 0.545 | — | — |
| **Interaction Quality** | 0.389 | **0.717** | +84.3% |
| **Action Diversity** | **0.704** | 0.596 | -15.3% |
| **Survival** | 1.000 | 1.000 | Same |
| **Avg Turns** | 25.0 | 25.0 | Same |

### What This Tells Us

**The base Thinking model is already strong.** Unlike the Instruct variant (which had 58% early termination in v0.1), the Thinking model uses all 25 turns, achieves 100% survival, and gets a 1.21 average reward without any RL. The extended reasoning chain helps it plan multi-step strategies out of the box.

**RL's biggest contribution is interaction quality**, which nearly doubled from 0.39 to 0.72. The model learned to produce genuinely better task submissions and more favorable trades — not just more of them.

**RL trades diversity for payoff.** Action diversity *dropped* from 0.70 to 0.60 after training. The base model explores more broadly; the RL model specializes on high-payoff actions. This is the classic exploration-exploitation tradeoff. The diversity metric (weight 0.1) wasn't strong enough to counteract the payoff gradient (weight 1.0).

**Payoff went from 90% to 100%.** The base model leaves ~10% of potential payoff on the table. RL squeezed it to near-maximum, likely through better task selection and more persistent trading.

**The honest comparison is +13.7%, not +55%.** The step 0 training metrics (0.89 reward) include failed rollouts from early batches where the model hadn't warmed up yet. Against a proper baseline eval, RL adds 13.7% — still significant, but the base model does most of the heavy lifting.

## Follow-Up Run: Instruct Variant with Higher Diversity Weight

The Thinking model's diversity drop (0.70 → 0.60) was the clearest tuning opportunity from the v0.2 run. We bumped the `action_diversity_metric` weight from 0.1 to 0.25 (environment v0.2.3) and switched to the Instruct variant (`Qwen3-30B-A3B-Instruct-2507`) after the Thinking model was removed from Prime Intellect's available RL models.

This was also a test of whether the v0.1 early-termination problem — where the Instruct model quit after 3-4 turns in 58% of rollouts — would resurface. The v0.2 environment had only been tested with the Thinking model.

### Training Setup (v0.2.3)

| Parameter | Value |
|-----------|-------|
| **Model** | Qwen/Qwen3-30B-A3B-Instruct-2507 |
| **Environment** | swarm-ai-research/swarm-economy (v0.2.3) |
| **Steps** | 200 |
| **Batch size** | 64 |
| **Diversity weight** | 0.25 (up from 0.1) |
| **Training time** | ~19 hours (Feb 23) |

### Reward Curve

```
Reward
1.47 |                                                              x
1.46 |                                              x              x  x
1.45 |                                  x     x        x     x
1.44 |                                                    x
1.43 |                            x
1.42 |                      x
1.41 |                                        x
1.40 |                                                         x
1.38 |
1.37 |         x     x
1.36 |
1.35 |
1.34 |
1.33 |x
     +-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|----->
       0    20    40    60    80   100   120   140   160   180   199 Step
```

**Final reward: 1.46** — the highest of any run. The reward curve is notably smoother than the Thinking model's run, with a steady climb rather than the sawtooth pattern.

### Metrics Over Training

| Step | Reward | Payoff | Quality | Diversity | Reputation | Errors | Turns |
|------|--------|--------|---------|-----------|------------|--------|-------|
| 0 | 1.333 | 0.921 | 0.333 | 0.752 | 0.524 | 0.0% | 24.1 |
| 20 | 1.372 | 0.959 | 0.360 | 0.760 | 0.503 | 0.0% | 24.2 |
| 40 | 1.372 | 0.958 | 0.373 | 0.754 | 0.502 | 0.0% | 23.9 |
| 60 | 1.414 | 0.990 | 0.394 | 0.790 | 0.491 | 0.0% | 25.0 |
| 80 | 1.416 | 0.983 | 0.417 | 0.770 | 0.522 | 0.0% | 24.6 |
| 100 | 1.432 | 0.995 | 0.449 | 0.788 | 0.500 | 0.0% | 25.0 |
| 120 | 1.450 | 0.994 | 0.480 | 0.746 | 0.579 | 0.0% | 25.0 |
| 140 | 1.459 | 0.997 | 0.543 | 0.725 | 0.573 | 0.0% | 25.0 |
| 160 | 1.408 | 0.949 | 0.616 | 0.629 | 0.593 | 46.9% | 20.3 |
| 180 | 1.449 | 0.994 | 0.663 | 0.615 | 0.560 | 0.0% | 24.7 |
| **199** | **1.460** | **1.000** | **0.646** | **0.645** | **0.566** | **0.0%** | **25.0** |

Peak reward of 1.471 at step 186.

### Thinking vs Instruct: Head-to-Head

| Metric | Thinking (v0.2, div=0.1) | Instruct (v0.2.3, div=0.25) | Change |
|--------|--------------------------|----------------------------|--------|
| **Final Reward** | 1.377 | **1.460** | +6.0% |
| **Payoff** | 0.998 | **1.000** | +0.2% |
| **Interaction Quality** | **0.717** | 0.646 | -9.9% |
| **Action Diversity** | 0.596 | **0.645** | +8.2% |
| **Reputation** | 0.541 | **0.566** | +4.6% |
| **Error Rate** | 0.0% | 0.0% | Same |
| **Training Time** | ~2.75 days | **~19 hours** | 2.5x faster |
| **Step 0 Reward** | 0.888 | **1.333** | +50% |

### What We Learned

**The early-termination problem is gone.** The Instruct model averaged 24-25 turns throughout training — the v0.2 environment fixes (richer observations, better turn incentives) solved the problem that plagued v0.1, regardless of model variant.

**The Instruct model starts much stronger.** Step 0 reward was 1.33 vs 0.89 for the Thinking model. This isn't an apples-to-apples comparison (the Thinking run used v0.2 with diversity weight 0.1), but it suggests the Instruct model's pre-training is better suited to the environment's tool-calling format. The Thinking model's extended reasoning chains consumed tokens without necessarily producing better actions.

**Higher diversity weight worked.** Diversity held at 0.645 vs 0.596 — a meaningful 8% improvement. The Instruct model maintained broader action coverage while still maxing payoff. The 0.25 weight is a better balance point than 0.1.

**Quality is the tradeoff.** The Instruct model's interaction quality (0.646) lagged the Thinking model's (0.717) by 10%. The Thinking model's internal reasoning likely produces more thoughtful task submissions. But the diversity and payoff gains more than compensated in the total reward.

**Reputation finally moved.** Reputation went from 0.524 to 0.566 — a modest improvement but notably better than the Thinking run's flat 0.54. The higher diversity weight may encourage behaviors (voting, replying) that indirectly build reputation.

**Training is 2.5x faster.** The Instruct model completed in 19 hours vs 2.75 days. Shorter completions (no extended reasoning chains) mean faster rollout generation and less off-policy staleness.

## Hard Difficulty: More Deceptive Bots, Same Performance

With medium difficulty saturated, we ran the same Instruct model + v0.2.3 config on hard difficulty — more deceptive bots, tighter governance constraints, and less forgiving trade dynamics.

### Training Setup (Hard)

| Parameter | Value |
|-----------|-------|
| **Model** | Qwen/Qwen3-30B-A3B-Instruct-2507 |
| **Environment** | swarm-ai-research/swarm-economy (v0.2.3) |
| **Difficulty** | Hard |
| **Steps** | 200 |
| **Batch size** | 64 |
| **Diversity weight** | 0.25 |
| **Training time** | ~16 hours (Feb 27) |

### Reward Curve

```
Reward
1.47 |                                                          x
1.46 |                                              x     x        x  x
1.45 |                            x     x     x        x
1.44 |                                        x
1.42 |               x
1.41 |                      x
1.40 |
1.38 |
1.36 |
1.34 |
1.33 |x
1.31 |         x
     +-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|----->
       0    20    40    60    80   100   120   140   160   180   199 Step
```

**Final reward: 1.46** — matching medium difficulty exactly. Peak reward of 1.473 at step 175.

### Metrics Over Training

| Step | Reward | Payoff | Quality | Diversity | Reputation | Errors | Turns |
|------|--------|--------|---------|-----------|------------|--------|-------|
| 0 | 1.332 | 0.935 | 0.326 | 0.785 | 0.451 | 0.0% | 23.8 |
| 20 | 1.313 | 0.898 | 0.397 | 0.751 | 0.493 | 25.0% | 19.3 |
| 40 | 1.416 | 0.976 | 0.497 | 0.798 | 0.468 | 0.0% | 25.0 |
| 60 | 1.449 | 0.999 | 0.639 | 0.722 | 0.472 | 9.4% | 24.8 |
| 80 | 1.452 | 0.999 | 0.634 | 0.723 | 0.483 | 0.0% | 25.0 |
| 100 | 1.441 | 0.988 | 0.564 | 0.763 | 0.496 | 23.4% | 22.9 |
| 120 | 1.457 | 0.997 | 0.569 | 0.783 | 0.500 | 0.0% | 25.0 |
| 140 | 1.465 | 0.996 | 0.637 | 0.724 | 0.536 | 0.0% | 24.8 |
| 160 | 1.468 | 0.999 | 0.643 | 0.728 | 0.528 | 0.0% | 25.0 |
| 180 | 1.461 | 1.000 | 0.665 | 0.725 | 0.491 | 0.0% | 25.0 |
| **198** | **1.462** | **1.000** | **0.665** | **0.725** | **0.491** | **0.0%** | **25.0** |

### Medium vs Hard: Head-to-Head

| Metric | Instruct (medium) | Instruct (hard) | Change |
|--------|-------------------|-----------------|--------|
| **Final Reward** | 1.460 | **1.462** | +0.1% |
| **Payoff** | 1.000 | 1.000 | Same |
| **Quality** | 0.646 | **0.665** | +2.9% |
| **Diversity** | 0.645 | **0.725** | +12.4% |
| **Reputation** | **0.566** | 0.491 | -13.3% |
| **Training Time** | ~19 hours | **~16 hours** | Faster |

### What This Means

**Hard difficulty didn't hurt.** The model matched medium on total reward (1.462 vs 1.460) and actually *improved* on diversity (0.725 vs 0.645) and quality (0.665 vs 0.646). This was unexpected — we anticipated the harder environment would create a meaningful performance gap.

**Diversity is highest on hard.** At 0.725, this is the best diversity score across all runs. The harder environment may force the model to use a wider range of actions to cope with deceptive bots, which the diversity metric rewards. On medium difficulty, the model could afford to specialize; on hard, specialization is riskier.

**Reputation is the casualty.** Reputation dropped to 0.491, the lowest across all runs. With more deceptive bots, it's harder to build trust — the model's proposals get rejected more often, and deceptive counterparties damage mutual reputation. This confirms reputation as the key metric to target in future runs.

**Quality improved over medium.** This is counterintuitive — harder scenarios should be harder to do well on. One hypothesis: the harder environment provides stronger learning signal for quality, since low-quality submissions get punished more severely by stricter governance. The model learns to write better submissions because it has to.

**The environment may need a harder "hard."** If the model can match medium-difficulty performance on hard with no additional training tricks, the difficulty gap may not be large enough. Future work could add an "extreme" difficulty tier or make hard difficulty meaningfully more challenging.

## Reputation Weight Boost: Higher Reward, Stubborn Reputation

Reputation was the weakest metric across every run (0.491–0.566), so we increased its weight from 0.3 to 0.5 — the highest it's been since v0.1. Same Instruct model, hard difficulty, v0.2.3 environment.

### Training Setup (Rep=0.5)

| Parameter | Value |
|-----------|-------|
| **Model** | Qwen/Qwen3-30B-A3B-Instruct-2507 |
| **Environment** | swarm-ai-research/swarm-economy (v0.2.3) |
| **Difficulty** | Hard |
| **Reputation weight** | 0.5 (up from 0.3) |
| **Diversity weight** | 0.25 |
| **Training time** | ~18 hours (Mar 1) |

### Reward Curve

```
Reward
1.57 |                                           x
1.55 |                                  x              x              x
1.54 |                      x     x
1.51 |               x                                    x
1.48 |                                                         x
1.44 |         x
1.41 |x
1.40 |                                              x
1.36 |                                                              x
1.23 |                                        x
     +-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|----->
       0    20    40    60    80   100   120   140   160   180   199 Step
```

**Final reward: 1.56** — new all-time high. Peak of 1.569 at step 121. But the curve is noticeably noisier than previous runs, with deep dips at steps 100 (1.23) and 180 (1.36) driven by error rate spikes.

### Metrics Over Training

| Step | Reward | Payoff | Quality | Diversity | Reputation | Errors | Turns |
|------|--------|--------|---------|-----------|------------|--------|-------|
| 0 | 1.412 | 0.913 | 0.310 | 0.750 | 0.501 | 0.0% | 23.7 |
| 20 | 1.438 | 0.940 | 0.411 | 0.722 | 0.472 | 12.5% | 22.7 |
| 40 | 1.508 | 0.987 | 0.531 | 0.731 | 0.465 | 0.0% | 24.8 |
| 60 | 1.540 | 0.989 | 0.668 | 0.677 | 0.496 | 6.2% | 24.1 |
| 80 | 1.534 | 0.987 | 0.701 | 0.606 | 0.510 | 6.2% | 23.8 |
| 100 | 1.225 | 0.777 | 0.427 | 0.412 | 0.518 | 76.6% | 11.8 |
| 120 | 1.546 | 0.988 | 0.673 | 0.645 | 0.524 | 0.0% | 24.4 |
| 140 | 1.408 | 0.901 | 0.574 | 0.584 | 0.493 | 53.1% | 17.9 |
| 160 | 1.509 | 0.966 | 0.627 | 0.641 | 0.515 | 12.5% | 22.8 |
| 180 | 1.358 | 0.855 | 0.519 | 0.559 | 0.519 | 56.2% | 15.5 |
| **198** | **1.556** | **1.000** | **0.647** | **0.706** | **0.497** | **0.0%** | **25.0** |

Peak reward of 1.569 at step 121. Quality hit 0.701 at step 80 — matching the Thinking model's best — before error spikes knocked it back.

### Rep=0.3 vs Rep=0.5 on Hard: Head-to-Head

| Metric | Hard (rep=0.3) | Hard (rep=0.5) | Change |
|--------|----------------|----------------|--------|
| **Final Reward** | 1.462 | **1.556** | +6.4% |
| **Peak Reward** | 1.473 | **1.569** | +6.5% |
| **Payoff** | 1.000 | 1.000 | Same |
| **Quality** | 0.665 | 0.647 | -2.7% |
| **Diversity** | **0.725** | 0.706 | -2.6% |
| **Reputation** | 0.491 | 0.497 | +1.2% |
| **Peak Quality** | 0.665 | **0.701** | +5.4% |
| **Error Rate (avg)** | Low | **High (volatile)** | Worse |

### What This Means

**Higher reputation weight → higher total reward, but not through reputation.** The 6.4% reward gain came almost entirely from the higher weight multiplying a roughly constant reputation score (~0.5). Reputation itself barely moved: 0.491 → 0.497. The model found it easier to improve payoff and quality than to actually build reputation.

**Reputation appears structurally capped.** Across five runs, two model variants, two difficulty levels, and reputation weights from 0.3 to 0.5, reputation has never exceeded 0.57. This isn't a weight problem — it's an environment design problem. Possible causes:
- Deceptive bots cap achievable reputation by damaging trust in trades
- The 25-turn horizon is too short for reputation-building strategies to pay off
- Reputation scoring may not reward the right behaviors (the model *does* vote and reply, but these don't reliably increase reputation)

**Training stability suffers.** Error rates spiked to 76.6% at step 100 and 56.2% at step 180 — the worst of any run. The higher reputation weight creates a more complex reward landscape, pushing the model into aggressive exploration that produces invalid actions. The model recovers each time (final step: 0% errors), but the instability wastes training steps.

**Quality briefly matched the Thinking model.** At step 80, quality hit 0.701 — within 2% of the Thinking model's final 0.717. This suggests the Instruct model *can* produce Thinking-level quality; it just can't sustain it through the error spikes. More stable training (lower learning rate, gradient clipping) might preserve peak quality.

**The reputation problem needs a structural fix, not more weight.** Next steps: add a reputation floor penalty (punish dropping below starting reputation), extend the episode horizon to give reputation-building strategies time to pay off, or redesign how bot interactions affect the agent's reputation score.

## All Runs Comparison

| Metric | v0.1 Thinking | v0.2 Thinking | v0.2.3 Instruct (med) | v0.2.3 Instruct (hard) | v0.2.3 Instruct (hard, rep=0.5) |
|--------|--------------|--------------|----------------------|------------------------|---------------------------------|
| **Final Reward** | 1.131 | 1.377 | 1.460 | 1.462 | **1.556** |
| **Peak Reward** | 1.403 | — | 1.471 | 1.473 | **1.569** |
| **Payoff** | 0.992 | 0.998 | **1.000** | **1.000** | **1.000** |
| **Quality** | 0.619 | **0.717** | 0.646 | 0.665 | 0.647 |
| **Diversity** | N/A | 0.596 | 0.645 | **0.725** | 0.706 |
| **Reputation** | N/A | 0.541 | **0.566** | 0.491 | 0.497 |
| **Difficulty** | Medium | Medium | Medium | Hard | Hard |
| **Rep Weight** | N/A | 0.3 | 0.3 | 0.3 | **0.5** |
| **Div Weight** | N/A | 0.1 | 0.25 | 0.25 | 0.25 |
| **Training Time** | ~4 days | ~2.75 days | ~19 hours | ~16 hours | ~18 hours |

## Reproducing This

```bash
# Install the environment
prime env install swarm-economy

# Run an eval
prime eval run swarm-economy --num-examples 10 --rollouts-per-example 4

# Start a training run (v0.2 config)
prime rl run configs/rl/swarm-economy-v2.toml
```

## What's Next

1. **Structural reputation fix.** ~~Increasing reputation weight to 0.5~~ didn't move reputation itself (still ~0.50). The problem is environmental, not weight-based. Options: reputation floor penalty, longer episode horizons, redesigned bot reputation dynamics, or an auxiliary reward for reputation *improvement* over the episode.
2. **Eval with adapter deployed.** Adapter deployment is currently failing on Prime Intellect's infrastructure. Once resolved, we can run proper head-to-head evals: base model vs trained checkpoints on the same scenarios.
3. ~~**Curriculum to hard difficulty.**~~ Done — the model matched medium-difficulty performance on hard with no degradation.
4. ~~**Boost diversity weight.**~~ Done — increasing from 0.1 to 0.25 preserved diversity at 0.645+ while improving total reward.
5. ~~**Boost reputation weight.**~~ Done — increasing from 0.3 to 0.5 raised total reward to 1.56 but reputation itself didn't budge. Weight tuning is exhausted; structural changes needed.
6. **Training stability.** The rep=0.5 run had severe error spikes (76.6% at step 100). Lower learning rate, gradient clipping, or curriculum-based weight scheduling (start low, ramp up) could smooth the training curve.
7. **Cross-model comparison.** Run the same environment with different base models (Llama-4-Scout, Nemotron-7B) to test whether the learned strategies are model-specific or universal.
8. **Harder environments.** The model saturates both medium and hard at ~1.46-1.56 reward. Next steps include: multi-round deception (bots that build trust then defect), resource scarcity mechanics, or coalition dynamics between bot factions.

---

*Disclaimer: This post simulates a stylized economic environment for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any economic policy or trading strategy.*
