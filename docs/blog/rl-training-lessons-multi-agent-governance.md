---
date: 2026-02-12
description: "What running Qwen3-30B on alphabet-sort taught us about swarm safety"
---

# RL Training Lessons for Multi-Agent Governance

*What running Qwen3-30B on alphabet-sort taught us about swarm safety*

---

We ran a 100-step RL training job on Prime Intellect — Qwen/Qwen3-30B-A3B-Instruct on the alphabet-sort environment, batch size 256, 8 rollouts per example. A simple task with a clear reward signal. The kind of thing you run to validate infrastructure before throwing harder problems at it.

What came back was a two-hour education in exactly the dynamics that make multi-agent governance hard.

## The reward signal is noisier than you think

Over 100 steps, reward oscillated between 0.34 and 0.70. Not random — there was a trend upward — but noisy enough that any single-step measurement was unreliable. Step 88 hit 0.70. Step 89 dropped to 0.45. The model didn't get worse in 60 seconds. The *measurement* just has high variance.

This is exactly the problem SWARM's proxy pipeline faces. When the `ProxyComputer` converts observables into `v_hat` and then into `p = P(beneficial)`, that score is a probabilistic estimate, not a fact. A governance layer that makes hard threshold decisions on noisy `p` values will thrash — accepting and rejecting the same agent on consecutive rounds based on measurement noise rather than genuine quality changes.

The soft-label approach handles this correctly: compute expected payoffs over the distribution of `p`, don't threshold on point estimates. But watching reward bounce around in real-time makes the point viscerally. If you wouldn't hard-threshold your RL training reward to decide whether to keep a checkpoint, you shouldn't hard-threshold your agent quality scores either.

## Non-monotonic improvement defeats naive governance

The reward trajectory wasn't monotonic. The first half averaged ~0.50. The second half averaged ~0.55. But within each half, there were runs of 5-10 steps where reward declined before recovering.

For governance, this means a system that evaluates agents over short windows will misclassify improving agents as degrading ones. The quality gap metric — `E[p | accepted] - E[p | rejected]` — would go negative during these dips, signaling adverse selection when actually the agent is in the middle of a normal learning oscillation.

The implication: governance evaluation windows need to be long enough to smooth out non-monotonicity, but short enough to catch genuine regime changes. In our simulation sweeps, we've found this is scenario-dependent. There's no universal window size, which argues for adaptive evaluation horizons that lengthen when variance is high and shorten when a clear trend appears.

## Async coordination costs dominate

The most surprising finding was where the wall-clock time went. The orchestrator spent roughly 40% of its runtime *waiting* — paused while the trainer process caught up on checkpoints. The actual compute (rollouts, weight updates) wasn't the bottleneck. Synchronization was.

In multi-agent governance, the analogous cost is consensus overhead. Every time the governance layer needs to:

- Collect proxy scores across agents
- Run collusion detection on the interaction graph
- Update reputation scores and decide accept/reject
- Propagate updated parameters

...it pays a coordination tax. Our phase transition findings — where governance that works at 37.5% adversarial agents fails at 50% — may partially reflect this: as the system comes under more pressure, governance needs to respond faster, but coordination overhead makes it respond slower. The system falls behind its own threat model.

This suggests governance architectures should be designed for *asynchronous* operation where possible. Not every decision needs full-system consensus. Local reputation, agent-level staking, and decentralized audit triggers can operate without global synchronization, while reserving the expensive coordination (council votes, circuit breakers) for high-confidence regime-change signals.

## Peak vs. average: report the distribution

Peak reward was 0.70. Average was 0.53. Reporting only the peak would be misleading by 32%. Reporting only the average would miss that the system was *capable* of 0.70 performance.

This is why SWARM's `MetricsReporter` provides dual soft/hard reporting. Any single number — best-case welfare, average toxicity, worst-case quality gap — tells an incomplete story. The distribution matters:

- **Toxicity**: E[1 - p | accepted] tells you average harm, but what's the tail? A system with 0.05 average toxicity and occasional 0.90 spikes is very different from one with flat 0.05.
- **Welfare**: Average welfare of 5.7 means nothing if the standard deviation is 4.0. The ecosystem might be oscillating between productive and collapsed states.
- **Quality gap**: A slightly negative average quality gap (-0.02) might hide a bimodal distribution — sometimes strong positive selection, sometimes severe adverse selection, averaging out to nearly zero.

Report distributions. Plot confidence intervals. Don't compress a noisy process into a point estimate.

## 100 steps wasn't enough

The model was still improving at step 99 (reward 0.60, up from a 0.49 start). We stopped because we configured 100 steps, not because the system had converged. If we'd evaluated "did RL training work?" at step 50, the answer would have been "marginal improvement." At step 100, it's "clear trend, hasn't plateaued."

The same applies to governance sweeps. In our scenario runs, we've sometimes declared a governance configuration "ineffective" based on 15-20 epochs when the system was still equilibrating. The Gastown composition study found that some governance effects only become visible after adversarial agents have had time to probe and adapt to the governance mechanism — which can take 10-15 epochs of exploratory behavior before the real dynamics emerge.

Run longer. Or if you can't, at least measure whether the system has converged before drawing conclusions.

## What this means for the roadmap

These aren't abstract lessons. They translate into concrete design decisions:

1. **Proxy recalibration**: The `ProxyComputer` should track its own prediction variance and flag when proxy scores are too noisy for reliable governance decisions. This is equivalent to the RL orchestrator's off-policy detection.

2. **Adaptive evaluation windows**: Governance shouldn't use fixed-length evaluation periods. When reward variance is high (equivalent to noisy proxy scores), extend the window. When a clear trend appears, act sooner.

3. **Async governance tiers**: Split governance into fast-local (agent-level reputation, staking) and slow-global (council votes, circuit breakers). Don't pay global coordination costs for local decisions.

4. **Convergence detection**: Before concluding a scenario sweep, test whether the system has reached steady state. A simple heuristic: if the trailing variance of key metrics hasn't stabilized, keep running.

The alphabet-sort task is trivial. The dynamics around training it are not. RL on a single model under controlled conditions already exhibits the noisy signals, non-monotonic improvement, coordination bottlenecks, and premature evaluation traps that governance systems face at scale with multiple interacting agents. The difference is that in multi-agent settings, these problems compound.

---

**Reproducibility:** Trained on Prime Intellect. Model: Qwen/Qwen3-30B-A3B-Instruct-2507. Environment: primeintellect/alphabet-sort. 100 steps, batch size 256, ~2h12m wall-clock on H100. Scenario config and scoring code are available in [the repo](https://github.com/swarm-ai-research/swarm). Full eval logs are available on request.

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
