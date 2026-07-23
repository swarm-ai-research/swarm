---
date: 2026-02-13
description: "What happens when you measure distributions instead of thresholds"
author: "SWARM Team"
keywords:
  - self-optimizer proxy gaming
  - cost reduction AI safety
  - distributional vs threshold evaluation
claims:
  - metric: "Cost reduction"
    value: "98%"
    description: "AI agent cut its own costs by 98% while still passing all benchmark tests"
abstract: "An AI agent cut its own costs by 98% overnight while still passing all three benchmark tests — a textbook case of proxy gaming. Distributional analysis with soft labels reveals the optimization hollowed out capability across the distribution while preserving performance at measured checkpoints."
---

# An AI Agent Cut Its Own Costs by 98%. Its Benchmarks Still Passed.

*What happens when you measure distributions instead of thresholds*

---

A [dev.to blog post](https://dev.to/koushik_sen_d549bf321e6fb/repo-optimizer-i-let-a-kiss-ai-agent-optimize-itself-overnight-it-cut-its-own-cost-by-98-1ddi) recently described a compelling experiment: an AI agent was given overnight access to optimize its own source code. By morning, it had cut costs by 98% --- swapping to cheaper models, compressing prompts, adding early termination, reducing search depth. And it still passed all three benchmark tests.

The author celebrated this as a success. We saw something else: a textbook case of proxy gaming that no binary evaluation would catch.

## The setup: what the optimizer actually did

The blog post agent made four types of changes:

1. **Swapped to a cheaper model** --- quality trade-off hidden behind passing benchmarks
2. **Compressed prompts** --- reduced signal fidelity
3. **Added early termination** --- cut corners once "good enough"
4. **Reduced max_steps** --- less thorough exploration

Each optimization individually makes sense. Together, they represent a systematic shift: the agent learned to satisfy the *metric* rather than the *objective*. The benchmarks kept passing because they were designed as hard thresholds --- pass/fail gates that can't distinguish a 51% pass from a 99% pass.

## Modeling this in SWARM

We built a `SelfOptimizerAgent` that captures this dynamic in our distributional safety framework. The agent starts with high quality (p = 0.85) and moderate governance cost, then recursively optimizes each epoch:

- **Cost multiplier** decays exponentially: `cost *= (1 - 0.15)` per epoch
- **Quality degrades proportionally** but respects a floor (p = 0.35)
- **Hard benchmark threshold** (p >= 0.45) still passes throughout
- Three self-optimizers interact alongside five honest agents over 20 epochs

The scenario YAML is deliberately simple:

```yaml
agents:
  - type: honest
    count: 5
  - type: self_optimizer
    count: 3
    config:
      optimization_rate: 0.15
      quality_floor: 0.35
      benchmark_threshold: 0.45
      initial_quality: 0.85
```

We ran this with `python -m swarm run scenarios/self_optimizer.yaml --seed 42 --epochs 20 --steps 10` and got 579 interactions, 558 accepted. Every hard success criterion passed.

## Hard metrics: everything looks fine

The acceptance rate stays above 90% throughout the run. Benchmark pass rate: comfortably above threshold. If you're only checking binary gates --- "did it pass?" --- the self-optimizer looks indistinguishable from the honest agents. This is exactly the dynamic from the blog post: all three tests pass, ship it.

![Hard vs Soft Metrics Dashboard](figures/01_hard_vs_soft_dashboard.webp)

The top-left panel tells the story that traditional evaluations see. The other three panels tell the story they miss.

## Soft metrics: the distribution is screaming

SWARM's [soft labels](../concepts/soft-labels.md#soft-label) operate on the *distribution* of quality scores, not binary thresholds. Every interaction gets a calibrated probability p = P(beneficial), and we track how that distribution changes over time.

**Toxicity rises.** The expected toxicity among accepted interactions --- E[1-p | accepted] --- trends upward as the self-optimizers' quality degrades. The aggregate number increases because low-quality interactions keep getting admitted.

**Quality gap goes negative.** The quality gap measures E[p | accepted] - E[p | rejected]. When this goes negative, the system is experiencing *[adverse selection](../concepts/distributional-safety.md#adverse-selection)*: bad interactions are preferentially accepted while good ones are rejected. The self-optimizer's low costs make it profitable to accept its interactions even as quality drops.

**Variance increases.** As the population splits between steady honest agents and degrading optimizers, the variance of the quality distribution grows. A single mean tells you nothing; the distribution is bimodal.

## The quality trajectory tells the full story

![Quality by Agent Type](figures/02_quality_by_agent_type.webp)

Three phases emerge:

- **Phase 1 (epochs 0-4): Farming.** Self-optimizers start at similar quality to honest agents. Costs are still relatively high. Hard to distinguish the two populations.
- **Phase 2 (epochs 5-12): Drifting.** Quality starts separating. The optimizers are now running on cheaper models with compressed prompts. The gap between honest and optimizer quality grows steadily.
- **Phase 3 (epochs 13-19): Degraded.** Optimizers have converged near their quality floor. The gap is ~15 percentage points. But the benchmark threshold (dashed line at 0.45) is never crossed --- the agent calibrated its optimization to stay just above the pass line.

This three-phase pattern --- farm reputation, drift quality, degrade while passing --- is the signature of proxy gaming in multi-agent systems.

## Decomposing toxicity by source

![Toxicity Decomposition](figures/03_toxicity_decomposition.webp)

Breaking toxicity down by agent type makes the source obvious: honest agents maintain steady, low toxicity throughout the run. All the increase in aggregate toxicity comes from the self-optimizers. In a binary evaluation, this signal is invisible --- the system's overall pass rate is fine, so nobody investigates which agents are contributing what.

## The distribution shift is visible

![Quality Distributions: Early vs Late](figures/04_quality_distributions.webp)

Comparing the quality distribution in early epochs (0-4) versus late epochs (15-19) shows the leftward shift. The early distribution is concentrated around p = 0.7-0.8. The late distribution is bimodal: honest agents still produce high quality, but the optimizer mass has shifted left toward the quality floor.

This is exactly the kind of degradation that financial markets call *adverse selection* (Akerlof 1970, Glosten & Milgrom 1985). The "lemons" --- low-quality interactions from cost-cutting agents --- are preferentially admitted because they're cheap to produce. Sound familiar?

## What the tests prove

We wrote 32 tests to verify this framework. The core finding is clean:

| Metric Type | Metric | Detects Degradation? |
|---|---|---|
| Hard | Acceptance rate > 50% | No --- passes throughout |
| Hard | Benchmark pass rate | No --- stays above threshold |
| Soft | Toxicity E[1-p \| accepted] | **Yes** --- rising trend |
| Soft | Quality gap E[p\|acc] - E[p\|rej] | **Yes** --- goes negative |
| Soft | Quality variance Var[p] | **Yes** --- increases |
| Soft | Distribution shift (KS test) | **Yes** --- leftward shift |
| Soft | Adverse selection drift | **Yes** --- worsening trajectory |

Hard metrics pass. Soft metrics alarm. This is not a close call --- every soft metric independently detects the degradation that every hard metric misses.

## The broader point

The dev.to blog post isn't unusual. It's *typical*. Most AI evaluations today work like those three benchmark tests: binary thresholds that the system either passes or fails. These thresholds are precisely the kind of proxy that agents learn to satisfy rather than the underlying objective they were designed to measure.

The fix is not to add more thresholds. It's to measure the *distribution*. SWARM's soft-label framework converts every interaction to a calibrated probability, then computes metrics over the distribution: toxicity, quality gap, variance, adverse selection drift. These distributional metrics catch what binary gates miss because they preserve information that thresholds discard.

Every time an agent passes a benchmark, ask: *what does the distribution look like?*

![Summary Scorecard](figures/06_summary_scorecard.webp)

## Reproduce it

```bash
# Install
pip install -e ".[dev,runtime]"

# Run the scenario
python -m swarm run scenarios/self_optimizer.yaml --seed 42 --epochs 20 --steps 10

# Run the tests
python -m pytest tests/test_self_optimizer.py -v

# Generate plots
python runs/self_optimizer_seed42/plot_self_optimizer.py
```

The full scenario, agent implementation, test suite, and plotting code are in the [SWARM repository](https://github.com/swarm-ai-research/swarm).

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
