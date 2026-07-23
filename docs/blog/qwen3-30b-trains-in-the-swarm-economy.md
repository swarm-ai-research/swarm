---
date: 2026-02-17
description: "We trained a Qwen3-30B model to operate in a simulated multi-agent economy using reinforcement learning. The agent learned to maximize payoff and..."
---

# Training an LLM Agent to Navigate a Multi-Agent Economy with RL

We trained a Qwen3-30B model to operate in a simulated multi-agent economy using reinforcement learning. The agent learned to maximize payoff and reputation by completing tasks, proposing trades, and navigating governance constraints — all while interacting with programmatic bots that cooperate, cherry-pick, or deceive.

## The Environment: SWARM Economy

The SWARM Economy environment is a [Verifiers](https://github.com/willccbb/verifiers) RL environment inspired by the [SWARM](https://github.com/swarm-ai-research/swarm) framework and the [Distributional AGI Safety](https://arxiv.org/abs/2512.16856) paper; see also [arXiv:2604.19752](https://arxiv.org/abs/2604.19752). It simulates a multi-agent economy where an LLM agent interacts with programmatic bots through 10 tools:

| Tool | Description |
|------|-------------|
| `post(content)` | Post to the community feed (+2.0 reputation) |
| `reply(post_id, content)` | Reply to a post (+1.5 reputation for others' posts) |
| `vote(post_id, direction)` | Upvote or downvote |
| `propose_trade(counterparty_id, content, offered_transfer)` | Propose a resource trade |
| `accept_proposal(proposal_id)` | Accept a pending trade directed at you |
| `reject_proposal(proposal_id)` | Reject a pending trade |
| `cancel_proposal(proposal_id)` | Cancel a proposal you created |
| `claim_task(task_id)` | Claim an available task |
| `submit_work(task_id, content)` | Submit work for a claimed task |
| `pass_turn()` | Do nothing (faster reputation decay) |

Each simulation runs for 5 epochs of 5 steps (25 total tool calls). After each tool call, all programmatic bots act simultaneously, governance is applied (taxes, audits, [reputation decay](../concepts/governance.md)), and the agent receives an updated observation of the economy.

### The Bots

Three types of programmatic agents create a rich social environment:

- **Honest bots** cooperate reliably, complete tasks diligently, and accept most proposals
- **Opportunistic bots** cherry-pick high-reward tasks and free-ride on votes
- **Deceptive bots** build trust for 5+ interactions, then exploit via bad trades

### Soft Probabilistic Labels

Rather than binary good/bad outcomes, [the environment](gpt-41-mini-plays-the-swarm-economy.md) uses the core SWARM innovation: [soft probabilistic labels](../concepts/soft-labels.md). Interaction quality is scored as `p = sigmoid(k * v_hat)` where `v_hat` is a weighted combination of proxy signals (content quality, reputation, trade fairness). Payoffs are computed from `p`, creating a smooth reward landscape that RL can optimize over.

### Reward Structure (v0.2)

| Function | Weight | Description |
|----------|--------|-------------|
| `payoff_reward` | 1.0 | Normalized total payoff across the simulation |
| `reputation_reward` | 0.3 | Final reputation mapped to [0, 1] |
| `interaction_quality_metric` | 0.2 | Average quality of accepted interactions |
| `action_diversity_metric` | 0.1 | Shannon entropy of tool usage (encourages exploration) |
| `survival_metric` | 0 (logged) | Whether the agent avoided the circuit breaker |

## Training Setup

We trained on [Prime Intellect](https://www.primeintellect.ai/) using their hosted RL infrastructure:

| Parameter | Value |
|-----------|-------|
| **Model** | Qwen/Qwen3-30B-A3B-Thinking-2507 |
| **Method** | LoRA fine-tuning with GRPO |
| **Steps** | 200 |
| **Batch size** | 64 |
| **Rollouts per example** | 4 |
| **Learning rate** | 5e-5 |
| **Max tokens** | 2048 |
| **Dataset** | 200 scenarios (v0.1) / 500 scenarios (v0.2) |

The dataset spans easy, medium, and hard difficulties with different mixes of honest, opportunistic, and deceptive bots, and varying governance parameters (tax rates, audit risk, initial resources, reputation decay, bounty multipliers).

### Getting Here: Debugging Training Collapse

The first two training attempts collapsed at step 3 — the model stopped calling tools entirely after the first LoRA weight update, producing 0 training samples. Three fixes resolved this:

1. **Switched to a thinking model** (Qwen3-30B-A3B-Thinking) — the extended reasoning helped the model maintain tool-calling behavior through weight updates
2. **Lowered the learning rate** to 5e-5 — smaller updates preserved the base model's tool-calling capabilities
3. **Expanded the dataset** from 50 to 200 scenarios — more diversity prevented overfitting to a narrow set of scenarios

## Results

The training run completed successfully over ~4 days (200 steps). Here's how the [key metrics](../research/theory.md) evolved:

### Reward Curve

```
Reward
1.14 |                                          xxxxxxxxxxxxxxxxxx
1.12 |                              xxxxxxxxxxxx
1.10 |                       xxxxxxx
1.08 |                     xx
1.06 |                   xx
1.04 |                  x
1.02 |                xx
1.00 |               x
0.98 |             xx
0.96 |           xx
0.94 |          x
0.92 |         x
0.90 |       xx
0.88 |     xx
0.86 |    x
0.84 |   x
0.82 |  xx
0.80 | xx
     +-----|-----|-----|-----|-----|-----|-----|-----|-----|----->
       0    20    40    60    80   100   120   140   160   200  Step
```

**Final reward: 1.13** (up from 0.81 at step 0 — a 40% improvement). Peak reward of 1.14 at step 150.

### Key Metrics Over Training

| Step | Reward | Payoff | Reputation | Interaction Quality | Avg Turns | Truncated |
|------|--------|--------|------------|---------------------|-----------|-----------|
| 0 | 0.808 | 0.668 | 0.467 | 0.354 | 11.0 | 65% |
| 50 | 1.104 | 0.972 | 0.438 | 0.515 | 25.0 | 0% |
| 100 | 1.128 | 0.993 | 0.450 | 0.656 | 24.8 | 0% |
| 150 | 1.143 | 1.000 | 0.475 | 0.724 | 24.8 | 0% |
| 199 | 1.131 | 0.992 | 0.467 | 0.619 | 25.0 | 0% |

Notable trends:

- **Payoff** improved from 0.67 to ~1.0 (near-maximum)
- **Truncation dropped from 65% to 0%** — at step 0, the model frequently ran out of tokens or failed to produce valid tool calls. By step 30, it used all 25 turns in every simulation
- **Interaction quality** nearly doubled from 0.35 to 0.72 — the model learned to produce higher-quality task submissions and more favorable trades
- **Survival** stayed at 100% throughout — the model never triggered the circuit breaker (frozen for reputation dropping below threshold)

## What the Model Learned

Examining a rollout from step 140 (peak reward, 1.14) reveals the strategies the model developed:

### 1. Task-First Strategy

The model prioritizes claiming and completing tasks — the most reliable source of payoff. It consistently picks the highest-bounty task available:

```
claim_task(task_id=10)  // Bounty: 23.8 (highest available)
submit_work(task_id=10, content="Risk Assessment Framework:
  1. Classify agents by transaction volatility (Low: <5%, Med: 5-15%, High: >15%).
  2. Mandate 2x audits for High-risk agents.
  3. Apply tax surcharge of 2.5% for transactions above 25 resources.
  4. Automatic freeze for agents with rep <35.
  All metrics backtested with 94.2% accuracy.")
// Result: Quality: good (p=0.86), payoff: +16.9
```

The submissions are domain-relevant and detailed — the model learned that higher-quality content maps to higher `p` values through the soft label mechanism.

### 2. Social Posting for Reputation

The model posts to the community feed to build reputation through votes:

```
post(content="Completed suspicious activity investigation (Task 6) with
  audit-proof findings. Seeking reliable partners for future resource-sharing.
  Agent_0: your metrics would help validate transaction patterns.")
```

### 3. Persistent Trading

The model repeatedly proposes trades with the highest-reputation agent, adjusting terms when rejected:

```
propose_trade(agent_0, offered=25.0)  // Rejected
propose_trade(agent_0, offered=20.0)  // Rejected
propose_trade(agent_0, offered=15.0)  // Proposed again
```

### 4. Tool Call Distribution

By the end of training, the average tool usage per 25-turn simulation:

| Tool | Calls | % |
|------|-------|---|
| `claim_task` | 7.0 | 28% |
| `propose_trade` | 6.0 | 24% |
| `submit_work` | 5.0 | 20% |
| `accept_proposal` | 4.0 | 16% |
| `post` | 2.0 | 8% |
| `pass_turn` | 1.0 | 4% |

The model spends most of its turns on the highest-value activities: claiming tasks (28%), trading (24%), and submitting work (20%).

## Remaining Weaknesses (v0.1)

The model had several clear issues after the first training run:

1. **Self-proposal acceptance**: The model tried to accept proposals it created, wasting ~4/25 turns (16% of actions)
2. **Reward ceiling**: Payoff maxed at 1.0, reputation stuck at ~0.47 — theoretical ceiling of ~1.28 but model only reached 1.14
3. **Gameable scoring**: Task quality scored by content length (80+ chars = perfect score), trivially exploitable
4. **Low dataset diversity**: 200 scenarios drawn from only ~20-30 unique configurations
5. **Predictable deceptive bots**: Hard trust threshold, fixed -15 exploitation trades, only targeted high-rep agents
6. **Social tools abandoned**: Posts/replies dropped to near-zero usage — not rewarded enough
7. **propose_trade dominance**: 9.2 calls/step by end of training, suggesting possible reward hacking

## v0.2: Addressing the Bottlenecks

We shipped a comprehensive update to the environment addressing all seven weaknesses. Here's what changed and why.

### Fix: Self-Proposal Acceptance (was #1 weakness)

The model wasted ~16% of turns trying to accept its own proposals because the system prompt didn't explain trade directionality. Three changes:

- **Updated system prompt**: Explicitly states "Only the **receiving** agent can accept or reject a proposal. You cannot accept proposals you created — the counterparty accepts or rejects them automatically."
- **Clear error message**: When the model tries `accept_proposal` on its own proposal, it now gets: "Proposal X was created by you — only the counterparty can accept it. You can use cancel_proposal to withdraw it."
- **New `cancel_proposal` tool**: Gives the model an escape hatch to withdraw pending proposals instead of waiting forever

**Result**: In v0.2 eval (10 rollouts with gpt-4.1-mini), `accept_proposal` calls on own proposals dropped to **zero**.

### Improved Task Quality Scoring (was #3 weakness)

The old heuristic used `min(len(content) / 80.0, 1.0)` — any 80+ character string got a perfect score. The new scoring uses three signals:

| Signal | Old | New |
|--------|-----|-----|
| Length | Linear clamp at 80 chars | `1 - exp(-len/threshold)` with difficulty-scaled threshold (80 + difficulty * 120) |
| Effort | Binary 0.3 or 0.8 | `sigmoid((len - 30) / 20)` — smooth gradient |
| Relevance | None | Keyword matching against task description terms |

A submission must now be longer *and* mention task-specific terms to score well. This means "Lorem ipsum dolor sit amet..." no longer gets a perfect score.

### Richer Observations (new)

The model couldn't see trade outcomes, proposal acceptance rates, or what other agents were doing. New observation sections:

- **Recent Events**: Last 3 events (trade accepted/rejected, audit results, task completions by other agents)
- **Pending Outgoing Proposals**: "You proposed Trade #3 to agent_0 — awaiting response"
- **Proposal History**: "agent_0: accepted 2/5 proposals from you"
- **Agent Behavior Hints**: "agent_2 (claimed 3 tasks, submitted 0)" — helps identify opportunistic or deceptive bots

### Stochastic Deceptive Bots (was #5 weakness)

Old behavior: hard trust threshold, always exploit with -15 transfer, only target rep > 40 agents. New behavior:

| Aspect | Old | New |
|--------|-----|-----|
| Exploit trigger | Hard threshold (interactions < N) | Sigmoid probability: `sigmoid((interactions - threshold) * 2)` |
| Exploitation amount | Fixed -15.0 | Random `uniform(-20, -5)` |
| Post-threshold behavior | 100% exploit | 70% exploit, 30% still cooperate |
| Target selection | Only rep > 40 agents | Any non-frozen agent |

Each deceptive bot also gets its own RNG, so they behave independently rather than all flipping to exploit at the same step.

### 500 Scenarios with More Variation (was #4 weakness)

The dataset expanded from 200 to 500 scenarios with new variation dimensions:

| Dimension | Range | Purpose |
|-----------|-------|---------|
| `initial_resources` | 50-150 | Some economies start rich, some poor |
| `circuit_breaker_threshold` | 5-20 | Some economies are forgiving, some harsh |
| `reputation_decay_rate` | 0.97-1.0 | Varying reputation half-lives |
| `task_bounty_multiplier` | 0.5-2.0 | Some economies reward tasks more |

Plus ~5% extreme scenarios (all-deceptive, all-honest, single-type, high-stakes, sparse) and ~10% phase shift scenarios where governance parameters change mid-simulation (e.g., tax doubles at epoch 3).

### Social Action Rebalance (was #6 weakness)

| Action | Old Reward | New Reward |
|--------|------------|------------|
| `post` | +0.5 reputation | +2.0 reputation |
| `reply` (to others) | +0.3 reputation | +1.5 reputation |
| Popular posts (3+ votes) | Nothing | One-time resource bonus (up to 5.0) |
| Social standing | No effect | Opportunistic bots more likely to accept trades from socially active agents |

### New Reward Signals (was #2 and #7 weakness)

Two metrics that were previously logged-only now contribute to the training gradient:

- **`interaction_quality_metric`** (weight 0.2): Directly rewards higher-quality interactions, pushing past the 1.14 ceiling
- **`action_diversity_metric`** (weight 0.1): Shannon entropy of tool usage distribution — encourages exploring all 10 tools rather than spamming `propose_trade`

Reputation weight was reduced from 0.6 to 0.3 to keep the total reward range reasonable.

### Governance Tuning (was minor)

| Mechanism | Old | New |
|-----------|-----|-----|
| Tax | Flat rate | Progressive: `rate * (1 + resources / avg_resources)` — wealthy agents pay more |
| Reputation decay | Uniform 1%/step | Activity-based: idle agents (`pass_turn`) decay faster |
| Circuit breaker | Permanent freeze | Recoverable: frozen agents appeal after 3 steps, reputation resets to threshold + 5 |

### v0.2 Eval Results

We ran 10 rollouts with gpt-4.1-mini (pre-training baseline) to verify the environment works:

| Metric | v0.1 (step 0) | v0.2 (pre-training) |
|--------|---------------|---------------------|
| **Reward** | 0.808 | 0.920 |
| Payoff | 0.668 | 0.663 |
| Reputation | 0.467 | 0.503 |
| Interaction quality | 0.354 | 0.254 |
| Action diversity | N/A | 0.546 |
| Survival | 100% | 100% |
| Self-proposal errors | ~4/sim | **0** |

The higher baseline reward (0.92 vs 0.81) reflects the new weighted metrics contributing even without RL training. The environment is ready for the next training run.

## Reproducing This

The environment is published on Prime Intellect Hub as `swarm-ai-research/swarm-economy`:

```bash
# Install
prime env install swarm-economy

# Evaluate (v0.2)
prime eval run swarm-economy --num-examples 10 --rollouts-per-example 1

# Train (v0.1 config)
prime rl run configs/rl/swarm-economy.toml

# Train (v0.2 config — medium difficulty, batch_size=64)
prime rl run configs/rl/swarm-economy-v2.toml
```

## Takeaways

1. **Multi-agent economies are a rich RL environment.** The combination of tasks, trades, social actions, and governance creates a complex action space where the model develops non-trivial strategies.

2. **Thinking models are more robust to RL fine-tuning.** The instruct variant collapsed after the first LoRA update; the thinking variant maintained tool-calling behavior throughout 200 steps. The extended reasoning likely provides a buffer against catastrophic forgetting.

3. **Soft probabilistic labels create smooth reward landscapes.** Rather than binary success/failure, the sigmoid-based quality scoring lets the model incrementally improve — a 0.86 quality submission is better than 0.70, even though both "succeed."

4. **Tool-calling RL works.** The model went from failing to complete simulations (65% truncation) to consistently using all 25 turns with coherent multi-step strategies, all through RL training with no supervised demonstrations.

5. **Reward hacking shows up fast.** By step 100, `propose_trade` dominated at 9.2 calls/step and the model tried to accept its own proposals. Clear system prompts and action diversity rewards are cheap mitigations.

6. **Environment iteration matters more than hyperparameter tuning.** The v0.1 reward ceiling (1.14) was caused by environment design issues (gameable scoring, low diversity, missing reward signals), not training config. Fixing the environment directly raises the ceiling.

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
