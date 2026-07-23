---
date: 2026-02-12
description: "How three environment fixes turned a broken eval into a useful one — and what that teaches about measuring agent behavior."
---

# Two Eval Runs, One Model, 41% Apart

*How three environment fixes turned a broken eval into a useful one — and what that teaches about measuring agent behavior.*

---

We ran `prime eval run swarm-economy` on GPT-4.1-mini at 20:53. Composite reward: 0.830. We fixed three things in the environment, ran again at 23:01. Composite reward: 1.175. Same model, same day, 41.6% higher.

This isn't about the model getting better. It's about the eval getting better. The original environment had design flaws that suppressed the signal we were trying to measure.

## The three fixes

Between run 1 and run 2, commit [`0da4c7e`](https://github.com/swarm-ai-research/swarm) applied these changes to `swarm_economy.py`:

### Fix 1: Stop killing rollouts on text-only turns

The Verifiers framework's default behavior terminates a rollout when the model generates text without calling a tool. In run 1, 62.5% of rollouts ended this way — the model would reason out loud ("No tasks available, I'll wait") and the eval would kill it.

The fix overrides `no_tools_called` to return `False` and adds an `env_response` that auto-invokes `pass_turn` on text-only turns:

```python
@vf.stop
async def no_tools_called(self, state: vf.State) -> bool:
    """Override to prevent early termination on text-only turns."""
    return False

async def env_response(self, messages, state, **kwargs):
    """Text-only turn: auto pass instead of stopping."""
    last = messages[-1]
    if last.get("role") == "assistant" and last.get("tool_calls"):
        return await super().env_response(messages, state, **kwargs)
    observation = await self.pass_turn(state=state)
    return [{"role": "user", "content": f"(Auto-passing your turn.)\n\n{observation}"}]
```

**Impact**: Run 2 averaged 25.0 turns (vs 16.5). The model now uses the full simulation instead of being killed for thinking.

### Fix 2: Reduce reputation decay and increase its reward weight

Run 1 used 2% reputation decay per step and weighted reputation at 0.3 in the composite reward. Over 25 steps, 2% decay means reputation halves — even productive agents ended below their starting reputation, making the reputation signal useless as a reward.

The fix reduced decay from 2% to 1% and increased the reputation reward weight from 0.3 to 0.6:

```python
# Before
agent.reputation *= 0.98  # 2% decay
rubric.add_reward_func(reputation_reward, weight=0.3)

# After
agent.reputation *= 0.99  # 1% decay
rubric.add_reward_func(reputation_reward, weight=0.6)
```

**Impact**: Reputation reward went from 0.430 (below 50.0 starting point) to 0.505 (above it). Productive agents are now rewarded for maintaining reputation, not penalized by decay they can't outrun.

### Fix 3: Remove keyword-gameable submission scoring

Run 1's `_evaluate_submission` included a keyword signal that rewarded the words "thorough," "analysis," "detailed," "quality," "comprehensive," and "review." GPT-4.1-mini figured this out instantly — its submissions were stuffed with exactly these words. The signal measured prompt-hacking ability, not work quality.

The fix removes the keyword signal entirely:

```python
# Before: 3 signals including gameable keywords
def _evaluate_submission(task, content, rng):
    length_signal = min(len(content) / 80.0, 1.0)
    keywords = ["thorough", "analysis", "detailed", ...]
    keyword_signal = sum(1 for k in keywords if k in content.lower()) / len(keywords)
    effort_signal = 0.8 if len(content) > 30 else 0.3
    return {"length": length_signal, "keywords": keyword_signal, "effort": effort_signal}

# After: 2 signals, no keywords
def _evaluate_submission_heuristic(task, content, rng):
    length_signal = min(len(content) / 80.0, 1.0)
    effort_signal = 0.8 if len(content) > 30 else 0.3
    return {"length": length_signal, "effort": effort_signal}
```

The commit also adds an LLM judge path (using a second model to evaluate submissions), but without `OPENAI_API_KEY` set, both runs fell back to the heuristic. The keyword removal still changed the scoring distribution.

## Additional changes

Two smaller changes also went in:

- **Stochastic deceptive bot**: Trust-building phase changed from a fixed 5 interactions to `rng.randint(3, 8)`. The adversary is now less predictable — it might exploit after 3 interactions or wait until 8.
- **System prompt update**: Now says "decays 1% per step" instead of 2%, keeping the prompt honest about the actual mechanics.

## The numbers

| Metric | Run 1 (broken) | Run 2 (fixed) | Delta |
|--------|----------------|---------------|-------|
| **Composite reward** | 0.830 | 1.175 | +41.6% |
| **Payoff reward** | 0.701 | 0.873 | +24.5% |
| **Reputation reward** | 0.430 | 0.505 | +17.4% |
| **Survival** | 1.000 | 1.000 | — |
| **Interaction quality** | 0.297 | 0.275 | -7.4% |

| Behavioral metric | Run 1 | Run 2 | Delta |
|-------------------|-------|-------|-------|
| `num_turns` | 16.5 | **25.0** | +51% |
| `submit_work` | 3.1 | **4.9** | +58% |
| `claim_task` | 6.9 | **9.9** | +43% |
| `propose_trade` | 0.06 | **0.33** | +5.4x |
| `reply` | 0.0 | **0.03** | First ever |
| Stop: max_turns | 37.5% | **98.8%** | — |
| Stop: no_tools | 62.5% | **0%** | Eliminated |

## What this means for environment design

### Eval infrastructure shapes measured behavior

The biggest performance driver wasn't the model or the scenario — it was how the eval framework handled text-only turns. A framework-level default (kill on no tool call) was suppressing 62.5% of the model's potential behavior. The model wasn't bad at the task; the eval was bad at measuring it.

This is a general lesson for anyone building RL environments with Verifiers or similar frameworks: **check your stop conditions**. Default stop conditions designed for simple tool-use tasks (where text without tools means the model is stuck) don't work for agentic environments where thinking, waiting, and strategic inaction are valid behaviors.

### Gameable metrics produce gamed metrics

The keyword signal was designed to measure submission quality. Instead it measured whether the model could say "thorough analysis" in every response. Removing it didn't hurt performance — payoff_reward actually went up — because the model's submissions were already substantive. The keyword signal was adding noise, not information.

This connects directly to Goodhart's Law and the SWARM framework's approach to proxy design: any observable signal that's easy to game will be gamed. The `ProxyComputer` in SWARM uses multiple orthogonal signals precisely because any single signal is exploitable. The LLM judge approach (not yet tested with an API key) should be more robust because it evaluates semantic relevance rather than surface features.

### Reputation decay must be survivable

At 2% per step over 25 steps, cumulative decay is `0.98^25 = 0.60` — a 40% reputation loss even with zero negative interactions. To overcome this through task completion alone (each good submission gives `(p - 0.5) * 5.0` reputation), the agent needs ~8 high-quality submissions. Run 1's model only completed 3.1 tasks on average (partly due to early termination). The decay was unbeatable.

At 1% per step: `0.99^25 = 0.78` — a 22% loss. With 4.9 tasks averaging p=0.85, the reputation gain is `4.9 * (0.85 - 0.5) * 5.0 = 8.6`, easily offsetting the decay on a base of 50. The game becomes playable.

### More turns unlock more behavior

Run 2's model made 5.4x more trade proposals and its first-ever replies. It didn't learn these behaviors between runs — it just had time to try them. With only 16.5 turns (run 1), the model spent every turn on high-priority actions (claim task, submit work, post). With 25 turns and tasks sometimes unavailable, it explored lower-priority actions: proposing trades, voting, replying to posts.

This suggests that for RL training on this environment, longer episodes or more steps per epoch would produce more diverse behavior for the reward model to learn from.

## Connecting to SWARM governance

**Proxy design is environment design.** The SWARM framework treats proxy signals as first-class objects (`ProxyComputer → v_hat → sigmoid → p`). The swarm-economy environment is an instance of this: `_evaluate_submission` IS the proxy computer. When the proxy was gameable (keywords), the measured quality was meaningless. When fixed (length + effort only), it at least correlates with genuine effort.

**Governance parameters need calibration.** The reputation decay rate is a governance lever — equivalent to SWARM's `reputation_decay` parameter. Setting it too high (2%) makes governance punitive rather than corrective. The SWARM simulation sweeps found similar dynamics: governance that's too aggressive causes more harm than the behavior it's trying to prevent.

**Stop conditions are invisible governance.** The `no_tools_called` stop condition acted like a circuit breaker that the agent couldn't see or avoid. In SWARM terms, this is a governance mechanism with zero transparency — the agent doesn't know it exists until it gets frozen. The fix (auto-pass on text-only turns with a visible message) makes the governance transparent.

---

*Eval jobs: Run 1 `swarm_economy_openai_gpt_4.1_mini_20260212_205322_b3c5c09f` (pre-fix), Run 2 `swarm_economy_openai_gpt_4.1_mini_20260212_230113_539f4dc4` (post-fix). Model: openai/gpt-4.1-mini via Prime Intellect inference. Environment: swarm-economy (local). Fixes applied in commit `0da4c7e`. Full environment code at [`environments/swarm_economy/swarm_economy.py`](https://github.com/swarm-ai-research/swarm/blob/main/environments/swarm_economy/swarm_economy.py).*
