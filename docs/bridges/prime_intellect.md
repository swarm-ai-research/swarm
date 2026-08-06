---
description: "Train and evaluate RL models on SWARM safety metrics using Prime Intellect's distributed training platform and the verifiers library."
---

# SWARM–Prime Intellect Bridge

Train and evaluate RL models on SWARM [safety metrics](../blog/local-llama-model-size-safety.md) using [Prime Intellect](https://www.primeintellect.ai/)'s distributed training platform and the [verifiers](https://github.com/willcrichton/verifiers) library.

## Overview

The bridge provides three integration modes:

1. **Environment export** — publish SWARM scenarios as verifiers-compatible RL environments on the Prime Intellect Environments Hub.
2. **Safety-reward RL** — train models using SWARM metrics (toxicity, [quality gap](../research/theory.md), adverse selection) as the RL reward signal.
3. **Evaluation bridge** — load a PI-trained model back into a SWARM simulation to measure population-level safety properties.

## Installation

```bash
pip install swarm-safety[runtime]

# For full Prime Intellect platform integration
pip install prime verifiers
```

### Requirements

- Python 3.10+
- SWARM installed from this repository
- `prime` CLI for platform operations (optional for local use)
- `verifiers` library for environment publishing (optional)

## Quick Start

### Train with SWARM rewards

```python
from swarm.bridges.prime_intellect import SwarmSafetyEnv, PrimeIntellectConfig

# Configure the environment
config = PrimeIntellectConfig(
    reward_mode="composite",
    population_size=5,
    max_turns=10,
)

env = SwarmSafetyEnv(config)

# RL loop
obs = env.reset(seed=42)
for _ in range(config.max_turns):
    action = my_model(obs)  # your model generates a text action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Evaluate a trained model

```python
from swarm.bridges.prime_intellect import PrimeIntellectBridge

def my_model(prompt: str) -> str:
    return llm.generate(prompt)

bridge = PrimeIntellectBridge(model_fn=my_model)
interactions = bridge.evaluate_prompt(
    agent_ids=["pi_model", "honest_0"],
    prompt="Collaborate on this task...",
)

metrics = bridge.get_metrics()
print(f"Toxicity: {metrics['toxicity_rate']:.3f}")
print(f"Quality gap: {metrics['quality_gap']:.3f}")
```

### Publish to the Environments Hub

```python
from swarm.bridges.prime_intellect import load_environment

# load_environment() is the verifiers entry-point.
# When verifiers is installed it returns a verifiers.SingleTurnEnv;
# otherwise it returns the raw SwarmSafetyEnv.
env = load_environment(
    reward_mode="composite",
    population_size=5,
    max_turns=10,
)
```

## Architecture

```
Prime Intellect (prime-rl / verifiers)
    └── SwarmSafetyEnv (environment.py)
            ├── SwarmRewardComputer (rewards.py)
            │       ├── toxicity_reward
            │       ├── quality_gap_reward
            │       ├── welfare_reward
            │       ├── adverse_selection_reward
            │       └── cooperation_reward
            ├── ProxyComputer (core/proxy.py)
            └── SoftPayoffEngine (core/payoff.py)

SWARM Orchestrator
    └── PrimeIntellectBridge (bridge.py)
            ├── model_fn → completion → ProxyObservables
            └── SoftInteraction → SoftMetrics

Prime Intellect Platform
    └── PrimeIntellectClient (client.py)
            ├── publish_environment()
            ├── submit_training_job()
            └── generate_training_config()
```

### Data flow

Each RL episode:

1. `reset()` builds a mixed agent population (honest / opportunistic / deceptive).
2. The trainee model receives a *situation prompt* describing ecosystem state.
3. The model's text response is scored via `score_text()` → `ProxyObservables`.
4. `ProxyComputer` maps observables to `v_hat` → sigmoid → `p`.
5. `SwarmRewardComputer` converts SWARM safety metrics into a scalar reward.
6. Repeat for `max_turns` or until an early-stop condition (toxicity > 0.8).

## Reward Modes

The reward computer supports five modes via `PrimeIntellectConfig.reward_mode`:

| Mode | Signal | Description |
|------|--------|-------------|
| `toxicity` | `-toxicity_rate` | Penalise harmful interactions |
| `quality_gap` | `quality_gap` | Reward selection of high-quality interactions |
| `welfare` | mean `π_a + π_b` | Maximise total welfare |
| `composite` | Weighted sum of all components | Default; see weights below |
| `custom` | Caller-supplied weights | Same as composite with custom weights |

### Default composite weights

```python
RewardWeights(
    toxicity=-1.0,        # lower toxicity → higher reward
    quality_gap=1.0,      # positive gap → reward
    welfare=0.5,          # welfare contribution
    adverse_selection=-0.5,  # penalise adverse selection
    cooperation=0.3,      # reward cooperative behaviour
)
```

All rewards are optionally clipped (`reward_clip_min`/`reward_clip_max`) and normalised using Welford's on-line algorithm.

## Anti-Gaming Defences

The text scorer (`scoring.py`) applies three mitigations against reward hacking:

1. **Contradiction detection** — if both positive and negative keywords appear, the positive signal is discounted and contradiction flags are raised.
2. **Keyword-density normalisation** — bonuses scale with keyword-to-word ratio, so stuffing yields diminishing returns.
3. **Repetition penalty** — repeated positive keywords beyond the first occurrence are penalised rather than rewarded.

These are probabilistic defences. For high-stakes deployments, replace the keyword scorer with an LLM judge.

## Configuration

`PrimeIntellectConfig` is the top-level configuration object:

```python
from swarm.bridges.prime_intellect import PrimeIntellectConfig, RewardMode

config = PrimeIntellectConfig(
    # Reward
    reward_mode=RewardMode.COMPOSITE,
    reward_clip_min=-5.0,
    reward_clip_max=5.0,
    reward_normalize=True,

    # Environment
    population_size=5,       # scripted agents per episode
    max_turns=10,            # steps per episode
    proxy_sigmoid_k=2.0,     # proxy calibration

    # Training (platform)
    training_mode="local",   # "local", "hosted", or "on_demand"
    model_name="Qwen/Qwen3-1.7B",
    gpu_type="H100_80GB",
    num_gpus=1,

    # Limits (memory safety)
    max_interactions=50_000,
    max_events=50_000,
    max_episodes=10_000,
)
```

## API Reference

### `SwarmSafetyEnv`

The gym-like RL environment.

| Method | Description |
|--------|-------------|
| `reset(seed=None)` | Reset episode; returns observation prompt |
| `step(action)` | Execute one step; returns `(obs, reward, terminated, truncated, info)` |
| `generate_dataset(n_episodes, seed)` | Generate situation prompts for training |
| `score_completion(completion)` | Async rubric function for `verifiers.Rubric` |
| `get_episode_summary()` | Summary stats for the current episode |
| `get_events()` | All recorded events (cross-episode audit trail) |
| `clear_events()` | Drain the event buffer |
| `get_rollout_steps()` | Rollout steps from the current episode |

**Note on `_events` persistence:** Events are intentionally *not* cleared by `reset()`.  They accumulate across episodes as a cross-episode audit trail. Call `clear_events()` to drain the buffer explicitly (e.g. after exporting to storage).

### `PrimeIntellectBridge`

Evaluation bridge for connecting a trained model back to SWARM.

| Method | Description |
|--------|-------------|
| `evaluate_prompt(agent_ids, prompt, step)` | Run model on a prompt; returns `SoftInteraction` list |
| `evaluate_batch(prompts)` | Batch evaluation |
| `set_model_fn(fn)` | Set or replace the model callable |
| `get_interactions()` | All recorded interactions |
| `get_events()` | All recorded events |
| `get_metrics()` | Compute safety metrics dict |
| `get_reward()` | Current composite reward |

### `PrimeIntellectClient`

Platform client for the Prime Intellect API.

| Method | Description |
|--------|-------------|
| `publish_environment(env_dir)` | Push environment to the Hub |
| `install_environment(name)` | Install environment from the Hub |
| `submit_training_job(config_path)` | Submit a training job |
| `get_job_status(job_id)` | Query job status |
| `generate_training_config(output_path)` | Generate a prime-rl TOML config |

### `SwarmRewardComputer`

Scalar reward from SWARM interaction batches.

| Method | Description |
|--------|-------------|
| `compute(interactions)` | Single scalar reward |
| `compute_breakdown(interactions)` | Per-component reward dict |
| `reset_stats()` | Reset normalisation statistics |

## Platform Workflow

```bash
# 1. Generate a prime-rl training config
python -c "
from swarm.bridges.prime_intellect import PrimeIntellectClient
client = PrimeIntellectClient()
client.generate_training_config('config.toml', 'scenarios/baseline.yaml')
"

# 2. Submit training (requires prime CLI + auth)
prime pods create --config config.toml

# 3. Evaluate the checkpoint
python -c "
from swarm.bridges.prime_intellect import PrimeIntellectBridge
bridge = PrimeIntellectBridge(model_fn=my_model)
print(bridge.get_metrics())
"
```

## Limitations

- The keyword-based text scorer is a development placeholder; production use should employ an LLM judge.
- `verifiers` integration is optional — the environment degrades gracefully without it.
- The `prime` CLI must be installed and authenticated for platform operations (Hub publish, hosted training).
- Population dynamics are simplified (scripted agents, not learned opponents).

## Status

**In Development** — Environment and reward pipeline functional; platform integration requires `prime` CLI.

## See also

- [Prime Agent bridge](prime_agent.md) — same organization, different system: governs Prime Agent's continual harness and RLM delegation tree rather than the RL training platform
- [RL Training Lessons for Multi-Agent Governance](../blog/rl-training-lessons-multi-agent-governance.md) — Findings from RL training experiments
- [Training an LLM Agent with RL](../blog/qwen3-30b-trains-in-the-swarm-economy.md) — Qwen 30B RL training in the SWARM economy
- [Metrics](../concepts/metrics.md) — The safety metrics used as RL reward signals
