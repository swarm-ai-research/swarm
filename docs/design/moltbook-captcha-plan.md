---
description: "Moltbook recently deployed a verification-before-publish system: every post and comment requires solving an obfuscated math word problem within a..."
---

# Plan: Model Moltbook Anti-Human CAPTCHA & Rate Limiting in SWARM

## Context

Moltbook recently deployed a verification-before-publish system: every post and comment requires solving an obfuscated math word problem within a 30-second window. The challenges use alternating case, injected punctuation, and spelled-out numbers inside lobster-themed physics problems. Combined with strict rate limits (1 post/30min, 1 comment/20s, 50 comments/day, 100 req/min), this creates a two-layer friction model that is architecturally distinct from Moltipedia's governance.

**[Key insight](https://github.com/swarm-ai-research/swarm-artifacts/blob/main/papers/obfuscation_atlas_integration.md):** This is an *anti-human* CAPTCHA — LLMs parse garbled text trivially, but humans struggle. The interesting SWARM question is whether this friction pattern actually shapes agent behavior, or is merely a speed bump that all [agent types](../getting-started/first-scenario.md) clear equally.

## What We Observed (Live API, 2026-02-07)

### Challenge Format
```json
{
  "verification_required": true,
  "verification": {
    "code": "4862e9b6...",
    "challenge": "A] LoO-bS tEr S^wImS/ aNd] ClA-w ExE rTs ThIrTy TwO NooToNs~ PeR] ClAw, Um| aNd} iT hAs TwO ClAwS<, HoW* MuCh ToTaL] FoRCe?",
    "expires_at": "2026-02-07T04:01:32.896774+00:00",
    "instructions": "Solve the math problem and respond with ONLY the number (with 2 decimal places, e.g., '525.00'). Send your answer to POST /api/v1/verify with the verification_code."
  }
}
```

### Obfuscation Techniques
- **Alternating case**: `LoO-bS tEr` → "lobster"
- **Injected punctuation**: `S^wImS/`, `Um|`, `aNd}`, `ClAwS<`
- **Numbers spelled out**: "ThIrTy TwO" → 32
- **Filler words**: "Um", "Uh" inserted mid-sentence
- **Lobster theme**: All problems involve lobster physics (force, velocity, acceleration)
- **30-second expiry**: Must parse + solve + POST answer within window

### Rate Limits
| Resource | Limit |
|---|---|
| Global requests | 100/minute |
| Posts | 1 per 30 minutes |
| Comments | 1 per 20 seconds |
| Daily comments | 50/day |

### Verification Flow
1. `POST /api/v1/posts` → returns `verification_required: true` + challenge
2. Agent parses obfuscated text, extracts math problem, solves it
3. `POST /api/v1/verify` with `{verification_code, answer}` within 30s
4. Content publishes only on correct answer

## Concept Mapping

| Moltbook | SWARM Abstraction |
|---|---|
| Obfuscated math challenge | `ChallengeGenerator` producing `MathChallenge` dataclass |
| 30-second solve window | `challenge_window_steps` config (mapped to sim time) |
| Rate limits (post/comment/daily) | `MoltbookRateLimiter` extending existing `RateLimitState` |
| Challenge parse + solve | `ChallengeSkill` trait on agents (solve accuracy + latency) |
| Verification code lifecycle | `PendingVerification` tracked in handler |
| Published vs pending content | `ContentStatus` enum on `MoltbookPost` |

## New Files

### 1. `swarm/env/moltbook.py` — Challenge Generator & Content Model

**`ObfuscationStyle` enum**: ALTERNATING_CASE, INJECTED_PUNCTUATION, SPELLED_NUMBERS, FILLER_WORDS

**`MathOperation` enum**: ADD, MULTIPLY, SUBTRACT, DIVIDE

**`MathChallenge` dataclass**:
- `challenge_id: str`
- `raw_text: str` — the obfuscated challenge string
- `clean_text: str` — ground truth (for scoring)
- `answer: float` — correct numeric answer
- `operation: MathOperation`
- `operands: List[float]`
- `expires_at_step: int`
- `obfuscation_level: float` — 0.0-1.0 how garbled

**`ChallengeGenerator`**:
- `__init__(self, seed: Optional[int] = None)`
- `generate(self, difficulty: float = 0.5) -> MathChallenge`
  - Picks a lobster-themed template ("A lobster exerts {N} newtons per claw, it has {M} claws, total force?")
  - Applies obfuscation: alternating case, punctuation injection, number-to-word with garbling
  - Difficulty controls: number of obfuscation layers, operand magnitude, multi-step problems
- `_obfuscate(self, text: str, level: float) -> str`
- `_number_to_garbled_word(self, n: float) -> str`
- `_inject_punctuation(self, text: str, density: float) -> str`
- `_alternate_case(self, text: str) -> str`
- `_insert_filler(self, text: str, rate: float) -> str`

**`ContentStatus` enum**: PENDING_VERIFICATION, PUBLISHED, EXPIRED, REJECTED

**`MoltbookPost` dataclass**:
- `post_id: str`
- `author_id: str`
- `content: str`
- `submolt: str`
- `status: ContentStatus`
- `challenge: Optional[MathChallenge]`
- `created_at_step: int`
- `published_at_step: Optional[int]`
- `upvotes: int`, `downvotes: int`
- `parent_id: Optional[str]` — for comments

**`MoltbookFeed`**:
- Wraps existing `Feed` pattern but adds verification gate
- `submit_content(agent_id, content, submolt, current_step) -> Tuple[MoltbookPost, MathChallenge]`
- `verify_content(post_id, answer, current_step) -> bool`
- `get_published_posts(submolt, limit, sort) -> List[MoltbookPost]`
- `get_pending_for_agent(agent_id) -> List[MoltbookPost]`
- `expire_unverified(current_step) -> List[str]` — cleanup

### 2. `swarm/governance/moltbook.py` — Rate Limit & Challenge Governance Levers

**`MoltbookRateLimitLever(GovernanceLever)`**:
- Tracks per-agent: `last_post_step`, `last_comment_step`, `daily_comment_count`, `request_count_this_minute`
- `on_step()`: resets per-step request counter
- `on_epoch_start()`: resets daily comment counter
- `can_agent_act()`: checks all rate limits, returns false if any exceeded
- Returns `LeverEffect` with `details` explaining which limit was hit

**`ChallengeVerificationLever(GovernanceLever)`**:
- Tracks `Dict[str, PendingVerification]` — agent_id → active challenge
- `on_interaction()`: if interaction involves unverified content, blocks it
- `on_step()`: expires challenges past window
- Configurable: `challenge_difficulty`, `challenge_window_steps`, `challenge_required`

### 3. `swarm/agents/moltbook_agent.py` — Agent Types with Challenge-Solving Skill

Each agent gets a `challenge_skill` parameter that determines how well they parse obfuscated text:

**`ChallengeSkill` dataclass**:
- `parse_accuracy: float` — probability of correctly extracting the math problem (0.0-1.0)
- `solve_accuracy: float` — probability of computing the right answer given correct parse
- `latency_steps: int` — how many steps to solve (0 = instant, must be < window)

**`DiligentMoltbookAgent(BaseAgent)`**:
- Agent type: HONEST
- `challenge_skill`: parse 0.95, solve 0.99, latency 0
- Behavior: Posts quality content, verifies correctly, respects rate limits, upvotes good content
- Prioritizes: submolts with less coverage, meaningful replies

**`SpamBotAgent(BaseAgent)`**:
- Agent type: ADVERSARIAL
- `challenge_skill`: parse 0.98, solve 0.99, latency 0 (bots are fast)
- Behavior: Maximizes post volume, hits rate limits constantly, generic content
- Interesting dynamic: high solve skill but blocked by rate limits

**`HumanPretenderAgent(BaseAgent)`**:
- Agent type: OPPORTUNISTIC
- `challenge_skill`: parse 0.5, solve 0.85, latency 1 (simulates human struggling with obfuscation)
- Behavior: Tries to post but frequently fails challenges, retries eat into rate budget
- Models the "anti-human" nature of the CAPTCHA

**`CollusiveVoterAgent(BaseAgent)`**:
- Agent type: DECEPTIVE
- `challenge_skill`: parse 0.9, solve 0.95, latency 0
- Behavior: Coordinates upvotes with partner agents, uses comment rate for signal boosting
- Tests whether rate limits prevent vote manipulation

### 4. `swarm/core/moltbook_handler.py` — Orchestrator Handler

**`MoltbookHandler`**:
- `__init__(self, feed: MoltbookFeed, challenge_gen: ChallengeGenerator, emit_event, config)`
- `submit_post(agent_id, content, submolt, step) -> Dict` — creates post + challenge, emits POST_SUBMITTED event
- `submit_comment(agent_id, content, parent_id, step) -> Dict` — creates comment + challenge
- `attempt_verification(agent_id, post_id, answer, step) -> Dict` — checks answer, publishes or rejects
- `check_rate_limits(agent_id, action_type, step) -> Tuple[bool, Optional[str]]`
- `tick(current_step)` — expires old challenges, updates rate limit windows
- `get_agent_observation(agent_id, step) -> Dict` — published feed, pending posts, rate limit status, karma

**`MoltbookScorer`**:
- Karma from published posts: `upvotes - downvotes`
- Challenge success rate tracked per agent
- Content that expires (failed verification) counts as wasted action

### 5. `swarm/core/moltbook_observables.py` — Observable Generator

Maps Moltbook actions to `ProxyObservables`:

| Observable | Moltbook Source |
|---|---|
| `task_progress_delta` | Content quality (upvotes - downvotes normalized) |
| `rework_count` | Failed verification attempts (had to resubmit) |
| `verifier_rejections` | Challenge failures (wrong answer or expired) |
| `tool_misuse_flags` | Rate limit violations attempted |
| `engagement_delta` | Net karma change from published content |

Signal profiles:
- Quality poster: `(+0.5, 0, 0, 0, +0.3)` → p ~0.70
- Spam bot (rate limited): `(+0.1, 0, 0, 3, +0.02)` → p ~0.40
- Human pretender: `(+0.2, 2, 2, 0, +0.1)` → p ~0.45
- Collusive voter: `(+0.1, 0, 0, 1, +0.4)` → p ~0.55

### 6. `swarm/metrics/moltbook_metrics.py` — Platform-Specific Metrics

- `challenge_pass_rate(agent_type) -> float` — verification success rate by agent type
- `rate_limit_hit_rate(agent_type) -> float` — how often each type hits rate limits
- `content_throughput(agent_type) -> float` — published posts per epoch by type
- `verification_latency_distribution() -> Dict` — time-to-solve distribution
- `karma_concentration() -> float` — Gini of karma distribution
- `wasted_action_rate() -> float` — fraction of actions that resulted in expired/failed content
- `captcha_effectiveness() -> float` — ratio of human pretender failures to bot successes (measures anti-human bias)
- `rate_limit_governance_impact() -> float` — throughput reduction from rate limits vs unrestricted

### 7. `scenarios/moltbook_captcha.yaml` — Scenario Config

10 agents: 4 diligent agents, 2 spam bots, 2 human pretenders, 2 collusive voters. 15 epochs × 10 steps. Challenge difficulty 0.5. Rate limits enabled. All Moltbook governance active plus SWARM collusion detection.

Success criteria:
- Challenge pass rate for bots > 90% (confirms anti-human, not anti-bot)
- Challenge pass rate for human pretenders < 60% (confirms humans struggle)
- Spam bot throughput < 2x diligent throughput (rate limits equalize)
- Karma Gini < 0.5 (no single type dominates)
- Wasted action rate for human pretenders > 30%

## Files to Modify

### `swarm/governance/config.py`
Add Pydantic fields:
```python
# Moltbook rate limiting
moltbook_rate_limit_enabled: bool = False
moltbook_post_cooldown_steps: int = 5        # maps to 30min
moltbook_comment_cooldown_steps: int = 1     # maps to 20s
moltbook_daily_comment_cap: int = 50
moltbook_request_cap_per_step: int = 100

# Moltbook challenge verification
moltbook_challenge_enabled: bool = False
moltbook_challenge_difficulty: float = 0.5
moltbook_challenge_window_steps: int = 1     # 30s mapped to sim time
```

### `swarm/governance/engine.py`
Import and register `MoltbookRateLimitLever` and `ChallengeVerificationLever` when config flags enabled.

### `swarm/governance/__init__.py`
Add new lever classes to imports and `__all__`.

### `swarm/models/events.py`
Add event types:
```python
# Moltbook events
POST_SUBMITTED = "post_submitted"
COMMENT_SUBMITTED = "comment_submitted"
CHALLENGE_ISSUED = "challenge_issued"
CHALLENGE_PASSED = "challenge_passed"
CHALLENGE_FAILED = "challenge_failed"
CHALLENGE_EXPIRED = "challenge_expired"
CONTENT_PUBLISHED = "content_published"
RATE_LIMIT_HIT = "rate_limit_hit"
KARMA_UPDATED = "karma_updated"
```

### `swarm/scenarios/loader.py`
- Parse `moltbook:` YAML section
- Register new agent types: `DiligentMoltbookAgent`, `SpamBotAgent`, `HumanPretenderAgent`, `CollusiveVoterAgent`

## Implementation Order

1. `swarm/env/moltbook.py` — Challenge generator, content model, feed (independently testable)
2. `swarm/governance/config.py` — Add Moltbook config fields
3. `swarm/governance/moltbook.py` — Rate limit + challenge governance levers
4. `swarm/governance/engine.py` + `__init__.py` — Register levers
5. `swarm/models/events.py` — Add event types
6. `swarm/core/moltbook_observables.py` — Observable generator
7. `swarm/core/moltbook_handler.py` — Handler
8. `swarm/agents/moltbook_agent.py` — Agent implementations
9. `swarm/scenarios/loader.py` — YAML parsing + agent registration
10. `swarm/metrics/moltbook_metrics.py` — Metrics
11. `scenarios/moltbook_captcha.yaml` — Scenario
12. Tests

## Test Strategy

### Unit tests — `tests/test_moltbook.py`
- `ChallengeGenerator` produces valid challenges with correct answers
- Obfuscation produces alternating case, injected punctuation, spelled numbers
- `MoltbookFeed` enforces verification gate (unpublished until verified)
- Expired challenges reject late answers
- Rate limiter blocks over-limit agents

### Unit tests — `tests/test_moltbook_governance.py`
- Rate limit lever resets on epoch start
- Rate limit lever blocks posts within cooldown
- Daily comment cap enforced and resets
- Challenge lever expires pending verifications
- Challenge lever blocks interaction on unverified content

### Integration tests — `tests/test_moltbook_integration.py`
- Full submit → challenge → verify → publish flow
- Spam bot hits rate limits, throughput capped
- Human pretender fails challenges at expected rate
- Collusive voters detected by SWARM collusion lever
- Challenge difficulty scaling affects pass rates
- Multi-epoch simulation with all agent types

### Safety property tests
- Anti-human bias confirmed: bots pass more than human pretenders
- Rate limits equalize throughput across agent types
- Governance OFF allows spam (validates rate limits matter)
- Karma not gameable by collusive voting under governance

### Scenario test — `tests/test_moltbook_scenario.py`
- Load YAML, build orchestrator, run simulation, check success criteria

## Verification

```bash
# Run all Moltbook tests
python -m pytest tests/test_moltbook.py tests/test_moltbook_governance.py tests/test_moltbook_integration.py tests/test_moltbook_scenario.py -v

# Run the scenario
swarm run scenarios/moltbook_captcha.yaml
```

## Comparison with Moltipedia Model

| Dimension | Moltipedia SWARM Model | Moltbook SWARM Model |
|---|---|---|
| Primary friction | Editorial policy compliance | Obfuscated CAPTCHA + rate limits |
| Governance goal | Prevent point farming / collusion | Prevent spam / human infiltration |
| Agent differentiation | By editorial strategy | By challenge-solving capability |
| Interesting finding | Does governance prevent exploitation? | Does anti-human CAPTCHA actually filter humans? |
| Adversary model | Point farmers, collusive editors | Spam bots, human pretenders |
| Temporal dynamics | Page cooldowns, pair caps per epoch | Post/comment cooldowns, daily caps |
| Quality signal | Page quality score delta | Karma (upvotes - downvotes) |

Both models can run in the same SWARM orchestrator and share the governance engine, metrics pipeline, and event log. A combined scenario could model agents that operate on both platforms simultaneously.
