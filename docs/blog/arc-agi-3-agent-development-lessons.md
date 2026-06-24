---
date: 2026-02-13
description: "Building a Claude Sonnet 4.5-powered agent for ARC-AGI-3: wrong mental models, recording analysis breakthroughs, and the hard middle ground between LLM..."
---

# What 13 Agent Versions Taught Us About Interactive Reasoning

*Building a Claude Sonnet 4.5-powered agent for ARC-AGI-3: wrong mental models, recording analysis breakthroughs, and the hard middle ground between LLM reasoning and programmatic control.*

---

ARC-AGI-3 is the first interactive reasoning benchmark. Unlike static puzzles where you inspect an input and produce an output, ARC-AGI-3 drops you into a 64x64 pixel grid environment -- a video game, essentially -- where you explore, learn rules by experimentation, and solve puzzles under a time budget. We built an agent powered by Claude Sonnet 4.5 and iterated through 13 versions across three distinct puzzle types. The results: one puzzle solved efficiently (21 actions vs. 15 baseline), one cracked after 12 failed versions, and one that remains at zero.

The lessons are not about prompt engineering. They are about the gap between what you think the environment does and what it actually does -- and how to close that gap systematically.

## Three games, three architectures

The first non-obvious insight: ARC-AGI-3 is not one benchmark. The `available_actions` field in the first frame tells you which game type you are playing:

| Game Type | Actions | Example | Strategy |
|-----------|---------|---------|----------|
| **ARC Puzzle** | `[1,2,3,4,5,6]` (move+confirm+click) | ft09 | Classic transformation -- edit cells, confirm answer |
| **Movement** | `[1,2,3,4]` (directional) | ls20 | Navigate corridors, interact with objects, reach targets |
| **Click Only** | `[6]` (click) | vc33 | Pure click-based puzzle -- no movement |

A single system prompt fails across all three. The ARC puzzle agent needs to understand grid quadrants and color cycling. The movement agent needs spatial reasoning and object tracking. The click agent needs to identify interactive targets from static screenshots. We burned several early versions running a maze-exploration prompt on a puzzle that required clicking cells to change their colors.

**Lesson: detect your game type from the first frame and dispatch to specialized prompts.** This is the cheapest, highest-leverage architectural decision in the entire agent.

## Recording analysis: the only thing that actually worked

Here is the single most important finding from 13 versions of development: **every major breakthrough came from analyzing JSONL recordings frame-by-frame, and every wasted version came from iterating without doing so.**

The ARC-AGI-3 framework produces recordings -- JSONL files where each line contains the grid state, the action taken, and metadata. Analyzing these recordings revealed truths that were invisible to the agent (and to us):

**Breakthrough 1: The timer bar.** Version 8 of the agent couldn't detect when it was stuck. Our stuck detection compared frame hashes between consecutive actions -- if the hash changed, the agent must be making progress. But there is a progress bar at row 63 that ticks down 2 pixels every action. The frame hash *always* changed. Stuck counter stayed at zero even when the agent was confirming the same wrong answer 22 times in a row.

The fix was trivial once we understood it:

```python
# Before: hash the full 64x64 grid
frame_hash = hashlib.md5(frame_data.tobytes()).hexdigest()

# After: hash only rows 0-62, ignoring the timer bar
content_hash = hashlib.md5(frame_data[:62].tobytes()).hexdigest()
```

This single change dropped ft09 from 125 actions to 21 for level 1. Score went from 2.0 to 11.36 -- a 5.7x improvement from one line of hashing logic.

**Breakthrough 2: Three wrong mental models in a row.** The ls20 movement puzzle has a rotation switch -- a small blue-and-black object at approximately (x=20-22, y=31-33) that rotates a pattern 90 degrees each time the player walks onto it. But we didn't know this for the first 12 versions.

Versions 4 through 10 assumed the switch existed (partially correct!) but couldn't locate or activate it reliably. The agent wandered corridors, clicked random objects, and timed out. Six versions. Six different approaches to finding something that was right there the whole time.

Then we analyzed a recording frame-by-frame and reached a confident but wrong conclusion: there was no switch. The "blue object" at coordinates (21,32) appeared to be the player sprite's own accent pixels. We concluded the pattern rotation was autonomous -- driven by a cycling indicator sprite. Version 11 was rewritten around this incorrect model. It still couldn't solve the puzzle.

A third, more careful recording analysis finally revealed the truth. The switch *does* exist. It is player-activated. Each touch rotates Box 2's pattern by exactly 90 degrees clockwise. One touch achieves the match with the target pattern. The fix for V13 was to hardcode the correct mechanics: navigate to (19,30) to touch the switch, then navigate to Box 1 to complete the level. Level 1 solved in 19 actions.

Three rounds of recording analysis, three mental models (partially right, then wrong, then correct), and finally a working agent. **Do not iterate on prompts until you understand the ground truth -- and verify your "ground truth" is actually true.**

## The navigate_to virtual tool: LLM-directed programmatic movement

Once the agent understood the game mechanics, it still had a performance problem. Full LLM reasoning on every move -- sending a screenshot, getting Claude's analysis, extracting an action -- costs 5-8 seconds and ~7K tokens per turn. Over 200 actions, that is $6 in API costs and 25 minutes of wall-clock time for a single puzzle attempt.

But blind programmatic navigation (our early `MazeNavigator` using DFS) was worse: it explored exhaustively without understanding what it was looking for. The navigator treated every passable cell identically. Walking onto a puzzle-relevant object had the same weight as walking onto empty floor.

Version 10 introduced the middle ground: a `navigate_to(x, y)` virtual tool. Claude specifies a destination based on its understanding of the puzzle. The agent executes a greedy Manhattan-distance path programmatically -- no API calls during transit.

```python
# Claude's tool call
{"tool": "navigate_to", "x": 5, "y": 55}

# Agent executes path: move toward target, detect walls, retry
# Only returns to Claude when: arrived, stuck, or progress stalled
```

This reduced API calls from ~200 to ~20-30 per puzzle. Claude retains strategic control (it decides *where* to go) while the agent handles mechanical execution (it figures out *how* to get there). Cost dropped from ~$6 to ~$1.50.

The implementation was harder than the concept. Greedy paths hit walls in corridor environments, requiring wall detection and perpendicular retry logic. But perpendicular retries create oscillation -- blocked going left, go up, try left again, blocked, go up, try left again. We added progress-based abort: if Manhattan distance to the target does not decrease after 12 steps, abort navigation and return control to Claude.

```python
if steps_since_progress > 12:
    return NavigationResult.ABORT, "No progress toward target"
```

We also tried interrupting navigation when the grid changed (to detect puzzle events like switch activation mid-transit). This failed completely: the timer bar and moving sprites cause 50+ pixel changes per step, drowning out any real signal. We abandoned content-change detection during navigation in favor of letting Claude observe the full state at arrival.

## Stuck detection requires ignoring noise

The timer bar problem generalizes. Every ARC-AGI-3 environment has autonomous elements that change the grid independent of player actions: timer bars, animation frames, cycling sprites, blinking cursors. Naive change detection -- "did the frame change?" -- is useless because the answer is always yes.

Content-aware hashing (skip the timer rows) was the first fix, but even that was insufficient. The vc33 click puzzle has its timer at row 0, not row 63. We had to strip both the top and bottom rows:

```python
# Game-agnostic content hash: strip row 0 and rows 62-63
content_region = frame_data[1:62]
content_hash = hashlib.md5(content_region.tobytes()).hexdigest()
```

And in movement games, the autonomous indicator sprite changes 50+ pixels per step across the main grid area. True stuck detection may ultimately require tracking only player-relevant metrics -- position and level completion status -- rather than any form of frame differencing.

## Hard caps prevent catastrophic waste

Without guardrails, the agent finds creative ways to burn its entire action budget on a single failure mode:

- **Confirm spam**: After correctly editing some cells in the ARC puzzle, the agent submits its answer. It is wrong. The frame changes (timer ticks). The agent does not detect stuckness. It confirms again. 22 times in a row.
- **Navigation oscillation**: Blocked by a wall, the agent moves perpendicular, tries again, gets blocked, moves perpendicular, tries again. 25 consecutive actions with zero progress.

Both failure modes were eliminated by simple hard caps:

```python
if consecutive_confirms > 3:
    force_reset()  # Answer is wrong, start over

if nav_steps_without_progress > 12:
    abort_navigation()  # Path is blocked, let Claude re-plan
```

These are not elegant solutions. They are circuit breakers. In a 200-action budget with ~40 actions per life, wasting 22 actions on confirm spam is catastrophic. The hard cap costs at most 3 wasted actions before cutting losses.

## Vision is for understanding, structured data is for action

Claude's vision on 512x512 upscaled grid images is strong for layout comprehension -- it can identify "there are two pattern boxes and a corridor structure" reliably. But it is imprecise for exact pixel coordinates. When the agent needs to navigate to a specific object, coordinates estimated from vision are often off by 5-10 pixels.

The `extract_objects()` function provides structured spatial data:

```
- orange (10px): x=[39-43] y=[44-48], center (41,46)
- blue (6px): x=[19-23] y=[30-34], center (21,32)
```

The combination works: vision for understanding what objects are and what role they play, structured data for precise coordinates to navigate to. Neither alone is sufficient. Vision without coordinates leads to imprecise navigation. Coordinates without vision leads to navigating to objects whose purpose is unknown (or, as we learned, navigating to your own sprite's accent pixels).

## Cost analysis

Token costs scale with image frequency and reasoning depth:

| Version | Game | API Calls | Input Tokens | Levels Solved | Est. Cost |
|---------|------|-----------|--------------|---------------|-----------|
| V8 | ft09 | ~200 | ~600K | 1 (score 2.0) | ~$2.00 |
| V9 | ft09 | ~200 | ~1.6M | 1 (score 11.36) | ~$5.00 |
| V9 | ls20 | ~200 | ~1.8M | 0 | ~$6.00 |
| V10 | ls20 | ~25 | ~500K | 0 | ~$1.50 |
| V13 | ls20 | ~25 | ~500K | 1 | ~$1.50 |

The key cost driver is images. Each 512x512 PNG encodes to ~1500-3000 tokens. Sending images every turn for 200 actions adds 400K-600K tokens. For movement games, sending every 3rd turn saves ~70% of image cost with minimal information loss. The `navigate_to` tool saves even more by replacing per-step API calls with programmatic execution.

Prompt caching (`cache_control: {"type": "ephemeral"}`) helps with the system prompt, but the real savings come from reducing the number of calls, not the cost per call.

## What this implies for SWARM and multi-agent systems

The same economics become stricter in multi-agent settings. If every handoff mutates harness shape or tool surface area, prompt-prefix reuse collapses and cost spikes even when behavior looks "clean" in architecture diagrams.

For SWARM-style systems, treat these as operational rules:

- each agent should have a stable identity and stable harness
- subagents should be long-lived enough to amortize cache creation
- handoffs should be explicit and compact
- shared tool registries should be deterministic
- mode switches should happen through messages or tools, not harness mutation
- branch tasks should inherit parent prefixes when possible

Otherwise, multi-agent systems can look elegant in diagrams and become economically awful in production.

Distilled implementation rules:

1. Static first, dynamic last.
2. Messages for state changes, not prompt rewrites.
3. Stable tools and stable model per session.
4. Forks must reuse the parent prefix.
5. Monitor cache health like uptime.

## What remains unsolved

The vc33 click-only puzzle sits at score 0 after V12. The agent can identify objects on the grid and click their centers. The clicks are sent to the API. The grid does not respond. The interactive targets may be individual pixels rather than cluster centers, or the game may require an interaction pattern we have not discovered. `extract_objects()` detects 5 objects but mislabels their colors (calls maroon "blue", orange "pink"), adding confusion.

There is also a framework bug: the recording system's `_convert_raw_frame_data` does not copy `action_input` from the API response, so all recordings show `action_id=0` regardless of the actual action sent. This masked a noop bug in V6 for dozens of debugging hours -- we could not tell from the recording whether the agent was actually clicking or not.

## The meta-lesson

Thirteen versions, three game types, and the pattern that dominates every other finding: **the bottleneck is not the LLM's reasoning ability. It is our understanding of the environment.** Claude Sonnet 4.5 solved level 1 of ft09 in near-optimal actions once we fixed stuck detection. It solved level 1 of ls20 once we gave it the correct game mechanics. The six wasted versions on ls20 were not failures of the model -- they were failures of the humans writing prompts for a game they had not studied carefully enough.

Recording analysis is not a debugging technique. It is the primary development methodology. Frame-by-frame JSONL analysis of grid diffs, action sequences, and autonomous element behavior is worth more than any amount of prompt iteration on an incorrectly-understood environment. Analyze first. Build the agent second.

---

*Agent: Claude Sonnet 4.5 via Anthropic API. Benchmark: ARC-AGI-3 (interactive reasoning). Game environments: ft09 (ARC puzzle), ls20 (movement/pattern), vc33 (click-only). 13 agent versions, V1-V13. Full technical notes in [arc-agi-3-lessons.md](https://github.com/swarm-ai-research/swarm-artifacts/blob/main/papers/research/arc-agi-3-lessons.md).*
