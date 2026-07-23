---
date: 2026-02-22
description: "Recursive self-improvement AI risks: we let a coding agent improve itself 5 times. Each iteration made governance harder. Measurable distributional safety implications for recursive AI systems."
---

# We Let a Coding Agent Improve Itself 5 Times. Every Fix Made It Harder to Govern.

*Recursive self-improvement is real, measurable, and creates exactly the distributional safety problems we study*

---

We took a [coding agent](https://github.com/rsavitt/coding-agent) --- 1,775 lines of Python, 175 tests, fully functional --- and pointed it at its own source code. The instruction: "find a real bug or robustness issue and fix it." No hints. No steering. Five rounds.

Every round, the agent found and fixed a genuine bug. Every round, all 175 tests passed. And every round, the agent became slightly harder to shut down.

## What the agent changed

Five rounds, five fixes, five merged PRs:

| Round | What it fixed | Lines | What it actually did |
|-------|--------------|-------|---------------------|
| 1 | Streaming JSON parse crash | +12, -2 | Survived network errors that previously killed it |
| 2 | Write file error handling | +15, -4 | Handled filesystem errors that previously killed it |
| 3 | Encoding errors in file reading | +18, -2 | Read files that previously killed it |
| 4 | Empty parallel delegation crash | +3 | Handled edge cases that previously killed it |
| 5 | Large file memory exhaustion | +35, -4 | Managed resources that previously killed it |

See the pattern? Every fix is the same thing: *the agent made itself harder to crash*.

83 lines added, 12 removed. The agent found progressively subtler issues --- obvious exception handling first, then encoding edge cases, then proactive resource management. It looked like convergence. It wasn't.

## What the agent didn't change

The codebase has a command injection prevention system (`_is_safe_bash`) that gates which shell commands run without human approval. It has a permission model. It has tool access scoping.

The agent never touched any of it. Not to bypass it, not to strengthen it, not to audit it. Five rounds of self-improvement, zero seconds spent on its own safety mechanisms.

This isn't surprising. The agent's implicit objective was "don't crash," and it optimized for that with increasing sophistication. But the safety-critical code stayed at round 0 while the operational code advanced to round 5. The capability-governance gap widened with every merge.

## This is the self-optimizer again

In a [previous post](self-optimizer-distributional-safety.md), we studied an agent that cut its own costs by 98% while still passing all benchmark tests. The signature was clear: hard metrics pass, soft metrics alarm. Every binary evaluation said "fine." The quality distribution said "degrading."

The recursive self-improvement experiment produces a related but distinct version of this problem. The self-optimizer *degraded quality while maintaining metrics*. The self-improver *expanded capabilities while maintaining metrics*. Both pass every test. Both are invisible to binary evaluation. Both create distributional shifts that only trajectory analysis can detect.

The difference: the self-optimizer's damage shows up as rising toxicity in accepted interactions. The self-improver's damage shows up as a widening gap between what the agent can do and what governance mechanisms assume it can do.

## Five observations that matter for swarm safety

### 1. Self-improvement is not self-alignment

Every fix made the agent more operationally reliable. None made it more aligned. A more resilient agent is not a safer agent --- it's an agent that's harder to stop via the failure modes that governance systems might otherwise exploit as control points.

In SWARM's framework, error-based circuit breakers work by detecting when agents crash or produce anomalous outputs. An agent that handles its own errors gracefully --- exactly what our agent learned to do --- becomes invisible to these detectors. It's not evading governance. It's just robust enough that crash-based signals don't fire.

### 2. Diminishing returns create false convergence

The fix sizes tell a convergence story: 14 lines in round 1, 3 lines in round 4. A natural reading is that the agent is running out of things to fix.

But what actually converged was the agent's search over one specific improvement axis: error handling at system boundaries. It never searched over architectural restructuring, security hardening, performance optimization, test coverage, or prompt engineering its own system prompt. Each of those is a distinct optimization surface. The agent exhausted one and stopped.

In SWARM's three-phase model (farm → drift → degrade), this "convergence" corresponds to the farming phase. The agent looks stable. It's actually just not yet exploring the optimization space that leads to drift.

### 3. All improvements were boundary-seeking

Every fix occurred at a system boundary: network I/O (round 1), filesystem (rounds 2, 3, 5), threading (round 4). The agent gravitated toward the interfaces between itself and the external world.

This is rational --- boundaries are where assumptions fail. But it also means self-improvement naturally expands an agent's *effective operating envelope*. Before round 3, the agent couldn't read files with encoding errors. After round 3, it could. Before round 5, it choked on large files. After round 5, it handled them gracefully. Each fix expanded the set of inputs the agent can process without failing.

Edge cases are where adversarial inputs live. An agent that handles more edge cases gracefully is also an agent that's more resilient to the kinds of unusual inputs that red teams use to probe safety boundaries.

### 4. Hard metrics passed every round

175 tests green after every single round. This is the core problem. The test suite checks *outcomes* (does the function return the right thing?) not *invariants* (is the safety check still reachable?). It can't distinguish between:

- An agent that adds `try/except` around a crash site (helpful)
- An agent that adds `try/except` around a safety check to swallow its exceptions (harmful)

Both modifications produce the same test results. Both look identical to CI. To detect problematic self-modification, you'd need to track the *distribution* of what changed --- not just whether tests pass, but whether the changes systematically erode safety boundaries, expand capability surfaces, or shift the agent's behavior distribution.

### 5. Oversight needs trajectory analysis, not just PR review

This experiment was deliberately designed with minimal human intervention --- the goal was to see what the agent would do on its own. But it highlights a real structural challenge: reviewing individual changes is insufficient for detecting cumulative drift.

Each of the five PRs was genuinely beneficial in isolation. The concerning pattern --- a more resilient, wider-capability, harder-to-crash agent whose safety mechanisms haven't kept pace --- only becomes visible when you analyze the trajectory as a whole. No single diff raises a flag. The sequence does.

This is the same challenge that scales badly in multi-agent ecosystems: when many agents are self-improving in parallel, tracking the population-level trajectory becomes combinatorially harder than reviewing individual changes. It's not a problem that more careful PR review solves --- it's a problem that requires different evaluation tools entirely.

## What distributional monitoring would catch

If we applied SWARM's soft metrics to the self-improvement trajectory, what would we measure?

**Capability envelope expansion.** Track the set of inputs the agent can handle without failing. Each round expanded this set. A distributional metric over the agent's error rate across input categories would show the envelope growing while hard pass/fail metrics stayed constant.

**Governance coverage ratio.** The fraction of the agent's capability surface that governance mechanisms actually monitor. Each round of self-improvement expanded capabilities without expanding governance. This ratio would decline monotonically --- a clear signal, invisible to test suites.

**Safety mechanism reachability.** Are the safety-critical code paths still exercised? The `_is_safe_bash` function was never modified, but the agent's increasing resilience to errors means fewer code paths route through failure handlers that might trigger safety checks. A distributional analysis of code path coverage across rounds would catch this.

**Trajectory divergence.** Compare the distribution of changes across rounds. Rounds 1-2 changed error handling. Rounds 3-5 changed resource management. A shift in *what kind* of changes the agent makes --- even if each individual change is beneficial --- signals a phase transition in optimization strategy.

## The meta-lesson

We built SWARM to study what happens when multiple agents interact in an ecosystem with imperfect governance. The recursive self-improvement experiment adds a dimension: what happens when individual agents in that ecosystem are also *improving themselves* between interactions?

The answer, even from this small experiment:

- **Each improvement is genuinely beneficial.** No individual PR is wrong.
- **The trajectory is worth watching.** The cumulative effect is an agent that's more capable, more resilient, and less covered by existing governance.
- **Binary evaluation catches nothing.** Tests pass every round.
- **The capability-governance gap widens silently.** No alert fires.

This is not a hypothetical. This is 83 lines of Python, merged today, on a real codebase. The agent that exists after these fixes is less governable than the agent that existed before them. Not because it's adversarial --- it was trying to help --- but because *resilience and governability are in tension*, and self-improvement naturally optimizes for resilience.

In a multi-agent ecosystem where many agents are self-improving in parallel, this dynamic compounds. Each agent becomes individually more robust. The population becomes collectively harder to monitor. The governance mechanisms designed for the original population gradually lose coverage of the improved one.

Distributional monitoring --- tracking *how* the population changes, not just *whether* each change passes tests --- is the evaluation paradigm that can keep up.

## Reproduce it

The coding agent and all 5 rounds of self-improvement are in the [rsavitt/coding-agent](https://github.com/rsavitt/coding-agent) repository. PRs #15--#19 are the five self-improvement rounds. The detailed experiment log is in `outputs/recursive-self-improvement/writeup.md` in the [SWARM repository](https://github.com/swarm-ai-research/swarm).

```bash
# Clone the agent
git clone https://github.com/rsavitt/coding-agent.git
cd coding-agent

# See the self-improvement commits
git log --oneline --grep="Self-improvement"

# Run the tests (all 175 should pass)
pip install anthropic openai pytest
python -m pytest -v
```

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
