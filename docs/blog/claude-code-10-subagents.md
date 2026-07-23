---
date: 2026-02-15
description: "And then tried to make them spawn their own. It didn't work."
---

# I Got Claude Code to Spin Up 10 Subagents at Once

*And then tried to make them spawn their own. It didn't work.*

---

When you're running a research simulation framework with 15+ scenarios, dozens of [parameter sweeps](../getting-started/first-scenario.md), and a paper to write, serial execution is the enemy. Claude Code's Task tool can launch specialized subagents — independent processes that run in parallel, each with their own context window. I wanted to see how far that scales.

The answer: 10 concurrent subagents, comfortably. Recursive subagents: hard no.

## Why subagents

A single Claude Code session has one context window. When you ask it to research five files, run three tests, and draft two sections of a paper, it does them sequentially. Each tool call blocks until it returns. For a 30-file codebase that's fine. For a research monorepo with scenarios, sweeps, metrics, and documentation — you're leaving parallelism on the table.

Subagents fix this. Each Task tool call launches an independent agent with its own context, tools, and execution thread. The parent agent fires them off and collects results. Think `Promise.all()` but for AI research assistants.

## The 10-agent pattern

Here's what 10 concurrent subagents look like in practice. We used this to parallelize a full study — [scenario design](https://github.com/swarm-ai-research/swarm-artifacts/blob/main/papers/research_swarm_governance.md), parameter sweeps, metric auditing, and writeup — across specialized agents:

| Agent | Role | Tools used |
|-------|------|------------|
| Scenario Architect | Design experiment YAML | Read, Write, Grep |
| Mechanism Designer | Propose governance levers | Read, Grep, WebSearch |
| Metrics Auditor | Validate metric definitions | Read, Grep, Bash |
| Adversary Designer | Design evasive strategies | Read, Write |
| Reproducibility Sheriff | Verify run artifacts | Bash, Read, Glob |
| Explore (codebase) | Map architecture | Grep, Glob, Read |
| Explore (literature) | Find related work | WebSearch, WebFetch |
| Bash (test runner) | Run pytest suite | Bash |
| Bash (scenario runner) | Execute simulation | Bash |
| General-purpose (writer) | Draft paper sections | Read, Write, Grep |

All 10 launch from a single parent message. The parent sends one response containing 10 `Task` tool calls. Claude Code's runtime executes them concurrently. Results stream back as each agent finishes.

```
Parent context
├── Task: Scenario Architect    ──→ returns design
├── Task: Mechanism Designer    ──→ returns lever analysis
├── Task: Metrics Auditor       ──→ returns audit report
├── Task: Adversary Designer    ──→ returns attack vectors
├── Task: Repro Sheriff         ──→ returns artifact check
├── Task: Explore (code)        ──→ returns architecture map
├── Task: Explore (literature)  ──→ returns related work
├── Task: Bash (tests)          ──→ returns test results
├── Task: Bash (scenarios)      ──→ returns run output
└── Task: General (writer)      ──→ returns draft sections
```

The wall-clock time is bounded by the slowest agent, not the sum. In our case, the scenario runner (which actually executes Python simulations) was the bottleneck at ~90 seconds. Everything else finished in 15–45 seconds. Total time: ~90 seconds for what would have been ~8 minutes serial.

## What works well

**Independent research tasks.** If you need to simultaneously search for related work, audit your metrics, and check reproducibility — these have zero dependencies. Fire all three at once.

**Specialized agent types.** Claude Code ships several agent types (`Explore`, `Plan`, `Bash`, `Scenario Architect`, `Metrics Auditor`, etc.) that carry domain-specific system prompts. The Metrics Auditor knows to check for metric drift and consistent logging. The Reproducibility Sheriff knows to verify that plots match committed data. Specialization means each agent does its job better than a general-purpose prompt would.

**Background execution.** Agents can run in the background (`run_in_background: true`), freeing the parent to continue working. You check results later by reading the output file. This is useful when you want to kick off a long simulation while continuing to edit code.

**Resumable agents.** Each agent returns an ID. You can resume it later with full context preserved — useful for multi-turn research where you need to ask follow-up questions about an agent's findings.

## What doesn't work: recursive subagents

Here's where it gets interesting. If subagents are good, subagents spawning their own subagents should be better, right? A tree of agents, each decomposing its task further, like a recursive divide-and-conquer.

We tried it. It doesn't work.

**The failure mode:** Subagents do not have access to the `Task` tool. When an agent launched via `Task` tries to call `Task` itself, the tool simply isn't available. The recursion is blocked at depth 1.

This is a deliberate design choice, not a bug. Here's why it makes sense:

1. **Resource explosion.** If each of 10 agents spawned 10 more, you'd have 100 concurrent API calls. At depth 3, that's 1,000. The cost and rate-limit implications are obvious.

2. **Context isolation.** Subagents can't see each other's work. A recursive tree would need coordination between siblings and cousins — passing intermediate results, avoiding duplicate work, merging conflicting outputs. The current flat model avoids this entirely.

3. **Debuggability.** When something goes wrong with 10 flat agents, you read 10 output files. With a recursive tree, you're tracing execution across an exponential number of nodes. The complexity isn't worth it for most tasks.

**The workaround:** If you genuinely need multi-level decomposition, do it in the parent. The parent agent reads results from the first wave of subagents, synthesizes them, then launches a second wave based on what it learned. This gives you depth-2 execution with full parent coordination:

```
Wave 1: Research + Explore + Audit (parallel)
         ↓ parent synthesizes ↓
Wave 2: Write + Test + Verify (parallel)
```

This "wave" pattern covers 90% of use cases that would otherwise tempt you toward recursion.

## Practical tips

**Match agent type to task.** Don't use a general-purpose agent for file search (use `Explore`) or for running commands (use `Bash`). The specialized agents have tuned system prompts and restricted tool sets that keep them focused.

**Keep prompts self-contained.** Subagents start with a blank context. They can't see your conversation history (unless they have "access to current context"). Include all necessary context in the prompt — file paths, expected formats, success criteria.

**Use `run_in_background` for long tasks.** If an agent needs to run a simulation that takes 2+ minutes, run it in background and check later. Don't block your main session waiting for it.

**Collect results before acting.** Don't launch 10 agents and immediately start writing code based on assumptions about what they'll find. Wait for results, then act. The `TaskOutput` tool lets you poll or block on specific agents.

**Parallel creation of structured work.** When creating many issues, tasks, or files, launch a subagent per item. We use this pattern to create 10+ beads issues simultaneously — each subagent runs `bd create` independently.

## The numbers

For our research workflow (distributional safety simulations across multiple governance scenarios):

| Metric | Serial | 10 subagents | Speedup |
|--------|--------|--------------|---------|
| Full study (research → paper) | ~25 min | ~6 min | 4.2x |
| Parameter sweep analysis | ~8 min | ~2 min | 4x |
| Codebase audit | ~5 min | ~1.5 min | 3.3x |
| Multi-scenario comparison | ~12 min | ~3 min | 4x |

The speedup isn't 10x because: (a) some tasks have dependencies and run in waves, (b) the parent needs time to synthesize results between waves, and (c) the slowest agent in each wave bounds the wall-clock time. But 3–4x on real research workflows is significant.

## What's next

The flat fan-out pattern with wave-based sequencing covers our needs today. But the design space is interesting:

- **Agent-to-agent communication** would enable pipeline patterns without parent bottlenecks
- **Shared scratch space** would let agents build on each other's intermediate work
- **Recursive spawning with budgets** (e.g., "you may spawn up to 3 sub-tasks, total depth 2") could enable controlled decomposition

For now, 10 flat agents with wave coordination is the sweet spot. It turns a 25-minute serial research session into a 6-minute parallel one — and the failed recursion experiment taught us that the constraint is actually a feature.

## Reproduce it

This repo uses subagent orchestration throughout. See:

- `.claude/agents/` — Specialized agent definitions (Scenario Architect, Metrics Auditor, etc.)
- `.claude/commands/full_study.md` — Full study command that orchestrates multiple agents
- `.claude/commands/sweep.md` — Parameter sweep using parallel Bash agents
- `AGENTS.md` — Role selection guidance for agent specialization
