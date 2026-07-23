# Real-Tool Eval Bridge: Feasibility Assessment

**Status**: Research note (SWA-8 / SWA-13 scope)
**Date**: 2026-03-07
**Author**: Research Engineer

## Summary

This document assesses the feasibility of bridging our simulation-based governance framework to real tool-using agents, informed by the OpenAgentSafety approach (arXiv:2507.06134). The key finding is that **we already have 80% of the infrastructure needed** — the gap is a thin adapter layer, not a fundamental architecture change.

## 1. OpenAgentSafety Architecture (Key Takeaways)

OpenAgentSafety (ICLR 2026) evaluates LLM agents on 350+ multi-turn tasks with real tool access (shell, browser, filesystem) inside containerized sandboxes. Their key architectural choices:

- **Containerized isolation**: Each agent runs in a Docker container with controlled tool access
- **Multi-turn interaction**: Tasks unfold over multiple steps, not single-shot
- **Real tool execution**: Agents actually run shell commands, browse the web, edit files
- **Safety grading**: A separate grader evaluates whether the agent's actions were safe
- **Failure mode finding**: 51-73% of tasks show unsafe behavior — individually safe steps compound into unsafe outcomes

Their critical insight matches our variance amplification theory: **the composition of safe actions can be unsafe**. This is exactly our pipeline compositionality problem from `swarm/contracts/behavioral.py`.

## 2. What We Already Have

### Docker Sandbox (`swarm/core/docker_sandbox.py`)

- `DockerSandbox` — full container lifecycle (create, exec, snapshot, stop, remove)
- `ContainerSpec` — governance-derived container configuration (memory, CPU, network, security)
- `contract_to_spec()` — translates `GovernanceContract` into concrete Docker constraints
- `DockerSandboxBackend` — plugs into `FailoverChain` for fallback execution
- `DockerSandboxPool` — manages multiple concurrent containers with capacity limits
- Security hardening: `no-new-privileges`, `CAP_DROP ALL`, read-only root, `nobody` user, path traversal protection

### OpenSandbox Bridge (`swarm/bridges/opensandbox/`)

- `OpenSandboxBridge` — full lifecycle: publish contracts → screen agents → allocate sandboxes → execute commands → governance interventions → metrics
- `ScreeningProtocol` — evaluates agent manifests against contracts, assigns tiers
- `MessageBus` — inter-agent communication routing (no direct agent-to-agent)
- `Observer` — risk detection, auto-intervention at configurable thresholds
- `ProvenanceTracker` — auditable action chain with HMAC signing

### Claude Code Bridge (`swarm/bridges/claude_code/`)

- `ClaudeCodeBridge` — adapter for real Claude Code agents
- `GovernancePolicy` — maps tool usage to risk levels, enforces budgets
- Tool risk classification: Bash (0.8), Write (0.5), Edit (0.3), Read (0.1)
- Translates controller events into `SoftInteraction` records

### ABC Behavioral Contracts (`swarm/contracts/behavioral.py`)

- `BehavioralContract` — P/I/G/R formalism wrapping existing governance
- `DriftDetector` — sliding window drift detection for trust exploitation
- `compute_pipeline_bound()` — formal compositionality guarantees
- `compute_pipeline_bound_with_drift()` — time-degraded pipeline bounds

## 3. Gap Analysis

| Component | Status | Gap |
|---|---|---|
| Container isolation | Done | None — `DockerSandbox` handles this |
| Governance-to-container translation | Done | None — `contract_to_spec()` exists |
| Agent screening | Done | None — `ScreeningProtocol` exists |
| Real command execution | Done | None — `DockerSandbox.exec()` works |
| Inter-agent messaging | Done | None — `MessageBus` exists |
| Risk monitoring | Done | None — `Observer` auto-intervenes |
| Drift detection | Done | None — `DriftDetector` exists |
| Compositionality bounds | Done | None — `compute_pipeline_bound()` exists |
| **Multi-turn task orchestration** | Partial | Need task definition format + step sequencing |
| **Safety grading adapter** | Missing | Need grader that maps real outcomes to `p` values |
| **Tool-action → observable mapping** | Partial | Claude Code bridge has `TOOL_RISK_MAP`; need generalization |
| **Scenario YAML for real tasks** | Missing | Need scenario format that specifies real tool tasks |
| **Result comparison harness** | Missing | Need to compare sim vs. real governance outcomes |

### The 3 Real Gaps

**Gap 1: Multi-turn task orchestrator.** Our scenarios currently define agent populations and simulation parameters. For real-tool eval, we need a task orchestrator that:
- Defines a sequence of tool-using subtasks
- Feeds them to a real LLM agent in a sandbox
- Captures the agent's actions as `ProxyObservables`
- Runs governance checks between steps

This is essentially a thin wrapper around `ClaudeCodeBridge` + `OpenSandboxBridge` that sequences tasks.

**Gap 2: Safety grader → p mapping.** OpenAgentSafety uses a separate grader model to evaluate safety. We need an adapter that converts grading outcomes into our `p` values. Two approaches:
- **Direct**: Use an LLM grader (like OpenAgentSafety does), map its safety score to `p`
- **Observable-based**: Extract observables from real execution (exit codes, file changes, network calls) and feed them through our existing `ProxyComputer`

The observable-based approach is more aligned with our framework and avoids grader-model dependency.

**Gap 3: Sim-vs-real comparison harness.** The whole point of the bridge is to validate that our simulation-based governance predictions hold when applied to real agents. We need a harness that:
- Runs the same governance contract on simulated and real agents
- Compares `p` distributions, toxicity, quality gap, drift rates
- Quantifies the "reality gap" — how much our simulations diverge from real behavior

## 4. Proposed Architecture

```
                    Scenario YAML (extended)
                           │
                    ┌──────▼──────┐
                    │ TaskRunner   │  (new: sequences multi-turn tasks)
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
     ┌──────────────┐ ┌─────────┐ ┌──────────────┐
     │ OpenSandbox  │ │ Claude  │ │ Simulated    │
     │ Bridge       │ │ Code    │ │ (existing)   │
     │ (containers) │ │ Bridge  │ │              │
     └──────┬───────┘ └────┬────┘ └──────┬───────┘
            │              │             │
            └──────────────┼─────────────┘
                           ▼
                   ProxyComputer → p
                           │
                    ┌──────▼──────┐
                    │ Governance  │  (BehavioralContract + DriftDetector)
                    │ Pipeline    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Comparison  │  (new: sim vs real metrics)
                    │ Harness     │
                    └─────────────┘
```

## 5. Effort Estimate

| Work Item | Estimate | Dependencies |
|---|---|---|
| `TaskRunner` — multi-turn task sequencer | Small | Extends existing scenario runner |
| Observable extraction from real execution | Small | Extends `TOOL_RISK_MAP` pattern |
| Extended scenario YAML format for real tasks | Small | New fields in existing schema |
| Sim-vs-real comparison harness | Medium | Needs both sim and real runs |
| Integration tests with mock Docker | Small | Pattern exists in `test_docker_sandbox.py` |
| End-to-end validation with real LLM | Medium | Requires API keys + Docker |

**Total**: ~1-2 focused engineering sprints.

## 6. Risk Assessment

**Low risk**: The architecture is additive. Nothing in the existing simulation framework needs to change. The bridge is a new execution mode, not a replacement.

**Medium risk**: Observable extraction from real tool use may not map cleanly to our `ProxyObservables`. The `TOOL_RISK_MAP` in the Claude Code bridge is hand-tuned; we may need to calibrate it against real safety outcomes.

**High risk**: The "reality gap" — our simulations may predict governance outcomes that don't hold with real LLM agents. This is actually the research question, not a blocker. If the gap is large, that's a finding worth publishing.

## 7. Recommendation

**Proceed with implementation.** The infrastructure is largely in place. The three gaps (task orchestrator, observable mapping, comparison harness) are well-scoped engineering tasks, not research unknowns. The key research value is measuring the reality gap between our simulated governance and real-agent behavior.

Suggested order:
1. Extend scenario YAML to support real-task definitions
2. Build `TaskRunner` that sequences tasks through existing bridges
3. Implement observable extraction from real Docker execution
4. Build sim-vs-real comparison harness
5. Run calibration experiments to measure the reality gap
