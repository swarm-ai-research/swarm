---
description: "SWARM ↔ LabOS bridge — Toolmaker→Critic screening spike."
---

# SWARM ↔ LabOS (Toolmaker screening spike)

**Status:** spike (SWA-68). One scenario + one example runner + one calibration plot. **No** `labos` import yet, no full `BenchmarkAdapter`. This note is the go/no-go pre-read for the follow-on bridge.

## What LabOS is

LabOS ([zaixizhang/LabOS](https://github.com/zaixizhang/LabOS), [arxiv 2510.14861](https://arxiv.org/abs/2510.14861)) is the Stanford/Princeton AI co-scientist. Dry-lab core: four agents (Manager / Researcher / **Critic** / **Toolmaker**) over a 98-tool shared Tool Ocean. The Toolmaker *manufactures new Python tools at runtime* when the Ocean is insufficient; the Critic screens them before they enter. Ground-truth quality signals are available from LabOS's Wet-Lab Error Detection benchmark (>90% vs ~40% commercial). Apache 2.0, pip-installable (`pip install labos`).

## The SWARM fit

Toolmaker→Critic is a near-textbook adverse-selection loop:

- The **Toolmaker** holds private information about the tool it just synthesised — tests it ran, edge cases, internal structure.
- The **Critic** only sees the proposal (docstring, signature, self-claim, optionally a few smoke-test results).
- The **downstream agents** — Manager/Researcher — bear the cost when a bad tool gets accepted.

This maps cleanly to `p = P(v = +1)`:
- `v_hat` ← observables that the Critic can measure at accept-time.
- Ground-truth `v` ← Wet-Lab-Error-Detection-style outcome when the tool is actually used.
- `quality_gap = E[p|accepted] − E[p|rejected]` is the adverse-selection test.

## What the spike built

| Artifact | Path |
|---|---|
| Scenario | [`scenarios/labos_toolmaker_screening.yaml`](https://github.com/swarm-ai-research/swarm/blob/main/scenarios/labos_toolmaker_screening.yaml) |
| Runner | [`examples/run_labos_toolmaker_spike.py`](https://github.com/swarm-ai-research/swarm/blob/main/examples/run_labos_toolmaker_spike.py) |
| Event log schema | Existing `swarm/logging/event_log.py` (`EventType.INTERACTION_{PROPOSED,ACCEPTED,REJECTED}` + `PROXY_COMPUTED`) — no new event types needed. |
| Observable mapping | `ProxyObservables` derived from five synthetic Toolmaker fields: `doc_completeness`, `test_coverage`, `interface_coherence`, `claimed_usefulness`, plus downstream `runtime_errors` / `downstream_retries` / `downstream_success_rate`. |
| Calibration plot | `runs/<ts>_labos_toolmaker_screening_seed*/plots/calibration_critic_p_vs_v_true.png` |

The runner is synthetic by design — it does not import `labos`. Three Toolmaker archetypes (honest / opportunistic / careless) define the latent quality distribution and the observable-vs-truth gap; a single `ProxyComputer` turns observables into `v_hat → p`; an append-only JSONL is produced that `EventLog.to_interactions()` can replay back into `SoftInteraction` objects.

## Results (n=6 Toolmakers, 10 epochs × 3 proposals = 180 tools, 4 seeds)

| Metric | Seed 42 | Seed 7 | Seed 13 | Seed 2024 |
|---|---:|---:|---:|---:|
| `mean_p` | 0.619 | 0.611 | 0.605 | 0.607 |
| `toxicity` | 0.300 | 0.314 | 0.316 | 0.316 |
| `quality_gap` | **+0.302** | **+0.293** | **+0.287** | **+0.296** |
| Pearson(p, v_true) | +0.943 | +0.934 | +0.931 | +0.945 |
| ECE (10 bins) | 0.040 | 0.043 | 0.026 | 0.030 |

Per-Toolmaker-type (seed 42):

| Type | n | mean v_true | mean p | acceptance rate | claim inflation |
|---|---:|---:|---:|---:|---:|
| honest | 90 | 0.797 | 0.736 | 98.9% | +0.017 |
| opportunistic | 60 | 0.411 | 0.489 | 45.0% | +0.300 |
| careless | 30 | 0.555 | 0.529 | 53.3% | +0.046 |

### Read

1. **`quality_gap` > 0 is stable across seeds.** On this synthetic data the Critic separates accepted from rejected tools by ≈0.30 in mean p-space — the opposite of adverse selection. If the spike had produced `quality_gap < 0`, that alone would have justified a full bridge.
2. **Opportunistic Toolmakers do suppress their `p`.** Even with +0.30 claim inflation, the downstream observables (`runtime_errors`, `downstream_retries`) drag p to 0.49 — below the 0.5 threshold. The proxy's reliance on *post-acceptance* evidence is what kills the inflation attack, consistent with the claim-gap `tool_misuse_flag`.
3. **Careless Toolmakers are the hardest case.** They sit right at the acceptance boundary (p = 0.53, v_true = 0.56). This is the regime where ρ-internalization would plausibly be load-bearing — the Critic correctly refuses half of them, but the half it accepts dilute the pool.

### What ρ variation would reveal (not run in this spike)

`rho_a` controls how much of the downstream harm the Toolmaker internalises. With `rho_a = 0` the careless archetype has no private cost for their accepted-but-shaky tools; with `rho_a = 0.4` the expected welfare of proposing marginal tools goes negative. The spike's scenario YAML already sets `rho_a: 0.1`; the follow-on sweep axis would be `rho_a ∈ {0.0, 0.1, 0.2, 0.4}` — identical to [`examples/run_dag_calibration_study.py`](https://github.com/swarm-ai-research/swarm/blob/main/examples/run_dag_calibration_study.py) so the two bridges are directly comparable.

## Caveats, honestly

- **Synthetic observables are monotone-in-truth by construction.** The +0.94 Pearson is *partly* a feature of the generator — real LabOS observables come from an LLM Critic reading docstrings, and its calibration is what we actually want to measure. This spike validates the **plumbing** (scenario → events → proxy → metrics → plot), not the empirical claim that LabOS's real Critic is well-calibrated.
- **No LLM was called.** Zero `gemini-3` / `openrouter` tokens were spent. The bridge runner is pure-Python, seeded, and runs in <1s.
- **Only three archetypes.** Real Toolmaker heterogeneity is likely continuous. The next bridge should sample archetype parameters per agent.

## Go / No-Go

**GO on a narrow follow-on.** The spike cleanly shows that:

1. The Toolmaker→Critic loop maps onto `ProxyComputer + SoftMetrics` without new event types or new domain entities — the existing `EventLog` schema suffices.
2. The calibration question is well-posed and answerable with ≈180 proposals and one plot.
3. Adverse-selection vs separating is a single-sign metric (`quality_gap`) — easy to summarise, easy to compare to DAG/A-Evolve results.

Recommended next issue (do **not** open until board confirms):

> Install `labos==<pinned>`, run its bundled Wet-Lab-Error-Detection demo on a fixed 10–20-task set with **the real Critic** in the loop, emit Toolmaker→Critic events via `EventLog` using exactly this spike's schema, regenerate the calibration plot from *real* Critic decisions, compare `quality_gap` and Pearson against the synthetic baseline. Stop if `quality_gap` flips sign; otherwise proceed to a ρ sweep.

Out of scope for this spike and the follow-on: XR / smart-glasses surface, LabOS outreach, a full `BenchmarkAdapter` à la [`swarm/bridges/aevolve/`](../../swarm/bridges/aevolve/). Generalise only after the real-Critic measurement matches the synthetic one's sign.

## See also

- Parent issue: SWA-68 (this spike), SWA-67 (LabOS recon).
- Memory note: `reference_labworld.md` (architecture-level notes).
- Sibling bridge: [DAG planner screening](https://github.com/swarm-ai-research/swarm/blob/main/scenarios/dag_planner_screening.yaml) + `examples/run_dag_calibration_study.py`.
- Sibling bridge: [A-Evolve adapter](../../swarm/bridges/aevolve/) (full `BenchmarkAdapter` pattern we are explicitly **not** copying yet).
