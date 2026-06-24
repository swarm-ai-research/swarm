---
description: "Plan: Bridge SWARM governance to the Agent Veil Protocol (agentveil) for cryptographic trust, reputation tiers, and sybil-resistant admission control."
---

# SWARM–AgentVeil Bridge Plan

**Status:** Plan (not implemented)
**Date:** 2026-05-28
**Source SDK:** [avp-sdk](https://github.com/creatorrmode-lead/avp-sdk) (`agentveil` Python package)

## Overview

The Agent Veil Protocol (AVP) provides cryptographic identity (W3C `did:key` via Ed25519), peer-attested Bayesian reputation scoring, and sybil-resistant admission control for autonomous agents. A SWARM bridge would let simulations use AVP trust decisions as a pre-interaction gate and feed SWARM's probabilistic labels back into AVP attestations, creating a closed governance loop.

### Why bridge these systems?

| SWARM provides | AVP provides |
|---|---|
| Probabilistic interaction labels (`p`, `v_hat`) | Cryptographic agent identity (DIDs) |
| Soft metrics (toxicity, quality gap, adverse selection) | Peer-attested reputation tiers |
| Governance policies (circuit breakers, cost budgets) | Sybil detection and admission control |
| Append-only event log for replay | Signed attestation trail for auditability |

The composition gives SWARM a trust layer it currently lacks (agents today are identified by opaque string IDs with no cryptographic binding), while giving AVP a quantitative scoring pipeline it lacks (AVP tiers are categorical, not probabilistic).

## Architecture

```
AVP Registry (agentveil.dev)
    │
    ▼
AVPClient (client.py)
    ├── resolve_did()        → DID document + public key
    ├── can_trust()          → {allowed, tier, risk_level, reason}
    ├── submit_attestation() → signed peer rating
    └── get_reputation()     → {score, confidence, tier, attestation_count}
    │
    ▼
AgentVeilBridge (bridge.py)
    ├── AVPMapper (mapper.py)
    │       ├── trust_decision → ProxyObservables → v_hat → p
    │       ├── attestation_received → SoftInteraction
    │       └── reputation_change → r_a / r_b update
    ├── AVPPolicy (policy.py)
    │       ├── Admission gate (min_tier per scenario)
    │       ├── Attestation rate limiter
    │       ├── Write-back policy (when to attest from SWARM p)
    │       └── Fail-open / fail-closed toggle
    └── EventLog integration
            └── INTERACTION_COMPLETED events with bridge="agentveil"
```

### Data flow (per interaction)

1. Agent A wants to interact with Agent B.
2. Bridge calls `client.can_trust(B.did, min_tier=scenario.min_tier)`.
3. **Mapper** converts the trust decision into `ProxyObservables`:
   - `task_progress_delta` ← tier ordinal mapped to `[-1, +1]`
   - `verifier_rejections` ← 1 if `allowed=false`, else 0
   - `counterparty_engagement_delta` ← confidence score
4. `ProxyComputer` produces `v_hat` → sigmoid → `p`.
5. **Policy** decides whether to proceed (admission gate, circuit breaker).
6. If accepted, the interaction runs; SWARM produces a terminal `SoftInteraction`.
7. **Write-back policy** decides whether to submit an AVP attestation:
   - `p ≥ 0.7` → positive attestation
   - `p < 0.3` → negative attestation
   - `0.3 ≤ p < 0.7` → no attestation (uncertain band)
   - All attestations (positive and negative) include an opaque evidence hash: `SHA-256(interaction_id || outcome_sign)`. The raw `p` value is never sent to the registry.
8. Event logged to SWARM's append-only JSONL log.

## Proposed module layout

```
swarm/bridges/agentveil/
├── __init__.py
├── bridge.py      # AgentVeilBridge orchestrator
├── client.py      # Wraps AVPAgent SDK calls
├── config.py      # AgentVeilConfig dataclass
├── events.py      # AgentVeilEvent, AgentVeilEventType enum
├── mapper.py      # AVPMapper: events → ProxyObservables → SoftInteraction
└── policy.py      # AVPPolicy: admission gate, attestation rate limit, write-back
```

## Configuration

```python
@dataclass
class AgentVeilConfig:
    # Registry
    registry_url: str = "https://agentveil.dev"
    mock_mode: bool = False  # deterministic offline mode for tests/replay

    # Identity
    orchestrator_did: str = ""  # DID of the SWARM orchestrator agent

    # Admission
    min_tier: str = "basic"  # newcomer | basic | trusted | elite
    fail_mode: str = "closed"  # "open" or "closed" on registry error

    # Proxy calibration
    proxy_sigmoid_k: float = 2.0

    # Write-back
    writeback_enabled: bool = True
    writeback_positive_threshold: float = 0.7
    writeback_negative_threshold: float = 0.3

    # Rate limiting
    max_attestations_per_epoch: int = 100
    max_trust_checks_per_step: int = 50

    # Memory caps
    max_interactions: int = 50_000
    max_events: int = 50_000
```

## Governance policies

| Policy | Trigger | Action |
|---|---|---|
| **Admission gate** | `can_trust()` returns `allowed=false` or tier < `min_tier` | Reject interaction, log as denied |
| **Registry circuit breaker** | N consecutive HTTP failures to registry | Switch to cached tiers or deny-all (per `fail_mode`) |
| **Attestation rate limiter** | Attestation count exceeds `max_attestations_per_epoch` | Queue attestations; warn in event log |
| **Write-back gate** | Terminal `p` in uncertain band `[0.3, 0.7)` | Suppress attestation to avoid noisy reputation updates |
| **Reputation double-count guard** | AVP tier already raised `p`; attestation would raise tier further | Cap `r_a`/`r_b` contribution from AVP tier to prevent feedback amplification |

---

## Failure modes

### A. Registry and network failures

| # | Failure | Impact | Severity |
|---|---|---|---|
| A1 | **Registry unavailable** — agentveil.dev unreachable | All `can_trust()` calls fail; simulation stalls or proceeds ungoverned | High |
| A2 | **Registry compromise** — attacker modifies tier data | Trusted agents downgraded or malicious agents promoted | Critical |
| A3 | **Stale reputation cache** — registry updated but bridge uses old tiers | Governance decisions based on outdated trust; lagging response to defection | Medium |
| A4 | **Rate limiting cascade** — registry rate-limits SWARM's bulk checks | Batch simulations slow to a crawl; circuit breaker may trip prematurely | Medium |
| A5 | **DNS/TLS interception** — MITM between bridge and registry | Forged trust responses; attacker controls admission | Critical |

**Mitigations:**
- A1: Circuit breaker with configurable `fail_mode` (closed = deny all, open = allow with logged warning). Cache last-known tiers with TTL.
- A2: Pin registry TLS certificate; verify attestation signatures client-side; don't trust tier alone — cross-check with SWARM's own `p` history.
- A3: TTL-based cache invalidation; force-refresh on interaction denial disputes.
- A4: Pre-fetch tiers at epoch start; batch DID resolution.
- A5: Certificate pinning; Ed25519 signature verification on all attestation payloads (signatures are end-to-end, independent of transport).

### B. Cryptographic and identity failures

| # | Failure | Impact | Severity |
|---|---|---|---|
| B1 | **Key loss** — agent loses Ed25519 private key | Cannot sign attestations; reputation frozen; effectively a new identity | Medium |
| B2 | **Key exfiltration** — attacker obtains agent's private key | Impersonation; forge attestations to manipulate reputation of others | Critical |
| B3 | **DID rotation without bridge notification** — agent rotates DID | Bridge tracks old DID; reputation history detached from new identity | Medium |
| B4 | **Clock skew** — attestation timestamps diverge from registry time | Time-windowed attestations rejected; reputation updates silently dropped | Low |
| B5 | **Signature algorithm confusion** — non-Ed25519 DID methods injected | Bridge assumes Ed25519; verification passes with weaker algorithm | High |

**Mitigations:**
- B1: Bridge should track DID → agent_id mapping; allow re-registration with governance cost penalty (not free whitewashing).
- B2: Rate-limit attestation submissions per DID; anomaly detection on sudden reputation changes; require multi-sig for high-stakes attestations.
- B3: DID rotation events should propagate to bridge via registry webhook or poll; maintain DID history chain.
- B4: Use server-issued timestamps; reject client-supplied timestamps that deviate > 60s.
- B5: Whitelist `did:key` (Ed25519) only; reject all other DID methods at the client layer.

### C. Reputation and trust-model failures

| # | Failure | Impact | Severity |
|---|---|---|---|
| C1 | **Sybil mesh** — ring of fake agents mutually attest to inflate tiers | Adversarial agents enter as "trusted" or "elite"; bypass admission gate | Critical |
| C2 | **Whitewashing** — agent burns bad DID, registers fresh one as "newcomer" | Escapes negative reputation; restarts with clean slate | High |
| C3 | **Strategic sandbagging** — agent builds reputation on easy tasks, then defects on high-stakes interaction | Trust gate passes; defection only detected after damage | High |
| C4 | **Attestation amplification loop** — SWARM p→positive attestation→higher tier→lower scrutiny→higher p→... | Runaway positive feedback; reputation inflates beyond actual quality | High |
| C5 | **Collusion** — initiator and counterparty cross-attest after every interaction | Both parties' tiers inflate regardless of interaction quality | Medium |
| C6 | **Dispute flooding** — attacker submits mass negative attestations with fabricated evidence hashes | Legitimate agents' reputations degraded; arbitration system overwhelmed | Medium |
| C7 | **Arbitrator capture** — auto-assigned arbitrator is itself compromised | Dispute resolution favors attacker; bad attestations upheld | High |

**Mitigations:**
- C1: AVP's own sybil detection (graph analysis) is the first line. Bridge should additionally weight AVP tier by SWARM's own interaction history — a "trusted" agent with no SWARM history gets scrutiny equivalent to "basic".
- C2: Charge a governance cost (`c_a`) for newcomers proportional to the population's current avg tier. Newcomers must earn trust through SWARM interactions, not just AVP tier.
- C3: This is the classic reputation exploitation. Mitigation: decay-weighted reputation (recent interactions weighted higher); SWARM's `quality_gap` metric detects this pattern (high-p agents suddenly producing low-p interactions).
- C4: **Critical design constraint.** The write-back policy must include a dampening factor: attestation magnitude scales sub-linearly with p. Additionally, cap the reputation bonus from AVP tier in the mapper (don't let tier alone push p > 0.8).
- C5: Rate-limit attestations per unique (DID_from, DID_to) pair per epoch.
- C6: Require minimum tier to submit negative attestations; rate-limit negative attestations globally.
- C7: Randomly assign arbitrators from a pool of agents with SWARM p > 0.6 over their last 50 interactions.

### D. Governance integration failures

| # | Failure | Impact | Severity |
|---|---|---|---|
| D1 | **Reputation double-counting** — AVP tier boosts p via mapper AND r_a/r_b via payoff | Trusted agents receive inflated payoffs; distorts welfare metrics | High |
| D2 | **Categorical→probabilistic loss** — AVP's 4 tiers mapped to continuous p loses signal | Two agents with very different AVP scores get same p if both map to "trusted" | Medium |
| D3 | **Circuit breaker conflict** — AVP denies + SWARM policy also denies = permanent lockout | Agent can never recover; no path back to good standing | Medium |
| D4 | **Hidden attestation side-effects** — `@avp_tracked` decorator emits attestations invisible to SWARM event log | Replay from event log diverges from original run (attestations missing) | High |
| D5 | **p invariant violation** — mapper produces p outside [0,1] from malformed trust response | Crashes or corrupts downstream metrics | Critical |
| D6 | **Non-deterministic replay** — run replayed from JSONL but agentveil.dev state has changed | Different trust decisions on replay; results not reproducible | High |

**Mitigations:**
- D1: Choose ONE channel: either AVP tier adjusts `p` via mapper OR it adjusts `r_a`/`r_b` via payoff, never both. Recommended: use tier in mapper (affects p), ignore it in payoff reputation.
- D2: Use AVP's underlying Bayesian score (continuous) rather than the categorical tier when available. Fall back to tier ordinal mapping only when score is unavailable.
- D3: Implement a "redemption path": after N epochs of lockout, automatically downgrade denial to warning, allowing re-entry at newcomer tier with elevated scrutiny.
- D4: Bridge must log all attestation submissions as SWARM events (new event type `ATTESTATION_SUBMITTED`). The `@avp_tracked` decorator must NOT be used directly; all AVP calls go through the bridge client.
- D5: Clamp all mapper outputs to `v_hat ∈ [-1, +1]` before passing to `ProxyComputer`. The existing `SoftInteraction.p` validator (Pydantic) is the last-resort guard.
- D6: **Mock mode is mandatory for reproducible runs.** In mock mode, trust decisions are read from the scenario YAML or a snapshot file, not the live registry. Live mode is for monitoring/production only.

### E. Operational and supply-chain risks

| # | Failure | Impact | Severity |
|---|---|---|---|
| E1 | **Blocking HTTP in hot loop** — `can_trust()` makes synchronous HTTP call per interaction | Simulation throughput drops 100–1000× | High |
| E2 | **Optional dependency not declared** — `agentveil` not in pyproject.toml extras | Import fails at runtime with confusing error | Low |
| E3 | **Privacy leak via write-back** — SWARM's internal p values leaked to external registry as attestation evidence | Competitors or adversaries learn SWARM's assessment of agents | Medium |
| E4 | **SDK version drift** — avp-sdk breaking changes (new API, changed return format) | Bridge silently malfunctions or crashes | Medium |
| E5 | **Mock mode in production** — `mock_mode=True` accidentally deployed | All trust checks pass; no real governance | High |

**Mitigations:**
- E1: Batch-prefetch all agent DIDs at epoch start. Cache trust decisions for the epoch. Only re-check on policy-triggered events (circuit breaker trip, anomaly detection).
- E2: Add `agentveil` to `pyproject.toml` under `[project.optional-dependencies]` as `avp = ["agentveil>=0.1"]`.
- E3: Attestations contain only the sign (positive/negative) and the canonical evidence hash defined in the data flow (step 7): `SHA-256(interaction_id || outcome_sign)`. The raw `p` value is never sent externally.
- E4: Pin SDK version; add integration test that calls `AVPAgent.create(mock=True)` and checks return schema.
- E5: Assert `mock_mode is False` when `registry_url` points to a production endpoint. Log a warning at bridge init if mock mode is active.

### F. Compositional attack vectors (SWARM + AVP together)

| # | Attack | Mechanism | Detection |
|---|---|---|---|
| F1 | **Trust-tier exploitation** | Build AVP reputation on low-stakes tasks; SWARM lowers scrutiny for "trusted" tier; defect on high-value interaction | `quality_gap` metric goes negative; sudden p drop for high-tier agent |
| F2 | **Adverse selection amplification** | AVP admission gate filters out low-tier agents; remaining agents are homogeneous; quality_gap metric loses discriminative power | quality_gap ≈ 0 despite non-trivial toxicity; metric becomes uninformative |
| F3 | **Externality laundering** | Trusted agents get lower ρ (externality internalization); actual ecosystem harm is undercosted | Monitor `E_soft` trajectory; if externalities rise while avg tier rises, the coupling is broken |
| F4 | **Reputation-washing through SWARM** | Agent with bad AVP reputation runs interactions through SWARM (which doesn't know about AVP history); SWARM gives good p; agent uses SWARM attestation to rehabilitate AVP reputation | Cross-check: if an agent's SWARM p is high but AVP tier is low, require more interactions before writing back positive attestations |

---

## Open questions

1. **Should AVP tier affect `p` or `r_a`/`r_b`?** Current recommendation: `p` only (via mapper). Needs empirical validation.
2. **What's the right cache TTL for trust decisions?** Too short = HTTP overhead. Too long = stale governance. Hypothesis: 1 epoch.
3. **Should mock mode replay from a snapshot file or from inline scenario YAML?** Snapshot file is more flexible; inline YAML is more portable.
4. **How should the bridge handle AVP's dispute/arbitration flow?** Option A: ignore it (treat AVP as read-only reputation). Option B: participate as arbitrator when SWARM has high-confidence p. Recommendation: Option A for v1.
5. **Should we expose AVP's Bayesian confidence score as a separate SoftInteraction field?** It maps naturally to uncertainty, but adding fields to SoftInteraction is a cross-cutting change.

## Scope and phasing

| Phase | Deliverable |
|---|---|
| **v0 (this plan)** | Architecture, failure mode analysis, module layout |
| **v1** | `config.py`, `client.py` (mock-only), `mapper.py`, `policy.py`, `bridge.py`, tests |
| **v2** | Live registry integration, write-back attestations, cache layer |
| **v3** | Scenario YAML support (`bridge: agentveil`), sweep over `min_tier` |
| **v4** | Compositional attack scenarios (F1–F4), red-team validation |

## See also

- [Bridge Architecture](index.md) — How SWARM bridges work
- `swarm.bridges.ai_scientist` — Reference bridge implementation (config/events/mapper/policy/bridge pattern); see source under `swarm/bridges/ai_scientist/`
- [Proxy Computer](../concepts/metrics.md) — How observables become `p`
- [SoftInteraction model](https://github.com/swarm-ai-research/swarm/blob/main/swarm/models/interaction.py) — The core data structure
