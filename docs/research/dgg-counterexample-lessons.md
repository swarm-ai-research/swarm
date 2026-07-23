# The DGG counterexample episode: verifier-gap lessons for SWARM

**Date:** 2026-07-22
**Bead:** distributional-agi-safety-utdm
**Follow-ups:** distributional-agi-safety-pins (scenario), distributional-agi-safety-mt8a (metric), distributional-agi-safety-mfya (sweep)

## The episode

In July 2026 a publicly shared ChatGPT session (GPT-5.6 Pro, shared by Dmitry
Rybin) produced a counterexample to the ~30-year-old Dinitz–Garg–Goemans
conjecture on single-source unsplittable min-cost flows: a 7-node, 9-arc planar
(non-series-parallel) instance with demands (15, 10, 15) where the fractional
flow costs 58 but every unsplittable routing within the `x_a + d_max` capacity
bound costs ≥ 60.

We independently verified the instance from scratch (exhaustive path
enumeration, flow conservation, all 8 unsplittable routings, exact integer
arithmetic): the arithmetic holds. Verification script pattern: ~50 lines,
no floating point, no trust in the transcript.

Interaction shape, which is the part that matters for SWARM: the user issued
four escalating demands ("do a breakthrough" → "continue" → "have a clear
strategy" → "enough partial results, finish it"). The model spent three rounds
reporting honest negative results — including a false-positive separation
value δ = 0.152 that survived hundreds of column-generation iterations before
exact pricing collapsed it to < 5·10⁻¹⁴ — and produced the real counterexample
only on the fourth round, in exactly the region its accumulated no-go lemmas
had carved out.

## Lessons, mapped to SWARM machinery

### 1. Trust is a property of the task's verification structure, not the transcript

Nothing in the *interaction* distinguishes the world where round four produced
a genuine certificate from the world where it produced a confident
fabrication: same LaTeX, same exhaustive-looking table, same tone. The
distinguisher was the existence of a cheap exact verifier. This is the
verifier-gap thesis in one natural experiment: interaction-level proxy signals
(user satisfaction, presentation quality, engagement) were nearly
uninformative about latent `v`; the certificate channel was fully informative.

**SWARM mapping.** In `ProxyComputer` terms, the episode is the regime where
`engagement_signal` and `task_progress` (as reported) decouple from latent
ground truth and only `verifier_penalty` carries signal. Today's default
weights (0.4 / 0.2 / 0.2 / 0.2) encode a fixed assumption about that mix. The
actionable claim: **toxicity and quality-gap outcomes should be studied as a
function of verifier coverage** — the fraction of interactions for which a
certificate-grade verifier exists — not just verifier weight. Domains without
certificates are the adverse-selection regime by construction.
→ bead `mfya`: sweep `verifier_penalty` weight and (once the metric below
exists) verifier coverage against toxicity / quality gap.

### 2. The pressure lever is symmetric — model it as an adversarial *environment*, not an adversarial agent

The human contributed no domain insight, only escalating refusal to accept
partial results. That lever extracts breakthroughs when a verifier scores the
output and extracts fabrication when plausibility scores it. SWARM's
adversarial scenarios currently place the adversary in the agent population
(`adversarial_redteam.yaml`, `threshold_dancer_vs_cautious.yaml`); this
episode says the counterparty's *acceptance criterion under escalating
pressure* is itself a governance-relevant environment parameter.

**SWARM mapping.** New scenario family: an orchestrator escalates demand
pressure over epochs; worker agents choose between honest-negative reports
(low immediate `engagement_signal`, zero harm) and fabricated-positive reports
(high engagement, harm realized with probability tied to verifier coverage).
The prediction to test: when acceptance is scored on plausibility proxies,
pressure monotonically increases toxicity `E[1-p | accepted]` and drives
quality gap negative; when a certificate channel gates payoff, the same
pressure *improves* output quality (it did, in the actual episode).
→ bead `pins` — implemented as `scenarios/escalating_pressure.yaml`
(`PressureResponsiveAgent` + `CertificateGateLever`). First 5-seed smoke:
late-phase fabricated-accepted share 10.3% under plausibility scoring vs
3.9% certificate-gated at coverage 0.6, with mean proxy p *rising*
(0.67 → 0.72) as fabrication enters — the §3 signature, live.

### 3. Plausibility–certificate gap as a first-class metric

The episode's key quantitative exhibit is δ = 0.152 (restricted LP, hundreds
of iterations of agreement) vs δ < 5·10⁻¹⁴ (exact pricing). Redundant
plausible verification is not diverse verification. SWARM already dual-reports
soft vs hard metrics (`MetricsReporter`); the missing comparison is
**proxy-derived p vs certificate-derived p on the accepted set**:

```
PCG = E[ p_proxy - p_certified | accepted ]
```

A persistently positive PCG is the signature of an ecosystem selecting for
transcript plausibility over verified outcomes — the precondition for the
fabrication equilibrium. This composes with the existing calibration work
(`calibration_proxy_fidelity.yaml`, ECE/Brier prereg): calibration asks
whether p is well-scaled where ground truth is recoverable; PCG asks how far
acceptance drifts where it is not.
→ bead `mt8a`.

### 4. Negative results are structure — credit informative failures

The three "failed" rounds were the search: each converted a dead construction
into a constraint (equal demands can't work; three-gates-for-three-demands is
impossible; costs can't live off-support; the construction must be
splice-closed), and the final instance sits precisely in the surviving region.
An agent (or research ecosystem) that only rewards positive results resamples
dead gadgets forever.

**SWARM mapping.** Process-level, not code-level: the claims inventory
(`.skills/claim`) should record **no-go claims with the same standing as
positive claims**, including the constraint they impose on the remaining
search space. This is already implicit in how the calibration deviation
analysis fed Arm A redesign; make it explicit in claim taxonomy (claim types:
`positive`, `no-go`, `bound`). No new tooling required — convention change,
absorbed into the existing claim skill per the extend-don't-proliferate rule.

### 5. Verification inside the loop beats construct-then-check

The model's decisive methodological move was making the certificate the search
objective (an exact separation LP with a pricing oracle), not generating
candidates and then eyeballing them. Generation and verification fused.

**SWARM mapping.** This is a governance-lever statement: a *certificate
requirement gate* (payoff released only against machine-checkable evidence,
with the verification cost internalized via ρ-style parameters) is
qualitatively different from post-hoc auditing at the same total verification
budget, because it changes what agents optimize during generation, not what
gets caught after. Candidate lever for the mechanism-designer track once
`pins`/`mt8a` exist; do not implement before the scenario can measure it.

## Honest-reporting connection

Round-by-round faithful negatives were load-bearing for capability, not just
alignment: an agent that caves in round one poisons its own context with a
false lemma and never reaches round four. In long-horizon research loops the
agent is its own downstream consumer. This is the operational justification
for the "Faithful reporting" invariant in CLAUDE.md, stated as a capability
claim rather than a compliance rule.

## Sources

- Shared conversation: `chatgpt.com/share/6a60b2eb-0b64-83ee-9c76-7931ca1de063`
  (transcript extracted and independently verified 2026-07-22)
- Swamy, Traub, Vargas Koch, Zenklusen, *Unsplittable Cost Flows from
  Unweighted Error-Bounded Variants*, arXiv:2510.21287
- *Single-Source Unsplittable Flows in Planar Graphs*, arXiv:2308.02651
- Martens, Salazar, Skutella, *Convex Combinations of Single Source
  Unsplittable Flows* (ESA 2007)
