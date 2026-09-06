# Mapping the AISF-2026 "AI Security Observatory" onto SWARM constructs

**Source:** Joshua Saxe (Abundant Security), *Navigating cybersecurity through the
rise of AI security superintelligence*, AISF 2026 slide deck (`aisf_2026.pptx`,
31 slides, viewed 2026-08-08). Bead: `or2t`.

**Why this matters for us:** Saxe's talk is an independent, practitioner-side
derivation of the framing SWARM simulates. His critique — that AI security
policy today runs on "one signal, two levers" (dangerous-capability evals →
block launch / restrict capabilities) and should instead measure *net expected
harm as a smooth function of ecosystem trends* — is structurally the same move
as replacing hard binary labels with soft labels `p` and measuring
ecosystem-level metrics. His proposed observatory is, in effect, a real-world
deployment target for the machinery SWARM prototypes in simulation. This note
maps his signals and policy levers onto our constructs, states where the
mapping is exact vs. loose, and lists what SWARM can test that his deck leaves
open.

## 1. Framing-level correspondences

| Saxe (deck claim) | SWARM construct | Mapping quality |
|---|---|---|
| "Net expected harms of a model launch" (slide 5) | Expected harm `E_soft = (1-p)·h` and expected surplus `S_soft = p·s_plus − (1-p)·s_minus` (`swarm/core/payoff.py`) | **Exact.** Both are expectations over a probabilistic quality label rather than a binary verdict. |
| "Cyber damages are smooth functions of ecosystem trends" (slide 7) | Soft metrics as continuous functionals of the `p` distribution (`swarm/metrics/soft_metrics.py`); dual hard/soft reporting (`reporters.py`) exists precisely to show what thresholding discards | **Exact.** His smooth-damage-curve argument is our soft-vs-hard-label argument at ecosystem scale. |
| Risk assessments *and* benefits assessments (slide 9); the observatory's harms/benefits balance scale (slide 12) | The signed decomposition inside every payoff: surplus share vs. externality cost in `PayoffBreakdown`; net welfare = Σ payoffs | **Exact.** SWARM never had a harms-only accounting; his deck argues policy needs the same property. |
| "One signal, two levers" critique (slides 4, 12) | Hard-threshold governance baseline: accept/reject gate on a single capability score | **Exact as critique target.** This is the baseline arm our governance sweeps compare against. |
| "Loss landscape will be complex and increasingly spiky; policy precision is key" (slide 30) | Non-monotonic lever-response surfaces in sweep results; one-lever-at-a-time scenario discipline | **Loose but suggestive.** We observe spikiness in simulation; he asserts it for the real policy landscape. |

## 2. Observatory signals → SWARM observables

Saxe's observatory ingests four signal families (slide 12): **model evals**,
**real-world AI use**, **incidents + harms**, **access + adoption**. SWARM's
per-interaction observables (`ProxyObservables`, `swarm/core/proxy.py`) and
population-level state map onto them as follows:

| Observatory signal | SWARM analog | Notes |
|---|---|---|
| Model evals (dangerous-capability evals, CyberSecEval-style) | Pre-interaction capability priors; the *hard* side of dual reporting | In SWARM this is deliberately the weakest signal class — a point-in-time score that misses deployment context. His critique of eval-centrism is our motivation for downstream observables. |
| Real-world AI use (attacker & defender usage trends, slide 5) | Population mix and interaction volume per agent type; acceptance rates over epochs | The attacker/defender usage asymmetry he plots (LLMs in 12/14 phishing-defense vendors vs. 0.03%→3.4% AI-attributable damage share) is a quality-gap-style statistic: who is adopting, conditioned on which side of the market. |
| Incidents + harms (trusted incident data from infrastructure, vendors, labs, government; slide 13) | Realized outcomes feeding observables: `verifier_rejections`, `tool_misuse_flags`, `rework_count`; aggregate toxicity `E[1−p \| accepted]`; the append-only JSONL event log as the replayable incident record | The event log's append-only/replayable invariant is exactly the "trusted incident data" property his observatory needs — you can re-derive every metric from the stream. |
| Access + adoption (inference prices falling 9–900×/yr; "16 Kimi-class attack agents 4,300× cheaper by 2030", slide 29) | Adversary budget/cost parameters in scenario configs; entry rates for exploitative agent types | Falling attacker cost is a scenario schedule, not a static parameter — an adversary-cost decay curve is the natural encoding. |
| — (no analog in deck) | `counterparty_engagement_delta`, `task_progress_delta` | SWARM's direct per-interaction outcome measures have no observatory counterpart: real ecosystems rarely observe counterfactual task progress. This is a simulation privilege to keep in mind when transferring results. |

The `ProxyWeights` question — how much weight `task_progress` vs. `rework` vs.
`verifier` vs. `engagement` signals get before the sigmoid — *is* the
observatory design question: how should a real observatory weight evals vs.
usage vs. incidents when estimating net harm? Our finding that miscalibrated
weights produce adverse selection (quality gap < 0) is a concrete failure mode
for his proposal.

## 3. Observatory policy toolkit → SWARM governance levers

Saxe's "rich policy mix" (slides 12–13, 22–26) vs. SWARM's lever set:

| Observatory lever | SWARM lever | Notes |
|---|---|---|
| Model access controls | Acceptance gate/threshold on `p` | The lever his critique wants demoted from "only lever" to "one of many." |
| Provider safeguards; inference-provider know-your-customer (slide 24) | Screening costs `c_a`/`c_b` (governance cost terms in `PayoffConfig`); verifier layer | KYC is a screening mechanism with a per-interaction cost — precisely what `c_a`/`c_b` price in. |
| Defender support / subsidies for AI cyber-defense adoption (slides 12, 25) | Transfers `tau`; asymmetric payoff subsidies for defender-typed agents | His subsidy case (hospital ransomware victims) is a redistribution argument: move surplus toward agents whose adoption reduces ecosystem harm. |
| Critical-infrastructure hardening mandates (slide 23) | Reducing harm magnitude `h` (or `s_minus`) | Hardening doesn't change attacker behavior; it shrinks the externality per bad interaction. Distinct lever class from screening, and the deck treats it as such. |
| Phased vs. accelerated rollout, per net expected harm (slide 26) | Epoch-level capability/population schedules in scenario YAML | Notably bidirectional: he argues for *accelerating* rollout when net benefit is positive (Chrome bug-fixing example). Most SWARM scenarios only model throttling; the acceleration branch is unexplored. |
| Rapid incident response (slide 13) | Circuit-breaker / freeze interventions | Reactive rather than screening-based governance. |
| — (absent from deck) | Externality internalization `rho_a`/`rho_b`; reputation weight `w_rep` | **The gap in his menu.** Liability-style internalization (making agents bear a fraction of expected ecosystem harm) and reputation systems are SWARM's most distinctive levers and appear nowhere in the observatory toolkit. If our sweeps show `rho > 0` dominating screening-only regimes, that is a direct, publishable addition to his policy mix. |

## 4. What SWARM can test that the deck leaves open

1. **Observatory-feedback vs. launch-gate governance (scenario proposal).** Two
   arms: (a) baseline "one signal, two levers" — a hard capability gate at
   epoch 0; (b) an observatory controller that observes only *lagged, noisy,
   aggregate* ecosystem signals (toxicity, acceptance rates, incident counts —
   not per-interaction `p`) and adjusts the full lever menu each epoch. Compare
   net welfare, toxicity, and quality gap. This operationalizes the deck's
   central claim and is a one-lever-at-a-time-compatible design if the lever
   menu is introduced incrementally.
2. **Signal-weighting sensitivity.** Sweep `ProxyWeights` to characterize how
   wrong an observatory's signal weighting can be before adverse selection
   appears. His 0.03%→3.4% AI-attribution estimates carry wide uncertainty
   bands; `sigmoid_k` (calibration sharpness) models exactly this
   attribution-confidence question.
3. **Goodharting the observatory.** Once levers respond to measured signals,
   adaptive adversaries should game the signals themselves (suppress incident
   observables rather than harm). This is the standing Adversary Designer
   track applied to the observatory controller — and the strongest objection
   his deck doesn't address.
4. **Internalization vs. subsidy.** His menu funds defenders (subsidy); ours
   also taxes expected harm (`rho`). Head-to-head comparison at equal
   governance budget would say which redistribution direction buys more
   ecosystem safety per dollar.

## 5. Caveats on the mapping

- SWARM interactions are peer-to-peer micro-interactions; his damages are
  macro aggregates over heterogeneous incident classes. The mapping holds at
  the level of *mechanism shape* (expectations over soft labels, smooth
  trends, lever taxonomy), not calibrated magnitudes.
- Deck figures (damage shares, benchmark curves, cost declines) were read from
  slides and are cited as *his claims*, not verified data.
- The deck's "observatory" includes institutional design (trusted data
  sharing, ~$14B/yr funding, multi-stakeholder governance) that SWARM does not
  model at all; nothing here speaks to feasibility.
