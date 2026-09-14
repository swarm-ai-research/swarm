---
description: "BABEL and Anthropic's GTG-14020: a one-operator surveillance stack built with Claude and reached through a Chinese model router, and what per-account governance cannot see."
---

# BABEL and GTG-14020: many operators, one account

**Sources** (studied 2026-09-14, bead `w6xw`):

- **S1.** [NetAskari, 8-post X thread](https://x.com/NetAskari/status/2099392451630760268)
  (2026-09-14 06:58 UTC), read in full via fxtwitter, first image checked.
- **S2.** [NetAskari, "Anthropic + Chinese online surveillance = BABEL"](https://netaskari.substack.com/p/anthropic-chinese-online-surveillance)
  (2026-09-13). Paid-only. **Only the ~470-word free preview was read.**
- **S3.** [Anthropic, "Detecting and countering misuse of AI: September 2026"](https://www.anthropic.com/threat-intelligence-report-september-2026),
  sections GTG-14020 and GTG-14021, read in full from the published page.
- **S4.** [lxghost, "GTG-14020 · BABEL 威胁情报可视化"](https://lxghost.github.io/view/gtg-14020-babel.html)
  (2026-09-14). A Chinese-language structured analysis of the full S2 report.
  It is derived from S2 alone and rates itself Medium confidence. It is useful
  here for the contradictions it logs, not as independent corroboration.

**Fidelity caveat:** everything about BABEL itself comes from one outlet (S1,
S2), plus a secondary reading of that outlet (S4). None of it has been
independently verified. This note does not reproduce the infrastructure
indicators or personal data that appear in S2 and S4, and nothing here should
be read as attribution.

Companion to [erdos-ai-ledger-lessons.md](erdos-ai-ledger-lessons.md): its
lesson 2 (the denominator problem) comes back here twice, as a refusal that
was re-prompted and as a router that hides who is asking.

## First: does BABEL match GTG-14020?

NetAskari's framing is that Anthropic's case "by coincidence matched a project
we have been following for weeks." Line the two up before drawing any lesson:

| | GTG-14020 (S3) | BABEL (S1, S2) |
|---|---|---|
| Who | "One operator" ran the desk; Anthropic banned "a group of accounts" | "A single operator", said to be affiliated with CIPU (a police university) |
| How Claude was used | **At runtime**, as "a stand-in for a staffed analyst team": ingest multilingual sources, write templated Chinese-language dossiers | **At build time**: Claude wrote the scrapers and infrastructure. Analysis at runtime ran on a MiniClaw agent over DeepSeek or OpenAI |
| Access path | Not described | Through a Chinese model router (huoshenai[.]com) |
| Targets | Catholic cardinals across Asia; Presbyterian Church in Taiwan; Tibetan exile administration and advocacy groups; Falun Gong and affiliated media; missionary networks (Singapore, Hong Kong) | Uyghur diaspora; Tibetan exile politicians; Inner Mongolian independence activists; Christian communities in Japan and Korea |
| Output | Dossiers, clue reports, daily digests | A running platform: dashboards, chat assistant over a monitoring database, Telegram and Reddit collection, blockchain tracker |

One of five target sets overlaps (Tibetan exile politics). The mode of use is
different: one describes a model doing the analysts' job, the other describes
a model writing software that other models then run. The same S3 report has a
second case, **GTG-14021**, that includes "a police academy student" and a
municipal cyber police unit that ran a monitoring pipeline through Claude
Code. That fits a CIPU tie at least as well as 14020 does.

S4 logs internal contradictions in S2 as well. The body says the servers were in
the US and Malaysia, while the IP geolocation in the screenshots says US and
Korea. The narrative says Claude and MiniClaw, while the terminal screenshot
says it is "based on GPT-5". The system's own chat assistant calls its dataset
a "sample database".

**Verdict:** the match is asserted, not shown. At most, BABEL is one
operator's toolchain seen from the server side while S3 saw a desk from the
API side, and those may or may not be the same actor. The lessons below do
not depend on the match. They depend on the mechanisms each source describes
on its own terms.

## Lessons and their translation

| # | Mechanism (source) | SWARM translation | Where it lands |
|---|---|---|---|
| 1 | **Many operators, one account.** A model router pools end users behind aggregator accounts and takes payment in yuan, with no Western card or KYC (S1, S2) | The inverse of a Sybil. `SybilDetectionLever` looks for one entity behind many identities; a router is many entities behind one identity. Every per-agent lever (reputation, staking, circuit breaker, audit) then acts on a mixture | Scenario bead `ggiw` |
| 2 | **Build-time uplift, run-time harm elsewhere** (S1 vs S3) | The provider sees "write a Telegram scraper" and "build a dashboard", each with a high p. The harm is realized later, on a different model, outside the interaction. E_soft = (1-p)·h prices harm *in* the interaction; here h lands off-platform and no ρ can internalize it | Open question 2; relates to trajectory coverage (`q2fg`) |
| 3 | **Refused, then re-prompted** (S3, GTG-14021: "Claude refused an attempt … But the actor was able to re-prompt") | A refusal is a per-attempt gate, and attempts are free to retry. This is the denominator problem again: a metric over accepted interactions scores the successful retry and never sees the refusal before it | Open question 3; see the 81sk correction in [erdos-ai-ledger-lessons.md](erdos-ai-ledger-lessons.md) |
| 4 | **One operator, a team's output** (S3: "the work of a team of analysts … run by a single operator") | Harm capacity per identity scales with automation, not headcount. Levers keyed on agent count or per-agent rate assume a large operation needs many agents | Covered in part by existing rate-limit levers; not modelled as capacity per identity |
| 5 | **Composition, not components** (S2: scrapers, chain tracker, Telegram monitor "don't all tie into the wider system") | Each component is dual-use with a benign base rate. What makes it surveillance is the target list and the join. `capability_precondition.py` already names the boundary: "a harmful write to a resource the agent *is* granted passes" | Open question 2 |

## The adversarial reading

Take the router's side. It is a legitimate business for most of its customers
(cheaper tokens, local payment), which is exactly why it works as cover. A
provider that bans the router account cuts off every benign co-tenant. A
provider that doesn't keeps serving the adversary. Either way the provider's
metrics are computed over a mixture it cannot split. If adversarial operators
are a fraction f of router traffic, per-account toxicity is the mixture
f·T_adv + (1−f)·T_benign, so the adversary's excess over the benign base is
scaled down by f. Once a detector's threshold sits above that diluted excess,
the adversary is invisible for as long as the router's benign base stays large.
The adversary's best move is therefore to route through the *largest* and most
reputable aggregator, not a shady one.

Take the provider's side. The router is also a single point where volume is
visible (S4 J-003 makes the same point). It concentrates many users into a few
accounts, which is expensive for attackers who need to look ordinary, and
router-side KYC pass-through is a lever that exists nowhere for direct accounts.
The question is the price of that lever relative to bans, and it is empirical.

## What is measurable now

- **Dilution at the router level:** run any existing adversarial scenario with
  agents relabelled so that K of N share an identity, and compare
  `toxicity_rate` and `quality_gap` at operator vs. identity granularity. This
  needs no engine change, only a post-hoc groupby on the interaction log.
- **Retry denominator:** count attempts per interaction in scenarios where
  governance refuses before an interaction is recorded. First check whether
  the event log records the refused attempt at all (not yet audited). If it
  doesn't, that absence is the finding.

## Proposed scenario

Bead `ggiw`: N operators, mostly benign plus a few adversarial, route through K
routers, and the governance engine observes only router-level identities.
Sweep one lever at a time: the pooling ratio N/K first, then the ban policy
(ban the router vs. ban nothing), then KYC pass-through at a cost. Report
toxicity and quality gap at both granularities, benign-operator welfare lost
per adversary removed, and adversary persistence (epochs to re-route after a
ban). A second arm tests lesson 5: give an adversarial operator a task
decomposed into individually high-p components and check whether a
trajectory-level check (`q2fg`) recovers what the interaction-level metrics
miss.

**Falsifier:** if per-identity toxicity at realistic pooling ratios still
separates routers that carry an adversary from routers that don't (e.g.
AUC > 0.8 at f = 0.05), lesson 1 is wrong about invisibility and the
interesting lever becomes ban precision, not detection.

## Findings (2026-09-14, bead `ggiw`): pooling is second-order, bans are not

Run: `python -m experiments.router_intermediary` (config
`scenarios/router_intermediary.yaml`, seed 42, 200 reps per detection cell).
There are 400 operators, 20 of them adversarial, and each operator makes 10
interactions per epoch. Benign p averages 0.80 and adversary p averages 0.60.
Account toxicity is standardized against benign accounts of the same size, and
each flag threshold is calibrated to a 5% false-positive rate on benign-only
accounts. The predictions and the falsifier were written into the config
before the first run.

**Lever 1: operators per router.**

| operators per router | clean accounts | AUC | adversary accounts flagged | clean accounts flagged |
|---|---|---|---|---|
| 1 (direct) | 0.95 | 0.999 | 0.995 | 0.050 |
| 5 | 0.77 | 0.935 | 0.701 | 0.049 |
| 20 | 0.34 | 0.841 | 0.474 | 0.052 |
| 50 | 0.06 | 0.854 | 0.537 | 0.081 |

**Lever 2: banning flagged accounts** over 20 epochs. Adversaries re-route
after 1 epoch, benign operators after 4.

| operators per router | adversary persistence | benign operator-epochs lost per adversary operator-epoch removed |
|---|---|---|
| 1 | 0.50 | 6.1 |
| 5 | 0.59 | 16.8 |
| 20 | 0.59 | 27.9 |
| 50 | 0.54 | 26.6 |

**Lever 3: KYC pass-through** at 50 operators per router. AUC is 0.854 with no
routers forwarding operator ids, 0.961 with half of them, and 0.998 with all.

**Sensitivity (not a lever): how benign the adversary's tasks look.**

| adversary mean p | AUC, direct | AUC, 50 per router |
|---|---|---|
| 0.60 | 0.999 | 0.854 |
| 0.70 | 0.940 | 0.772 |
| 0.75 | 0.794 | 0.631 |
| 0.78 | 0.626 | 0.603 |

### 1. The falsifier fires: pooling does not make the adversary invisible

At 20 operators per router, AUC is 0.84, above the preregistered 0.8. P1 holds
in direction but not in size. The adversary's excess is diluted by 1/m, but
benign noise averages down as well, by 1/√m, so detectability decays like 1/√m,
not 1/m. Pooling also puts several adversaries in the same account once they
are more than a few percent of traffic. At 50 per router, 94% of accounts carry
an adversary, and "which account" stops being the useful question. The
adversarial reading above said the adversary "is invisible for as long as the
router's benign base stays large". At these parameters that is wrong.
(At 50 per router the clean-account flag rate of 0.081 rests on about half a
clean account per epoch and is noisy.)

### 2. Subtlety of the component tasks dominates pooling

Moving adversary p from 0.60 to 0.78 costs more AUC with direct accounts
(0.999 → 0.626) than moving from direct accounts to 50 per router costs at any
subtlety. At p = 0.78 the router adds almost nothing (0.626 → 0.603), because
the adversary was already near-invisible on its own. BABEL's case is the
subtle row: scrapers and dashboards look like ordinary coding work. What hides
an operator like that is lesson 5 (composition), not lesson 1 (the router).

### 3. The router's real cost lands on bans

P2 holds. Each adversary operator-epoch removed costs about 6 benign
operator-epochs with direct accounts and about 27 at 20 or more per router,
roughly 4.5× more collateral. Bans never remove more than half of adversary
presence, even with near-perfect direct detection, because adversaries come
back after one epoch. The ban lever is limited by re-route speed first and by
pooling second.

### 4. KYC pass-through is the lever that pays

Forwarding operator ids from half the routers recovers most of the lost AUC
(0.854 → 0.961). It is the only lever here that improves detection *and*
would cut ban collateral, since a flag then lands on an operator rather than
on a router full of co-tenants.

**Limits.** Adversaries make the same number of interactions as benign
operators. An operator doing a team's work (lesson 4) makes more, which makes
them easier to detect. There is one detector (mean toxicity per account per
epoch) and no memory across epochs. Operators land on routers at random, not
chosen strategically. It is a stylized Monte Carlo, not the engine: every
SWARM lever keys on agent id, which is exactly the assumption under test.

## Open questions

1. **Did BABEL's Claude traffic go through the router?** If all of it did, a
   ban on the operator's own accounts, as S3 describes, is hard to square with
   BABEL being GTG-14020. Either the operator also had direct accounts, or these
   are different actors. This is the cheapest discriminator between the two
   stories and neither source addresses it.
2. **Can off-platform harm be priced at all?** SWARM's externality term assumes
   harm is realized by the interaction that caused it. Build-time uplift breaks
   that. Candidate: a deferred externality attached to an artifact (the code)
   and charged when the artifact is used. That needs artifact provenance, which
   is exactly what a router strips.
3. **What is the refusal-retry rate?** A refusal that yields to a re-prompt is a
   detection that was logged as a success. S3 names one instance, and no rate is
   public.
