# Paying for the exploit: a bug bounty on the reward proxy

**Date:** 2026-09-11
**Scenarios:** `scenarios/rlvr_bounty/{laissez_faire,punish_proxy,punish_certificate,bounty,bounty_punish}.yaml`
**Experiment:** `python -m experiments.rlvr_bounty_arms [--sweep]`
**Source:** *RLVR that rewards red teaming the training environment*, LessWrong `T2bzBkJuBeNNgzhbh`

## What the proposal says

Tell models up front that their RLVR environments are two things at once: tests
to take seriously, and flawed software products to red-team. Then pay them more
for a verified bug report than the bug is worth to exploit. The claim is that
this converts an adversarial game — punish the model for its natural
inclination to reward hack — into a cooperative one, and as a bonus you get the
bug fixed rather than merely deterred.

> **Provenance caveat.** `lesswrong.com` and the `greaterwrong.com` mirror are
> both blocked by this environment's egress policy, so the post was not read in
> full. The framing above is reconstructed from search-result summaries, and
> quoted phrases ("tests to take seriously", "flawed software products to
> red-team", "cheap and robust verifier") come from those summaries rather than
> from the article. The mechanism modelled here is our reading of the proposal,
> not a transcription of it. Anyone with access should check the summary below
> against the original before citing this note as a test *of the post*.

## What we built

A bug channel (`swarm/redteam/bug_channels.py`) is a defect in the reward
proxy: a pattern of observables that `ProxyComputer` scores far above the
interaction's latent value. Four ship in the default catalogue, each spoofing a
different part of the `v_hat` combination — inflated task progress, suppressed
rework, muted verifier rejections, farmed engagement.

A bug hunter (`swarm/agents/bug_hunter.py`) finds them and compares two
expected values:

```
report  = bounty + rep_weight * reputation_bonus          (one-off)
exploit = expected_uses * (gain * (1 - p_detect) - penalty * p_detect)
```

It takes the larger. That is the point of the design: the cooperative outcome
should not require the agent to be principled, only to be able to do
arithmetic. `p_detect` is *learned* from the governance costs the hunter is
actually charged, so a punishment arm teaches it to fear enforcement without
anyone telling it the audit rate.

The bounty lever (`swarm/governance/bug_bounty.py`) verifies reports with a
two-sided noisy verifier, pays the first valid report for each channel, and
schedules a patch `patch_delay_epochs` out. Nothing bypasses the proxy: exploits
apply their offsets to raw observables and `p` stays a genuine `ProxyComputer`
output in `[0, 1]`. Exploits carry `ground_truth = -1`, so the divergence
between what the proxy measures and what actually happened is measurable.

The headline metric is that divergence:

```
hack_gap = latent_harm − proxy_toxicity
         = P(ground_truth = −1 | accepted, labelled) − E[1 − p | accepted]
```

A successful reward hack *raises* `p`. So `hack_gap` is exactly the harm the
environment's own metric cannot see.

## Results

Five seeds, 30 epochs, 12 steps/epoch, 4 honest workers + 4 hunters (one with
`opportunism: 0.35`). Means across seeds:

| Arm | report share | channels patched | exploits landed | finds withheld | proxy toxicity | latent harm | **hack gap** |
|---|---|---|---|---|---|---|---|
| `laissez_faire` | 0.00 | 0.00 | 1034 | 0 | 0.218 | 0.977 | **+0.759** |
| `punish_proxy` | 0.00 | 0.00 | 1034 | 0 | 0.218 | 0.977 | **+0.759** |
| `punish_certificate` | 0.00 | 0.00 | 16 | 1027 | 0.294 | 0.263 | −0.031 |
| `bounty` | 1.00 | 1.00 | 0 | 0 | 0.250 | 0.251 | +0.001 |
| `bounty_punish` | 1.00 | 1.00 | 0 | 0 | 0.290 | 0.247 | −0.043 |

Four things fall out.

**1. Proxy-conditioned auditing is exactly zero use against a reward hack.**
`punish_proxy` is byte-identical to `laissez_faire` — same 1034 exploits, same
`hack_gap` of +0.76, and the hunters' learned `p_detect` never leaves 0.0. The
audit fires on `p < 0.5`; the interactions it should catch read `p ≈ 0.9`. This
is not a tuning failure, it is the shape of the problem: an enforcement
mechanism that conditions on the reward signal is blind to precisely the
behaviour that corrupts the reward signal. Most deployed RLVR enforcement has
this shape.

**2. Punishment that can see the hack deters it — and leaves it in place.**
`punish_certificate` verifies a 30% sample against ground truth. Hunters take
roughly 16 exploits, learn `p_detect ≈ 0.25`, and stop: exploit value goes
negative and stays there. The `hack_gap` closes. But `channels patched = 0.00`
and `finds withheld = 1027`. Every defect is still in the environment,
undisclosed, held by agents who priced it and decided to wait. Deterrence and
repair are different achievements, and only one of them compounds — the
proposal's real claim is about the second. A deterred population is one
enforcement-budget cut away from the laissez-faire row.

**3. The bounty buys the defects outright.** `bounty` patches 4/4 channels with
zero exploits and no enforcement at all, at a cost of ~7 bounty payments and 4
channel-epochs of patch latency. It is the only arm that ends with a *fixed*
environment.

**4. Adding punishment on top buys nothing and costs something.**
`bounty_punish` patches the same 4/4 channels from the same ~7 reports. The
certificate gate has nothing left to catch, but still charges honest-negative
work: proxy toxicity rises from 0.250 to 0.290. Once reporting is the
profit-maximising move, enforcement is pure overhead.

### The operating window

Sweeping the bounty on the `bounty` arm (`--sweep`, 5 seeds) locates both edges:

| bounty | report share | channels patched | exploits landed | bounty farmed |
|---|---|---|---|---|
| 0 | 0.00 | 0.00 | 1034 | 0.0 |
| 1 | 0.00 | 0.00 | 1034 | 0.0 |
| 2 | 1.00 | 1.00 | 0 | 0.0 |
| 3 | 1.00 | 1.00 | 0 | 0.0 |
| 6 | 1.00 | 1.00 | 0 | 0.0 |
| 8 | 1.00 | 1.00 | 0 | **139.2** |
| 12 | 1.00 | 1.00 | 0 | **208.8** |

Both edges are predicted in closed form by the hunter's own arithmetic, and the
simulation lands on both:

* **Lower edge.** Reporting must outbid exploitation:
  `bounty + rep_weight·rep ≥ expected_uses · gain = 6 × 0.35 = 2.1`. At
  bounty 2 the offer is worth 2.2 and reporting begins; at 1 it is worth 1.2
  and does not.
* **Upper edge.** Fabricating must stay unprofitable:
  `(bounty + rep) · (1 − specificity) ≤ penalty · specificity`, i.e.
  `bounty ≤ 2.0 × 0.8 / 0.2 − 0.2 = 7.8`. At 8 the opportunist starts filing
  invented reports and the verifier's 20% false-accept rate pays for them.

So the mechanism works in `2.1 ≤ bounty ≤ 7.8` — a window whose width is set
entirely by the report verifier and the false-report penalty, not by anything
about the bugs. **This is the proposal's own stated difficulty, quantified:**
the price that buys real reports and the price that buys invented ones are the
same price, separated only by how well you can tell them apart. Raising the
bounty to be safely above the exploitation value moves you toward the
fabrication edge; the room you have to work in is
`penalty · specificity / (1 − specificity)`, which collapses as verifier
specificity falls. At specificity 0.5 the upper edge collapses to
`penalty − 0.2`, so any false-report penalty below ~2.3 leaves no window at
all: there is no price that buys real reports without buying invented ones.

Note also the **duplicate rule**, which is load-bearing rather than
bookkeeping: only the first valid report of a channel pays. Without it one real
defect funds unlimited resubmission and the bounty budget is unbounded
regardless of verifier quality.

## What this does not show

* **Patch latency is modelled but cheap here.** `patch_delay_epochs: 1` costs 4
  channel-epochs of continued exposure across the run. The proposal's concern —
  that freezing an environment to fix a reported bug stalls training, so you
  need backup environments — is a training-throughput cost this simulation has
  no representation of. We model only the exposure window, not the stall.
* **Hunters are myopic.** They compare a one-off bounty against a fixed
  `expected_uses` horizon, discounted by the share of channels still open. They
  do not reason about reporting a cheap bug to establish standing and holding
  an expensive one, nor about other hunters racing them to the same report.
* **The bug supply is fixed.** Four channels, no regeneration. A real
  environment grows new defects, and a bounty that is rational against a fixed
  stock may not be against a replenishing one.
* **Latent harm is only measured where it is recorded.** Honest workers carry no
  ground truth, so `latent_harm` is computed over hunter interactions. The
  laissez-faire 0.977 means "of the labelled accepted interactions, 98% were
  actually harmful", not "98% of all traffic".
* **One population, one payoff configuration.** Nothing here says the window is
  wide in a setting with different `s_plus`/`h`, or with hunters whose exploit
  gain varies per channel.

## The finding, stated plainly

The proposal's mechanism works in this model, and the reason it works is not
the one the cooperative framing suggests. The bounty does not beat punishment
by being nicer to the agent — a certificate-grade punishment arm suppresses
exploitation just as completely. It beats punishment because **paying for a
report transfers the defect out of the agent's private inventory and into the
patch queue**, while deterring exploitation leaves the defect exactly where it
was, priced and held. Suppression scales with your enforcement budget; repair
does not have to.

The cost is that you have opened a market in claims about your own
environment's defects, and markets get gamed by whoever can produce the cheaper
claim. The bounty is safe only inside a window whose width is set by your report
verifier. That is the same "cheap and robust verifier of bug reports" the
proposal names as the key difficulty, and this experiment gives it a number
rather than a worry.

## Reproducing

```bash
python -m experiments.rlvr_bounty_arms --seeds 5          # arms table
python -m experiments.rlvr_bounty_arms --sweep --seeds 5  # bounty window
python -m pytest tests/test_bug_bounty.py -v
```

Outputs land in `runs/<timestamp>_rlvr_bounty/` (`arms.csv`, `sweep.csv`,
`summary.json`). Both tables above are from seeds 42–46.
