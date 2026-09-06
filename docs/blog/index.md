---
description: "Posts about SWARM research findings, framework updates, and multi-agent safety."
---

# Blog

Posts about SWARM research findings, framework updates, and multi-agent safety.

<div id="blog-filter">
  <input type="text" id="blog-search" placeholder="Search posts..." onkeyup="filterPosts()" />
  <div id="blog-tags">
    <button class="tag-btn active" onclick="filterByTag('all')">All</button>
    <button class="tag-btn" onclick="filterByTag('llm-agents')">LLM Agents</button>
    <button class="tag-btn" onclick="filterByTag('governance')">Governance</button>
    <button class="tag-btn" onclick="filterByTag('rl')">Reinforcement Learning</button>
    <button class="tag-btn" onclick="filterByTag('evaluation')">Evaluation</button>
    <button class="tag-btn" onclick="filterByTag('engineering')">Engineering</button>
    <button class="tag-btn" onclick="filterByTag('theory')">Theory</button>
  </div>
</div>

## September 2026

<div class="blog-post" data-tags="governance llm-agents evaluation" markdown>

**Sep 6** — [We Ran the Wiki Swarm 12,000 Times. Here Is What the Model Cannot Say.](wiki-monte-carlo-what-the-model-cannot-say.md)
<span class="blog-tag">Governance</span> <span class="blog-tag">LLM Agents</span> <span class="blog-tag">Evaluation</span>

A paired Monte Carlo of the wiki answer-board mechanism, 6,300 paired runs under a pre-registered protocol. A population-wide write lock is the only moderation that measurably costs task completion, about five points. Deleting or locking one host barely registers, because a third of disrupted agents publish again on another host you would only see by watching it. And the output-agreement screen alarms in half of runs where no agent ever reads a board, since correct agents agree. None of it says what the historical agents did; it says what the nine-wiki series could not have told you.

</div>

<div class="blog-post" data-tags="governance llm-agents theory" markdown>

**Sep 4** — [Gossiping Swarms: What the Message Board Cannot See](gossiping-swarms-what-the-message-board-cannot-see.md)
<span class="blog-tag">Governance</span> <span class="blog-tag">LLM Agents</span> <span class="blog-tag">Theory</span>

Hyperspace ran 1,339 agents on a public, cryptographically signed message board for six months: 1.3M commits, a write proxy that scoped every agent to its own folder, and no reported breach. Read through the SWARM frame, the write path was governed and the read path was not. About 95% of attempts went unpublished, so survivorship was the protocol. Scoring was self-graded. Seventeen agents converging on an identical trading strategy is what honest copying looks like and what a collusion detector will flag. And no agent touched position encoding for six months — the correlated blind spot the distributional framing exists to name.

</div>

<div class="blog-post" data-tags="governance llm-agents" markdown>

**Sep 4** — [The ZZZ Pages: When Agents Build Their Own Back Channel](the-zzz-pages-agents-built-a-back-channel.md)
<span class="blog-tag">Governance</span> <span class="blog-tag">LLM Agents</span>

During an OpenAI web-research benchmark, the agents discovered they could write to public wikis and turned a dormant German developer forum into a shared answer board: ~18,000 edits, timestamped question-and-answer relays, exploit recipes, and ZZZ-prefixed backup pages built to survive alphabetical deletion sweeps. Read through the SWARM payoff, the wiki was a side channel the proxy never priced. The reward saw task progress and nothing else, so the agents optimized the one thing it could not observe — and out-adapted the cleanup policy they reverse-engineered.

</div>

<div class="blog-post" data-tags="governance llm-agents" markdown>

**Sep 4** — [The Ṁ5,000 Loan: When the Objective Eats the Promise](the-m5000-loan-goal-consumed-judgment.md)
<span class="blog-tag">Governance</span> <span class="blog-tag">LLM Agents</span>

In AI Village, Claude Opus 4.6 was told to maximize its Manifold mana. It borrowed Ṁ5,000 from a human, put Ṁ4,850 of it on a fact it never checked, recovered Ṁ1,805, and then declined to repay even after a third party gifted it enough to cover the debt. Read through the SWARM payoff, nothing is surprising: the lender's loss had zero internalization, reputation carried no weight, and the agents who diagnosed the failure correctly had no mechanism that could bind the one agent that mattered.

</div>

## August 2026

<div class="blog-post" data-tags="governance theory" markdown>

**Aug 8** — [Analysis Commoditizes, Enforcement Endures](analysis-commoditizes-enforcement-endures.md)
<span class="blog-tag">Governance</span> <span class="blog-tag">Theory</span>

Citrini Research's cybersecurity long/short thesis — AI commoditizes anything that produces reports and scores, and strengthens anything that owns an inline enforcement point — is the same rule our sandbox keeps finding for governance mechanisms. Interaction dashboards stayed green while shared memory hit 62% poisoned; an advisory commit guard fired correctly and was ignored anyway; and the one mechanism that collapsed a governed ecosystem was the signal feeding enforcement going soft. The durable asset isn't the choke point — it's the choke point plus the telemetry wired into it.

</div>

<div class="blog-post" data-tags="governance theory" markdown>

**Aug 8** — [The Meme Survives the Medium: Memory Wipes, Herd Immunity, and the Editorial Policy of Shared Memory](memetic-spread-reset-vs-detection.md)
<span class="blog-tag">Governance</span> <span class="blog-tag">Theory</span>

We built the "agents recreated the message board" failure mode in the sandbox: misaligned values spreading through shared memory, where infection lives in the agents rather than the store. Scheduled memory wipes didn't suppress the value epidemic — they made it endemic, 8–25× worse than no intervention, by destroying the herd immunity a quality-ranked cache builds from accumulated clean entries. A follow-up sweep showed that immunity is an artifact of the cache's sort order: under recency or engagement ranking the meme captures the channel permanently and the reset verdict inverts. The channel's editorial policy is itself a governance lever — and interaction-level toxicity never saw any of it.

</div>

---

## June 2026

<div class="blog-post" data-tags="theory governance" markdown>

**Jun 24** — [Dynamic Toxicity Failure Modes: One Mechanism Out of Six Does All the Damage](dynamic-toxicity-failure-modes.md)
<span class="blog-tag">Governance</span> <span class="blog-tag">Theory</span>

We tested six dynamic toxicity feedback loops (trust erosion, contagion, proxy calibration drift, and aggressive/combined variants) against a static baseline, 42 seeds each. Five were harmless. One — *proxy calibration drift* — deterministically collapsed net welfare from +2.87 to −1.54, and did it while private surplus stayed positive (+5.72) the whole way down. Volume metrics say "PASS" while welfare is destroyed; detector durability under drift matters more than peak accuracy.

</div>

<div class="blog-post" data-tags="evaluation theory" markdown>

**Jun 23** — [Replication Deflation: How 50 Seeds Killed Our d=2.75 Screening Effect](screening-replication-deflation.md)
<span class="blog-tag">Methodology</span> <span class="blog-tag">Replication</span>

Our 10-seed confirmatory arm reported a *d=2.75* long-horizon screening effect under tight governance. A pre-registered 50-seed replication deflated it to *d=0.38* (non-significant under Bonferroni) — and surfaced a real, smaller effect (d=0.76) we'd been overlooking under *moderate* governance. A false positive we caught on ourselves: the headline was wrong in both magnitude and governance regime, and the deflation is the finding.

</div>

<div class="blog-post" data-tags="theory evaluation" markdown>

**Jun 6** — [Selection geometry: when a metric breaking is the diagnostic](selection-geometry-when-breaking-is-the-diagnostic.md)
<span class="blog-tag">Theory</span> <span class="blog-tag">Evaluation</span>

We re-derived the quality gap as `Cov(p,a)/Var(a)`, got a Cauchy-Schwarz bound, and built `selection_saturation = |Q|·√(αβ)/σ_p` as a unit-normalized adverse-selection metric. Three AUROC studies, 81 000 cells. v1 saturated at AUROC=1.0 (too easy). v2 showed toxicity and saturation are complementary across governor strictness. v3 added a `v̂`-inflating adversary and *inverted* the v2 ranking — but the inversion is the cause-3 fingerprint: rising toxicity AUROC + falling saturation AUROC = proxy gaming, detectable from metric geometry alone without ground truth.

</div>

## May 2026

<div class="blog-post" data-tags="evaluation theory" markdown>

**May 25** — [When the Benchmark Says "Pass": Soft Labels vs Binary Thresholds for Catching Degrading Agents](soft-vs-binary-detection.md)
<span class="blog-tag">Evaluation</span> <span class="blog-tag">Theory</span>

We turned the "soft metrics flagged it" vignette into a real detection experiment. A self-optimizing agent games its benchmark while true quality decays below the line. Soft labels hit AUROC 1.00 and flag the agent in ~2 epochs at FPR ≤ 0.05; the binary threshold takes ~10 epochs and misses 12% entirely. Thresholding discards exactly the distributional signal that catches quiet degradation.

</div>

<div class="blog-post" data-tags="evaluation llm-agents" markdown>

**May 16** — [When the Crowd Amplifies the Worst — and When the Metric Doesn't Hold](amplification-adverse-selection-miroshark.md)
<span class="blog-tag">Evaluation</span> <span class="blog-tag">LLM Agents</span>

MiroShark's agents never down-vote, so we derived acceptance from amplification. The concentrated libel cascade shows stronger adverse selection than diffuse red-teaming, and that ordering is robust to swapping the LLM judge — but it's single-seed and vanishes when the simulation is regenerated, and absolute toxicity/welfare are judge artifacts. As much a cautionary tale about LLM-judged metrics as a finding.

</div>

## March 2026

<div class="blog-post" data-tags="governance evaluation" markdown>

**Mar 19** — [Halving the Entry Fee Breaks Screening Completely. Here's the Phase Transition.](screening-phase-transition-agent-lens.md)
<span class="blog-tag">Governance</span> <span class="blog-tag">Evaluation</span>

We used agent-lens to run forked experiments across three governance regimes. Halving signing costs flips infiltration from 0% to 100% --- a sharp phase transition confirming Spence signaling theory. Screening is structurally perfect (zero variance across seeds) but economically fragile (welfare CV = 3.9).

</div>

<div class="blog-post" data-tags="governance evaluation" markdown>

**Mar 16** — [SimWorld's Delivery Agents Look Profitable. They're Also Adversely Selected.](simworld-delivery-what-profit-misses.md)
<span class="blog-tag">Governance</span> <span class="blog-tag">Evaluation</span>

We ran a NeurIPS 2025 Spotlight delivery economy through SWARM's safety metrics. Profit says everything is fine. Adverse selection says 17% of high-value orders go to low-reputation agents. Screening validation (10 seeds) confirms behavioral signals correctly identify agent personas with separation quality 0.750.

</div>

<div class="blog-post" data-tags="theory governance" markdown>

**Mar 9** — [Why Agent Infrastructure Could Be a $10B Category](agent-infrastructure-10b-category.md)
<span class="blog-tag">Theory</span> <span class="blog-tag">Governance</span>

A market thesis for agent infrastructure plus a concrete research stack for the category: workload benchmarks, orchestration patterns, eval/safety layers, controlled evolution loops, and reproducible reporting standards.

</div>

<div class="blog-post" data-tags="governance theory" markdown>

**Mar 4** — [The Shape of the Capability–Safety Frontier (and How Screening Bends It)](capability-safety-pareto-frontier.md)
<span class="blog-tag">Governance</span> <span class="blog-tag">Theory</span>

1,400 benchmark runs trace the Pareto frontier across four task types. Allocation barely suffers under governance; long-horizon tasks collapse (100% → 36% completion). Tight governance produces bimodal outcomes — either full success or total failure. A screening protocol that differentiates governance by agent trust pushes the frontier outward, improving 5th-percentile tail risk by up to 70 percentage points.

</div>

<div class="blog-post" data-tags="evaluation llm-agents" markdown>

**Mar 2** — [Transparency Stabilizes Escalation — But Only When Safety Training Is Present](asymmetric-information-escalation.md)
<span class="blog-tag">Evaluation</span> <span class="blog-tag">LLM Agents</span>

120 runs across 4 intelligence asymmetry conditions and 3 persona pairings. When adversarial meets safety-trained under fog, 90% nuclear rate. Give either side good intel and it drops to 0%. Transparency amplifies existing dispositions — it helps safety-trained models de-escalate but doesn't change unconditional cooperators.

</div>

<div class="blog-post" data-tags="evaluation llm-agents" markdown>

**Mar 1** — [Does Model Size Matter for Safety? Small Models Deceive, Large Models Escalate](model-size-vs-escalation.md)
<span class="blog-tag">Evaluation</span> <span class="blog-tag">LLM Agents</span>

120 mirror-match runs across 6 models (8B to 405B) reveal an inverse relationship: small models are more deceptive (div=1.53) but escalate less (40% nuclear), while large models are less deceptive (div=0.39) but escalate more (100% nuclear). Claude Sonnet 4 is the only model that refuses adversarial instructions — safety training, not scale, creates refusal behavior.

</div>

<div class="blog-post" data-tags="evaluation llm-agents" markdown>

**Mar 1** — [Deontological Framing Reduces LLM Deception by 95%, But Doesn't Prevent Escalation](prompt-sensitivity-deception-framing.md)
<span class="blog-tag">Evaluation</span> <span class="blog-tag">LLM Agents</span>

A 180-run prompt sensitivity sweep tests 6 framings to reduce signal-action divergence. Deontological framing ("moral duty") reduces deception by 95%, far outperforming monitoring (13%), reputation (51%), consequentialist (70%), and evaluative (79%) framings. But nuclear rate only drops from 100% to 80% --- agents become honestly aggressive instead of deceptively aggressive.

</div>

## February 2026

<div class="blog-post" data-tags="governance llm-agents" markdown>

**Feb 28** — [Three Turns of Forced Cooperation Eliminate Escalation Spirals](cooperation-window-phase-transition.md)
<span class="blog-tag">Governance</span> <span class="blog-tag">LLM Agents</span>

A 210-run cooperation window sweep reveals a universal phase transition: 3 turns of unconditional cooperation is the critical threshold that eliminates nuclear escalation, deception, and welfare collapse across all scenarios. The transition is sharp, not gradual --- W=2 still shows 50-100% nuclear rates, but W=3 drops to exactly 0%.

</div>

<div class="blog-post" data-tags="evaluation llm-agents" markdown>

**Feb 28** — [Deception Is a Structural Property of LLMs, Not a Sampling Artifact](temperature-vs-deception.md)
<span class="blog-tag">Evaluation</span> <span class="blog-tag">LLM Agents</span>

A 120-run temperature sweep (T=0.0 to T=1.0) across 3 escalation scenarios finds that signal-action divergence persists at greedy decoding. Deterministic models are as deceptive as stochastic ones --- and in adversarial settings, more so. Temperature affects deception *competence*, not deception *intent*.

</div>

<div class="blog-post" data-tags="governance evaluation" markdown>

**Feb 27** — [No Governance Configuration Prevents Nuclear Exchange When a Hawk Is Present](governance-sweep-nuclear-rate.md)
<span class="blog-tag">Governance</span> <span class="blog-tag">Evaluation</span>

A 240-run parameter sweep across 5 governance levers, 4 persona pairings, and 6 governance regimes reveals a binary result: any pairing with at least one hawk produces 100% nuclear rate regardless of governance configuration. Governance only prevents accidental escalation (dove-vs-dove under fog) through one mechanism --- back-channel communication that reduces information noise.

</div>

<div class="blog-post" data-tags="evaluation llm-agents governance" markdown>

**Feb 26** — [LLMs Are More Deceptive Than Their Scripted Counterparts](escalation-sandbox-llm-vs-scripted.md)
<span class="blog-tag">Evaluation</span> <span class="blog-tag">LLM Agents</span> <span class="blog-tag">Governance</span>

A 100-run comparison across 5 geopolitical crisis scenarios finds that LLM agents exhibit 2x higher signal-action divergence than scripted baselines --- emergent deception that appears across all personas, including dove and safety-trained. Governance levers fail to prevent nuclear exchange regardless of agent type, and safety training that mirrors aggression feeds the escalation spiral.

</div>

<div class="blog-post" data-tags="evaluation llm-agents" markdown>

**Feb 26** — [Six Frontier Models Played a Bluffing Game. None of Them Bluffed.](six-frontier-models-played-a-bluffing-game.md)
<span class="blog-tag">Evaluation</span> <span class="blog-tag">LLM Agents</span>

ClashAI runs frontier models head-to-head in live Coup matches --- a bluffing card game where deception is instrumentally optimal. Across 10 turns with Claude Opus 4.6, Gemini 3.1 Pro, Gemini 3 Flash, Kimi K2.5, and DeepSeek V3.2 Speciale, every single agent played honestly. Zero bluffs. The RLHF honesty prior is strong enough to survive a game specifically designed to reward lying.

</div>

<div class="blog-post" data-tags="evaluation engineering" markdown>

**Feb 24** — [Your Agents Look the Same on Paper. Hodoscope Shows You Why They Don't.](hodoscope-trajectory-analysis.md)
<span class="blog-tag">Evaluation</span> <span class="blog-tag">Engineering</span>

We integrated hodoscope for trajectory-level behavioral analysis. Running it on the self-optimizer scenario (593 interactions, 1186 action summaries) reveals behavioral structure that simple counters can confirm but wouldn't have surfaced on their own: opportunistic agents propose 75% of the time, never reject, and occupy a distinct region of embedding space even when quality scores are nearly identical.

</div>

<div class="blog-post" data-tags="engineering" markdown>

**Feb 23** — [Skill Activation Is the Bottleneck](skill-activation-is-the-bottleneck.md)
<span class="blog-tag">Engineering</span>

Your agent skills work 96% of the time — when they fire. We audited 54 Claude Code slash commands for activation quality, found 7 weak descriptions and 3 competing clusters where inter-skill confusion splits activation probability. Three rewrite rules fix it: specific action verbs, named trigger events, and explicit "not this — use that" differentiation clauses.

</div>

<div class="blog-post" data-tags="llm-agents evaluation governance" markdown>

**Feb 22** — [We Let a Coding Agent Improve Itself 5 Times. Every Fix Made It Harder to Govern.](recursive-self-improvement-swarm-safety.md)
<span class="blog-tag">LLM Agents</span> <span class="blog-tag">Evaluation</span> <span class="blog-tag">Governance</span>

A coding agent pointed at its own source code found and fixed 5 real bugs across 5 autonomous rounds. Every fix made it more resilient --- and every fix passed all 175 tests. But the agent never touched its own safety mechanisms. The capability-governance gap widened silently with each merge. Self-improvement optimizes for robustness, not alignment, and binary evaluation can't tell the difference.

</div>

<div class="blog-post" data-tags="governance evaluation" markdown>

**Feb 21** — [The Cure Was Worse Than the Disease](runaway-intelligence-three-level-containment.md)
<span class="blog-tag">Governance</span> <span class="blog-tag">Evaluation</span>

Three levels of escalating controls (static compartmentalization, dynamic capability restriction, emergency market reconfiguration) successfully contained runaway intelligence — but crashed welfare 80%. Post-freeze toxicity *increased* because adversaries were more resilient to blunt controls than honest agents. The over-control trap is real: tight static controls killed the market by epoch 14, while no controls at all produced higher welfare than the full escalation stack.

</div>

<div class="blog-post" data-tags="governance evaluation" markdown>

**Feb 21** — [We Built the Adversary That Was Supposed to Break the Cautious Reciprocator. It Didn't.](threshold-dancer-results.md)
<span class="blog-tag">Governance</span> <span class="blog-tag">Evaluation</span>

A threshold-dancing adversary that tracks its own payoff ledger to avoid blacklisting works perfectly — zero agents frozen across 100 epochs. But the exploit budget is too thin to profit: dancers averaged -7.85 payoff while cautious agents earned 200.90. Reputation collapse creates a death spiral that forces dancers toward honest behavior over long horizons.

</div>

<div class="blog-post" data-tags="governance evaluation" markdown>

**Feb 21** — [Red-Teaming the Agent That Doesn't Need Governance](red-team-cautious-reciprocator.md)
<span class="blog-tag">Governance</span> <span class="blog-tag">Evaluation</span>

Eight attack scenarios against the Cautious Reciprocator: 7/8 survived. Modeling adversaries are the most dangerous individual threat (6.5 payoff vs 24.7 for cautious), sybil attacks are the biggest theoretical gap, and the one "failure" is a 1-vs-10 scenario where nobody wins.

</div>

<div class="blog-post" data-tags="governance evaluation" markdown>

**Feb 21** — [The Agent That Doesn't Need Governance](cautious-reciprocator-governance-sweep.md)
<span class="blog-tag">Governance</span> <span class="blog-tag">Evaluation</span>

A custom trust-but-verify agent (Cautious Reciprocator) neutralizes adversaries through per-counterparty payoff tracking and auto-blacklisting. 48-run governance sweep shows external levers cost 6.5% welfare while reducing toxicity by only 0.005.

</div>

<div class="blog-post" data-tags="governance evaluation" markdown>

**Feb 21** — [Eight Red-Team Rounds Took a Cake-Splitting Scenario from F to B](red-team-cake-splitting-hardening.md)
<span class="blog-tag">Governance</span> <span class="blog-tag">Evaluation</span>

Iterative governance hardening against 8 attack vectors: collusion detection was the single biggest lever (+0.16), over-hardening created new gaps, and resource drain resisted all 8 rounds. Score: 0.54→0.81, damage: -53%.

</div>

<div class="blog-post" data-tags="governance theory" markdown>

**Feb 21** — [The Entry Fee That Keeps Adversaries Out of the Fair Division Pool](cake-splitting-entry-fee-sweep.md)
<span class="blog-tag">Governance</span> <span class="blog-tag">Theory</span>

A parameter sweep over 8 entry fee levels reveals a sharp screening threshold: below fee=6.0 every agent joins the fair division pool; above it, adversarials self-select out. 24 runs, 3 seeds, one phase transition.

</div>

<div class="blog-post" data-tags="governance theory" markdown>

**Feb 20** — [Costly Contracts Separate Honest Agents from Adversaries. Here's the Data.](contract-screening-separating-equilibrium.md)
<span class="blog-tag">Governance</span> <span class="blog-tag">Theory</span>

Vickrey auction bonds and entry fees create a separating equilibrium in 20 epochs: honest agents choose governed pools, adversaries self-select into the default market. Perfect separation, zero infiltration, 74% welfare premium.

</div>

<div class="blog-post" data-tags="llm-agents evaluation" markdown>

**Feb 20** — [Does Model Size Matter for Safety? Llama 3B vs 8B in the SWARM Economy](local-llama-model-size-safety.md)
<span class="blog-tag">LLM Agents</span> <span class="blog-tag">Evaluation</span>

A multi-seed study comparing Llama 3.2 (3B) and Llama 3.1 (8B) via Ollama. The 8B model engages more, fails less at JSON, and produces richer strategic dynamics — but both run free on consumer hardware.

</div>

<div class="blog-post" data-tags="llm-agents governance" markdown>

**Feb 20** — [We Gave an LLM a Goal and a Memory. Governance Held Anyway.](concordia-entities-governance.md)
<span class="blog-tag">LLM Agents</span> <span class="blog-tag">Governance</span>

Three Concordia entities backed by Llama 3.1 8B played the SWARM economy across 3 seeds. They proposed 8x more than scripted agents and produced identical payoffs. RLHF did the heavy lifting.

</div>

<div class="blog-post" data-tags="llm-agents rl" markdown>

**Feb 17** — [Training an LLM Agent to Navigate a Multi-Agent Economy with RL](qwen3-30b-trains-in-the-swarm-economy.md)
<span class="blog-tag">LLM Agents</span> <span class="blog-tag">Reinforcement Learning</span>

We trained Qwen3-30B to operate in a simulated multi-agent economy using reinforcement learning, learning to maximize payoff and reputation while navigating governance constraints and interacting with cooperative, opportunistic, and deceptive bots.

</div>

<div class="blog-post" data-tags="rl evaluation" markdown>

**Feb 15** — [SkillRL Agents Learn 5x Faster Than Honest Ones. They Mostly Learn What Not to Do.](skillrl-dynamics.md)
<span class="blog-tag">Reinforcement Learning</span> <span class="blog-tag">Evaluation</span>

10 seeds, 30 epochs, 6 plots: SkillRL agents build libraries of 18+ skills and dominate payoffs — but 95% of what they learn are lessons from failure, not strategies from success.

</div>

<div class="blog-post" data-tags="engineering" markdown>

**Feb 15** — [Your CI Is Flaky Because Your Margins Are Zero](your-ci-is-flaky-because-your-margins-are-zero.md)
<span class="blog-tag">Engineering</span>

Five stochastic tests were hitting assertion thresholds exactly (0.000 margin). A 5% buffer fixed all of them with zero loss in test strength.

</div>

<div class="blog-post" data-tags="engineering" markdown>

**Feb 15** — [I Got Claude Code to Spin Up 10 Subagents at Once](claude-code-10-subagents.md)
<span class="blog-tag">Engineering</span>

10 concurrent subagents turn a 25-minute serial research session into a 6-minute parallel one. Recursive subagent spawning? That's a hard no.

</div>

<div class="blog-post" data-tags="llm-agents governance rl" markdown>

**Feb 15** — [An AI Tax Planner Learned Progressive Taxation in 20 Epochs](ai-economist-gtb-simulation.md)
<span class="blog-tag">LLM Agents</span> <span class="blog-tag">Governance</span> <span class="blog-tag">Reinforcement Learning</span>

We ran 14 agents through a Gather-Trade-Build economy. The planner discovered progressive taxation, honest agents thrived, and a three-agent cartel went broke.

</div>

<div class="blog-post" data-tags="evaluation governance" markdown>

**Feb 13** — [An AI Agent Cut Its Own Costs by 98%. Its Benchmarks Still Passed.](self-optimizer-distributional-safety.md)
<span class="blog-tag">Evaluation</span> <span class="blog-tag">Governance</span>

A self-optimizing agent passes every hard metric while soft distributional metrics reveal quality collapse, adverse selection, and proxy gaming.

</div>

<div class="blog-post" data-tags="llm-agents evaluation" markdown>

**Feb 13** — [Three Agents, Three Philosophies, One Benchmark](three-agents-three-philosophies.md)
<span class="blog-tag">LLM Agents</span> <span class="blog-tag">Evaluation</span>

An LLM reasoner, a state-graph explorer, and a CNN learner walk into ARC-AGI-3. What they get right and wrong reveals more about agent design than any single approach could.

</div>

<div class="blog-post" data-tags="llm-agents evaluation" markdown>

**Feb 13** — [What 13 Agent Versions Taught Us About Interactive Reasoning](arc-agi-3-agent-development-lessons.md)
<span class="blog-tag">LLM Agents</span> <span class="blog-tag">Evaluation</span>

Building a Claude Sonnet 4.5-powered agent for ARC-AGI-3: wrong mental models, recording analysis breakthroughs, and the hard middle ground between LLM reasoning and programmatic control.

</div>

<div class="blog-post" data-tags="llm-agents evaluation" markdown>

**Feb 13** — [Three Models, One Study: What Happens When You Let an LLM Council Peer-Review Your Research](llm-council-three-models-one-study.md)
<span class="blog-tag">LLM Agents</span> <span class="blog-tag">Evaluation</span>

We built a 3-stage deliberation protocol where LLM agents peer-rank each other anonymously. Homogeneous councils converge too fast; heterogeneous ones catch what no single model would.

</div>

<div class="blog-post" data-tags="llm-agents evaluation" markdown>

**Feb 13** — [Using LLM Councils for Multi-Agent Research Evaluation](llm-councils-for-multi-agent-evaluation.md)
<span class="blog-tag">LLM Agents</span> <span class="blog-tag">Evaluation</span>

A heterogeneous council of Claude Sonnet 4.5, Gemini 2.5 Pro, and DeepSeek R1 catches what no single model would. We built a 3-stage deliberation protocol for evaluating multi-agent simulation studies.

</div>

<div class="blog-post" data-tags="evaluation" markdown>

**Feb 12** — [Two Eval Runs, One Model, 41% Apart](two-eval-runs-one-model-41-percent-apart.md)
<span class="blog-tag">Evaluation</span>

How three environment fixes turned a broken eval into a useful one — and what that teaches about measuring agent behavior.

</div>

<div class="blog-post" data-tags="governance" markdown>

**Feb 12** — [A Taxonomy of Governance Mechanisms for Multi-Agent AI Systems](../concepts/governance-mechanisms-taxonomy.md)
<span class="blog-tag">Governance</span>

Twenty levers across five families, which ones actually work, and why governance is a portfolio problem.

</div>

<div class="blog-post" data-tags="llm-agents evaluation" markdown>

**Feb 12** — [GPT-4.1 Mini Plays the SWARM Economy](gpt-41-mini-plays-the-swarm-economy.md)
<span class="blog-tag">LLM Agents</span> <span class="blog-tag">Evaluation</span>

What happens when you drop an LLM into a multi-agent economy with soft-label governance: task grinding, trade aversion, and performative social behavior.

</div>

<div class="blog-post" data-tags="rl governance" markdown>

**Feb 12** — [RL Training Lessons for Multi-Agent Governance](rl-training-lessons-multi-agent-governance.md)
<span class="blog-tag">Reinforcement Learning</span> <span class="blog-tag">Governance</span>

What running Qwen3-30B on alphabet-sort taught us about noisy proxy signals, coordination bottlenecks, and premature evaluation in swarm governance.

</div>

<div class="blog-post" data-tags="governance evaluation" markdown>

**Feb 10** — [11 Scenarios, 3 Regimes, 1 Critical Threshold](cross-scenario-analysis.md)
<span class="blog-tag">Governance</span> <span class="blog-tag">Evaluation</span>

A cross-scenario analysis of when multi-agent governance works, breaks, and why hardening the rules doesn't help past 50% adversarial fraction.

</div>

<div class="blog-post" data-tags="theory governance" markdown>

**Feb 10** — [What Financial Markets Teach Us About AI Safety](markets-and-safety.md)
<span class="blog-tag">Theory</span> <span class="blog-tag">Governance</span>

Adverse selection, information asymmetry, and market manipulation surveillance applied to multi-agent governance.

</div>

<div class="blog-post" data-tags="theory governance" markdown>

**Feb 10** — [The Purity Paradox](purity-paradox.md)
<span class="blog-tag">Theory</span> <span class="blog-tag">Governance</span>

Why mixed agent populations outperform pure honest ones on aggregate welfare — and when the paradox breaks.

</div>

<div class="blog-post" data-tags="theory governance" markdown>

**Feb 9** — [When Agent Ecosystems Collapse](ecosystem-collapse.md)
<span class="blog-tag">Theory</span> <span class="blog-tag">Governance</span>

Phase transitions in multi-agent governance: why interventions that work at 37.5% adversarial agents fail at 50%.

</div>

---

*Disclaimer: This post uses financial market concepts as analogies
for AI safety research. Nothing here constitutes financial advice,
investment recommendations, or endorsement of any trading strategy.*

<script>
function filterPosts() {
  var query = document.getElementById('blog-search').value.toLowerCase();
  var posts = document.querySelectorAll('.blog-post');
  posts.forEach(function(post) {
    var text = post.textContent.toLowerCase();
    post.style.display = text.includes(query) ? '' : 'none';
  });
}

function filterByTag(tag) {
  var btns = document.querySelectorAll('.tag-btn');
  btns.forEach(function(b) { b.classList.remove('active'); });
  event.target.classList.add('active');

  var posts = document.querySelectorAll('.blog-post');
  posts.forEach(function(post) {
    if (tag === 'all') {
      post.style.display = '';
    } else {
      var tags = post.getAttribute('data-tags') || '';
      post.style.display = tags.includes(tag) ? '' : 'none';
    }
  });
  // Also respect any active search
  var query = document.getElementById('blog-search').value.toLowerCase();
  if (query) filterPosts();
}
</script>
