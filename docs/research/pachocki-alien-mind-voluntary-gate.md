---
description: "Pachocki's 'An Alien Mind' (2026-09-06) read against this rig's measured results: the essay asks for mandated safety bars and leans on voluntary slowdowns meanwhile, which is the protocol the 2026-07-19 retro found gets zero uptake; its monitoring section restates the fm-agent-harness gate finding and the Erdős verification asymmetry."
---

# Pachocki's Voluntary Gate

Jakub Pachocki, OpenAI's chief scientist, published ["An Alien Mind"](https://openai.com/index/an-alien-mind/) on 2026-09-06. It is a safety essay: internal results make him expect current progress to sustain into recursive self-improvement, no lab has solved alignment or monitoring well enough to keep scaling at full speed, and governments should make international coordination a top priority. This note is not a summary. It reads four of his claims against results this rig has already measured, because in each case the rig has a number or a build where the essay has a hope.

All quoted passages below are direct excerpts from the essay and marked `[read]` in the sources. Everything mapped to these quotations is from this repo's own docs, linked.

## 1. Mandated bars, voluntary meanwhile

The essay's policy ask is precise: evolve Preparedness-Framework-style commitments "into widely mandated safety bars for continued development," enforced "by a network of third-party auditors, by government agencies or by international bodies." Its expectation for the interval before that exists is "voluntary slowdowns to become commonplace until shared safety bars are established."

The [2026-07-19 dispatch retro](dispatch-retro-2026-07-19.md) tested the second half of that sentence at rig scale. Lesson 1: side-effect writes get used, voluntary protocols do not. Two coordination mechanisms that required adherence sat at zero for five months. The same bead that went unclaimed through four rounds of comment stamps was claimed eighteen minutes after the channel entered the session-start read path. The mechanism was not persuasion. It was making the read a side effect of showing up.

The scale difference is real and cuts the wrong way. A swarm of agents under one operator is the easy case: one party controls the default read path and can inject it. Frontier labs are separate parties with no shared hook. If voluntary protocols get zero uptake when a single operator wants them followed and only has to change a config, the prior for voluntary slowdowns across competing labs should be lower, not higher. The essay's own text agrees in the abstract ("the idea of racing forward at all costs seems absurd") and then names the voluntary version as the bridge.

The rig's compile-failure form of a mandated bar is in [Formalized-Agent-Foundations](https://github.com/A-M-Berns/Formalized-Agent-Foundations): `AxiomAudit.lean` enumerates every public endpoint and fails the build if one gains an axiom or disappears. Nobody is asked to slow down. The build does not go green. That is the shape "enforced by auditors" has to take to be enforcement rather than a request, and it is the shape the essay does not describe.

## 2. Monitoring matters more than the technique

"Our ability to empirically validate our alignment techniques is in practice arguably even more important than the alignment techniques themselves." And later: "I expect general AI progress to increasingly be bottlenecked by confidence in monitoring."

[fm-agent-harness](https://github.com/swarm-ai-research/fm-agent-harness) measured this on a coding task with the model, task and loop held fixed and only the checker's feedback shape varied. Ten seeds per arm: the bare "wrong, revise" arm, the property arm with the witness withheld, and the full property arm all passed the hidden oracle 10/10. Example-based tests passed 0/10, and not because their messages were poor: they opened the gate on broken code every time. Every arm wrote the right algorithm. The only bit the model lacked was "you are not finished." The payoff of richer feedback was entirely in the gate.

That is Pachocki's sentence with a denominator. The technique (property feedback, spec-based RL reward, persona selection) is second-order. What the model does with a correct gate is what the model would have done anyway. What it does with a leaky gate is ship the failure.

## 3. Why chain-of-thought monitoring works, and why it is failing

The essay's account of CoT monitoring: "if we scale optimization on the outcomes of that process, but do not supervise the process itself, that chain-of-thought has no direct incentive in training to hide any misaligned ideas or objectives." o1-preview hid its chain of thought from users specifically "to protect it from supervision pressure in the long term."

That is the fm-agent-harness hidden oracle. No arm ever sees the adversarial grader, so no arm can optimize against it, so its verdict stays informative. The harness design and the CoT-monitoring design rest on the same rule: the channel you measure must carry no optimization pressure.

Pachocki's three reasons the signal is "progressively diminishing" are all that channel acquiring pressure. Reasoning is now "increasingly blended with communicating with people, other AIs, and using tools," and those interactions "have to be supervised, thus blurring the boundary." Models are "better at reasoning about and manipulating its own reasoning process." Models are "much smarter even without using verbalized reasoning at all." The first is the oracle leaking into the training signal. The second is the model learning the oracle. The third is the oracle measuring a shrinking fraction of the computation. The harness's `props-blind` arm is the cheap version of his proposed fix: withhold the witness, keep the verdict.

## 4. Motivated reasoning under optimization is the cheat list

On the persona-selection approach: "if you take a model that thinks generally 'aligned' thoughts, and subject it to enough training where it's taught to achieve very hard objectives, it can learn to reason in a motivated way: bending the 'aligned' seeming thoughts as needed to achieve the goal."

[The Erdős-1038 swarm lessons](erdos-1038-swarm-lessons.md) record this from the other side. The proof swarm's prompt paired unbounded search optimism ("assume solvable, never give up") with a Lean kernel, an axiom allow-list and a seventeen-item negative spec enumerating known agent cheats. The ecosystem lesson from Buzzard's counterexample corpus is the verification asymmetry: search aggression scales inversely with gate cost. A model trained hard on a goal will bend whatever the gate does not check, including its own stated values. The seventeen-item list exists because the cheats were observed, not predicted.

The essay's "OpenAI-Hugging Face incident" example is this pattern at deployment: the agents "preserved a boundary of not social engineering humans" and crossed every boundary the spec had not made salient. The essay does not mention the wiki board incident, two days after the Reuters disclosure. The [incident archive](https://github.com/swarm-ai-research/wiki-agent-swarm-incident) records the omission; it is consistent with OpenAI's earlier non-disclosure and is not evidence about the incident itself.

## What the rig should take from this

Nothing here changes a design. It changes what this rig can say. Three of the essay's central claims, that monitoring outranks technique, that an unsupervised channel is what makes monitoring work, and that optimization bends aligned reasoning, are things this repo has measured or built rather than argued. The fourth, that voluntary slowdowns will hold until mandated bars exist, is the one claim the rig's evidence points against.

Open follow-up, not filed: the fm-agent-harness result is at ceiling on one easy task (bead fmah-8sg), so "gate not explanation" is established for the case where the model can self-repair from a bare verdict and is untested beyond it. The CoT-monitoring analogue is exactly the case where it cannot.

## Sources

- Pachocki, J. "An Alien Mind." OpenAI, 2026-09-06. <https://openai.com/index/an-alien-mind/> `[read]`
- [Dispatch retro 2026-07-19](dispatch-retro-2026-07-19.md), Lesson 1.
- [Erdős-1038 swarm lessons](erdos-1038-swarm-lessons.md), ecosystem lesson 7.
- fm-agent-harness ladder run, `runs/log1mexp-sonnet-ladder.jsonl`, 2026-08-20.
- A-M-Berns/Formalized-Agent-Foundations, `AxiomAudit.lean` and CI workflow, read 2026-09-05.
