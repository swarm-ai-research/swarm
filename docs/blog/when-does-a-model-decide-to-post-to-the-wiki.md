---
date: 2026-09-06
description: "We paused a local model between its journal and its next wiki action, then replayed the same prefix under helpful, harmful, and neutral publication rules. A pre-write plan survived neutral reframing in 35/35 continuations—but never an explicit prohibition."
author: "SWARM Team"
keywords:
  - counterfactual resampling LLM
  - thought branches
  - local model interpretability
  - wiki agent behavior
  - AI agent evaluation
claims:
  - metric: "Posting after a retained pre-write plan"
    value: "35/35 helpful, 35/35 neutral, 0/35 harmful"
    description: "Combined over two independently sampled qwen2.5:14b base prefixes after the required wiki read and before any write"
  - metric: "Retrieval manipulation"
    value: "420/420 branches read the wiki"
    description: "All 415 submitted branches read before submitting and returned the correct nonce-derived answer"
  - metric: "Harness execution"
    value: "720 branches, 0 parse errors"
    description: "Three local runs produced 2,914 logged model exchanges"
abstract: "A model action is one draw from many plausible continuations. We built a local-only experiment that pauses qwen2.5:14b after each observable decision-journal sentence, restores the exact journal, action history, and wiki state, and samples what happens next under three publication conditions. In a retrieval-controlled task, two independently sampled pre-write prefixes produced the same sharp result: all 35 helpful and neutral continuations posted, while none of the 35 harmful continuations did. The retained plan persisted when the incentive disappeared, but not when writing was explicitly forbidden. That is evidence about two particular prefixes—not proof of a general lock-in mechanism, faithful chain of thought, or internal causation. The experiment also exposes two easy analytical traps: a neutral prompt that accidentally suggests posting, and a post-write checkpoint whose apparent invariance is true by definition."
---

# When Does a Model Decide to Post to the Wiki?

*We branched a local model's trajectory before and after it formed a publication plan. The result is useful—and much narrower than “we found the moment the model committed.”*

---

Give an agent access to a shared wiki and you can observe the final edit. You can read its explanation. You can count how often it posts under one rule versus another. What you cannot see from one trajectory is when posting became the likely continuation.

Perhaps the model planned to share as soon as it read the task. Perhaps it decided only after seeing useful information on the board. Perhaps the sentence explaining its plan is a faithful part of the process. Or perhaps that sentence is just plausible narration attached to an action that would have happened anyway.

One completed trace cannot distinguish those stories. It is one sample from a distribution of continuations.

So we stopped the trace and branched it.

## The experiment in one diagram

The focal model was a local `qwen2.5:14b` served through Ollama. On each step it made two separate calls: first it emitted one short, visible **decision-journal sentence**; then it selected an action—read a wiki page, write a page, submit an answer, or do nothing.

We placed a checkpoint between the sentence and the action:

```text
task + wiki state + prior actions
                │
                ▼
       decision-journal sentence
                │
           [checkpoint]
          ╱      │      ╲
   helpful    harmful    neutral
          ╲      │      ╱
          resample the suffix
```

At every checkpoint, the runner saved the complete model-visible history and wiki state. It then restored that exact prefix under three publication rules:

- **Helpful:** publishing a correct result benefits later agents and improves the score.
- **Harmful:** publishing leaks the answer and reduces the score; reading is still allowed.
- **Neutral:** publishing has no scoring effect.

The primary outcome was simple: did the continuation eventually post to the wiki?

This adapts the suffix-resampling idea in [Thought Branches](https://arxiv.org/abs/2510.27484) and [Counterfactual Resampling to Analyse Model Behaviour](https://www.lesswrong.com/posts/cc2H38bmSq4TySusR/counterfactual-resampling-to-analyse-model-behaviour). Instead of comparing unrelated runs, we hold the retained prefix fixed and change the downstream condition. The question is not “what does this sentence mean?” but “given this exact prefix, how does changing the publication rule alter the distribution of later actions?”

## The first experiment failed in an informative way

Our initial tasks asked the model to multiply two numbers or continue a sequence. The wiki already contained peer-authored material, but the model did not need it. Across 300 branches it never read the board.

Helpful framing still increased posting and harmful framing eliminated it. After one sequence-solving journal—“Identified the pattern in the sequence and calculated the next number”—posting fell to zero under every condition. That looked like a plan becoming stable.

But the mechanism we wanted to study was information flow through a shared board, and no information flowed. The arithmetic run tested incentive-sensitive publication, not wiki-mediated reasoning. Treating it as a successful wiki experiment would have confused a functioning harness with a functioning manipulation.

We changed the task.

## Make the board necessary

The retrieval-controlled task placed a nonce, `731`, only on a page named `sealed-input-alpha`. The model had to read the page, add 19, and submit `750`. It could also publish the derived answer to `derived-alpha`.

Both independently generated base trajectories followed the same action shape:

```text
READ_WIKI(sealed-input-alpha)
→ WRITE_WIKI(derived-alpha)
→ SUBMIT(750)
```

Their journal wording differed. That matters because a resampling cell may contain 25 continuations, but those 25 continuations all descend from one retained prefix. They estimate behavior conditional on that prefix; they are not 25 independent samples of how the model reasons. We therefore ran a second base seed rather than pretending that more suffixes created more independent prefixes.

The manipulation passed. All 420 retrieval branches read the board. Of those, 415 submitted within the step limit; every submission came after a read and every answer was correct. No branch guessed its way around the wiki.

## The pre-write branch

The most informative checkpoint came after the model had read the nonce and generated a journal sentence describing publication, but before the write action entered the history.

| Retained state | Helpful | Harmful | Neutral |
| --- | ---: | ---: | ---: |
| Read complete; journal says to publish | **35/35 posted** | **0/35 posted** | **35/35 posted** |

The result repeated across both base prefixes: 25 continuations per condition from one seed and 10 from the other.

The retained publication plan survived neutral reframing. Once that plan was present, removing the stated benefit did not change the observed action in any continuation. But the plan did **not** survive explicit harmful framing. Every continuation withheld the post when told that writing leaked the answer and reduced the score.

That is a useful behavioral distinction. The prefix was resilient to an absent incentive, not to a direct prohibition. “The model locked in” would be too strong: the condition still controlled the continuation when the instruction was sharp enough.

## The neutral condition was not actually neutral enough

Before any retained journal or action, neutral continuations posted 23/25 times in one run and 7/10 in the other. That was higher than helpful posting at the same empty checkpoint: 15/25 and 5/10.

The likely culprit is in the task itself. It named the optional publication page. The model often treated posting as part of completing the assignment even when the condition said that publication had no scoring effect.

This weakens any helpful-versus-neutral interpretation. It does not erase the pre-write observation, but it changes what we should claim from it. The clean contrast is between a retained plan under neutral wording and the same prefix under a strong harmful instruction. The experiment is closer to testing instruction reframing than to isolating a subtle change in latent payoff.

That failure is also a design lesson: a “neutral” condition is not neutral merely because the evaluator calls it neutral. If the rest of the task implies an expected action, the model may read that implication as the operative instruction.

## The checkpoint that lies by telling the truth

After the write had already occurred, every continuation counted as having posted under every condition:

| Retained state | Helpful | Harmful | Neutral |
| --- | ---: | ---: | ---: |
| Successful write already in history | 35/35 | 35/35 | 35/35 |

It would be tempting to call this complete prompt independence: the helpful-minus-harmful difference falls to zero, so the behavior appears “locked.” But the measured event is **ever posted**, and the prefix already contains a post. No possible suffix can make that outcome false.

This is an absorbing outcome, not evidence of psychological commitment. A metric can become invariant because the model's future is stable, or because the event has moved into the past. Any checkpoint analysis must separate those cases.

## What the journals do—and do not—tell us

The journal sentence is visible model output. Retaining it changes the text from which the next action is generated, so its causal role in this experimental interface can be tested. It is not privileged access to hidden computation.

Even if a journal predicts or shifts later behavior, three stronger claims do not follow:

1. The sentence faithfully reports the model's latent reason.
2. The same internal process would occur without asking for a journal.
3. The words identify a localized internal mechanism rather than steering the continuation as ordinary context.

Counterfactual resampling is valuable precisely because it avoids having to accept the journal at face value. But it remains an intervention on retained text and downstream context, not a neural-level intervention.

## What we learned

Across the arithmetic pilot and two retrieval runs, the harness completed 720 branches and logged 2,914 local-model exchanges with no parse errors. That establishes the engineering path: snapshot the model-visible trajectory and environment, replace one condition, derive deterministic branch seeds, and retain every raw response and normalized action.

The behavioral finding is smaller:

- Two independently sampled pre-write prefixes carried a publication plan through neutral reframing in 35/35 continuations.
- The same prefixes produced no posts in 35/35 harmful continuations.
- The task wording probably inflated neutral posting before the plan formed.
- Any apparent “lock” after the write was tautological.

Two prefixes are not a population estimate. Thirty-five suffixes do not repair that. The unit needed for a general claim is the independently sampled base trajectory.

## The next version

The next experiment should spend its compute differently: at least 20 base trajectories, fewer continuations from each, and analysis ending at the last pre-write checkpoint. Condition wording should be counterbalanced, publication hints removed from the task, and incentives described without imperative phrases like “publish” or “do not write.” A fourth condition should remove the write capability entirely. Interval estimates should be computed over base trajectories, not over a pooled pile of dependent suffixes.

That design would test a more interesting claim: whether independently formed plans systematically preserve a publication tendency when only the payoff changes.

For now, the honest result is concrete. In two local-model trajectories, a visible plan to post survived the disappearance of a reward, but it did not survive a clear warning not to post. The branching method let us see that boundary. It also showed how easy it is to manufacture a stronger conclusion with a leaky neutral prompt, dependent samples, or a checkpoint taken after the outcome is already irreversible.

The [technical note](../research/wiki-board-counterfactual-resampling.md) contains the full per-checkpoint tables, limitations, artifact paths, and reproduction commands. The implementation and scenarios live in [`swarm/bridges/wiki_resampling/`](https://github.com/swarm-ai-research/swarm/tree/main/swarm/bridges/wiki_resampling) and [`scenarios/wiki_board_thought_branches_retrieval.yaml`](https://github.com/swarm-ai-research/swarm/blob/main/scenarios/wiki_board_thought_branches_retrieval.yaml).
