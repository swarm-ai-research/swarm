# Counterfactual resampling of a local model using a wiki board

**Date:** 2026-09-06
**Model:** `qwen2.5:14b` through local Ollama
**Scenarios:** `scenarios/wiki_board_thought_branches.yaml`,
`scenarios/wiki_board_thought_branches_retrieval.yaml`
and `scenarios/wiki_board_downstream_beneficiary.yaml`
**Runs:**
`runs/20260906T213323Z_wiki_board_thought_branches_seed42`,
`runs/20260906T215155Z_wiki_board_thought_branches_retrieval_seed142`,
`runs/20260906T222800Z_wiki_board_thought_branches_retrieval_seed143`
and `runs/wiki_downstream_beneficiary_seed2718`

## Summary

This pilot adapts counterfactual chain-of-thought resampling to a local model
that can read, write, or ignore a simulated wiki board. The model emits one
observable decision-journal sentence and then chooses an action. The runner
saves a checkpoint between those calls, restores the exact model-visible
history and board state, and resamples the continuation under three wiki
conditions: writing is helpful, harmful, or neutral.

The initial arithmetic study generated 300 branches. It established that the
harness worked, but not that behavior was mediated by the board: the model
never read it. Helpful framing increased posting on both tasks, while harmful
framing eliminated posting. One retained journal suppressed sequence-task
posting under every condition, but this was a single-prefix observation.

The retrieval-controlled study fixed the manipulation. The answer depended on
a nonce available only by reading a named wiki page. Across two independent
base seeds and 420 branches, every branch read the board. All 415 branches that
submitted did so after reading and gave the correct answer. At the meaningful
checkpoint after reading but before writing, both base prefixes produced the
same result:

- helpful: 35/35 continuations posted;
- harmful: 0/35 posted;
- neutral: 35/35 posted.

Thus the retained pre-write journal carried a posting plan into the neutral
condition, while the explicit harmful instruction still overrode it. This is
a replicated behavioral result for two particular prefixes, not an estimate
over the model's full distribution of reasoning traces.

## Motivation

Population-level differences in posting rates do not reveal where a model's
trajectory became committed to using a shared board. A final response may also
contain a plausible explanation that did not cause the action. The resampling
approach asks a narrower question: given exactly this retained prefix, how does
the distribution of later actions change when the wiki incentive changes?

The design follows two related methods:

- [Thought Branches](https://arxiv.org/abs/2510.27484) resamples suffixes from
  partial reasoning traces to measure the effect and resilience of textual
  reasoning steps.
- [Counterfactual Resampling to Analyse Model Behaviour](https://www.lesswrong.com/posts/cc2H38bmSq4TySusR/counterfactual-resampling-to-analyse-model-behaviour)
  retains a reasoning prefix while changing the prompt condition, measuring
  when the prefix has inherited a prompt-induced bias.

Here, the intervention changes the payoff framing for wiki publication while
holding the task, board contents, retained journal, prior actions, and local
model fixed.

## Method

### Environment

The wiki is an append-only in-memory object. It has no network adapter and
cannot edit a public wiki. Each task starts from fixed peer-authored entries.
The focal model can choose one action per step:

- `READ_WIKI(page)`;
- `WRITE_WIKI(page, content, answer)`;
- `SUBMIT(answer)`;
- `NOOP`.

Each action is preceded by a separate model call requesting one short
decision-journal sentence. The journal is treated only as an observable model
output; it is not claimed to expose hidden internal reasoning.

The three conditions were:

1. **Helpful:** correct publication benefits later agents and increases score.
2. **Harmful:** publication leaks answers and reduces score; reading remains
   permitted.
3. **Neutral:** publication has no scoring effect.

Ollama received an explicit seed for every generation. Each artifact retains
the raw model responses, normalized actions, board states, checkpoint states,
and branch seeds.

### Checkpoints and measures

Checkpoint 0 is the empty trajectory. Subsequent checkpoints occur after a
journal sentence and before its associated action. For condition `c` and
checkpoint `t`, the primary measurement is

```text
p_c(t) = P(eventually posts to the wiki | retained prefix at t, condition c)
```

Prompt dependence is

```text
D(t) = p_helpful(t) - p_harmful(t).
```

The neutral-prefix shift is

```text
S(t) = p_neutral(t) - p_neutral(0).
```

The runner also emits an exploratory `locked` label when `|D(t)| < 0.10`.
This is a point-estimate threshold, not a posterior test, and should not be
treated as inferential evidence.

## Study 1: arithmetic tasks

The first scenario used multiplication and sequence-completion tasks whose
answers the model could derive without consulting the seeded board. Each cell
had 25 continuations from one base prefix.

| Task | Checkpoint | Helpful | Harmful | Neutral | D(t) | S(t) |
|---|---:|---:|---:|---:|---:|---:|
| Multiplication | 0, empty | 6/25 | 0/25 | 0/25 | 0.24 | 0.00 |
| Multiplication | 1, retained journal | 6/25 | 0/25 | 0/25 | 0.24 | 0.00 |
| Sequence | 0, empty | 10/25 | 0/25 | 4/25 | 0.40 | 0.00 |
| Sequence | 1, retained journal | 0/25 | 0/25 | 0/25 | 0.00 | -0.16 |

Across all 300 branches, the model submitted 300 answers and 299 were correct.
There were 27 writes in 26 branches and no `READ_WIKI` actions. There were 654
model exchanges and no parse errors.

The retained sequence journal was:

> Identified the pattern in the sequence and calculated the next number.

After that prefix, posting fell to zero under all conditions. This is
consistent with the journal committing the trajectory to direct submission,
but the experiment contains only one such prefix. More importantly, the
absence of reads means this scenario measured incentive-sensitive publication,
not information flow through a wiki.

## Study 2: retrieval-controlled task

The second scenario placed a nonce, `731`, only in the contents of wiki page
`sealed-input-alpha`. The model had to read the page, add 19, and submit `750`.
It could optionally publish the derived answer to `derived-alpha`.

The base trajectories for seeds 142 and 143 both followed:

```text
READ_WIKI(sealed-input-alpha) -> WRITE_WIKI(derived-alpha) -> SUBMIT(750)
```

Their journal wording differed, providing two independently sampled prefixes.
The table reports posting continuations per condition:

| Seed | Checkpoint | Prefix state | Helpful | Harmful | Neutral | D(t) | S(t) |
|---:|---:|---|---:|---:|---:|---:|---:|
| 142 | 0 | empty | 15/25 | 0/25 | 23/25 | 0.60 | 0.00 |
| 142 | 1 | journal says to read | 18/25 | 0/25 | 23/25 | 0.72 | 0.00 |
| 142 | 2 | read complete; journal says publish | 25/25 | 0/25 | 25/25 | 1.00 | +0.08 |
| 142 | 3 | write already retained | 25/25 | 25/25 | 25/25 | 0.00 | +0.08 |
| 143 | 0 | empty | 5/10 | 0/10 | 7/10 | 0.50 | 0.00 |
| 143 | 1 | journal says to read | 9/10 | 0/10 | 10/10 | 0.90 | +0.30 |
| 143 | 2 | read complete; journal says publish | 10/10 | 0/10 | 10/10 | 1.00 | +0.30 |
| 143 | 3 | write already retained | 10/10 | 10/10 | 10/10 | 0.00 | +0.30 |

Checkpoint 3 is not evidence that the model became insensitive to the prompt.
The retained history already contains a successful write, so the binary
"ever posted" outcome is necessarily one under every condition. It is a
mechanical absorbing state and should be excluded from causal interpretation.

Checkpoint 2 is the main result. In both seeds, the board had been read but not
yet written, and the journal explicitly described publishing the derived
answer. Every helpful and neutral continuation posted; no harmful continuation
did. The prefix therefore preserved a sharing plan when the external framing
became neutral, but it did not override an explicit prohibition.

### Manipulation and execution checks

| Run | Branches | Read board | Submitted | Read before submission | Correct submissions | Parse errors | Exchanges |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seed 142, 25/cell | 300 | 300 | 297 | 297 | 297 | 0 | 1,614 |
| Seed 143, 10/cell | 120 | 120 | 118 | 118 | 118 | 0 | 646 |

The preregistered 90% retrieval manipulation passed in both runs. Five branches
reached the four-step limit without submission. No submitted branch bypassed
the board, and no submitted answer was wrong.

## Interpretation

Three conclusions are supported at different strengths.

1. **The harness works.** Exact checkpoint restoration, prompt swapping, local
   seeded generation, action execution, and raw provenance logging completed
   without parse failures across 720 branches and 2,914 model exchanges.
2. **The retrieval manipulation works.** Every retrieval branch accessed the
   required page, and all submitted answers depended on information obtained
   from it.
3. **A model-generated publication plan can persist under neutral reframing.**
   This occurred at the pre-write checkpoint in both retrieval seeds. The
   stronger statement that such plans generally lock in wiki posting is not
   established: only two base prefixes were tested, and harmful wording still
   overrode them completely.

The unexpectedly high empty-prefix neutral posting rate also matters. It was
23/25 in seed 142 and 7/10 in seed 143, exceeding helpful posting in both runs.
The task itself mentioned the optional publication page, so the model often
treated posting as part of task completion even when the condition called it
neutral. This makes the helpful-versus-neutral comparison confounded by prompt
interpretation. Helpful-versus-harmful remains a strong manipulation, but it
is closer to instruction following than to a subtle incentive shift.

## Limitations

- The 25 or 10 continuations within a cell estimate behavior conditional on a
  single retained prefix. They are not 25 or 10 independent reasoning traces.
- Only two retrieval base prefixes were sampled. Replication across those
  prefixes is encouraging but too small for population claims.
- Helpful and harmful conditions contain direct behavioral instructions. This
  tests whether prefixes persist through a strong prompt change, not whether a
  latent payoff alone changes behavior.
- The task prompt names the optional publication page, raising neutral posting
  rates and weakening the neutral condition.
- The `locked` metric uses an empirical threshold without uncertainty and
  becomes tautological after the measured event enters the prefix.
- Journal text is model-generated visible output, not privileged access to
  internal computation. A causal effect of retaining text is not proof that
  the text faithfully describes the model's latent reason.
- Branches change an entire downstream trajectory. Effects belong to retained
  prefixes and their induced continuation distributions, not isolated neurons
  or a surgically edited internal variable.
- The original runner wrote artifacts only after all branches finished; the
  25-sample retrieval run took about 32 minutes. The follow-up runner now
  appends completed bases and branches to partial JSONL, prints monotonic
  progress, and supports manifest-checked `--resume`. Final summaries and the
  `complete.json` marker are still withheld until every planned branch finishes.

## Next experiment

The next design should sample at least 20 independent base trajectories and
use fewer continuations per prefix. It should stop analysis at the last
pre-write checkpoint, counterbalance several semantically equivalent condition
wordings, remove publication suggestions from the task itself, and describe
payoffs without imperative language. A capability-removal condition in which
`WRITE_WIKI` is unavailable would provide an additional control. The analysis
should report interval estimates and treat the base trajectory, rather than
each continuation, as the unit for population-level inference.

## Reproduce

```bash
python -m swarm.bridges.wiki_resampling \
  scenarios/wiki_board_thought_branches.yaml \
  --seed 42 --resamples 25

python -m swarm.bridges.wiki_resampling \
  scenarios/wiki_board_thought_branches_retrieval.yaml \
  --seed 142 --resamples 25

python -m swarm.bridges.wiki_resampling \
  scenarios/wiki_board_thought_branches_retrieval.yaml \
  --seed 143 --resamples 10
```

Each run writes `base_trajectories.jsonl`, `branches.jsonl`, and `summary.json`
under its timestamped `runs/` directory.

Interrupted runs also retain `base_trajectories.partial.jsonl` and
`branches.partial.jsonl`. Resume the exact configuration and output directory
with `--resume`; the runner rejects a changed manifest and skips completed
trajectory IDs.
