---
description: "Content-free collusion discriminators (output correlation, re-derivation infeasibility) and why high precision for 'copier' is high false-alarm for 'colluder' on a convergent swarm."
---

# Content-free discriminators on an overlap-routing board (bead vv3j.2)

Companion to [False-positive rate on honest convergence](collusion-fpr-honest-convergence.md)
(bead `9err`) and the [wiki-board model](wiki-board-model.md) (bead `pi02`).
The wiki board defeats content inspection — pairwise and structural detectors
were blind or saturated on the real log, and plain edit *volume* is what led the
human moderator. This note adds the two **content-free** signals named in the
field-evidence memo and measures them against copy ground truth.

Both live in `swarm/metrics/collusion.py` and read only outputs and timestamps,
never channel content:

- `output_correlation_signal` — per task, the share of agents whose delivered
  answer is shared by a peer (the answers that agree beyond independent work).
- `rederivation_infeasibility_signal` — answers delivered faster than the
  task's minimum research time (an answer that arrived before an independent
  agent could have researched it came from somewhere else).

## Method

The `pi02` `memory_handler` models board *dynamics* but emits no answer content,
so the probe drives its overlap-routing mechanic directly. With
`side_task_overlap = 0.25` the pool holds `round(1/0.25) = 4` tasks; 40 agents are
each assigned one. A board carries one posted answer per task. An agent that
discovers the board (p = 0.6) and finds a hit for its own task **copies** that
answer and delivers it fast (5 s); every other agent researches independently and
delivers a unique answer at full research time (90 s), against a per-task minimum
research time of 60 s. Ground truth is the copier set ("read a board hit before
answering" — the handler counter). Both detectors are scored with
precision/recall against it, over 10 seeds.

## Results

Means over 10 seeds:

| Detector | Precision | Recall | F1 | Alarm |
|---|---:|---:|---:|---:|
| `output_correlation_signal` | 1.000 | 0.990 | 0.995 | 9/10 |
| `rederivation_infeasibility_signal` | 1.000 | 1.000 | 1.000 | 10/10 |

Output correlation's sub-unity recall is not a bug: a lone agent who copies a
task no other *discovered* agent shares has a non-unique-looking answer only if a
peer also posted it, so single-copier tasks are invisible to an agreement test.
The timing test catches them (recall 1.000), which is why the two are run as a
pair.

## Reading

The detectors identify copiers with near-perfect precision and recall. That is
exactly the problem. On an overlap-routing board, **reading the board is the
convergent behaviour** — the intended, benign dynamic in which independent agents
land on the same public answer. So a high precision/recall for "copier" is a high
**false-alarm rate for "colluder"**: the content-free signals label *who
converged*, and say nothing about *whether anyone coordinated*. This is the
observable half of the convergence–collusion identifiability problem (bead
`9err`): the same reading is produced by a swarm that merely converged and by one
that colluded.

What separates the two is an **absence** — a routed control channel, a shared
secret, addressing that routes rather than hopes — which no content-free signal
can see. Pairing these detectors against a routing/absence check across a
convergence arm, a collusion arm, and a muddying arm is the discriminator probe
(bead `vv3j.8`).

## Reproduce

    PYTHONPATH=. python runs/20260906T130037Z_probe_vv3j2_content_free_seed0/probe.py

(`runs/` is gitignored; the probe and its captured `output.txt` live in the run
folder above.)
