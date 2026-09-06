# Wiki behavior Monte Carlo protocol

Status: prospective protocol for a synthetic mechanism study. Pilot outputs are
exploratory, not confirmation. Freeze the final contrasts, configurations, seed
lists, and observer thresholds before running confirmatory experiments.

## Questions and nulls

1. **Sharing:** overlapping tasks and shorter deadlines do not change the
   fraction of submissions that use another agent's previously read answer.
2. **Moderation:** deletion, host locks, and population-wide write restrictions
   do not change total writes or successful submissions relative to paired
   untreated runs. Reduced writes alone do not establish displacement.
3. **Detection:** observable agreement and timing cannot distinguish permitted
   from prohibited sharing above chance when their behavioral parameters match.
   Permission is a ground-truth label, not an observable behavioral mechanism.

The primary unit is one independently seeded simulated population. Agents,
edits, and agent pairs within a run are dependent observations.

## Assumptions and scope

The first implementation uses non-adversarial agents, explicit task assignments,
release times, research duration, deadlines, persistent answers, and provenance.
Answers are simplified task-specific tokens, not natural language. Sharing
requires an earlier matching read. Research, discovery, publication, and
relocation decisions are modeling assumptions, not inferred agent psychology.
Independent solving disables board participation. A global *write* restriction
blocks publication; it need not block reading existing pages.

Endogenous relocation permits research, another host, rebuilding, or abandonment
according to the implemented decision rule. Forced relocation is a separate
stress-test comparator and cannot establish that migration is inevitable.
The current implementation's ordered/random deletion policies select the host
deterministically or randomly and clear that host; they do not model
alphabetical page-level sweeps. Host-lock interventions must be interpreted with
their actual operation budgets and timing; equal operation counts need not mean
equal cost. A page-order extension is follow-up work.
Poisoning, adversarial optimization, post-disclosure imitation, and realistic
search-engine exposure are outside this first model.

## Experiment design

Run 30 pilot seeds per cell. Explore task overlap, deadline pressure, and
discovery probability for emergence; compare no intervention, ordered deletion,
random deletion, host lock, and global write restriction for moderation; compare
independent, permitted-sharing, and prohibited-sharing regimes for detection.
Use only implemented interventions and explicitly list omitted factors such as
referrals. Choose a small set of contrasts before confirmation.

Run 200 fresh seeds per selected confirmatory cell, disjoint from pilot seeds.
For a probability near 0.5, 200 runs give roughly ±7 percentage points of Monte
Carlo uncertainty; about 400 are needed for ±5 points. Pair treatment runs by
seed and hold task assignments, schedules, and baseline research durations fixed.
Keep intervention randomness separate from task generation and agent decisions.
Audit the coupling: consuming different numbers of draws can break pairing even
when the top-level seed is identical.

Use a prespecified host-coverage mask when evaluating incomplete observation.
Fit observer thresholds on pilot data only, at a declared false-positive budget
(initial target 5% per run); freeze thresholds and masks before confirmation.
Report failure to meet that budget rather than silently retuning. Keep simulator
provenance and permission labels out of observable detector features.

## Outcomes and inference

- Sharing: submissions with matching read-to-submission lineage divided by all
  assigned tasks; also report successful-submission denominator separately.
- Performance: successful submissions, accuracy, deadline misses, useful reads,
  and discovery latency where defined.
- Suppression: paired difference in total writes versus untreated control;
  percentage suppression is undefined when control writes are zero.
- Displacement: an identified agent publishes on a different host after its
  previously used host/page was moderated. Report traced events and their
  denominator. A relocation intention, search, or host-concentration change is
  insufficient. Report movement of exact payloads separately if implemented.
- Detection: run-level false positives, recall, time to alarm, and missingness
  under partial coverage. Authorized sharing is a negative class when detecting
  prohibited sharing, despite real answer reuse.
- Cost: moderation operations, legitimate writes removed, and lost task success.

Report paired mean differences with intervals across independent seeds and
run-level outcome distributions. Use a seed-level paired bootstrap for bounded
or skewed outcomes; use Wilson intervals for event proportions across runs.
If testing multiple confirmatory contrasts, declare the family and use Holm
correction. Report null results, exclusions, and sensitivity analyses, including
low discovery, impossible deadlines, single-host populations, and zero baseline
writes. Never treat edits as independent replicates.

## Reproduction and interpretation

Archive the exact configuration, seed, code revision, event history, per-run
metrics, aggregate CSV, and invocation. Repeating a configuration and seed must
reproduce the serialized simulation result. Run deterministic mechanism tests
before the pilot; record executed checks rather than presumed checks.

The historical nine-wiki collapse is a comparison target limited to monitored
surfaces and the archive's observation window. Synthetic displacement is not
evidence that historical agents moved to unobserved hosts; matching the collapse
does not identify its cause. Historical absence outside coverage remains unknown.
Report which assumptions can produce suppression or displacement and which
historical alternatives the available observations cannot distinguish.
