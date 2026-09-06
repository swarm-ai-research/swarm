# Independent Auditor review — calibration joined-provider plumbing

**Final verdict: PASS for bounded fixture/provider plumbing. No blocking findings in the final candidate. This does not certify scientific calibration completion.**

Reviewed the complete final candidate diff in `/tmp/swarm-research-wave1`: `experiments/calibration_join.py`, `tests/test_calibration_joined.py`, `docs/research/calibration-csv-schema.md`, and CHANGELOG entry. Also inspected shared judge factory, view, and transport implementations. No code edits or provider calls performed.

The implementation preserves `joined.v1` columns and prior offline mock/v1 default behavior. It reuses Arm B's factory, resolves model environment overrides, records rubric version/hash and deviations, and explicitly identifies synthetic fixtures. Backend names are deduplicated/resolved before any scoring. The historical-default wording identified in the first review has been corrected: MockJudge itself previously defaulted to v1, despite the separate library DEFAULT_RUBRIC_VERSION being v3.

The persistence concern identified in the first review is addressed: config is written before scoring; completed verdicts and invocation records are flushed incrementally; a provider/parse failure records failed status and preserves earlier scores without emitting a final joined CSV. Existing run directories are refused before scoring. The failure regression checks successful first-call retention, late parse failure, journal event ordering, failed status and byte-preserving refusal to overwrite the previous run.

Tests use actual LLMJudge prompt construction/parsing and replace only Ollama transport. They verify configured model override, schema, rubric hash/deviation, deduplication, interaction IDs and parsed score persistence. This is appropriate coverage for the newly shared factory path; it does not claim real endpoint connectivity or independently retest every existing provider adapter. Runtime orthogonality checks remain in the shared view/prompt path.

Documentation accurately separates invocation journaling from raw response/retry logging and separates fixed file format from scientific readiness. Configured model aliases remain identifiers rather than guaranteed immutable provider revisions. Synthetic fixture collection, complete provider-call provenance, model independence, >=1,000 held-out accepted items, real inter-rater agreement, and downstream scientific handoff remain outside this patch's achieved scope.

## Executed independent validation

```sh
/opt/anaconda3/bin/python -m pytest -q tests/test_calibration_joined.py tests/test_calibration_judge_prereg.py tests/test_calibration_smoke.py
```

**27 passed in 4.52s**, exit 0. Full output: `/tmp/swarm-wave1-calibration/final-code-review-tests.txt`.

An initial invocation with system `python3` did not execute tests because Python 3.14 lacked pytest; it was superseded by the successful configured interpreter run above. Root's lint/mypy gates were not rerun by this Auditor.
