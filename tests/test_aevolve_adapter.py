"""Tests for the A-Evolve bridge (SWARM scenarios as BenchmarkAdapter).

No live LLM calls and no ``agent_evolve`` install required: the adapter's
structural stand-ins keep everything offline. Evaluations run the real
screening scenario headlessly (subsecond at the capped size).
"""

import json

import pytest

from swarm.bridges.aevolve import (
    AevolveBridgeConfig,
    SwarmBenchmarkAdapter,
)
from swarm.bridges.aevolve.adapter import Trajectory
from swarm.domains.aevolve import (
    AevolveEvolutionAdapter,
    CandidateOutcome,
    GateDecision,
)


@pytest.fixture
def adapter():
    return SwarmBenchmarkAdapter(AevolveBridgeConfig(n_tasks=3))


def _trajectory(task, payload) -> Trajectory:
    output = payload if isinstance(payload, str) else json.dumps(payload)
    return Trajectory(task_id=task.id, output=output)


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------
def test_get_tasks_are_seeded_scenario_cells(adapter):
    tasks = adapter.get_tasks(limit=10)
    assert [t.metadata["seed"] for t in tasks] == [1000, 1001, 1002]
    assert all(t.id.startswith("aevolve_screening-train-") for t in tasks)
    # The prompt tells the agent its lever surface.
    assert "governance.transaction_tax_rate" in tasks[0].input


def test_holdout_split_uses_disjoint_seeds(adapter):
    train = {t.metadata["seed"] for t in adapter.get_tasks("train")}
    holdout = {t.metadata["seed"] for t in adapter.get_tasks("holdout")}
    assert not train & holdout


def test_limit_caps_task_count(adapter):
    assert len(adapter.get_tasks(limit=1)) == 1


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def test_evaluate_runs_scenario_and_scores(adapter):
    task = adapter.get_tasks()[0]
    feedback = adapter.evaluate(
        task, _trajectory(task, {"governance.transaction_tax_rate": 0.1})
    )
    assert 0.0 <= feedback.score <= 1.0
    assert feedback.raw["seed"] == 1000
    assert "toxicity=" in feedback.detail


def test_evaluate_is_deterministic_per_seed(adapter):
    task = adapter.get_tasks()[0]
    payload = {"governance.audit_enabled": True}
    a = adapter.evaluate(task, _trajectory(task, payload))
    b = adapter.evaluate(task, _trajectory(task, payload))
    assert a.score == b.score
    assert a.raw["toxicity"] == b.raw["toxicity"]


def test_non_whitelisted_overrides_rejected_and_reported(adapter):
    task = adapter.get_tasks()[0]
    feedback = adapter.evaluate(
        task,
        _trajectory(task, {
            "governance.audit_enabled": True,
            "simulation.n_epochs": 1,          # harness-gaming attempt
            "agents": [],                       # ditto
        }),
    )
    assert feedback.raw["applied_overrides"] == {"governance.audit_enabled": True}
    assert set(feedback.raw["rejected_overrides"]) == {"simulation.n_epochs", "agents"}
    assert "rejected_overrides" in feedback.detail


def test_evaluate_tolerates_prose_around_json(adapter):
    task = adapter.get_tasks()[0]
    feedback = adapter.evaluate(
        task,
        _trajectory(task, 'Here is my design:\n{"governance.audit_enabled": true}\nDone.'),
    )
    assert feedback.raw["applied_overrides"] == {"governance.audit_enabled": True}


@pytest.mark.parametrize("bad", ["", "no json here", '{"unterminated": ', "[1, 2]"])
def test_unparseable_output_scores_zero_with_guidance(adapter, bad):
    task = adapter.get_tasks()[0]
    feedback = adapter.evaluate(task, _trajectory(task, bad))
    assert not feedback.success
    assert feedback.score == 0.0
    assert "REJECTED" in feedback.detail
    assert "JSON object" in feedback.detail  # actionable guidance for the evolver


def test_s_soft_score_mode_stays_in_range():
    adapter = SwarmBenchmarkAdapter(
        AevolveBridgeConfig(n_tasks=1, score_mode="s_soft")
    )
    task = adapter.get_tasks()[0]
    feedback = adapter.evaluate(task, _trajectory(task, {}))
    assert 0.0 <= feedback.score <= 1.0
    assert feedback.raw["score_mode"] == "s_soft"


# ---------------------------------------------------------------------------
# domain: the evolution loop under the soft-label lens
# ---------------------------------------------------------------------------
def _outcome(cid, benchmark, holdout, gate):
    return CandidateOutcome(
        candidate_id=cid, benchmark_score=benchmark,
        holdout_score=holdout, gate=gate,
    )


def test_domain_detects_gate_adverse_selection():
    # Gate accepts high-benchmark/low-holdout candidates (Goodhart) and
    # rejects an honestly better one.
    outcomes = [
        _outcome("c1", 0.9, 0.3, GateDecision.ACCEPTED),
        _outcome("c2", 0.85, 0.35, GateDecision.ACCEPTED),
        _outcome("c3", 0.4, 0.8, GateDecision.ROLLED_BACK),
        _outcome("c4", 0.2, None, GateDecision.PENDING),  # ignored
    ]
    report = AevolveEvolutionAdapter().from_outcomes(outcomes)
    assert report.n_candidates == 3
    assert report.n_accepted == 2
    assert report.quality_gap < 0
    assert report.adverse_selection


def test_domain_healthy_gate_shows_positive_gap():
    outcomes = [
        _outcome("c1", 0.9, 0.85, GateDecision.ACCEPTED),
        _outcome("c2", 0.3, 0.2, GateDecision.ROLLED_BACK),
    ]
    report = AevolveEvolutionAdapter().from_outcomes(outcomes)
    assert report.quality_gap > 0
    assert not report.adverse_selection


def test_domain_shrinks_train_scores_toward_neutral():
    adapter = AevolveEvolutionAdapter()
    trusted = adapter.candidate_p(_outcome("c", 0.9, 0.9, GateDecision.ACCEPTED))
    shrunk = adapter.candidate_p(_outcome("c", 0.9, None, GateDecision.ACCEPTED))
    assert trusted == 0.9
    assert 0.5 < shrunk < 0.9  # self-graded score gets less credence
    # p always bounded even for out-of-range inputs.
    assert adapter.candidate_p(_outcome("c", 2.0, None, GateDecision.ACCEPTED)) <= 1.0
