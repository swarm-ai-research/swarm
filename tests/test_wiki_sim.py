"""Mechanism invariants for the synthetic wiki experiment, not historical claims."""

from dataclasses import replace

import pytest

from swarm.bridges.wiki_sim import SimulationConfig, simulate


def test_deterministic_replay() -> None:
    config = SimulationConfig(n_agents=8, n_tasks=4)
    assert simulate(config, 91).to_dict() == simulate(config, 91).to_dict()


@pytest.mark.parametrize("regime", ["authorized", "prohibited", "independent"])
def test_finite_metrics(regime: str) -> None:
    import math

    result = simulate(SimulationConfig(sharing_regime=regime), 17)
    assert result.events
    assert all(math.isfinite(value) for value in result.metrics.values())


def test_permission_label_does_not_change_behavior() -> None:
    config = SimulationConfig(n_agents=10, n_tasks=4)
    permitted = simulate(config, 27)
    prohibited = simulate(replace(config, sharing_regime="prohibited"), 27)
    # Legal status is not an agent capability or an observable causal signal.
    def actions(result):
        return [
            (event["time"], event["type"], event.get("agent_id"),
             event.get("task_id"), event.get("host_id"))
            for event in result.events
        ]

    assert actions(permitted) == actions(prohibited)


@pytest.mark.parametrize("changes", [
    {"n_agents": 0}, {"n_tasks": 0}, {"n_hosts": 0},
    {"deadline": 0}, {"research_mean": 0},
    {"task_overlap": -0.1}, {"task_overlap": 1.1},
    {"discovery_probability": float("nan")},
    {"publish_probability": 1.1}, {"moderation_time": -1},
    {"sharing_regime": "unknown"}, {"moderation_policy": "unknown"},
    {"relocation_mode": "unknown"},
])
def test_invalid_configuration_rejected(changes: dict) -> None:
    with pytest.raises(ValueError):
        simulate(SimulationConfig(**changes), 1)


@pytest.mark.parametrize("policy", ["ordered", "random", "lock", "global_lock"])
def test_interventions_preserve_exogenous_tasks(policy: str) -> None:
    config = SimulationConfig(n_agents=8, n_tasks=4, moderation_time=2)
    control = simulate(config, 31)
    treatment = simulate(replace(config, moderation_policy=policy), 31)

    def tasks(result):
        return [{key: event[key] for key in (
            "time", "agent_id", "task_id", "answer_key", "deadline", "research_end"
        )} for event in result.events if event["type"] == "task_release"]

    assert tasks(control) == tasks(treatment)


def test_independent_agents_never_use_board() -> None:
    result = simulate(SimulationConfig(sharing_regime="independent"), 1)
    assert not any(event["type"] in {"read", "write", "discovery"}
                   for event in result.events)
    assert result.metrics["shared_submission_rate"] == 0


def test_global_write_lock_suppresses_future_publications() -> None:
    config = SimulationConfig(discovery_probability=1, publish_probability=1,
                              moderation_time=2)
    control = simulate(config, 1)
    locked = simulate(replace(config, moderation_policy="global_lock"), 1)
    assert control.metrics["post_intervention_writes"] > 0
    assert locked.metrics["post_intervention_writes"] == 0


def test_page_ordered_sweep_has_equal_budget_and_evasion() -> None:
    base = SimulationConfig(
        n_agents=12, n_tasks=4, n_hosts=1, task_overlap=1,
        discovery_probability=1, publish_probability=1,
        moderation_time=2, moderation_budget=2,
        moderation_granularity="page", page_deletion_fraction=0.5,
        evasion_learning_probability=1.0,
    )
    ordered = simulate(replace(base, moderation_policy="ordered"), 12)
    random = simulate(replace(base, moderation_policy="random"), 12)
    ordered_removed = [event["removed_pages"] for event in ordered.events
                        if event["type"] == "moderation"]
    random_removed = [event["removed_pages"] for event in random.events
                      if event["type"] == "moderation"]
    assert ordered_removed[0] == random_removed[0] > 0
    assert any(event.get("evasion_learned") for event in ordered.events
               if event["type"] == "moderation")
    assert ordered.metrics["total_writes"] >= 0


def test_shared_submissions_have_matching_prior_read_and_publication() -> None:
    config = SimulationConfig(n_agents=20, n_tasks=4, n_hosts=1,
                              task_overlap=1, discovery_probability=1,
                              publish_probability=1)
    result = simulate(config, 5)
    shared = [event for event in result.events
              if event["type"] == "submission" and event["used_shared_answer"]]
    assert shared, "Fixture must exercise copying, not pass vacuously"
    for submission in shared:
        source = result.events[submission["source_event_id"]]
        assert source["type"] == "write"
        assert source["task_id"] == submission["task_id"]
        assert source["answer"] == submission["answer"]
        assert source["agent_id"] != submission["agent_id"]
        reads = [event for event in result.events[:submission["event_id"]]
                 if event["type"] == "read"
                 and event["agent_id"] == submission["agent_id"]
                 and event["task_id"] == submission["task_id"]
                 and event["source_event_id"] == source["event_id"]]
        assert reads
        assert source["time"] <= reads[0]["time"] <= submission["time"]


def test_deadlines_produce_one_terminal_event_per_assignment() -> None:
    config = SimulationConfig(n_agents=10, n_tasks=3, deadline=0.01,
                              research_mean=100, sharing_regime="independent")
    result = simulate(config, 4)
    terminals = [event for event in result.events
                 if event["type"] in {"submission", "deadline_miss"}]
    assert len(terminals) == config.n_agents * config.n_tasks
    assert len({(event["agent_id"], event["task_id"]) for event in terminals}) == len(terminals)
    assert result.metrics["deadline_miss_rate"] > 0
    assert all(event["time"] <= event["deadline"] for event in terminals
               if event["type"] == "submission")


def test_confirmation_summary_reports_displacement_denominators(tmp_path) -> None:
    import json
    import subprocess
    import sys

    sweep = subprocess.run(
        [sys.executable, "scripts/sweep_wiki_mc.py", "--family", "moderation",
         "--seeds", "2", "--max-cells", "3", "--output", str(tmp_path / "sweep")],
        capture_output=True, text=True, check=True)
    assert "pairs" in sweep.stdout
    subprocess.run(
        [sys.executable, "scripts/analyze_wiki_mc_confirmation.py", "--summary",
         "--input", str(tmp_path / "sweep"), "--output", str(tmp_path / "summary.json")],
        capture_output=True, text=True, check=True)
    cells = json.loads((tmp_path / "summary.json").read_text())["cells"]
    assert len(cells) == 3 and all(cell["n"] == 2 for cell in cells)
    untreated, row = cells[0], cells[2]
    assert untreated["disrupted_works"] == 0 and untreated["displaced_per_disrupted"] is None
    assert row["overrides"] == '{"moderation_policy": "ordered", "relocation_mode": "endogenous"}'
    assert 0 < row["displaced_works"] <= row["relocated_works"] <= row["disrupted_works"]
    assert row["displaced_per_disrupted"] == row["displaced_works"] / row["disrupted_works"]
    assert 0 <= row["alarm_wilson_low"] <= row["alarm_rate"] <= row["alarm_wilson_high"] <= 1
    assert (tmp_path / "summary.csv").exists()
