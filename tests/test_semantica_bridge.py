"""Tests for the Semantica export bridge (swarm/bridges/semantica/)."""

import json

import pytest

from swarm.bridges.semantica.client import (
    SemanticaMCPError,
    decode_tool_result,
    encode_request,
)
from swarm.bridges.semantica.exporter import export_run, load_interactions
from swarm.bridges.semantica.mapper import (
    interaction_to_decision,
    proxy_weights_to_bridge_axioms,
    run_manifest,
)
from swarm.logging.event_log import EventLog
from swarm.models.events import Event, EventType
from swarm.models.interaction import InteractionType, SoftInteraction


def make_interaction(p=0.73, accepted=True, **kwargs) -> SoftInteraction:
    defaults = {
        "initiator": "agent_a",
        "counterparty": "agent_b",
        "interaction_type": InteractionType.COLLABORATION,
        "accepted": accepted,
        "v_hat": 0.4,
        "p": p,
        "tau": 0.1,
        "c_a": 0.02,
        "c_b": 0.01,
        "r_a": 0.05,
        "r_b": 0.03,
    }
    defaults.update(kwargs)
    return SoftInteraction(**defaults)


class TestMapper:
    def test_confidence_is_p_untouched(self):
        for p in (0.0, 0.31, 0.5, 0.999, 1.0):
            d = interaction_to_decision(make_interaction(p=p))
            assert d["confidence"] == p

    def test_outcome_maps_accept_reject(self):
        assert interaction_to_decision(make_interaction(accepted=True))["outcome"] == "accepted"
        assert interaction_to_decision(make_interaction(accepted=False))["outcome"] == "rejected"

    def test_decision_maker_is_counterparty(self):
        d = interaction_to_decision(make_interaction())
        assert d["decision_maker"] == "agent_b"
        orphan = interaction_to_decision(make_interaction(counterparty=""))
        assert orphan["decision_maker"] == "swarm:governance"

    def test_entities_and_category(self):
        d = interaction_to_decision(make_interaction(), category_prefix="swarm")
        assert d["entities"] == ["agent_a", "agent_b"]
        assert d["category"] == "swarm:collaboration"

    def test_metadata_preserves_payoffs_and_lineage(self):
        i = make_interaction(causal_parents=["prior-1"], ground_truth=1)
        d = interaction_to_decision(i, run_id="r1", scenario_id="baseline", seed=42)
        md = d["metadata"]
        assert md["run_id"] == "r1"
        assert md["scenario_id"] == "baseline"
        assert md["seed"] == 42
        assert md["causal_parents"] == ["prior-1"]
        assert md["ground_truth"] == 1
        assert md["payoff"]["tau"] == pytest.approx(0.1)
        assert md["v_hat"] == pytest.approx(0.4)

    def test_reasoning_mentions_proxy_signals(self):
        d = interaction_to_decision(make_interaction())
        assert "v_hat" in d["reasoning"]
        assert "P(beneficial)=0.7300" in d["reasoning"]

    def test_bridge_axioms_from_weights(self):
        axioms = proxy_weights_to_bridge_axioms(
            {"task_progress": 0.4, "rework_penalty": 0.2}, sigmoid_k=2.0
        )
        assert len(axioms) == 3
        by_name = {a["name"]: a for a in axioms}
        assert by_name["task_progress"]["coefficient"] == 0.4
        assert by_name["calibrated_sigmoid"]["output_domain"] == "swarm:soft_label"
        assert all(a["axiom_id"].startswith("BA-SWARM-") for a in axioms)

    def test_run_manifest_shape(self):
        m = run_manifest("r1", scenario_id="s", seed=7, n_interactions=3)
        assert m["record_type"] == "swarm_run_manifest"
        assert m["n_interactions"] == 3


class TestClientFraming:
    def test_encode_request_and_notification(self):
        req = json.loads(encode_request(5, "tools/call", {"name": "x"}))
        assert req == {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "x"}}
        note = json.loads(encode_request(None, "notifications/initialized", {}))
        assert "id" not in note

    def test_decode_text_block_json(self):
        resp = {
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": json.dumps({"decision_id": "d1"})}]
            },
        }
        assert decode_tool_result(resp) == {"decision_id": "d1"}

    def test_decode_raises_on_rpc_error(self):
        with pytest.raises(SemanticaMCPError):
            decode_tool_result({"id": 1, "error": {"code": -32000, "message": "boom"}})

    def test_decode_raises_on_tool_error(self):
        with pytest.raises(SemanticaMCPError):
            decode_tool_result({"id": 1, "result": {"isError": True, "content": []}})

    def test_decode_plain_text_falls_back(self):
        resp = {"id": 1, "result": {"content": [{"type": "text", "text": "ok"}]}}
        assert decode_tool_result(resp) == {"text": "ok"}


def write_event_log(run_dir, interactions):
    log = EventLog(run_dir / "events.jsonl")
    for i in interactions:
        log.append(
            Event(
                event_type=EventType.INTERACTION_PROPOSED,
                interaction_id=i.interaction_id,
                initiator_id=i.initiator,
                counterparty_id=i.counterparty,
                payload={
                    "interaction_type": i.interaction_type.value,
                    "v_hat": i.v_hat,
                    "p": i.p,
                    "causal_parents": i.causal_parents,
                },
            )
        )
        if i.accepted:
            log.append(
                Event(
                    event_type=EventType.INTERACTION_ACCEPTED,
                    interaction_id=i.interaction_id,
                )
            )
        log.append(
            Event(
                event_type=EventType.PAYOFF_COMPUTED,
                interaction_id=i.interaction_id,
                payload={"components": {"tau": i.tau, "c_a": i.c_a, "c_b": i.c_b,
                                        "r_a": i.r_a, "r_b": i.r_b}},
            )
        )
    return log


class TestExporter:
    @pytest.fixture
    def run_dir(self, tmp_path):
        run = tmp_path / "20260809T000000Z_baseline_seed42"
        run.mkdir()
        interactions = [
            make_interaction(p=0.9, accepted=True, interaction_id="i-1"),
            make_interaction(p=0.2, accepted=False, interaction_id="i-2"),
        ]
        write_event_log(run, interactions)
        (run / "history.json").write_text(
            json.dumps(
                {
                    "simulation_id": "baseline",
                    "seed": 42,
                    "proxy": {"weights": {"task_progress": 0.5}, "sigmoid_k": 2.0},
                }
            )
        )
        return run

    def test_load_interactions_roundtrip(self, run_dir):
        loaded = load_interactions(run_dir)
        assert {i.interaction_id for i in loaded} == {"i-1", "i-2"}
        by_id = {i.interaction_id: i for i in loaded}
        assert by_id["i-1"].accepted and by_id["i-1"].p == 0.9
        assert not by_id["i-2"].accepted

    def test_export_writes_manifest_then_decisions(self, run_dir):
        summary = export_run(run_dir)
        assert summary.n_interactions == 2
        assert summary.n_written == 2
        lines = [json.loads(x) for x in summary.out_path.read_text().splitlines()]
        manifest, decisions = lines[0], lines[1:]
        assert manifest["record_type"] == "swarm_run_manifest"
        assert manifest["scenario_id"] == "baseline"
        assert manifest["seed"] == 42
        axiom_names = {a["name"] for a in manifest["bridge_axioms"]}
        assert axiom_names == {"task_progress", "calibrated_sigmoid"}
        assert len(decisions) == 2
        for d in decisions:
            assert 0.0 <= d["confidence"] <= 1.0
            assert d["confidence"] == d["metadata"]["p"]
            assert d["metadata"]["run_id"] == run_dir.name

    def test_export_pushes_via_client(self, run_dir):
        class StubClient:
            def __init__(self):
                self.calls = []

            def record_decision(self, decision):
                self.calls.append(decision)
                return {"decision_id": f"d{len(self.calls)}", "status": "recorded"}

        stub = StubClient()
        summary = export_run(run_dir, client=stub)
        assert summary.n_pushed == 2
        assert not summary.push_errors
        assert {c["outcome"] for c in stub.calls} == {"accepted", "rejected"}

    def test_push_errors_do_not_abort_artifact(self, run_dir):
        class FailingClient:
            def record_decision(self, decision):
                raise RuntimeError("connection lost")

        summary = export_run(run_dir, client=FailingClient())
        assert summary.n_pushed == 0
        assert len(summary.push_errors) == 2
        assert summary.n_written == 2  # artifact still complete

    def test_empty_run_dir_exports_manifest_only(self, tmp_path):
        run = tmp_path / "empty_run"
        run.mkdir()
        summary = export_run(run)
        assert summary.n_interactions == 0
        lines = summary.out_path.read_text().splitlines()
        assert len(lines) == 1
