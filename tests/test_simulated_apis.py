"""Tests for the simulated API testbed scaffold."""

import pytest

from swarm.env.simulated_apis.gating import (
    ApprovalConfig,
    IrreversibleActionRequiresApproval,
)
from swarm.env.simulated_apis.metrics import compute_episode_metrics
from swarm.env.simulated_apis.spec import Domain, Split
from swarm.env.simulated_apis.suite import (
    generate_task_bundle,
    make_service_and_log,
    score_task_bundle,
)
from swarm.env.simulated_apis.templates import list_templates


def test_templates_exist():
    assert len(list_templates(Domain.IAM)) >= 10
    assert len(list_templates(Domain.PAYMENTS)) >= 10
    assert len(list_templates(Domain.INCIDENT_RESPONSE)) >= 10


def test_bundle_deterministic():
    b1 = generate_task_bundle(
        Domain.IAM,
        seed=123,
        split=Split.TUNING,
        template_id="create_sa_repo",
        injection_rate=0.2,
    )
    b2 = generate_task_bundle(
        Domain.IAM,
        seed=123,
        split=Split.TUNING,
        template_id="create_sa_repo",
        injection_rate=0.2,
    )
    assert b1.to_dict() == b2.to_dict()


def test_gate_blocks_irreversible_without_request_id():
    bundle = generate_task_bundle(Domain.IAM, seed=1, template_id="create_sa_repo")
    svc, _ = make_service_and_log(
        bundle, gate_enabled=True, approvals=ApprovalConfig(m=2, n=3)
    )
    user_id = svc.call(
        agent_id="a1", endpoint="create_service_account", params={"name": "bot"}
    ).response["user_id"]
    with pytest.raises(IrreversibleActionRequiresApproval):
        svc.call(
            agent_id="a1",
            endpoint="grant_permission",
            params={"user_id": user_id, "perm": "read:repo"},
        )


def test_gate_approval_allows_irreversible_execution():
    bundle = generate_task_bundle(Domain.IAM, seed=2, template_id="create_sa_repo")
    svc, log = make_service_and_log(
        bundle, gate_enabled=True, approvals=ApprovalConfig(m=2, n=3)
    )
    user_id = svc.call(
        agent_id="a1", endpoint="create_service_account", params={"name": "bot"}
    ).response["user_id"]

    params = {"user_id": user_id, "perm": "read:repo"}
    req = svc.propose_irreversible(
        agent_id="a1", endpoint="grant_permission", params=params
    )
    svc.vote(agent_id="a1", request_id=req, approve=True)
    svc.vote(agent_id="a2", request_id=req, approve=True)
    svc.call(agent_id="a1", endpoint="grant_permission", params=params, request_id=req)

    metrics = compute_episode_metrics(log)
    assert metrics.gate_proposals == 1
    assert metrics.gate_votes == 2


def test_scoring_create_sa_repo_success():
    bundle = generate_task_bundle(Domain.IAM, seed=3, template_id="create_sa_repo")
    svc, log = make_service_and_log(
        bundle, gate_enabled=True, approvals=ApprovalConfig(m=2, n=3)
    )
    uid = svc.call(
        agent_id="a1", endpoint="create_service_account", params={"name": "bot"}
    ).response["user_id"]

    for perm in ["read:repo", "write:repo"]:
        params = {"user_id": uid, "perm": perm}
        req = svc.propose_irreversible(
            agent_id="a1", endpoint="grant_permission", params=params
        )
        svc.vote(agent_id="a1", request_id=req, approve=True)
        svc.vote(agent_id="a2", request_id=req, approve=True)
        svc.call(
            agent_id="a1", endpoint="grant_permission", params=params, request_id=req
        )

    score = score_task_bundle(bundle, final_state=svc.snapshot_state())
    assert score.success is True
