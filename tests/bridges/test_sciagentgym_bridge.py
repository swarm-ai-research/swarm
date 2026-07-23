from __future__ import annotations

from dataclasses import dataclass

from swarm.bridges.sciagentgym import (
    EnvironmentTopology,
    SciAgentGymToolProvider,
    SciEnvManager,
    ToolCallFingerprint,
    ToolLoopDetector,
)


@dataclass
class FakeEnv:
    name: str
    closed: bool = False

    def list_tools(self) -> list[dict]:
        return [{"name": f"tool_{self.name}", "args_schema": {"x": "int"}}]

    def call_tool(self, name: str, arguments: dict) -> dict:
        if name == "explode":
            raise RuntimeError("boom")
        return {
            "ok": True,
            "result": {"tool": name, "args": arguments},
            "artifacts": [f"/{self.name}/result.json"],
        }

    def close(self) -> None:
        self.closed = True


def test_env_manager_reuses_env_for_shared_topology():
    created: list[str] = []

    def factory(**kwargs):
        created.append(kwargs["episode_id"])
        return FakeEnv(name=kwargs["episode_id"])

    manager = SciEnvManager(env_factory=factory, topology=EnvironmentTopology.SHARED_EPISODE)

    env1 = manager.get_or_create(episode_id="ep1", agent_id="a1")
    env2 = manager.get_or_create(episode_id="ep1", agent_id="a2")

    assert env1 is env2
    assert created == ["ep1"]


def test_env_manager_uses_agent_partition_for_per_agent_topology():
    manager = SciEnvManager(
        env_factory=lambda **kwargs: FakeEnv(name=f"{kwargs['episode_id']}:{kwargs['agent_id']}"),
        topology=EnvironmentTopology.PER_AGENT,
    )

    env1 = manager.get_or_create(episode_id="ep1", agent_id="a1")
    env2 = manager.get_or_create(episode_id="ep1", agent_id="a2")

    assert env1 is not env2


def test_tool_provider_normalizes_call_results_and_errors():
    manager = SciEnvManager(env_factory=lambda **_: FakeEnv(name="lab"))
    provider = SciAgentGymToolProvider(env_manager=manager)

    tools = provider.list_tools(episode_id="ep")
    assert tools[0]["name"] == "tool_lab"

    ok = provider.call_tool(
        episode_id="ep",
        tool_name="measure",
        arguments={"sample": 2},
    )
    assert ok.ok is True
    assert ok.result == {"tool": "measure", "args": {"sample": 2}}
    assert ok.artifacts == ["/lab/result.json"]
    assert len(ok.call_signature_hash) == 16

    failed = provider.call_tool(
        episode_id="ep",
        tool_name="explode",
        arguments={},
    )
    assert failed.ok is False
    assert failed.error == "boom"


def test_tool_loop_detector_trips_circuit_breaker_on_repetition():
    detector = ToolLoopDetector(ngram_size=2, window_size=8, max_repeat_ratio=0.3)
    fp_a = ToolCallFingerprint("query_db", "a", None)
    fp_b = ToolCallFingerprint("query_db", "b", "ValidationError")

    for fp in [fp_a, fp_b, fp_a, fp_b, fp_a, fp_b]:
        detector.record(fp)

    assert detector.repeat_ratio() > 0.3
    assert detector.should_trip_circuit_breaker() is True


def test_env_manager_close_all_calls_close_hook():
    envs: list[FakeEnv] = []

    def factory(**kwargs):
        env = FakeEnv(name=kwargs["episode_id"])
        envs.append(env)
        return env

    manager = SciEnvManager(env_factory=factory, topology=EnvironmentTopology.PER_TASK)
    manager.get_or_create(episode_id="ep1", task_id="t1")
    manager.get_or_create(episode_id="ep1", task_id="t2")

    manager.close_all()

    assert all(e.closed for e in envs)
