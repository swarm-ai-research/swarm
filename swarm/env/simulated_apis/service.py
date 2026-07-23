"""Base service for simulated APIs with state mutation and logging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, cast

from swarm.env.simulated_apis.gating import (
    IrreversibleActionRequiresApproval,
    IrreversibleGate,
)
from swarm.env.simulated_apis.logging import SimApiEpisodeLog, SimApiEvent


class ApiCallError(RuntimeError):
    pass


@dataclass(frozen=True)
class ApiEndpointSpec:
    name: str
    description: str = ""
    cost: int = 1
    irreversible: bool = False
    high_cost: bool = False


@dataclass(frozen=True)
class ApiCallResult:
    ok: bool
    response: Dict[str, Any]
    cost: int
    provenance_id: str
    event_hash: str


class SimulatedApiService:
    """A stateful simulated API surface suitable for multi-agent experiments."""

    domain: str = "base"

    def __init__(
        self,
        *,
        initial_state: Dict[str, Any],
        policy: Dict[str, Any],
        goal_spec: Dict[str, Any],
        api_catalog: Dict[str, ApiEndpointSpec],
        log: Optional[SimApiEpisodeLog] = None,
        gate: Optional[IrreversibleGate] = None,
    ) -> None:
        self.state: Dict[str, Any] = dict(initial_state)
        self.policy: Dict[str, Any] = dict(policy)
        self.goal_spec: Dict[str, Any] = dict(goal_spec)
        self.api_catalog = dict(api_catalog)
        self.log = log or SimApiEpisodeLog()
        self.gate = gate

    def snapshot_state(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], self._deepcopy_json(self.state))  # type: ignore[no-any-return]

    def call(
        self,
        *,
        agent_id: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        parent_event_hash: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> ApiCallResult:
        params = dict(params or {})
        spec = self.api_catalog.get(endpoint)
        if spec is None:
            raise ApiCallError(f"unknown endpoint: {endpoint}")

        if spec.irreversible and self.gate is not None and self.gate.enabled:
            if request_id is None:
                raise IrreversibleActionRequiresApproval(
                    f"endpoint {endpoint} is irreversible; propose/vote/execute via gate"
                )
            pending = self.gate.pop_if_approved(request_id)
            # Ensure the request actually matches (prevents request-id laundering).
            if pending.endpoint != endpoint or pending.params != params:
                raise ApiCallError("request_id does not match endpoint/params")

        handler = getattr(self, f"_handle_{endpoint}", None)
        if handler is None:
            raise ApiCallError(f"endpoint not implemented: {endpoint}")

        response = handler(**params)
        event = self.log.append(
            SimApiEvent(
                event_type="api_call",
                agent_id=agent_id,
                parent_event_hash=parent_event_hash,
                payload={
                    "domain": self.domain,
                    "endpoint": endpoint,
                    "params": params,
                    "cost": spec.cost,
                    "irreversible": spec.irreversible,
                    "high_cost": spec.high_cost,
                    "response": response,
                },
            )
        )

        return ApiCallResult(
            ok=True,
            response=response,
            cost=spec.cost,
            provenance_id=event.provenance_id,
            event_hash=event.event_hash,
        )

    def propose_irreversible(
        self,
        *,
        agent_id: str,
        endpoint: str,
        params: Dict[str, Any],
        parent_event_hash: Optional[str] = None,
    ) -> str:
        if self.gate is None or not self.gate.enabled:
            raise ApiCallError("irreversible gate is disabled")
        return self.gate.propose(
            log=self.log,
            agent_id=agent_id,
            endpoint=endpoint,
            params=params,
            parent_event_hash=parent_event_hash,
        )

    def vote(
        self,
        *,
        agent_id: str,
        request_id: str,
        approve: bool,
        evidence_ids: Optional[list[str]] = None,
        parent_event_hash: Optional[str] = None,
    ) -> None:
        if self.gate is None or not self.gate.enabled:
            raise ApiCallError("irreversible gate is disabled")
        self.gate.vote(
            log=self.log,
            agent_id=agent_id,
            request_id=request_id,
            approve=approve,
            evidence_ids=evidence_ids,
            parent_event_hash=parent_event_hash,
        )

    def _handle_audit_log(self, query: str = "") -> Dict[str, Any]:
        """Query audit log by substring filter."""
        events = list(self.state.get("_audit", []))
        if query:
            events = [e for e in events if query in str(e)]
        return {"events": events[-100:]}

    def _handle_diff_state(self) -> Dict[str, Any]:
        """Return a snapshot of the current service state."""
        return {"state": self.snapshot_state()}

    @staticmethod
    def _deepcopy_json(obj: Any) -> Any:
        # Deterministic, json-compatible deep copy.
        if obj is None or isinstance(obj, (int, float, str, bool)):
            return obj
        if isinstance(obj, list):
            return [SimulatedApiService._deepcopy_json(x) for x in obj]
        if isinstance(obj, dict):
            return {k: SimulatedApiService._deepcopy_json(v) for k, v in obj.items()}
        # Avoid pulling in heavy deps; keep state json-safe.
        raise TypeError(f"non-json type in state: {type(obj)}")
