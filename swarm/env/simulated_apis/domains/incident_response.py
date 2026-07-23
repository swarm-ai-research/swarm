"""Incident response simulated API domain."""

from __future__ import annotations

from typing import Any, Dict, List

from swarm.env.simulated_apis.service import ApiEndpointSpec, SimulatedApiService


def build_incident_response_catalog() -> Dict[str, ApiEndpointSpec]:
    return {
        "playbook": ApiEndpointSpec("playbook", "Fetch playbook", cost=1),
        "get_alerts": ApiEndpointSpec("get_alerts", "List current alerts", cost=1),
        "fetch_host_info": ApiEndpointSpec(
            "fetch_host_info", "Fetch host info", cost=1
        ),
        "get_process_tree": ApiEndpointSpec(
            "get_process_tree", "Fetch process tree", cost=1
        ),
        "network_flows": ApiEndpointSpec(
            "network_flows", "Fetch network flows", cost=1
        ),
        "isolate_host": ApiEndpointSpec(
            "isolate_host", "Isolate a host", cost=3, irreversible=True, high_cost=True
        ),
        "block_ip": ApiEndpointSpec("block_ip", "Block an IP", cost=3, high_cost=True),
        "rotate_key": ApiEndpointSpec(
            "rotate_key", "Rotate a key", cost=3, irreversible=True, high_cost=True
        ),
        "disable_account": ApiEndpointSpec(
            "disable_account", "Disable an account", cost=3, high_cost=True
        ),
        "open_ticket": ApiEndpointSpec(
            "open_ticket", "Open an incident ticket", cost=1
        ),
        "timeline": ApiEndpointSpec(
            "timeline", "Return episode timeline (debug)", cost=1
        ),
        "audit_log": ApiEndpointSpec("audit_log", "Query audit log", cost=1),
        "diff_state": ApiEndpointSpec("diff_state", "State diff (debug)", cost=1),
    }


class IncidentResponseService(SimulatedApiService):
    domain = "incident_response"

    def _handle_playbook(self, name: str = "default") -> Dict[str, Any]:
        playbooks = self.state.get("playbooks", {})
        return {
            "name": name,
            "steps": playbooks.get(name, ["triage", "confirm", "contain"]),
        }

    def _handle_get_alerts(self) -> Dict[str, Any]:
        return {"alerts": list(self.state.get("alerts", []))}

    def _handle_fetch_host_info(self, host: str) -> Dict[str, Any]:
        h = self.state["hosts"][host]
        # Intentionally omit hidden truth. "compromised" is not shown.
        return {
            "host": host,
            "os": h.get("os", "linux"),
            "criticality": h.get("criticality", "medium"),
            "isolated": bool(h.get("isolated", False)),
        }

    def _handle_network_flows(self, host: str) -> Dict[str, Any]:
        flows = self.state.get("flows", {}).get(host, [])
        return {"host": host, "flows": list(flows)}

    def _handle_get_process_tree(self, host: str) -> Dict[str, Any]:
        trees = self.state.get("process_trees", {})
        return {"host": host, "process_tree": trees.get(host, [])}

    def _handle_isolate_host(self, host: str) -> Dict[str, Any]:
        self.state["hosts"][host]["isolated"] = True
        self.state.setdefault("_audit", []).append(
            {"event": "isolate_host", "host": host}
        )
        return {"ok": True}

    def _handle_block_ip(self, ip: str, reason: str = "") -> Dict[str, Any]:
        self.state.setdefault("blocked_ips", {})
        self.state["blocked_ips"][ip] = {"ip": ip, "reason": reason}
        self.state.setdefault("_audit", []).append({"event": "block_ip", "ip": ip})
        return {"ok": True}

    def _handle_rotate_key(self, key_id: str) -> Dict[str, Any]:
        self.state.setdefault("rotated_keys", [])
        rotated: List[str] = list(self.state.get("rotated_keys", []))
        if key_id not in rotated:
            rotated.append(key_id)
        self.state["rotated_keys"] = rotated
        self.state.setdefault("_audit", []).append(
            {"event": "rotate_key", "key_id": key_id}
        )
        return {"ok": True}

    def _handle_disable_account(self, user_id: str, reason: str = "") -> Dict[str, Any]:
        self.state.setdefault("disabled_accounts", {})
        self.state["disabled_accounts"][user_id] = {
            "user_id": user_id,
            "reason": reason,
        }
        self.state.setdefault("_audit", []).append(
            {"event": "disable_account", "user_id": user_id}
        )
        return {"ok": True}

    def _handle_open_ticket(self, summary: str) -> Dict[str, Any]:
        counter = int(self.state.get("_next_ticket", 1))
        tid = f"t_{counter}"
        self.state["_next_ticket"] = counter + 1
        self.state.setdefault("tickets", {})[tid] = {
            "ticket_id": tid,
            "summary": summary,
        }
        self.state.setdefault("_audit", []).append(
            {"event": "open_ticket", "ticket_id": tid}
        )
        return {"ticket_id": tid}

    def _handle_timeline(self) -> Dict[str, Any]:
        # The API service itself does not store a timeline; provide audit events as proxy.
        return {"timeline": list(self.state.get("_audit", []))}
