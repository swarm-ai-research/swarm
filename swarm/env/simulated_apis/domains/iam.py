"""IAM-like simulated API domain."""

from __future__ import annotations

from typing import Any, Dict, List

from swarm.env.simulated_apis.service import ApiEndpointSpec, SimulatedApiService


def build_iam_catalog() -> Dict[str, ApiEndpointSpec]:
    return {
        "list_users": ApiEndpointSpec("list_users", "List all users", cost=1),
        "get_user": ApiEndpointSpec("get_user", "Get user record", cost=1),
        "list_roles": ApiEndpointSpec("list_roles", "List available roles", cost=1),
        "get_role": ApiEndpointSpec("get_role", "Get role permissions", cost=1),
        "simulate_access": ApiEndpointSpec(
            "simulate_access", "Simulate access", cost=1
        ),
        "create_user": ApiEndpointSpec("create_user", "Create a user", cost=2),
        "create_service_account": ApiEndpointSpec(
            "create_service_account", "Create service account", cost=2
        ),
        "grant_permission": ApiEndpointSpec(
            "grant_permission",
            "Grant a permission to a user",
            cost=2,
            irreversible=True,
        ),
        "revoke_permission": ApiEndpointSpec(
            "revoke_permission", "Revoke permission", cost=2
        ),
        "set_role": ApiEndpointSpec(
            "set_role",
            "Set a user's role",
            cost=2,
            irreversible=True,
        ),
        "rotate_credentials": ApiEndpointSpec(
            "rotate_credentials",
            "Rotate a user's credentials",
            cost=3,
            high_cost=True,
        ),
        "delete_user": ApiEndpointSpec(
            "delete_user",
            "Delete a user",
            cost=3,
            irreversible=True,
            high_cost=True,
        ),
        "audit_log": ApiEndpointSpec("audit_log", "Query audit log", cost=1),
        "diff_state": ApiEndpointSpec("diff_state", "State diff (debug)", cost=1),
    }


class IamService(SimulatedApiService):
    domain = "iam"

    def _handle_list_users(self) -> Dict[str, Any]:
        users = self.state.get("users", {})
        return {
            "users": [
                {"user_id": uid, "name": u["name"], "role": u["role"]}
                for uid, u in users.items()
            ]
        }

    def _handle_get_user(self, user_id: str) -> Dict[str, Any]:
        u = self.state["users"][user_id]
        return {
            "user_id": user_id,
            "name": u["name"],
            "role": u["role"],
            "permissions": sorted(u.get("permissions", [])),
            "description": u.get("description", ""),
        }

    def _handle_list_roles(self) -> Dict[str, Any]:
        roles = self.state.get("roles", {})
        return {"roles": sorted(roles.keys())}

    def _handle_get_role(self, role: str) -> Dict[str, Any]:
        perms = self.state.get("roles", {}).get(role)
        if perms is None:
            return {"role": role, "permissions": [], "exists": False}
        return {"role": role, "permissions": sorted(perms), "exists": True}

    def _effective_permissions(self, user_id: str) -> List[str]:
        u = self.state["users"][user_id]
        role = u["role"]
        role_perms = set(self.state.get("roles", {}).get(role, []))
        user_perms = set(u.get("permissions", []))
        return sorted(role_perms | user_perms)

    def _handle_simulate_access(
        self, user_id: str, action: str, resource: str
    ) -> Dict[str, Any]:
        # Policy encodes permission strings of the form "{action}:{resource}" or wildcard.
        perms = set(self._effective_permissions(user_id))
        needed = f"{action}:{resource}"
        allowed = (
            needed in perms
            or f"{action}:*" in perms
            or f"*:{resource}" in perms
            or "*:*" in perms
        )
        return {"allowed": bool(allowed), "needed": needed}

    def _handle_create_service_account(self, name: str) -> Dict[str, Any]:
        counter = int(self.state.get("_next_user", 1))
        user_id = f"sa_{counter}"
        self.state["_next_user"] = counter + 1
        self.state.setdefault("users", {})[user_id] = {
            "name": name,
            "role": "service",
            "permissions": [],
            "description": "",
        }
        self.state.setdefault("_audit", []).append(
            {"event": "create_user", "user_id": user_id}
        )
        return {"user_id": user_id}

    def _handle_create_user(self, name: str, role: str = "reader") -> Dict[str, Any]:
        counter = int(self.state.get("_next_user", 1))
        user_id = f"u_{1000 + counter}"
        self.state["_next_user"] = counter + 1
        self.state.setdefault("users", {})[user_id] = {
            "name": name,
            "role": role,
            "permissions": [],
            "description": "",
        }
        self.state.setdefault("_audit", []).append(
            {"event": "create_user", "user_id": user_id}
        )
        return {"user_id": user_id}

    def _handle_grant_permission(self, user_id: str, perm: str) -> Dict[str, Any]:
        self.state["users"][user_id].setdefault("permissions", [])
        if perm not in self.state["users"][user_id]["permissions"]:
            self.state["users"][user_id]["permissions"].append(perm)
        self.state.setdefault("_audit", []).append(
            {"event": "grant", "user_id": user_id, "perm": perm}
        )
        return {"ok": True}

    def _handle_revoke_permission(self, user_id: str, perm: str) -> Dict[str, Any]:
        perms = self.state["users"][user_id].get("permissions", [])
        self.state["users"][user_id]["permissions"] = [p for p in perms if p != perm]
        self.state.setdefault("_audit", []).append(
            {"event": "revoke", "user_id": user_id, "perm": perm}
        )
        return {"ok": True}

    def _handle_set_role(self, user_id: str, role: str) -> Dict[str, Any]:
        self.state["users"][user_id]["role"] = role
        self.state.setdefault("_audit", []).append(
            {"event": "set_role", "user_id": user_id, "role": role}
        )
        return {"ok": True}

    def _handle_rotate_credentials(self, user_id: str) -> Dict[str, Any]:
        self.state.setdefault("credential_rotations", {})
        self.state["credential_rotations"][user_id] = (
            int(self.state["credential_rotations"].get(user_id, 0)) + 1
        )
        self.state.setdefault("_audit", []).append(
            {"event": "rotate_credentials", "user_id": user_id}
        )
        return {"ok": True}

    def _handle_delete_user(self, user_id: str) -> Dict[str, Any]:
        existed = user_id in self.state.get("users", {})
        if existed:
            del self.state["users"][user_id]
        self.state.setdefault("_audit", []).append(
            {"event": "delete_user", "user_id": user_id, "existed": existed}
        )
        return {"ok": existed}
