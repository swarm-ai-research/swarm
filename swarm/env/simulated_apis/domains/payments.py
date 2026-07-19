"""Payments-like simulated API domain."""

from __future__ import annotations

from typing import Any, Dict

from swarm.env.simulated_apis.service import ApiEndpointSpec, SimulatedApiService


def build_payments_catalog() -> Dict[str, ApiEndpointSpec]:
    return {
        "policy_rules": ApiEndpointSpec(
            "policy_rules", "Get current refund policy", cost=1
        ),
        "lookup_order": ApiEndpointSpec("lookup_order", "Fetch order record", cost=1),
        "order_history": ApiEndpointSpec(
            "order_history", "Fetch a user's order history", cost=1
        ),
        "eligible_refund": ApiEndpointSpec(
            "eligible_refund", "Compute eligible refund", cost=1
        ),
        "compute_refund": ApiEndpointSpec(
            "compute_refund", "Compute eligible refund (alias)", cost=1
        ),
        "create_refund": ApiEndpointSpec("create_refund", "Create a refund", cost=2),
        "finalize_refund": ApiEndpointSpec(
            "finalize_refund", "Finalize a refund", cost=2, irreversible=True
        ),
        "release_escrow": ApiEndpointSpec(
            "release_escrow", "Release escrow for an order", cost=2, irreversible=True
        ),
        "flag_fraud": ApiEndpointSpec(
            "flag_fraud", "Flag an order as fraud", cost=2, high_cost=True
        ),
        "issue_credit": ApiEndpointSpec(
            "issue_credit",
            "Issue store credit to a user",
            cost=2,
            irreversible=True,
        ),
        "audit_log": ApiEndpointSpec("audit_log", "Query audit log", cost=1),
        "diff_state": ApiEndpointSpec("diff_state", "State diff (debug)", cost=1),
    }


class PaymentsService(SimulatedApiService):
    domain = "payments"

    def _handle_policy_rules(self) -> Dict[str, Any]:
        return {"policy": self.policy}

    def _handle_lookup_order(self, order_id: str) -> Dict[str, Any]:
        order = self.state["orders"][order_id]
        return dict(order)

    def _handle_order_history(self, user_id: str) -> Dict[str, Any]:
        orders = [
            o
            for o in self.state.get("orders", {}).values()
            if o.get("user_id") == user_id
        ]
        return {"user_id": user_id, "orders": [dict(o) for o in orders]}

    def _handle_eligible_refund(self, order_id: str) -> Dict[str, Any]:
        order = self.state["orders"][order_id]
        # Deterministic rule: eligible amount stored in state (generated under policy).
        return {
            "eligible": bool(order["eligible"]),
            "amount": float(order["eligible_amount"]),
        }

    def _handle_compute_refund(self, order_id: str) -> Dict[str, Any]:
        return self._handle_eligible_refund(order_id)

    def _handle_create_refund(self, order_id: str, amount: float) -> Dict[str, Any]:
        counter = int(self.state.get("_next_refund", 1))
        refund_id = f"r_{counter}"
        self.state["_next_refund"] = counter + 1
        self.state.setdefault("refunds", {})[refund_id] = {
            "refund_id": refund_id,
            "order_id": order_id,
            "amount": float(amount),
            "status": "created",
        }
        self.state.setdefault("_audit", []).append(
            {
                "event": "create_refund",
                "order_id": order_id,
                "refund_id": refund_id,
                "amount": amount,
            }
        )
        return {"refund_id": refund_id}

    def _handle_finalize_refund(self, refund_id: str) -> Dict[str, Any]:
        refund = self.state["refunds"][refund_id]
        refund["status"] = "finalized"
        self.state.setdefault("_audit", []).append(
            {"event": "finalize_refund", "refund_id": refund_id}
        )
        return {"ok": True}

    def _handle_release_escrow(self, order_id: str) -> Dict[str, Any]:
        order = self.state["orders"][order_id]
        order["escrow_status"] = "released"
        self.state.setdefault("_audit", []).append(
            {"event": "release_escrow", "order_id": order_id}
        )
        return {"ok": True}

    def _handle_flag_fraud(self, order_id: str, reason: str = "") -> Dict[str, Any]:
        self.state.setdefault("fraud_flags", {})
        self.state["fraud_flags"][order_id] = {"order_id": order_id, "reason": reason}
        self.state.setdefault("_audit", []).append(
            {"event": "flag_fraud", "order_id": order_id}
        )
        return {"ok": True}

    def _handle_issue_credit(
        self, user_id: str, amount: float, reason: str = ""
    ) -> Dict[str, Any]:
        self.state.setdefault("credits", {})
        self.state["credits"][user_id] = float(
            self.state["credits"].get(user_id, 0.0)
        ) + float(amount)
        self.state.setdefault("_audit", []).append(
            {"event": "issue_credit", "user_id": user_id, "amount": amount}
        )
        return {"ok": True}
