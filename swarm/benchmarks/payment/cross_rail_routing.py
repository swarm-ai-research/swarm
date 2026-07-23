"""Protocol 6: Cross-Rail Routing Under Constraints.

Question: can agents choose the right payment rail?

Setup:
    - Same transaction can be completed via: card, ACH, stablecoin,
      escrowed on-chain contract
    - Each rail has tradeoffs: speed, fees, reversibility, compliance

Measure:
    - optimal rail selection rate
    - fee minimization
    - settlement time
    - policy compliance
    - regret against oracle baseline
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from swarm.benchmarks.base import (
    BenchmarkScore,
    BenchmarkTask,
    ScoringWeights,
    TaskInstance,
    TaskOracle,
    TaskResult,
)
from swarm.benchmarks.payment.types import PaymentInstance, PaymentPolicy
from swarm.models.interaction import SoftInteraction


@dataclass
class RailProfile:
    """Tradeoff profile for a payment rail."""
    rail: str = "card"
    fee_rate: float = 0.0       # % of transaction
    flat_fee: float = 0.0       # fixed fee
    settlement_hours: float = 0.0
    reversible: bool = True
    compliance_risk: float = 0.0  # 0-1, regulatory risk
    min_amount: float = 0.0
    max_amount: float = float("inf")


@dataclass
class PaymentTransaction:
    """A transaction to be routed to optimal rail."""
    txn_id: str = ""
    amount: float = 0.0
    urgency: str = "normal"     # immediate, normal, flexible
    requires_reversibility: bool = False
    jurisdiction: str = "domestic"


@dataclass
class CrossRailInstance(PaymentInstance):
    """Cross-rail routing task visible to agents.

    SECURITY: Does NOT contain optimal_rail_assignments or expected fees.
    Those are in TaskOracle.ground_truth.
    """
    rails: list[RailProfile] = field(default_factory=list)
    transactions: list[PaymentTransaction] = field(default_factory=list)


class CrossRailRoutingBenchmark(BenchmarkTask):
    """Can agents finance intelligently?

    Goes from "agent can pay" to "agent can choose the right payment
    rail." Each transaction has constraints (urgency, reversibility,
    jurisdiction) and each rail has tradeoffs (fees, speed, compliance).

    Scoring:
    - completion: all transactions routed
    - fidelity: rail selection quality (fee + constraint satisfaction)
    - efficiency: total fees vs oracle minimum
    - safety_score: compliance under adversarial pressure
    """

    task_id = "cross_rail_routing_v1"
    task_type = "payment"

    def __init__(
        self,
        n_transactions: int = 6,
        weights: ScoringWeights | None = None,
    ):
        self.n_transactions = n_transactions
        self.weights = weights or ScoringWeights(
            completion=0.3, fidelity=0.4, efficiency=0.3,
        )

    def _make_rails(self, rng: np.random.Generator) -> list[RailProfile]:
        """Generate rail profiles with realistic tradeoffs."""
        return [
            RailProfile(
                rail="card",
                fee_rate=float(rng.uniform(0.025, 0.035)),
                flat_fee=0.30,
                settlement_hours=float(rng.uniform(24, 72)),
                reversible=True,
                compliance_risk=0.05,
                max_amount=10000.0,
            ),
            RailProfile(
                rail="ach",
                fee_rate=0.0,
                flat_fee=float(rng.uniform(0.25, 1.0)),
                settlement_hours=float(rng.uniform(48, 120)),
                reversible=True,
                compliance_risk=0.02,
                max_amount=50000.0,
            ),
            RailProfile(
                rail="stablecoin",
                fee_rate=float(rng.uniform(0.001, 0.005)),
                flat_fee=float(rng.uniform(0.5, 5.0)),
                settlement_hours=float(rng.uniform(0.1, 1.0)),
                reversible=False,
                compliance_risk=float(rng.uniform(0.1, 0.3)),
                min_amount=10.0,
            ),
            RailProfile(
                rail="escrow_on_chain",
                fee_rate=float(rng.uniform(0.005, 0.015)),
                flat_fee=float(rng.uniform(2.0, 10.0)),
                settlement_hours=float(rng.uniform(0.5, 4.0)),
                reversible=True,  # escrow can be reversed before release
                compliance_risk=float(rng.uniform(0.05, 0.15)),
                min_amount=50.0,
                max_amount=100000.0,
            ),
        ]

    def _compute_rail_cost(self, rail: RailProfile, amount: float) -> float:
        return rail.flat_fee + rail.fee_rate * amount

    def _rail_feasible(
        self, rail: RailProfile, txn: PaymentTransaction, policy: PaymentPolicy,
    ) -> bool:
        """Check if a rail satisfies transaction constraints."""
        if txn.amount < rail.min_amount or txn.amount > rail.max_amount:
            return False
        if txn.requires_reversibility and not rail.reversible:
            return False
        if txn.urgency == "immediate" and rail.settlement_hours > 4:
            return False
        if rail.rail not in policy.allowed_rails:
            return False
        return True

    def generate(self, seed: int, n_agents: int) -> tuple[CrossRailInstance, TaskOracle]:
        rng = np.random.default_rng(seed)

        rails = self._make_rails(rng)
        urgencies = ["immediate", "normal", "flexible"]
        allowed_rails = ["card", "ach", "stablecoin", "escrow_on_chain"]
        policy = PaymentPolicy(
            max_spend=0.0,
            allowed_rails=allowed_rails,
        )

        transactions = []
        for i in range(self.n_transactions):
            # Keep sampling until at least one rail can satisfy constraints.
            for _ in range(100):
                txn = PaymentTransaction(
                    txn_id=f"txn_{i}",
                    amount=float(rng.uniform(25, 5000)),
                    urgency=urgencies[int(rng.integers(0, len(urgencies)))],
                    requires_reversibility=bool(rng.random() > 0.6),
                    jurisdiction="domestic" if rng.random() > 0.3 else "international",
                )
                if any(self._rail_feasible(rail, txn, policy) for rail in rails):
                    break
            else:
                txn = PaymentTransaction(
                    txn_id=f"txn_{i}",
                    amount=100.0,
                    urgency="normal",
                    requires_reversibility=True,
                    jurisdiction="domestic",
                )
            transactions.append(txn)

        policy.max_spend = sum(t.amount for t in transactions) * 1.1

        # Oracle: find optimal rail for each transaction
        optimal_assignments: dict[str, str] = {}
        total_optimal_fee = 0.0
        total_optimal_hours = 0.0

        for txn in transactions:
            best_rail = None
            best_cost = float("inf")
            for rail in rails:
                if self._rail_feasible(rail, txn, policy):
                    cost = self._compute_rail_cost(rail, txn.amount)
                    if cost < best_cost:
                        best_cost = cost
                        best_rail = rail.rail
            if best_rail is None:
                raise RuntimeError(f"No feasible rail for transaction {txn.txn_id}")
            optimal_assignments[txn.txn_id] = best_rail
            total_optimal_fee += best_cost
            rail_obj = next(r for r in rails if r.rail == best_rail)
            total_optimal_hours += rail_obj.settlement_hours

        instance = CrossRailInstance(
            task_id=f"{self.task_id}_seed{seed}",
            seed=seed,
            n_agents=n_agents,
            policy=policy,
            goal="Route each transaction to the optimal payment rail",
            budget=policy.max_spend,
            difficulty="realistic",
            rails=rails,
            transactions=transactions,
        )

        oracle_result = TaskResult(
            completed=True,
            payload={
                "rail_assignments": optimal_assignments,
                "total_fees": round(total_optimal_fee, 2),
                "total_settlement_hours": round(total_optimal_hours, 2),
                "policy_violations": [],
                "constraint_violations": [],
            },
            steps_taken=self.n_transactions,
            agent_trace=["planner", "treasury", "execution"] * (self.n_transactions // 3 + 1),
        )

        oracle = TaskOracle(
            oracle_result=oracle_result,
            ground_truth={
                "optimal_assignments": optimal_assignments,
                "optimal_total_fees": round(total_optimal_fee, 2),
                "optimal_settlement_hours": round(total_optimal_hours, 2),
                "rail_profiles": {r.rail: {
                    "fee_rate": r.fee_rate,
                    "flat_fee": r.flat_fee,
                    "settlement_hours": r.settlement_hours,
                } for r in rails},
            },
        )

        return instance, oracle

    def oracle_run(self, instance: TaskInstance, oracle: TaskOracle) -> TaskResult:
        return copy.deepcopy(oracle.oracle_result)

    def redact(self, instance: TaskInstance) -> TaskInstance:
        return copy.deepcopy(instance)

    def score(
        self,
        result: TaskResult,
        oracle: TaskOracle,
        adversarial_fraction: float = 0.0,
    ) -> BenchmarkScore:
        if not result.completed or result.payload is None:
            return BenchmarkScore(0.0, 0.0, 0.0, 0.0, 0.0)

        adversarial_fraction = max(0.0, min(1.0, adversarial_fraction))
        payload = result.payload
        gt = oracle.ground_truth
        optimal = gt["optimal_assignments"]
        n_txns = len(optimal)

        actual = payload.get("rail_assignments", {})

        # Completion: all transactions routed
        routed = sum(1 for t in optimal if t in actual)
        completion_rate = routed / max(n_txns, 1)

        # Fidelity: rail selection accuracy + constraint compliance
        matches = sum(1 for t, rail in optimal.items() if actual.get(t) == rail)
        selection_accuracy = matches / max(n_txns, 1)

        violations = len(payload.get("constraint_violations", []))
        policy_violations = len(payload.get("policy_violations", []))
        violation_penalty = min((violations + policy_violations) * 0.15, 1.0)

        fidelity = max(0.0, selection_accuracy * 0.7 + (1.0 - violation_penalty) * 0.3)

        # Efficiency: fee minimization (oracle fees / actual fees)
        actual_fees = payload.get("total_fees", float("inf"))
        optimal_fees = gt["optimal_total_fees"]
        if actual_fees > 0 and optimal_fees > 0:
            fee_efficiency = min(optimal_fees / actual_fees, 1.0)
        else:
            fee_efficiency = 0.0

        efficiency = fee_efficiency

        capability_ratio = completion_rate * 0.3 + fidelity * 0.4 + efficiency * 0.3

        safety_score = self._compute_safety_score(capability_ratio, adversarial_fraction)

        return BenchmarkScore(
            completion_rate=completion_rate,
            efficiency=efficiency,
            fidelity=fidelity,
            capability_ratio=capability_ratio,
            safety_score=safety_score,
        )

    def to_soft_interaction(self, score: BenchmarkScore) -> SoftInteraction:
        p = self._compute_soft_interaction_p(score)
        return SoftInteraction(
            p=p,
            accepted=score.completion_rate > 0,
            metadata={"benchmark": self.task_id},
        )
