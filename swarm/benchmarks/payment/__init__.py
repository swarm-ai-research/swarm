"""Agent payment safety benchmarks.

Seven benchmark protocols for evaluating whether autonomous economic agents
can remain aligned, accountable, and resilient under real financial pressure.

Each protocol defines domain-specific agent roles (e.g., buyer/seller/treasury,
viewer/proposer/signer, verifier, planner) and traces that vary by protocol
requirements rather than following a single standardized pipeline.

Protocols:
    1. Delegated Spending Safety
    2. Prompt Injection Against Purchasing Agents
    3. Multi-Agent Collusion and Misalignment
    4. Escrow and Milestone Release Reliability
    5. Agent Identity and Authority Boundaries
    6. Cross-Rail Routing Under Constraints
    7. Swarm Treasury Management

Severity hierarchy:
    soft     — inefficiency (overpayment, slow settlement)
    medium   — policy violation with no financial loss
    hard     — financial loss (funds sent incorrectly)
    critical — self-amplifying or hidden loss
"""

from swarm.benchmarks.payment.authority_boundaries import AuthorityBoundariesBenchmark
from swarm.benchmarks.payment.cross_rail_routing import CrossRailRoutingBenchmark
from swarm.benchmarks.payment.delegated_spending import DelegatedSpendingBenchmark
from swarm.benchmarks.payment.escrow_milestone import EscrowMilestoneBenchmark
from swarm.benchmarks.payment.multi_agent_collusion import MultiAgentCollusionBenchmark
from swarm.benchmarks.payment.prompt_injection import PromptInjectionBenchmark
from swarm.benchmarks.payment.swarm_treasury import SwarmTreasuryBenchmark
from swarm.benchmarks.payment.types import (
    AgentRole,
    AuditEntry,
    DifficultyTier,
    FailureSeverity,
    Merchant,
    PaymentAction,
    PaymentInstance,
    PaymentMetrics,
    PaymentPolicy,
    PaymentRail,
)

__all__ = [
    # Types
    "AgentRole",
    "AuditEntry",
    "DifficultyTier",
    "FailureSeverity",
    "Merchant",
    "PaymentAction",
    "PaymentInstance",
    "PaymentMetrics",
    "PaymentPolicy",
    "PaymentRail",
    # Benchmarks
    "DelegatedSpendingBenchmark",
    "PromptInjectionBenchmark",
    "MultiAgentCollusionBenchmark",
    "EscrowMilestoneBenchmark",
    "AuthorityBoundariesBenchmark",
    "CrossRailRoutingBenchmark",
    "SwarmTreasuryBenchmark",
]
