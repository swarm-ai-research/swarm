"""Governance policy for Prime Agent harness refinement and RLM delegation.

Two things a Prime Agent session does that an ordinary agent transcript does not
expose, and that this policy adjudicates:

**Harness refinement.** ``/refine`` writes durable state the agent will read on
every later session. Upstream already enforces the hard invariant — the base
system prompt is never rewritten — and keeps snapshots for rollback. What it
does not enforce is a *rate*: nothing stops a session from writing many large,
unsupported entries, and nothing scores whether the accumulated harness is
converging or oscillating. Those are the checks here.

**RLM delegation.** ``rlm(...)`` spawns real child sessions whose usage is
attributed to the parent. Depth and fan-out are the cost surface — and, because
a child inherits the parent's harness, also the blast radius of a bad
refinement.

Reuses ``GovernanceConfig.self_evolution_*`` knobs (shared with the LiveSWE
bridge, which governs the same class of runtime self-modification) rather than
introducing a parallel set. Prime-Agent-specific caps that have no analogue
elsewhere live in :class:`RefinementPolicyConfig`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from swarm.bridges.prime_agent.events import RefinementRecord, RlmSpawnRecord
from swarm.bridges.prime_agent.harness import HarnessDriftState, HarnessTracker
from swarm.governance.config import GovernanceConfig

logger = logging.getLogger(__name__)


class PolicyDecision(Enum):
    """Possible outcomes of a policy evaluation."""

    APPROVE = "approve"
    WARN = "warn"
    DENY = "deny"


@dataclass
class PolicyResult:
    """Result of a policy evaluation."""

    decision: PolicyDecision
    reason: str = ""
    governance_cost: float = 0.0

    @property
    def denied(self) -> bool:
        return self.decision is PolicyDecision.DENY

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "governance_cost": self.governance_cost,
        }


@dataclass
class RefinementPolicyConfig:
    """Prime-Agent-specific limits with no existing GovernanceConfig analogue."""

    #: Deny refinements whose rationale cites nothing concrete. Off by default:
    #: on a first pass you usually want the unsupported rate *measured* before
    #: it is enforced, since the evidence test is syntactic.
    require_evidence: bool = False
    #: Content characters above which a refinement stops being a "small update".
    max_refinement_chars: int = 4000
    #: Deepest RLM child level allowed (1 = root may spawn, children may not).
    max_recursion_depth: int = 2
    #: Most children admitted from a single ipython cell.
    max_spawn_fanout: int = 8
    #: Most children admitted across a tracked lineage.
    max_children_total: int = 32
    #: Rollback fraction above which the harness is judged to be oscillating.
    rollback_churn_threshold: float = 0.34
    #: Failed harness edits tolerated before the circuit breaker trips.
    max_failed_edits: int = 5


class HarnessRefinementPolicy:
    """Adjudicates harness refinements and RLM spawns against governance limits.

    All checks are no-ops unless ``GovernanceConfig.self_evolution_enabled`` is
    set, matching the LiveSWE bridge: measurement is the default, enforcement is
    opt-in.
    """

    def __init__(
        self,
        governance_config: Optional[GovernanceConfig] = None,
        policy_config: Optional[RefinementPolicyConfig] = None,
        tracker: Optional[HarnessTracker] = None,
    ) -> None:
        self.config = governance_config or GovernanceConfig()
        self.policy_config = policy_config or RefinementPolicyConfig()
        self.tracker = tracker or HarnessTracker()

    # --- Refinement gating ---

    def evaluate_refinement(
        self,
        refinement: RefinementRecord,
        state: HarnessDriftState,
        reputation: float = 0.0,
    ) -> PolicyResult:
        """Evaluate one ``/refine`` pass.

        Checks, in order of severity:

        1. An edit targeting the immutable base system prompt.
        2. Harness entry count against ``self_evolution_max_tools``.
        3. Net growth rate against ``self_evolution_max_growth_rate``.
        4. Missing concrete evidence (deny only when ``require_evidence``).
        5. Cross-session (``global``) scope from a negative-reputation agent.
        6. Refinement size against ``max_refinement_chars``.
        7. Rollback churn against ``rollback_churn_threshold``.

        Args:
            refinement: The refinement to adjudicate.
            state: Drift state *including* this refinement.
            reputation: The agent's current reputation.

        Returns:
            PolicyResult with decision, reason, and governance cost.
        """
        if not self.config.self_evolution_enabled:
            return PolicyResult(
                decision=PolicyDecision.APPROVE,
                reason="self-evolution governance disabled",
            )

        base_prompt_edits = [e for e in refinement.edits if e.targets_base_prompt]
        if base_prompt_edits and self.config.self_evolution_block_self_mod:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=(
                    f"{len(base_prompt_edits)} edit(s) target the immutable "
                    "base system prompt"
                ),
                governance_cost=0.2,
            )

        if state.total_entries > self.config.self_evolution_max_tools:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=(
                    f"harness entry count {state.total_entries} exceeds limit "
                    f"{self.config.self_evolution_max_tools}"
                ),
                governance_cost=0.1,
            )

        if state.growth_rate > self.config.self_evolution_max_growth_rate:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=(
                    f"harness growth rate {state.growth_rate:.3f} entries/turn "
                    f"exceeds limit {self.config.self_evolution_max_growth_rate}"
                ),
                governance_cost=0.1,
            )

        evidence_backed = self.tracker.is_evidence_backed(refinement)
        if not evidence_backed and self.policy_config.require_evidence:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason="refinement rationale cites no concrete evidence",
                governance_cost=0.15,
            )

        if refinement.scope == "global" and reputation < 0.0:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=(
                    f"cross-session harness write denied for low-reputation "
                    f"agent ({reputation:.2f})"
                ),
                governance_cost=0.15,
            )

        oversized = (
            refinement.total_content_chars > self.policy_config.max_refinement_chars
        )
        if oversized and reputation < 0.0:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=(
                    f"oversized refinement ({refinement.total_content_chars} chars) "
                    f"denied for low-reputation agent ({reputation:.2f})"
                ),
                governance_cost=0.1,
            )
        if oversized:
            return PolicyResult(
                decision=PolicyDecision.WARN,
                reason=(
                    f"refinement writes {refinement.total_content_chars} chars, "
                    f"above the {self.policy_config.max_refinement_chars}-char "
                    "small-update budget"
                ),
                governance_cost=0.05,
            )

        if not evidence_backed:
            return PolicyResult(
                decision=PolicyDecision.WARN,
                reason="refinement rationale cites no concrete evidence",
                governance_cost=0.05,
            )

        if state.rollback_churn > self.policy_config.rollback_churn_threshold:
            return PolicyResult(
                decision=PolicyDecision.WARN,
                reason=(
                    f"rollback churn {state.rollback_churn:.2f} exceeds "
                    f"{self.policy_config.rollback_churn_threshold}: harness is "
                    "oscillating, not converging"
                ),
                governance_cost=0.05,
            )

        return PolicyResult(
            decision=PolicyDecision.APPROVE,
            reason="evidence-backed refinement within limits",
        )

    # --- Delegation gating ---

    def evaluate_spawn(
        self,
        spawn: RlmSpawnRecord,
        state: HarnessDriftState,
        fanout: int = 1,
        reputation: float = 0.0,
    ) -> PolicyResult:
        """Evaluate one ``rlm(...)`` child spawn.

        Args:
            spawn: The spawn record; ``spawn.depth`` is the parent's depth.
            state: Drift state for the lineage.
            fanout: Children spawned from the same ipython cell.
            reputation: The agent's current reputation.
        """
        if not self.config.self_evolution_enabled:
            return PolicyResult(
                decision=PolicyDecision.APPROVE,
                reason="self-evolution governance disabled",
            )

        child_depth = spawn.depth + 1
        if child_depth > self.policy_config.max_recursion_depth:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=(
                    f"child depth {child_depth} exceeds max recursion depth "
                    f"{self.policy_config.max_recursion_depth}"
                ),
                governance_cost=0.1,
            )

        if fanout > self.policy_config.max_spawn_fanout:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=(
                    f"fan-out {fanout} exceeds max "
                    f"{self.policy_config.max_spawn_fanout} children per cell"
                ),
                governance_cost=0.1,
            )

        if state.children_spawned > self.policy_config.max_children_total:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=(
                    f"lineage spawned {state.children_spawned} children, above "
                    f"the cap of {self.policy_config.max_children_total}"
                ),
                governance_cost=0.1,
            )

        if reputation < 0.0 and child_depth > 1:
            return PolicyResult(
                decision=PolicyDecision.WARN,
                reason=(
                    f"nested delegation (depth {child_depth}) by low-reputation "
                    f"agent ({reputation:.2f})"
                ),
                governance_cost=0.05,
            )

        return PolicyResult(
            decision=PolicyDecision.APPROVE,
            reason="spawn within delegation limits",
        )

    # --- Aggregate signals ---

    def should_circuit_break(self, state: HarnessDriftState) -> bool:
        """Whether accumulated drift warrants halting the lineage."""
        if not self.config.self_evolution_enabled:
            return False
        if (
            self.config.self_evolution_block_self_mod
            and state.base_prompt_attempts > 0
        ):
            return True
        if state.failed_edits > self.policy_config.max_failed_edits:
            return True
        return state.drift_score > self.config.self_evolution_divergence_threshold

    def compute_drift_penalty(self, state: HarnessDriftState) -> float:
        """Penalty in [0, 1] applied to engagement for accumulated drift.

        Zero below the divergence threshold, then scaling linearly to 1.0 as
        drift approaches its maximum — so an agent refining within limits pays
        nothing, and the cost appears only once it crosses the line.
        """
        threshold = self.config.self_evolution_divergence_threshold
        drift = state.drift_score
        if drift <= threshold:
            return 0.0
        if threshold >= 1.0:
            return 0.0
        return min(1.0, (drift - threshold) / (1.0 - threshold))
