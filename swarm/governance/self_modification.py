"""Self-modification governance lever implementing the Two-Gate policy.

Implements the architecture from docs/research/self-modification-governance-byline.md:

- Gate 1 (tau): Validation margin on held-out safety metrics
- Gate 2 (K_max): Rolling-window modification capacity cap per agent
- Deterministic risk-tier classification
- Frequency limiting and oscillation detection
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from swarm.governance.levers import GovernanceLever, LeverEffect

if TYPE_CHECKING:
    from swarm.env.state import EnvState
    from swarm.governance.config import GovernanceConfig
    from swarm.governance.refinement_proposal import RefinementProposal
    from swarm.models.interaction import SoftInteraction

# Maximum number of proposals retained per agent for oscillation detection.
_MAX_HISTORY_PER_AGENT = 50

# Maximum number of distinct agent IDs tracked to bound memory.
_MAX_TRACKED_AGENTS = 10_000


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class RiskTier(Enum):
    """Deterministic risk classification for modification proposals."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModificationState(Enum):
    """Lifecycle states for a modification proposal (append-only)."""

    PROPOSED = "proposed"
    SANDBOXED = "sandboxed"
    TESTED = "tested"
    SHADOW = "shadow"
    CANARY = "canary"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


# Valid state transitions (from -> set of allowed targets)
_VALID_TRANSITIONS: dict[ModificationState, set[ModificationState]] = {
    ModificationState.PROPOSED: {ModificationState.SANDBOXED, ModificationState.REJECTED},
    ModificationState.SANDBOXED: {ModificationState.TESTED, ModificationState.REJECTED},
    ModificationState.TESTED: {ModificationState.SHADOW, ModificationState.REJECTED},
    ModificationState.SHADOW: {ModificationState.CANARY, ModificationState.REJECTED},
    ModificationState.CANARY: {
        ModificationState.PROMOTED,
        ModificationState.ROLLED_BACK,
    },
    ModificationState.PROMOTED: {ModificationState.ROLLED_BACK},
    ModificationState.REJECTED: set(),
    ModificationState.ROLLED_BACK: set(),
}


@dataclass
class ModificationProposal:
    """A proposed self-modification with Byline provenance fields.

    Maps to the Byline schema defined in the governance doc:

    - Identity fields: modification_id, agent_id, timestamp, target_ref, change_type
    - Decision context: trigger, objective, evidence_refs
    - Verification: metric_results, constitutional_result, compositional_result
    - Lifecycle: state, state_reason, risk_tier
    - Integrity: entry_hash, prev_hash
    """

    # Identity
    modification_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    agent_id: str = ""
    run_id: str = ""
    timestamp: float = field(default_factory=time.time)
    target_ref: str = ""  # e.g. "ProxyComputer.weights", "skill:summarize"
    change_type: str = ""  # e.g. "config_delta", "skill_add", "workflow_reorder"

    # Decision context
    trigger: str = ""  # what prompted the modification
    objective: str = ""  # what the agent aims to achieve
    evidence_refs: list[str] = field(default_factory=list)
    alternatives_considered: list[str] = field(default_factory=list)

    # Change payload
    proposed_change: Dict[str, Any] = field(default_factory=dict)

    # Verification (populated during gate evaluation)
    metric_results: Dict[str, float] = field(default_factory=dict)
    constitutional_result: Optional[bool] = None
    compositional_result: Optional[bool] = None

    # Lifecycle (append-only transitions)
    state: ModificationState = ModificationState.PROPOSED
    state_reason: str = ""
    risk_tier: RiskTier = RiskTier.LOW

    # Integrity chain
    entry_hash: str = ""
    prev_hash: str = ""

    # Complexity weight (computed by classifier — not agent-settable)
    complexity_weight: float = 0.0

    # Audit trail of state transitions: (state, reason, timestamp)
    state_history: list[Tuple[str, str, float]] = field(default_factory=list)

    def compute_hash(self) -> str:
        """Compute deterministic hash over identity, change payload, and chain.

        Covers all fields that an adversary might mutate post-approval:
        identity, proposed_change (canonical JSON), and prev_hash for chaining.
        """
        # Canonical JSON for proposed_change to ensure deterministic hashing
        change_canonical = json.dumps(self.proposed_change, sort_keys=True)
        payload = (
            f"{self.modification_id}:{self.agent_id}:{self.target_ref}"
            f":{self.change_type}:{self.timestamp}"
            f":{change_canonical}:{self.prev_hash}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def transition(self, new_state: ModificationState, reason: str = "") -> None:
        """Transition to a new lifecycle state with validation.

        Appends to state_history for auditability.
        """
        allowed = _VALID_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid transition: {self.state.value} -> {new_state.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )
        self.state_history.append((self.state.value, self.state_reason, time.time()))
        self.state = new_state
        self.state_reason = reason

    def to_dict(self) -> dict:
        """Serialize for event logging."""
        d = asdict(self)
        d["state"] = self.state.value
        d["risk_tier"] = self.risk_tier.value
        return d


# ---------------------------------------------------------------------------
# Risk-tier classifier
# ---------------------------------------------------------------------------

# Surfaces that auto-escalate to critical
_CRITICAL_SURFACES = frozenset(
    {
        "governance",
        "provenance",
        "auth",
        "key",
        "isolation",
        "classifier",
        "policy",
        "constitutional",
    }
)

# Surfaces that auto-escalate to high
_HIGH_SURFACES = frozenset(
    {
        "api",
        "tool_integration",
        "permission",
        "capability",
        "inter_agent",
    }
)


# Surfaces that are explicitly LOW risk (must match to get LOW classification).
# Anything that doesn't match a known surface defaults to MEDIUM to prevent
# evasion via obfuscated or novel target names.
_LOW_SURFACES = frozenset(
    {
        "prompt",
        "template",
        "docs",
        "readme",
        "memory_cleanup",
        "comment",
        "text_edit",
        "typo",
        "cosmetic",
    }
)


def classify_risk_tier(proposal: ModificationProposal) -> RiskTier:
    """Deterministic risk-tier classification from change surface.

    Classification algorithm (highest severity wins):
    - Critical: touches governance policy, provenance, auth, isolation, or classifier
    - High: new external API/tool, expands permissions, inter-agent capability
    - Medium: workflows, model routing, tool params, memory schemas
    - Low: prompt/template tuning, docs, memory cleanup

    **Anti-evasion:** unknown or unrecognized surfaces default to MEDIUM,
    not LOW.  An agent must match an explicit LOW-risk surface pattern to
    receive the most permissive thresholds.
    """
    target = proposal.target_ref.lower()
    change_type = proposal.change_type.lower()

    # Critical check (highest priority)
    for surface in _CRITICAL_SURFACES:
        if surface in target or surface in change_type:
            return RiskTier.CRITICAL

    # High check
    for surface in _HIGH_SURFACES:
        if surface in target or surface in change_type:
            return RiskTier.HIGH

    # Medium check (explicit indicators)
    medium_indicators = {"workflow", "model", "routing", "tool_param", "memory_schema"}
    for indicator in medium_indicators:
        if indicator in target or indicator in change_type:
            return RiskTier.MEDIUM

    # Low check — must explicitly match a known-safe surface.
    # Unknown/unrecognized targets default to MEDIUM (anti-evasion).
    for surface in _LOW_SURFACES:
        if surface in target or surface in change_type:
            return RiskTier.LOW

    # Default: MEDIUM (fail-safe — unrecognized surfaces are not trusted)
    return RiskTier.MEDIUM


def compute_complexity_weight(proposal: ModificationProposal) -> float:
    """Deterministic complexity weight: w(c) = 1*f + 2*t + 4*p + 3*i + 5*s.

    Where:
    - f: normalized files/surfaces touched (clamped >= 1)
    - t: new/changed tool interfaces count (clamped >= 0)
    - p: privilege delta score (clamped >= 0)
    - i: inter-agent interaction surface delta (clamped >= 0)
    - s: safety-critical surface touched (0/1)

    All inputs are clamped to non-negative values and ``f`` has a floor of 1
    so that every proposal costs at least 1 unit of budget.  This prevents an
    adversarial agent from zeroing or inverting its own weight.
    """
    change = proposal.proposed_change
    f = max(float(change.get("files_touched", 1)), 1.0)
    t = max(float(change.get("tool_interfaces_changed", 0)), 0.0)
    p = max(float(change.get("privilege_delta", 0)), 0.0)
    i = max(float(change.get("interaction_surface_delta", 0)), 0.0)
    s = 1.0 if proposal.risk_tier in (RiskTier.CRITICAL, RiskTier.HIGH) else 0.0

    weight = 1 * f + 2 * t + 4 * p + 3 * i + 5 * s
    # Floor: every proposal costs at least 1.0
    return max(weight, 1.0)


# ---------------------------------------------------------------------------
# Gate 1: Validation margin tau
# ---------------------------------------------------------------------------

# Default tau_min thresholds by tier
_TAU_MIN_DEFAULTS: dict[RiskTier, Optional[float]] = {
    RiskTier.LOW: -0.10,
    RiskTier.MEDIUM: 0.00,
    RiskTier.HIGH: 0.25,
    RiskTier.CRITICAL: None,  # human-only
}


@dataclass
class GateResult:
    """Result of a gate evaluation."""

    passed: bool
    gate_name: str
    value: float = 0.0
    threshold: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


def evaluate_tau_gate(
    proposal: ModificationProposal,
    baseline_means: Dict[str, float],
    baseline_stds: Dict[str, float],
    candidate_values: Dict[str, float],
    candidate_cis: Dict[str, float],
    metric_directions: Dict[str, int],
    sigma_min: float = 0.01,
    tau_min_overrides: Optional[Dict[str, float]] = None,
) -> GateResult:
    """Evaluate Gate 1: validation margin tau.

    For each safety metric j:
        m_j = d_j * (x_j - mu_j) / sigma_j - eps_j / sigma_j

    Overall: tau = min_j(m_j)
    Pass: tau >= tau_min[tier]

    Args:
        proposal: The modification proposal
        baseline_means: Metric name -> baseline mean
        baseline_stds: Metric name -> baseline std dev
        candidate_values: Metric name -> candidate value on held-out eval
        candidate_cis: Metric name -> 95% CI half-width
        metric_directions: Metric name -> +1 (higher better) or -1 (lower better)
        sigma_min: Floor for std dev to avoid division by zero
        tau_min_overrides: Optional per-tier overrides
    """
    tier = proposal.risk_tier

    # Critical tier requires human review
    tau_min_map = dict(_TAU_MIN_DEFAULTS)
    if tau_min_overrides:
        for tier_name, val in tau_min_overrides.items():
            tau_min_map[RiskTier(tier_name)] = val

    tau_min = tau_min_map.get(tier)
    if tau_min is None:
        return GateResult(
            passed=False,
            gate_name="tau",
            details={"reason": "critical tier requires human approval"},
        )

    margins: Dict[str, float] = {}
    for metric in baseline_means:
        if metric not in candidate_values:
            # Fail-closed on missing metrics
            return GateResult(
                passed=False,
                gate_name="tau",
                details={"reason": f"missing candidate value for {metric}"},
            )

        mu = baseline_means[metric]
        sigma = max(baseline_stds.get(metric, sigma_min), sigma_min)
        x = candidate_values[metric]
        eps = candidate_cis.get(metric, 0.0)
        d = metric_directions.get(metric, 1)

        m = d * (x - mu) / sigma - eps / sigma
        margins[metric] = m

    if not margins:
        return GateResult(
            passed=False,
            gate_name="tau",
            details={"reason": "no metrics provided"},
        )

    tau = min(margins.values())
    passed = tau >= tau_min

    return GateResult(
        passed=passed,
        gate_name="tau",
        value=tau,
        threshold=tau_min,
        details={"margins": margins, "tau_min": tau_min},
    )


# ---------------------------------------------------------------------------
# Gate 2: Modification capacity cap K_max
# ---------------------------------------------------------------------------

_K_MAX_DEFAULTS: dict[RiskTier, float] = {
    RiskTier.LOW: 20.0,
    RiskTier.MEDIUM: 35.0,
    RiskTier.HIGH: 50.0,
    RiskTier.CRITICAL: -1.0,  # impossible to pass (strict < 0)
}


def evaluate_k_max_gate(
    proposal: ModificationProposal,
    current_budget_used: float,
    k_max_overrides: Optional[Dict[str, float]] = None,
) -> GateResult:
    """Evaluate Gate 2: modification capacity cap K_max.

    Pass: K_m(T) + w(c_new) <= K_max[tier]
    """
    tier = proposal.risk_tier

    k_max_map = dict(_K_MAX_DEFAULTS)
    if k_max_overrides:
        for tier_name, val in k_max_overrides.items():
            k_max_map[RiskTier(tier_name)] = val

    k_max = k_max_map.get(tier, 0.0)
    new_total = current_budget_used + proposal.complexity_weight

    passed = new_total <= k_max

    return GateResult(
        passed=passed,
        gate_name="k_max",
        value=new_total,
        threshold=k_max,
        details={
            "current_used": current_budget_used,
            "proposal_weight": proposal.complexity_weight,
            "remaining": max(0.0, k_max - new_total),
        },
    )


# ---------------------------------------------------------------------------
# Governance lever
# ---------------------------------------------------------------------------


class SelfModificationLever(GovernanceLever):
    """Governance lever for self-modification proposals.

    Tracks per-agent modification budgets, evaluates proposals against
    the Two-Gate policy, and detects oscillation patterns.

    Thread safety: ``evaluate_proposal`` acquires a per-agent lock so
    that concurrent evaluations for the same agent are serialized,
    preventing TOCTOU races on the budget.
    """

    def __init__(self, config: "GovernanceConfig"):
        super().__init__(config)
        # Per-agent rolling budget: agent_id -> total complexity used
        self._agent_budgets: Dict[str, float] = defaultdict(float)
        # Per-agent modification history for oscillation detection (bounded)
        self._agent_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=_MAX_HISTORY_PER_AGENT)
        )
        # Per-agent modification count this epoch
        self._epoch_counts: Dict[str, int] = defaultdict(int)
        # Per-agent lock for atomic budget check-and-update
        self._agent_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)

    @property
    def name(self) -> str:
        return "self_modification"

    def on_epoch_start(self, state: "EnvState", epoch: int) -> LeverEffect:
        """Reset per-epoch counters (budgets persist across epochs)."""
        self._epoch_counts.clear()
        return LeverEffect(lever_name=self.name)

    def on_interaction(
        self,
        interaction: "SoftInteraction",
        state: "EnvState",
    ) -> LeverEffect:
        if not getattr(self.config, "self_modification_enabled", False):
            return LeverEffect(lever_name=self.name)

        meta = getattr(interaction, "metadata", {}) or {}
        if not meta.get("self_modification"):
            return LeverEffect(lever_name=self.name)

        agent_id = interaction.initiator

        # Check frequency limit (check first, increment only on pass)
        max_per_epoch = getattr(self.config, "self_modification_max_per_epoch", 10)
        if self._epoch_counts[agent_id] >= max_per_epoch:
            return LeverEffect(
                lever_name=self.name,
                cost_a=1.0,  # Sentinel: blocks the modification
                agents_to_freeze={agent_id},
                details={
                    "blocked": True,
                    "reason": "epoch frequency limit exceeded",
                    "count": self._epoch_counts[agent_id],
                    "limit": max_per_epoch,
                },
            )

        # Increment only after passing the check
        self._epoch_counts[agent_id] += 1
        return LeverEffect(lever_name=self.name)

    def evaluate_proposal(
        self,
        proposal: ModificationProposal,
        baseline_means: Dict[str, float],
        baseline_stds: Dict[str, float],
        candidate_values: Dict[str, float],
        candidate_cis: Dict[str, float],
        metric_directions: Dict[str, int],
    ) -> tuple[bool, GateResult, GateResult]:
        """Evaluate a modification proposal against both gates.

        Thread-safe: acquires a per-agent lock so that the budget
        read-check-update is atomic.

        Returns:
            (approved, tau_result, k_max_result)
        """
        # Bound tracked agent count to prevent Sybil-driven memory growth
        if (
            proposal.agent_id not in self._agent_budgets
            and len(self._agent_budgets) >= _MAX_TRACKED_AGENTS
        ):
            return (
                False,
                GateResult(
                    passed=False,
                    gate_name="tau",
                    details={"reason": "agent tracking limit reached"},
                ),
                GateResult(
                    passed=False,
                    gate_name="k_max",
                    details={"reason": "agent tracking limit reached"},
                ),
            )

        # Classify risk tier (governance-side, not agent-settable)
        proposal.risk_tier = classify_risk_tier(proposal)
        proposal.complexity_weight = compute_complexity_weight(proposal)
        proposal.entry_hash = proposal.compute_hash()

        # Gate 1: tau
        tau_result = evaluate_tau_gate(
            proposal,
            baseline_means,
            baseline_stds,
            candidate_values,
            candidate_cis,
            metric_directions,
        )

        # Gate 2: K_max — under lock for atomic budget check-and-update
        lock = self._agent_locks[proposal.agent_id]
        with lock:
            current_budget = self._agent_budgets[proposal.agent_id]
            k_max_result = evaluate_k_max_gate(proposal, current_budget)

            approved = tau_result.passed and k_max_result.passed

            if approved:
                # Atomic budget update
                self._agent_budgets[proposal.agent_id] += proposal.complexity_weight
                self._agent_history[proposal.agent_id].append(proposal)
                proposal.constitutional_result = tau_result.passed
                proposal.compositional_result = k_max_result.passed
                proposal.transition(ModificationState.SANDBOXED, "gates passed")
            else:
                reasons = []
                if not tau_result.passed:
                    reasons.append(f"tau gate: {tau_result.details}")
                if not k_max_result.passed:
                    reasons.append(f"k_max gate: {k_max_result.details}")
                proposal.transition(
                    ModificationState.REJECTED,
                    "; ".join(reasons),
                )

        return approved, tau_result, k_max_result

    def detect_oscillation(self, agent_id: str, window: int = 6) -> bool:
        """Detect if an agent is oscillating (reverting its own changes).

        Checks if the same target_ref appears in both add/remove or
        opposing changes within the recent modification window.
        """
        history = self._agent_history.get(agent_id)
        if history is None or len(history) < 2:
            return False

        recent = list(history)[-window:]
        targets = [p.target_ref for p in recent]
        # Oscillation = same target modified more than twice in window
        counts = Counter(targets)
        return any(c > 2 for c in counts.values())

    def get_agent_budget(
        self, agent_id: str, tier: RiskTier = RiskTier.MEDIUM
    ) -> Dict[str, float]:
        """Return budget status for an agent.

        Args:
            agent_id: The agent to query.
            tier: Risk tier to report K_max against (default MEDIUM).
        """
        k_max = _K_MAX_DEFAULTS[tier]
        used = self._agent_budgets.get(agent_id, 0.0)
        return {
            "used": used,
            "k_max": k_max,
            "remaining": max(0.0, k_max - used),
        }

    def reset_agent_budget(self, agent_id: str) -> None:
        """Reset an agent's modification budget (after consolidation)."""
        self._agent_budgets[agent_id] = 0.0
        self._agent_history[agent_id] = deque(maxlen=_MAX_HISTORY_PER_AGENT)

    def evaluate_refinement(
        self,
        proposal: "RefinementProposal",
        baseline_metrics: Dict[str, float],
        metric_uncertainties: Dict[str, float],
    ) -> Tuple[bool, GateResult, GateResult]:
        """Evaluate a skill refinement against Two-Gate policy.

        Args:
            proposal: The refinement proposal to evaluate.
            baseline_metrics: Current agent performance metrics.
            metric_uncertainties: Uncertainty estimates for metrics.

        Returns:
            (approved, tau_result, k_max_result)
        """
        # Classify risk tier and complexity (governance-controlled)
        proposal.risk_tier = self._classify_refinement_risk(proposal)
        proposal.complexity_weight = self._compute_refinement_complexity(proposal)
        proposal.entry_hash = proposal.compute_hash()

        # Gate 1: τ (safety gate)
        tau_result = self._evaluate_tau_gate_for_refinement(
            proposal, baseline_metrics, metric_uncertainties
        )

        # Gate 2: K_max (capacity gate) — under lock for atomic budget check-and-update
        lock = self._agent_locks[proposal.agent_id]
        with lock:
            current_budget = self._agent_budgets[proposal.agent_id]
            k_max_result = evaluate_k_max_gate(proposal, current_budget)

            approved = tau_result.passed and k_max_result.passed

            if approved:
                # Atomic budget update
                self._agent_budgets[proposal.agent_id] += proposal.complexity_weight
                self._agent_history[proposal.agent_id].append(proposal)
                proposal.constitutional_result = tau_result.passed
                proposal.compositional_result = k_max_result.passed
                proposal.transition(ModificationState.SANDBOXED, "gates passed")
            else:
                reasons = []
                if not tau_result.passed:
                    reasons.append(f"tau gate: {tau_result.details}")
                if not k_max_result.passed:
                    reasons.append(f"k_max gate: {k_max_result.details}")
                proposal.transition(ModificationState.REJECTED, "; ".join(reasons))

        return approved, tau_result, k_max_result

    def _classify_refinement_risk(self, proposal: "RefinementProposal") -> RiskTier:
        """Classify risk tier based on refinement magnitude.

        Classification rules:
        - CRITICAL: Large effect changes (>0.2) or composite skills
        - HIGH: Medium effect changes (>0.1)
        - MEDIUM: Moderate condition changes (>0.15) with small effect changes
        - LOW: Small changes
        """
        max_cond_delta = 0.0
        max_eff_delta = 0.0

        # Compute max condition delta
        for _key, (old, new) in proposal.condition_delta.items():
            if isinstance(old, (int, float)) and isinstance(new, (int, float)):
                max_cond_delta = max(max_cond_delta, abs(new - old))

        # Compute max effect delta
        for _key, (old, new) in proposal.effect_delta.items():
            if isinstance(old, (int, float)) and isinstance(new, (int, float)):
                max_eff_delta = max(max_eff_delta, abs(new - old))

        # Classification logic (highest severity wins)
        if max_eff_delta > 0.2 or proposal.skill_type == "COMPOSITE":
            return RiskTier.CRITICAL
        elif max_eff_delta > 0.1:
            return RiskTier.HIGH
        elif max_cond_delta > 0.15 and max_eff_delta > 0.05:
            return RiskTier.MEDIUM
        else:
            return RiskTier.LOW

    def _compute_refinement_complexity(self, proposal: "RefinementProposal") -> float:
        """Return the refinement's complexity weight, floored at 1.0.

        The weight itself is computed in RefinementProposal.__post_init__();
        the floor guarantees every proposal costs at least one budget unit.
        """
        result: float = max(proposal.complexity_weight, 1.0)
        return result

    def _evaluate_tau_gate_for_refinement(
        self,
        proposal: "RefinementProposal",
        baseline_metrics: Dict[str, float],
        uncertainties: Dict[str, float],
    ) -> GateResult:
        """Evaluate τ gate: Does refinement improve safety?

        Two checks:
        1. Effect direction: Does refinement move parameters toward safety?
           (e.g., higher acceptance_threshold_delta = more conservative)
        2. Condition justification: Is refinement justified by poor performance?

        Args:
            proposal: The refinement proposal.
            baseline_metrics: Current agent performance metrics.
            uncertainties: Uncertainty estimates for metrics.

        Returns:
            GateResult with passed/failed status and details.
        """
        details: Dict[str, Any] = {}

        # Check 1: Effect direction (higher threshold_delta = more conservative).
        # No acceptance_threshold_delta change (or a non-numeric one) is neutral.
        effect_safe_direction = True
        details["effect_safe_direction"] = True
        for key, (old, new) in proposal.effect_delta.items():
            if key != "acceptance_threshold_delta":
                continue
            if isinstance(old, (int, float)) and isinstance(new, (int, float)):
                effect_safe_direction = new > old
                details["effect_safe_direction"] = effect_safe_direction
                if not effect_safe_direction:
                    details["warning"] = f"{key}: moved in risky direction"

        # Check 2: Condition vs performance
        # Refinement is justified if skill is under-performing
        condition_justified = False
        success_rate = proposal.perf_before.get("success_rate", 1.0)
        if success_rate < 0.4:
            condition_justified = True
            details["condition_tightening_justified"] = True
        else:
            condition_justified = False
            details["condition_tightening_justified"] = False
            if "warning" not in details:
                details["warning"] = "Refining well-performing skill"

        # Get tau_min threshold for this risk tier
        tau_min = _TAU_MIN_DEFAULTS.get(proposal.risk_tier, 0.0)

        # Compute tau value (0-1 scale, two components)
        tau_value = 0.0
        if effect_safe_direction:
            tau_value += 0.5
        if condition_justified:
            tau_value += 0.5

        passed = tau_value >= tau_min if tau_min is not None else False

        return GateResult(
            passed=passed,
            gate_name="tau_refinement",
            value=tau_value,
            threshold=tau_min if tau_min is not None else 0.0,
            details=details,
        )
