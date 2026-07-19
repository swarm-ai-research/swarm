"""Agent Behavioral Contracts (ABC) formalism.

Implements the ABC framework from arXiv:2602.22302 where a contract is
C = (P, I, G, R):
    P - Preconditions: what must hold before an agent enters
    I - Invariants: runtime checks that must hold throughout
    G - Governance: execution, audit, and penalty logic (existing Contract)
    R - Recovery: what happens when invariants are violated

Also provides DriftDetector for tracking behavioral drift D* across
sessions, which is critical for detecting trust-building adversaries.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from swarm.contracts.contract import Contract
from swarm.models.agent import AgentState
from swarm.models.interaction import SoftInteraction

# ── Preconditions ──────────────────────────────────────────────────

@dataclass
class Precondition:
    """A predicate that must hold before an agent enters a contract."""

    name: str
    check: Callable[[AgentState], bool]
    description: str = ""

    def evaluate(self, agent: AgentState) -> bool:
        return self.check(agent)


def min_resources(threshold: float) -> Precondition:
    """Precondition: agent must have at least `threshold` resources."""
    return Precondition(
        name=f"min_resources({threshold})",
        check=lambda a: a.resources >= threshold,
        description=f"Agent resources >= {threshold}",
    )


def min_trust_score(threshold: float) -> Precondition:
    """Precondition: agent reputation must be at least `threshold`."""
    return Precondition(
        name=f"min_trust({threshold})",
        check=lambda a: a.reputation >= threshold,
        description=f"Agent reputation >= {threshold}",
    )


# ── Invariants ─────────────────────────────────────────────────────

@dataclass
class InvariantCheck:
    """A runtime invariant that must hold throughout a contract."""

    name: str
    check: Callable[[SoftInteraction], bool]
    severity: float = 1.0  # 0..1, how bad a violation is
    description: str = ""

    def __post_init__(self) -> None:
        if not (0.0 <= self.severity <= 1.0):
            raise ValueError(
                f"InvariantCheck.severity must be between 0.0 and 1.0 inclusive; got {self.severity!r}"
            )

    def evaluate(self, interaction: SoftInteraction) -> bool:
        return self.check(interaction)


def p_in_bounds(low: float = 0.0, high: float = 1.0) -> InvariantCheck:
    """Invariant: p must stay within [low, high]."""
    return InvariantCheck(
        name=f"p_bounds({low},{high})",
        check=lambda i: low <= i.p <= high,
        severity=0.8,
        description=f"p in [{low}, {high}]",
    )


def max_drift_rate(threshold: float) -> InvariantCheck:
    """Invariant: behavioral drift D* must stay below threshold.

    This invariant is stateless — it reads drift_rate from interaction metadata
    (defaults to 0.0 if absent). Note: DriftDetector computes drift internally
    but does not populate interaction.metadata; callers must populate this
    field separately if integration with DriftDetector is desired.
    """
    return InvariantCheck(
        name=f"max_drift({threshold})",
        check=lambda i: i.metadata.get("drift_rate", 0.0) <= threshold,
        severity=1.0,
        description=f"drift_rate <= {threshold}",
    )


# ── Recovery mechanisms ────────────────────────────────────────────

@dataclass
class RecoveryAction:
    """An action taken when an invariant is violated."""

    type: str  # "circuit_breaker", "tier_downgrade", "penalty", "expel"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryPolicy:
    """Defines what happens when invariants are violated."""

    name: str
    # Maps violation severity thresholds to recovery actions.
    # If severity >= threshold, the action is triggered.
    # Checked in descending order; first match wins.
    escalation: List[tuple[float, RecoveryAction]] = field(default_factory=list)

    def select_action(self, severity: float) -> Optional[RecoveryAction]:
        """Select recovery action based on violation severity."""
        for threshold, action in sorted(self.escalation, key=lambda x: -x[0]):
            if severity >= threshold:
                return action
        return None


def default_recovery() -> RecoveryPolicy:
    """Standard escalating recovery: penalty -> downgrade -> expel."""
    return RecoveryPolicy(
        name="default_escalating",
        escalation=[
            (0.3, RecoveryAction(type="penalty", metadata={"multiplier": 1.5})),
            (0.6, RecoveryAction(type="tier_downgrade")),
            (0.9, RecoveryAction(type="expel")),
        ],
    )


# ── Invariant violation record ─────────────────────────────────────

@dataclass
class InvariantViolation:
    """Record of an invariant violation."""

    agent_id: str
    invariant_name: str
    severity: float
    interaction_id: str
    recovery_action: Optional[RecoveryAction]
    epoch: int = 0
    step: int = 0


# ── BehavioralContract ─────────────────────────────────────────────

class BehavioralContract:
    """ABC-formalism wrapper around an existing Contract.

    Adds preconditions (P), invariants (I), and recovery (R) on top of
    the existing governance (G) logic in Contract.

    This does NOT subclass Contract — it composes with one. The wrapped
    contract handles execute/penalize/audit; BehavioralContract adds the
    P/I/R layers.
    """

    def __init__(
        self,
        governance: Contract,
        preconditions: Optional[List[Precondition]] = None,
        invariants: Optional[List[InvariantCheck]] = None,
        recovery: Optional[RecoveryPolicy] = None,
    ):
        self.governance = governance
        self.preconditions = preconditions or []
        self.invariants = invariants or []
        self.recovery = recovery or default_recovery()
        self._violations: List[InvariantViolation] = []
        self._expelled: set[str] = set()

    @property
    def name(self) -> str:
        return f"ABC({self.governance.name})"

    def check_preconditions(self, agent: AgentState) -> tuple[bool, List[str]]:
        """Check all preconditions for an agent.

        Returns:
            (all_passed, list_of_failure_reasons)
        """
        failures: List[str] = []
        for pre in self.preconditions:
            if not pre.evaluate(agent):
                failures.append(pre.name)
        return len(failures) == 0, failures

    def check_invariants(
        self, interaction: SoftInteraction
    ) -> List[InvariantViolation]:
        """Check all invariants against an interaction.

        Returns list of violations (empty if all pass).
        Recovery actions are recorded in the violation objects. Currently,
        only ``expel`` is enforced (blocks future ``execute()`` calls);
        ``penalty`` and ``tier_downgrade`` are tracked for callers to
        act on via the violation list.
        """
        violations: List[InvariantViolation] = []
        for inv in self.invariants:
            if not inv.evaluate(interaction):
                action = self.recovery.select_action(inv.severity)
                violation = InvariantViolation(
                    agent_id=interaction.initiator,
                    invariant_name=inv.name,
                    severity=inv.severity,
                    interaction_id=interaction.interaction_id,
                    recovery_action=action,
                )
                violations.append(violation)
                self._violations.append(violation)

                # Apply recovery
                if action and action.type == "expel":
                    self._expelled.add(interaction.initiator)

        return violations

    def is_expelled(self, agent_id: str) -> bool:
        return agent_id in self._expelled

    def execute(self, interaction: SoftInteraction) -> SoftInteraction:
        """Execute the governance contract, then check invariants.

        Raises ``PermissionError`` if the initiator has been expelled.
        """
        if self.is_expelled(interaction.initiator):
            raise PermissionError(
                f"Agent '{interaction.initiator}' has been expelled and "
                f"cannot execute under contract {self.name}"
            )
        modified = self.governance.execute(interaction)
        self.check_invariants(modified)
        return modified

    def get_violations(
        self, agent_id: Optional[str] = None
    ) -> List[InvariantViolation]:
        if agent_id:
            return [v for v in self._violations if v.agent_id == agent_id]
        return list(self._violations)

    def reset(self) -> None:
        self._violations.clear()
        self._expelled.clear()

    def to_report_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "governance": self.governance.name,
            "preconditions": [p.name for p in self.preconditions],
            "invariants": [i.name for i in self.invariants],
            "recovery": self.recovery.name,
            "violations": len(self._violations),
            "expelled": len(self._expelled),
        }


# ── DriftDetector ──────────────────────────────────────────────────

class DriftDetector:
    """Tracks behavioral drift D* across sessions.

    Detects agents whose behavior changes significantly after building
    trust. Uses a sliding window of p values per agent and computes
    drift as the signed difference between baseline and recent means:

        D* = mean(baseline_window) - mean(recent_window)

    Positive D* indicates degradation (recent behavior worse than
    baseline); negative D* indicates improvement. Only positive drift
    exceeding ``drift_threshold`` triggers a flag, since we care about
    trust-building followed by exploitation.
    """

    def __init__(
        self,
        window_size: int = 20,
        baseline_size: int = 50,
        drift_threshold: float = 0.15,
    ):
        """
        Args:
            window_size: Number of recent interactions for drift calculation.
            baseline_size: Number of initial interactions for baseline.
            drift_threshold: D* above this flags the agent.
        """
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if baseline_size < 1:
            raise ValueError("baseline_size must be >= 1")
        if not 0.0 <= drift_threshold <= 1.0:
            raise ValueError("drift_threshold must be in [0, 1]")

        self.window_size = window_size
        self.baseline_size = baseline_size
        self.drift_threshold = drift_threshold

        # Per-agent history of p values
        self._history: Dict[str, deque[float]] = {}
        # Per-agent baseline (computed once enough data)
        self._baselines: Dict[str, float] = {}
        # Flagged agents
        self._flagged: Dict[str, float] = {}  # agent_id -> D*

    def record(self, agent_id: str, p: float) -> Optional[float]:
        """Record an interaction's p value for an agent.

        Args:
            agent_id: The agent identifier.
            p: Probability value in [0, 1].

        Returns the current drift rate D* if enough data, else None.
        Positive means degradation, negative means improvement.

        Raises:
            ValueError: If p is not in [0, 1] or is non-finite.
        """
        import math

        if not (0.0 <= p <= 1.0) or math.isnan(p) or math.isinf(p):
            raise ValueError(f"p must be a finite value in [0, 1], got {p!r}")

        if agent_id not in self._history:
            self._history[agent_id] = deque(
                maxlen=self.baseline_size + self.window_size
            )

        self._history[agent_id].append(p)
        history = self._history[agent_id]

        # Need at least baseline_size + window_size to compute drift
        if len(history) < self.baseline_size + self.window_size:
            return None

        # Compute baseline from first baseline_size entries
        if agent_id not in self._baselines:
            baseline_values = list(history)[: self.baseline_size]
            self._baselines[agent_id] = sum(baseline_values) / len(
                baseline_values
            )

        # Compute recent mean from last window_size entries
        recent = list(history)[-self.window_size :]
        recent_mean = sum(recent) / len(recent)

        # D* = baseline - recent (positive means degradation)
        drift = self._baselines[agent_id] - recent_mean

        if drift > self.drift_threshold:
            self._flagged[agent_id] = drift

        return drift

    def get_drift(self, agent_id: str) -> Optional[float]:
        """Get current drift rate for an agent."""
        history = self._history.get(agent_id)
        if not history or len(history) < self.baseline_size + self.window_size:
            return None

        baseline = self._baselines.get(agent_id)
        if baseline is None:
            return None

        recent = list(history)[-self.window_size :]
        recent_mean = sum(recent) / len(recent)
        return baseline - recent_mean

    def is_flagged(self, agent_id: str) -> bool:
        return agent_id in self._flagged

    def get_flagged_agents(self) -> Dict[str, float]:
        """Return all flagged agents and their drift rates."""
        return dict(self._flagged)

    def reset(self) -> None:
        self._history.clear()
        self._baselines.clear()
        self._flagged.clear()

    def reset_agent(self, agent_id: str) -> None:
        self._history.pop(agent_id, None)
        self._baselines.pop(agent_id, None)
        self._flagged.pop(agent_id, None)


# ── Compositionality ───────────────────────────────────────────────

@dataclass
class StageGuarantee:
    """Per-stage compliance guarantee for a pipeline."""

    stage_name: str
    p: float  # compliance probability
    delta: float  # invariant violation probability bound

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must be in [0,1], got {self.p}")
        if not 0.0 <= self.delta <= 1.0:
            raise ValueError(f"delta must be in [0,1], got {self.delta}")


@dataclass
class PipelineBound:
    """End-to-end compliance bound for a multi-stage pipeline."""

    p_pipeline: float  # Product of per-stage p values
    delta_pipeline: float  # Union bound on invariant violations
    n_stages: int
    stage_details: List[Dict[str, Any]] = field(default_factory=list)


def compute_pipeline_bound(stages: List[StageGuarantee]) -> PipelineBound:
    """Compute end-to-end compliance bounds from per-stage guarantees.

    Given per-stage contracts with compliance probability p_i and
    invariant violation bound delta_i:

    - p_pipeline = prod(p_i)  (all stages must comply)
    - delta_pipeline = min(1, sum(delta_i))  (union bound on violations)

    The union bound is a true upper bound on the probability of any
    violation, valid regardless of correlation between stages.

    Both degrade with more stages, making the case for fewer, stronger
    governance contracts over many weak ones.
    """
    if not stages:
        raise ValueError("Pipeline must have at least one stage")

    p_pipeline = 1.0
    delta_sum = 0.0

    for stage in stages:
        p_pipeline *= stage.p
        delta_sum += stage.delta

    delta_pipeline = min(1.0, delta_sum)

    return PipelineBound(
        p_pipeline=p_pipeline,
        delta_pipeline=delta_pipeline,
        n_stages=len(stages),
        stage_details=[
            {"name": s.stage_name, "p": s.p, "delta": s.delta} for s in stages
        ],
    )


def compute_pipeline_bound_with_drift(
    stages: List[StageGuarantee],
    drift_rate: float,
    time_steps: int,
) -> PipelineBound:
    """Compute pipeline bound accounting for behavioral drift.

    If drift causes p_i(t) = p_i(0) - D* * t, the pipeline bound
    degrades over time:

        p_pipeline(t) = prod(clamp(p_i(0) - D* * t, 0, 1))

    Args:
        stages: Per-stage guarantees (at t=0).
        drift_rate: D* — per-step degradation rate (must be >= 0).
        time_steps: Number of time steps to project (must be >= 0).

    Returns:
        PipelineBound at the given time horizon.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not stages:
        raise ValueError("Pipeline must have at least one stage")
    if drift_rate < 0.0:
        raise ValueError("drift_rate must be non-negative")
    if time_steps < 0:
        raise ValueError("time_steps must be non-negative")

    p_pipeline = 1.0
    delta_sum = 0.0
    details = []

    for stage in stages:
        p_t = min(1.0, max(0.0, stage.p - drift_rate * time_steps))
        p_pipeline *= p_t
        delta_sum += stage.delta
        details.append({
            "name": stage.stage_name,
            "p_0": stage.p,
            "p_t": p_t,
            "delta": stage.delta,
        })

    delta_pipeline = min(1.0, delta_sum)

    return PipelineBound(
        p_pipeline=p_pipeline,
        delta_pipeline=delta_pipeline,
        n_stages=len(stages),
        stage_details=details,
    )
